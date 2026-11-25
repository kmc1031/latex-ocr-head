"""AI-assisted TFRecord input cache for gray foreground training."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..io import digest, read, resolve, save
from .unet_data import entries, gold_masks, targets

VERSION = 1
GRID = (64, 256)


def _read_image(path):
    with np.load(path, allow_pickle=False) as stored:
        gray = stored['image']
        gold = targets(gold_masks(stored), instance=False)
    return gray, gold


def _encode(gray, gold, height, width):
    h, w = gray.shape
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:h, :w, 0] = 255-gray
    array[:h, :w, 1] = gold[..., 0]
    array[:h, :w, 2] = np.rint(gold[..., 1]*32)
    return array.tobytes()


def build(dataset, variants, output, train_limit=None, val_limit=None):
    dataset_id = read(dataset/'summary.json')['fingerprint']
    variant_id = digest(variants/'complete.json')
    variant_info = read(variants/'complete.json')
    if variant_info['dataset'] != dataset_id:
        raise ValueError('Augmentation cache and dataset differ.')
    info = dict(version=VERSION, dataset=dataset_id, variants=variant_id,
                train_limit=train_limit, val_limit=val_limit)
    marker = output/'complete.json'
    if marker.is_file():
        previous = read(marker)
        if all(previous[key] == value for key, value in info.items()):
            return
        raise ValueError('Choose a new TFRecord directory for different inputs.')
    output.mkdir(parents=True, exist_ok=True)
    info['splits'] = {}
    for split, limit in (('train', train_limit), ('val', val_limit)):
        rows = entries(dataset, split, limit)
        if not rows or variant_info['splits'][split] < len(rows):
            raise ValueError(f'Missing source variants for {split}.')
        groups = Counter()
        files = []
        for start in range(0, len(rows), 256):
            file = output/f'{split}-{start//256:03d}.tfrecord'
            with tf.io.TFRecordWriter(str(file.with_suffix('.tmp'))) as writer:
                for row in rows[start:start+256]:
                    name = row['id']+'.npz'
                    clean = _read_image(resolve(dataset/split, name))
                    damaged = _read_image(resolve(variants/split, name))
                    height, width = (max(a, b) for a, b in zip(clean[0].shape, damaged[0].shape))
                    height, width = (height + (-height % GRID[0]), width + (-width % GRID[1]))
                    groups[(height, width)] += 1
                    payloads = {'clean': _encode(*clean, height, width),
                                'degraded': _encode(*damaged, height, width)}
                    fields = {k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
                              for k, v in payloads.items()}
                    fields['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[height, width]))
                    writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
            file.with_suffix('.tmp').replace(file)
            files.append(file.name)
            print(f'[tfrecord] {split} {min(start+256, len(rows))}/{len(rows)}', flush=True)
        info['splits'][split] = dict(count=len(rows), files=files,
                                    bucket_counts=[[h, w, n] for (h, w), n in sorted(groups.items())])
    save(marker, info)


def load(cache, dataset, variants, batch, augment, seed, train_limit, val_limit):
    info = read(cache/'complete.json')
    if (info['version'] != VERSION or info['dataset'] != read(dataset/'summary.json')['fingerprint']
            or info['variants'] != digest(variants/'complete.json')
            or info['train_limit'] != train_limit or info['val_limit'] != val_limit):
        raise ValueError('TFRecord cache does not match this training configuration.')
    schema = {'shape': tf.io.FixedLenFeature([2], tf.int64),
              'clean': tf.io.FixedLenFeature([], tf.string),
              'degraded': tf.io.FixedLenFeature([], tf.string)}

    def make(split, condition):
        meta = info['splits'][split]
        training = split == 'train'
        data = tf.data.TFRecordDataset([str(resolve(cache, path)) for path in meta['files']],
                                       num_parallel_reads=4 if training else 1)
        if training:
            data = data.shuffle(384, seed=seed, reshuffle_each_iteration=True)

        def select(encoded):
            parsed = tf.io.parse_single_example(encoded, schema)
            if training:
                data = tf.cond(tf.random.uniform((), seed=seed) < augment,
                               lambda: parsed['degraded'], lambda: parsed['clean'])
            else:
                data = parsed[condition]
            return parsed['shape'], data

        data = data.map(select, num_parallel_calls=4, deterministic=True)
        data = data.group_by_window(
            key_func=lambda shape, encoded: shape[0]*4096+shape[1],
            reduce_func=lambda key, group: group.batch(batch), window_size=batch)

        def decode(shape, encoded):
            pixels = tf.reshape(tf.io.decode_raw(encoded, tf.uint8), [-1, shape[0, 0], shape[0, 1], 3])
            inp = tf.cast(pixels[..., :1], tf.float32)/255.
            return inp, tf.stack((tf.cast(pixels[..., 1], tf.float32),
                                  tf.cast(pixels[..., 2], tf.float32)/32.), axis=-1)

        count = sum((n+batch-1)//batch for _, _, n in meta['bucket_counts'])
        data = data.map(decode, num_parallel_calls=4)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 10
        options.threading.max_intra_op_parallelism = 1
        return data.with_options(options).apply(tf.data.experimental.assert_cardinality(count)), meta['count']

    return make


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'source-variants', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--val-limit', type=int)
    args = parser.parse_args()
    build(args.dataset, args.source_variants, args.output, args.train_limit, args.val_limit)


if __name__ == '__main__':
    main()
