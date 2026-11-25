"""Prepare, train, evaluate, and run the segmentation stage without OCR."""

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..components import masked_patch, preprocess
from ..io import digest, read, save
from ..raw import pair_input, raw_components
from .dataset import groups_from_record, prepare, records
from .metrics import Measurement, owner_oracle


def patch(labels, first, second):
    return np.rint(pair_input(labels, first, second).transpose(1, 2, 0) * 255).astype(np.uint8)


def _gap(first, second):
    x0, y0, x1, y1 = first.box
    u0, v0, u1, v1 = second.box
    return max(0, x0-u1, u0-x1)**2 + max(0, y0-v1, v0-y1)**2


def _examples(dataset, split, limit, negative_cap=None, seed=2026):
    rng = np.random.default_rng(seed)
    for _, record in records(dataset, split, limit):
        by_id = {group.ids[0]: group for group in groups_from_record(record)}
        pairs = record['pairs']
        pairs = pairs[(pairs[:, 2] >= 0) & (pairs[:, 3] == 1)]
        if negative_cap is not None:
            positives = pairs[pairs[:, 2] == 1]
            negatives = pairs[pairs[:, 2] == 0]
            if len(negatives) > negative_cap:
                dist = [_gap(by_id[a], by_id[b]) for a, b, _, _ in negatives]
                order = np.argsort(dist, kind='stable')
                hard = (negative_cap + 1) // 2
                choice = np.r_[order[:hard], rng.choice(order[hard:], negative_cap-hard, replace=False)]
                negatives = negatives[choice]
            pairs = np.concatenate((positives, negatives))
        if len(pairs):
            labels = record['labels']
            yield labels, by_id, pairs


def _sampled(dataset, split, limit, negative_cap, seed):
    patches, targets = [], []
    for labels, by_id, pairs in _examples(dataset, split, limit, negative_cap, seed):
        patches.append(np.stack([patch(labels, by_id[a], by_id[b]) for a, b, _, _ in pairs]))
        targets.append(pairs[:, 2].astype(np.float32))
    if not targets:
        raise ValueError(f'{split}: no eligible labelled pairs')
    y = np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError(f'{split}: both positive and negative pairs required')
    return np.concatenate(patches), y[:, None]


def _batches(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _validation_scores(net, dataset, limit, batch_size):
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec((None, 48, 48, 2), tf.uint8)])
    def predict(images):
        normalized = tf.cast(images, tf.float32) / 255.
        paired = tf.concat((normalized, tf.reverse(normalized, [-1])), 0)
        left, right = tf.split(tf.sigmoid(net(paired, training=False))[:, 0], 2)
        return (left + right) / 2

    examples = ((patch(labels, by_id[a], by_id[b]), target)
                for labels, by_id, pairs in _examples(dataset, 'val', limit)
                for a, b, target, _ in pairs)
    scores, targets = [], []
    for batch in _batches(examples, batch_size):
        images, ys = zip(*batch)
        scores.extend(predict(np.stack(images)).numpy())
        targets.extend(ys)
    return np.asarray(scores), np.asarray(targets)


def calibration(scores, targets, min_precision):
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError('Calibration requires both classes.')
    values = []
    for threshold in np.linspace(.5, .999, 100):
        positive = scores >= threshold
        true = int(np.count_nonzero(positive & (targets == 1)))
        false = int(np.count_nonzero(positive & (targets == 0)))
        precision = true / max(1, true + false)
        if true and precision >= min_precision:
            values.append((true / int(targets.sum()), precision, float(threshold), true, false))
    best = max(values) if values else (0., None, 1.01, 0, 0)
    return dict(zip(('val_recall', 'val_precision', 'threshold', 'tp', 'fp'), best)) | {
        'test_used_for_selection': False, 'eligible_val_pairs': len(targets),
        'required_precision': min_precision,
    }


def train(args):
    import keras
    import tensorflow as tf

    from ..merge import network

    if (args.output / 'model.keras').exists():
        raise ValueError('Choose a new output directory to preserve existing checkpoints.')
    args.output.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    net = keras.models.load_model(args.init, compile=False) if args.init else network()
    if net.input_shape[1:] != (48, 48, 2) or net.output_shape[1:] != (1,):
        raise ValueError('Initial checkpoint must be a component-pair merger.')
    rate = args.learning_rate or (1e-4 if args.init else 1e-3)
    net.compile(optimizer=keras.optimizers.AdamW(rate, weight_decay=1e-4),
                loss=keras.losses.BinaryCrossentropy(from_logits=True))

    def tensors(split, augment):
        print(f'[train] loading {split} pairs', flush=True)
        features, targets = _sampled(args.dataset, split, args.limit, args.negatives, args.seed)
        count = len(targets)
        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        if augment:
            ds = ds.shuffle(min(2048, count), seed=args.seed)
        ds = ds.batch(args.batch)

        def normalize(images, labels):
            if augment:
                swap = tf.random.uniform([tf.shape(images)[0], 1, 1, 1]) < .5
                images = tf.where(swap, tf.reverse(images, [-1]), images)
            return tf.cast(images, tf.float32) / 255., labels

        return ds.map(normalize, num_parallel_calls=2).prefetch(1), count

    train_data, train_count = tensors('train', True)
    val_data, val_count = tensors('val', False)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    history = net.fit(train_data, validation_data=val_data, epochs=args.epochs,
                      callbacks=[early], verbose=1, shuffle=False)
    print('[train] calibrating on all eligible validation pairs', flush=True)
    scores, targets = _validation_scores(net, args.dataset, args.limit, args.batch)
    save(args.output / 'calibration.json', calibration(scores, targets, args.min_precision))
    net.save(args.output / 'model.keras')
    save(args.output / 'training.json', {
        'dataset': read(args.dataset / 'summary.json')['fingerprint'], 'split_limit': args.limit,
        'train_pairs': train_count, 'validation_sampled_pairs': val_count, 'seed': args.seed,
        'init_sha256': digest(args.init) if args.init else None, 'learning_rate': rate,
        'negative_cap_per_formula': args.negatives, 'history': history.history,
    })


def _reviewer(merger, checkpoint, rollback):
    from ..config import Settings
    from .rollback import ReviewedMerger

    policy = read(rollback)
    if policy['merge_sha256'] != digest(checkpoint):
        raise ValueError('The rollback policy requires the original merge checkpoint.')
    return ReviewedMerger(merger, policy, Settings().max_symbols)


def evaluate(dataset, split, checkpoints, batch_size, limit, rollback=None):
    from ..merge import Merger

    pipelines = {str(i): Merger(path) for i, path in enumerate(checkpoints, 1)}
    reviewer = _reviewer(pipelines['1'], checkpoints[0], rollback) if rollback else None
    variants = [*pipelines, *(['1_rollback'] if reviewer else [])]
    measurements = {name: Measurement() for name in ('raw', *variants, 'owner_oracle')}
    mistakes, comparisons = [], {name: Counter() for name in variants}
    files = read(dataset / f'{split}.json')[:limit]

    def items():
        keys = ('labels', 'components', 'owner', 'overlap', 'gold_areas', 'glyph_names', 'component_areas')
        for entry, record in records(dataset, split, limit):
            yield entry, {key: record[key] for key in keys}

    done = 0
    for batch in _batches(items(), batch_size):
        inputs = [(record['labels'], groups_from_record(record)) for _, record in batch]
        predictions = {name: pipeline.groups_many(inputs) for name, pipeline in pipelines.items()}
        if reviewer:
            predictions['1_rollback'] = [reviewer.review(labels, raw, predicted)
                                         for (labels, raw), predicted in zip(inputs, predictions['1'])]
        for i, ((entry, record), (_, raw)) in enumerate(zip(batch, inputs)):
            result = {'raw': measurements['raw'].add(record, raw),
                      'owner_oracle': measurements['owner_oracle'].add(record, owner_oracle(record, raw))}
            for name in variants:
                result[name] = measurements[name].add(record, predictions[name][i])
                delta = result[name]['matched_both80'] - result['raw']['matched_both80']
                comparisons[name].update(improved=int(delta > 0), worsened=int(delta < 0),
                                         unchanged=int(delta == 0))
            if len(mistakes) < 100 and any(row['gold'] != row['matched_both80'] or
                                          row['predicted'] != row['gold'] for row in result.values()):
                mistakes.append({'id': entry['id'], 'source': entry['source'], 'counts': result})
        done += len(batch)
        if done % (batch_size * 10) == 0 or done == len(files):
            print(f'[evaluate] {split} {done}/{len(files)}', flush=True)
    return {'split': split, 'limit': limit, 'dataset': read(dataset / 'summary.json')['fingerprint'],
            'rollback': {'path': str(rollback), 'sha256': digest(rollback)} if rollback else None,
            'checkpoints': {name: {'path': str(path), 'sha256': digest(path), 'threshold': pipelines[name].threshold}
                            for name, path in zip(pipelines, checkpoints)},
            'metrics': {name: measure.result() for name, measure in measurements.items()},
            'against_raw': comparisons, 'sample_errors': mistakes}


def segment(path, checkpoint, out, crops, rollback=None, batch_output=None):
    from ..merge import Merger

    if batch_output is not None and len({batch_output.resolve(), out.resolve(), path.resolve()}) != 3:
        raise ValueError('Image, JSON, and symbol batch must use different paths.')
    with Image.open(path) as source:
        image, foreground = preprocess(source)
    labels, raw = raw_components(foreground)
    merger = Merger(checkpoint)
    if rollback:
        merger = _reviewer(merger, checkpoint, rollback)
    groups = merger.groups(labels, raw)
    if crops:
        crops.mkdir(parents=True, exist_ok=True)
    segments = []
    for idx, group in enumerate(groups):
        row = {'box': list(group.box), 'component_ids': list(group.ids)}
        if crops:
            name = f'glyph_{idx:03d}.png'
            masked_patch(image, labels, group).save(crops / name)
            row['file'] = name
        segments.append(row)
    if batch_output is not None:
        from .handoff import save_batch

        save_batch(batch_output, image, labels, groups)
    save(out, {'image_size': list(image.size), 'coordinates': 'preprocessed_image_xyxy',
               'raw_components': len(raw), 'segments': segments})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--sources', required=True, type=Path)
    prep.add_argument('--output', required=True, type=Path)
    training = commands.add_parser('train')
    training.add_argument('--dataset', required=True, type=Path)
    training.add_argument('--output', required=True, type=Path)
    training.add_argument('--init', type=Path)
    training.add_argument('--epochs', type=int, default=12)
    training.add_argument('--batch', type=int, default=128)
    training.add_argument('--negatives', type=int, default=16)
    training.add_argument('--seed', type=int, default=2026)
    training.add_argument('--learning-rate', type=float)
    training.add_argument('--min-precision', type=float, default=.85,
                          help='minimum validation pair precision; compare final masks separately')
    ev = commands.add_parser('evaluate')
    ev.add_argument('--dataset', required=True, type=Path)
    ev.add_argument('--checkpoint', type=Path, action='append', required=True)
    ev.add_argument('--split', choices=('val', 'test'), default='val')
    ev.add_argument('--output', required=True, type=Path)
    ev.add_argument('--batch', type=int, default=16)
    infer = commands.add_parser('segment')
    infer.add_argument('image', type=Path)
    infer.add_argument('--checkpoint', required=True, type=Path)
    infer.add_argument('--output', required=True, type=Path)
    infer.add_argument('--crops', type=Path)
    infer.add_argument('--batch-output', type=Path, help='save the NCHW batch for the 92-class CNN')
    for command in (prep, training, ev):
        command.add_argument('--limit', type=int)
    for command in (ev, infer):
        command.add_argument('--rollback', type=Path)
    for command in (training, ev, infer):
        command.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    for name in ('limit', 'batch', 'epochs', 'negatives', 'learning_rate'):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if not 0 < getattr(args, 'min_precision', 1) <= 1:
        parser.error('--min-precision must be in (0, 1]')
    if args.command == 'prepare':
        print(json.dumps(prepare(args.sources, args.output, args.limit), ensure_ascii=False, indent=2))
        return
    from ..device import configure_device

    configure_device(args.device)
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        save(args.output, evaluate(args.dataset, args.split, args.checkpoint, args.batch, args.limit,
                                   args.rollback))
    else:
        segment(args.image, args.checkpoint, args.output, args.crops, args.rollback,
                args.batch_output)


if __name__ == '__main__':
    main()
