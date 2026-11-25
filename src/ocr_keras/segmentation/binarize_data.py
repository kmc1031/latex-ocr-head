"""AI-assisted preparation of shadowed source frames for TensorFlow binarization."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from ..annotations import stroke_masks
from ..components import preprocess
from ..io import read, save
from .restore_data import gray_image, source_rows
from .unet_data import evaluation_gray


def prepare(sources, dataset, output, seed, workers, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    summary = {'dataset': read(dataset/'summary.json')['fingerprint'], 'seed': seed,
               'degradation': 'before_background_and_scale', 'limit': limit, 'splits': {}}
    marker = output/'complete.json'
    if marker.is_file():
        previous = read(marker)
        if any(previous[k] != v for k, v in summary.items() if k != 'splits'):
            raise ValueError('Existing augmentation cache has different settings.')
        return

    def process(job):
        split, row, image_path, annotation = job
        with Image.open(image_path) as source:
            gray = gray_image(source)
        masks = stroke_masks(annotation, len(row['glyphs']), gray.shape[::-1])
        damaged = evaluation_gray(gray, row['id'], 'degraded', seed)
        image, _ = preprocess(Image.fromarray(damaged))
        packed = np.stack([np.packbits((np.asarray(mask.resize(
            image.size, Image.Resampling.LANCZOS)) > 100).ravel()) for mask in masks])
        target = output/split/(row['id']+'.npz')
        with target.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, image=np.asarray(image), gold_mask_bits=packed)
        target.with_suffix('.tmp').replace(target)
        return row['id']

    with ZipFile(sources/'crohme/crohme2014.zip') as archive, ThreadPoolExecutor(max_workers=workers) as pool:
        for split in ('train', 'val', 'test'):
            (output/split).mkdir(exist_ok=True)
            rows = list(source_rows(sources, dataset, split, limit))
            for start in range(0, len(rows), 3 * workers):
                batch = [(split, row, path, archive.read(row['source'])) for row, path in
                         rows[start:start+3*workers] if not (output/split/(row['id']+'.npz')).is_file()]
                list(pool.map(process, batch))
                if start % (max(1, 300//(3*workers))*3*workers) == 0:
                    print(f'[binarize data] {split} {min(start+3*workers, len(rows))}/{len(rows)}', flush=True)
            summary['splits'][split] = len(rows)
    save(marker, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3026)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.seed < 0 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Expected positive workers and limit, and nonnegative seed.')
    prepare(args.sources, args.dataset, args.output, args.seed, args.workers, args.limit)


if __name__ == '__main__':
    main()
