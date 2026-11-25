"""Run the final classical and TensorFlow U-Net symbol segmentation paths."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..device import configure_device
from ..config import PROJECT
from ..io import digest, read, resolve, save
from .dataset import records
from .unet_data import entries, gold_masks
from .unet_inference import PixelMeasurement, diagnostic_targets

DEFAULT_MERGE = PROJECT/'runs/keras_v6/merge/model.keras'
DEFAULT_ASSETS = PROJECT/'resources/segmentation'


def build_pipelines(args, modes):
    from ..merge import Merger
    from .pipeline import GeometryMerger, SegmentationPipeline

    geometry = args.profile == 'rendered'
    merger = GeometryMerger(args.geometry) if geometry else Merger(args.merger)
    foreground = args.foreground
    if foreground is None and 'unet' in modes:
        foreground = read(args.unet/'selection.json')['foreground']
    cutoff = args.gray_threshold
    if cutoff is None and geometry and not args.otsu:
        cutoff = 180
    return {mode: SegmentationPipeline(mode, merger, unet=args.unet, foreground=foreground or .65,
                                       gray_threshold=cutoff) for mode in modes}


def segment(args):
    destination = args.output
    if destination.suffix.lower() != '.npz' or destination.resolve() == args.image.resolve():
        raise ValueError('Use a separate .npz output file.')
    pipeline = build_pipelines(args, [args.method])[args.method]
    with Image.open(args.image) as source:
        batch, ink, _, groups = pipeline(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as stream:
        np.savez_compressed(stream, **batch)
    if args.debug:
        args.debug.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).save(args.debug/'ink.png')
        save(args.debug/'segments.json', {'method': args.method, 'profile': args.profile,
                                         'image_size': batch['image_size'].tolist(),
                                         'groups': [{'ids': list(map(int, g.ids)), 'box': list(map(int, g.box))}
                                                    for g in groups]})
    print(f'{args.method}/{args.profile}: {len(groups)} symbols → {destination}')


def evaluate(args):
    variants = read(args.variants/'complete.json') if args.variants else None
    fingerprint = read(args.dataset/'summary.json')['fingerprint']
    count = len(entries(args.dataset, args.split, args.limit))
    if not count:
        raise ValueError('No evaluation formulas.')
    if variants and (variants['dataset'] != fingerprint or variants['splits'][args.split] < count):
        raise ValueError('Augmentation data must match the evaluation split.')
    pipelines = build_pipelines(args, ('classical', 'unet'))
    results = {}
    for condition in ('clean', 'degraded') if args.variants else ('clean',):
        scores = {method: PixelMeasurement() for method in pipelines}
        for index, (entry, original) in enumerate(records(args.dataset, args.split, args.limit), 1):
            if condition == 'clean':
                row = {key: original[key] for key in ('image', 'gold_mask_bits', 'glyph_names',
                                                      'gold_areas', 'overlap', 'components')}
            else:
                with np.load(resolve(args.variants/args.split, entry['id']+'.npz'), allow_pickle=False) as item:
                    row = {key: item[key] for key in ('image', 'gold_mask_bits')}
                    row['glyph_names'] = original['glyph_names']
                gold = gold_masks(row)
                row.update(gold_areas=gold.sum((1, 2)), overlap=np.zeros((1, len(gold)), np.int32),
                           components=np.zeros((0, 5), np.int32))
            gold = gold_masks(row)
            diagnostics = diagnostic_targets(row, gold)
            for method, pipeline in pipelines.items():
                prediction = pipeline.on_gray(row['image'])
                scores[method].add_result(row, gold, *prediction, diagnostics)
            if index % 50 == 0:
                print(f'{condition}: {index} formulas', flush=True)
        results[condition] = {method: score.result() for method, score in scores.items()}
        for method, score in results[condition].items():
            print(f'{condition}/{method}: F1 {score["both80_f1"]:.2%}; '
                  f'whole {score["all_glyphs_both80"]}/{score["formulas"]}', flush=True)
    save(args.output, {'profile': args.profile, 'split': args.split, 'limit': args.limit,
                       'dataset': fingerprint, 'conditions': results,
                       'unet_training_dataset': pipelines['unet'].predict.info['dataset'],
                       'unet_sha256': digest(args.unet/'model.keras'),
                       'merger_sha256': digest(args.geometry if args.profile == 'rendered' else args.merger),
                       'merge_threshold': pipelines['classical'].merger.threshold,
                       'foreground': args.foreground if args.foreground is not None
                       else read(args.unet/'selection.json')['foreground'],
                       'classical_gray_threshold': pipelines['classical'].gray_threshold})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    infer = subcommands.add_parser('segment')
    infer.add_argument('image', type=Path)
    infer.add_argument('--method', choices=('classical', 'unet'), default='classical')
    infer.add_argument('--debug', type=Path)
    compare = subcommands.add_parser('evaluate')
    compare.add_argument('--dataset', required=True, type=Path)
    compare.add_argument('--variants', type=Path)
    compare.add_argument('--split', choices=('val', 'test'), default='val')
    compare.add_argument('--limit', type=int)
    for sub in (infer, compare):
        sub.add_argument('--profile', choices=('handwriting', 'rendered'), default='handwriting')
        sub.add_argument('--merger', type=Path, default=DEFAULT_MERGE)
        sub.add_argument('--geometry', type=Path, default=DEFAULT_ASSETS/'rendered_merge.json')
        sub.add_argument('--unet', type=Path, default=DEFAULT_ASSETS/'unet')
        sub.add_argument('--foreground', type=float)
        classical = sub.add_mutually_exclusive_group()
        classical.add_argument('--gray-threshold', type=int)
        classical.add_argument('--otsu', action='store_true', help='use Otsu even in the rendered profile')
        sub.add_argument('--output', required=True, type=Path)
        sub.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    if getattr(args, 'limit', None) is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.foreground is not None and not 0 < args.foreground < 1:
        parser.error('--foreground must be in (0, 1).')
    if args.gray_threshold is not None and not 0 < args.gray_threshold <= 255:
        parser.error('--gray-threshold must be in [1, 255].')
    configure_device(args.device)
    {'segment': segment, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
