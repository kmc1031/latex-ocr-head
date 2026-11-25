"""Train, compare, and export U-Net-based symbol segmentation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit

from ..components import masked_patch, otsu, preprocess
from ..io import digest, read, save
from ..raw import raw_components
from .dataset import records
from .unet_data import entries, evaluation_gray, gold_masks, model_input, pad, stream
from .unet_inference import PixelMeasurement, decode_instances, diagnostic_targets

MODES = ('binary', 'gray', 'instance')


def train(args):
    import keras
    import tensorflow as tf

    from .unet_model import build, initialize_instance, segmentation_loss

    variants = check_variants(args)
    if (args.output / 'model.keras').exists() or (args.output / 'best.weights.h5').exists():
        raise ValueError('Choose a new output directory for this training run.')
    args.output.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy('mixed_float16')
    instance = args.mode == 'instance'
    model = build(args.width, instance)
    if args.init:
        source = args.init / 'model.keras'
        info = read(args.init / 'settings.json')
        if not instance or info['mode'] != 'gray' or info['width'] != args.width:
            raise ValueError('Warm start needs the gray U-Net of the same width in instance mode.')
        if info['dataset'] != read(args.dataset / 'summary.json')['fingerprint']:
            raise ValueError('Warm start must use the same prepared dataset.')
        initialize_instance(model, keras.models.load_model(source, compile=False))
    model.compile(optimizer=keras.optimizers.AdamW(args.learning_rate, weight_decay=1e-4),
                  loss=segmentation_loss(instance), jit_compile=False)
    if args.resume_weights:
        model.load_weights(args.resume_weights)
    channels = 6 if instance else 2

    cache_loader = None
    if args.tfrecord_cache:
        from .binarize_cache import load

        if args.mode != 'gray' or args.source_variants is None:
            raise ValueError('TFRecord caching is for gray mode with source variants.')
        cache_loader = load(args.tfrecord_cache, args.dataset, args.source_variants, args.batch,
                            args.augment, args.seed, args.train_limit, args.val_limit)

    def dataset(split, limit, fixed_degradation=None):
        if cache_loader is not None:
            return cache_loader(split, 'degraded' if fixed_degradation is not None else 'clean')
        rows = entries(args.dataset, split, limit)
        if not rows:
            raise ValueError(f'No {split} rows.')
        training = split == 'train'
        epoch = 0

        def iterator():
            nonlocal epoch
            seed = args.seed + epoch
            if training:
                epoch += 1
            return stream(args.dataset, split, rows, args.mode, seed,
                          args.augment if training else 0., training, fixed_degradation, args.source_variants)

        data = tf.data.Dataset.from_generator(iterator, output_signature=(
            tf.TensorSpec((None, None, 1), tf.float32),
            tf.TensorSpec((None, None, channels), tf.float32),
        ))
        data = data.padded_batch(args.batch, padded_shapes=((None, None, 1), (None, None, channels)))
        count = (len(rows) + args.batch - 1) // args.batch
        return data.apply(tf.data.experimental.assert_cardinality(count)), len(rows)

    training, train_count = dataset('train', args.train_limit)
    validation, val_count = dataset('val', args.val_limit)
    if args.validate_both:
        degraded, _ = dataset('val', args.val_limit, args.seed)
        validation = validation.concatenate(degraded)
    print(f'[unet] {args.mode}: {model.count_params():,} parameters, '
          f'{train_count} train / {val_count} validation formulas', flush=True)
    weights = args.output / 'best.weights.h5'
    class Progress(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.monotonic()
            self.epoch = epoch+1

        def on_train_batch_end(self, batch, logs=None):
            if batch == 0 or (batch+1) % args.progress_every == 0:
                print(f'[train] epoch={self.epoch} step={batch+1}/{self.params.get("steps")} '
                      f'elapsed={time.monotonic()-self.epoch_start:.1f}s loss={logs["loss"]:.5f}',
                      flush=True)

        def on_test_begin(self, logs=None):
            print(f'[validation] epoch={self.epoch} started', flush=True)

        def on_test_batch_end(self, batch, logs=None):
            if batch == 0 or (batch+1) % args.progress_every == 0:
                print(f'[validation] epoch={self.epoch} step={batch+1} loss={logs["loss"]:.5f}',
                      flush=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(weights, save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=.5, min_lr=1e-5),
        keras.callbacks.CSVLogger(args.output / 'history.csv'),
    ]
    if args.progress_every:
        callbacks.append(Progress())
    history = model.fit(training.prefetch(2), validation_data=validation.prefetch(2),
                        epochs=args.epochs, callbacks=callbacks, verbose=args.verbose, shuffle=False)
    model.load_weights(weights)
    model.save(args.output / 'model.keras')
    save(args.output / 'settings.json', {
        'mode': args.mode, 'width': args.width, 'seed': args.seed, 'augment': args.augment,
        'dataset': read(args.dataset / 'summary.json')['fingerprint'],
        'train_count': train_count, 'val_count': val_count,
        'init_sha256': digest(args.init / 'model.keras') if args.init else None,
        'learning_rate': args.learning_rate, 'history': history.history,
        'device': args.device, 'tensorflow': tf.__version__,
        'visible_gpus': [gpu.name for gpu in tf.config.get_visible_devices('GPU')],
        'validation_conditions': ['clean', 'degraded'] if args.validate_both else ['clean'],
        'mixed_precision': args.mixed_precision,
        'source_variants': variants,
        'tfrecord_cache': str(args.tfrecord_cache) if args.tfrecord_cache else None,
        'resume_weights': digest(args.resume_weights) if args.resume_weights else None,
    })


class Predictor:
    def __init__(self, directory):
        import keras
        import tensorflow as tf

        self.info = read(directory / 'settings.json')
        self.mode = self.info['mode']
        if self.mode not in MODES:
            raise ValueError(f'Unsupported U-Net mode: {self.mode}')
        self.model = keras.models.load_model(directory / 'model.keras', compile=False)
        if self.model.output_shape[-1] != (4 if self.mode == 'instance' else 1):
            raise ValueError('Checkpoint output does not match its segmentation mode.')

        @tf.function(input_signature=[tf.TensorSpec((1, None, None, 1), tf.float32)])
        def forward(batch):
            return self.model(batch, training=False)

        self.forward = forward

    def __call__(self, gray, binary):
        shape = gray.shape
        array = pad(model_input(gray, binary if self.mode == 'binary' else None))
        return self.forward(array[None]).numpy()[0, :shape[0], :shape[1]]


def decode(prediction, mode, merger, args):
    ink = expit(prediction[..., 0]) >= args.foreground
    if mode == 'instance':
        labels, groups = decode_instances(prediction, args.foreground, args.center, args.vote_radius)
    else:
        if merger is None:
            raise ValueError('Foreground-only modes require --merger.')
        labels, raw = raw_components(ink)
        groups = merger.groups(labels, raw)
    return ink, labels, groups


def check_variants(args):
    if args.source_variants is None:
        return None
    cache = read(args.source_variants / 'complete.json')
    if cache['dataset'] != read(args.dataset / 'summary.json')['fingerprint']:
        raise ValueError('Augmentation cache and dataset differ.')
    checks = (('train', args.train_limit), ('val', args.val_limit)) if args.command == 'train' else (
        (args.split, args.limit),)
    for split, limit in checks:
        if cache['splits'][split] < len(entries(args.dataset, split, limit)):
            raise ValueError(f'Not enough augmented {split} formulas.')
    return cache


def augmented_sample(args, entry, record):
    with np.load(args.source_variants / args.split / (entry['id']+'.npz'), allow_pickle=False) as variant:
        gray = variant['image']
        masks = gold_masks(variant)
    mask = otsu(gray) if gray.min() != gray.max() else np.zeros(gray.shape, dtype=bool)
    labels, raw = raw_components(mask)
    overlap = np.stack([np.bincount(labels[gold], minlength=int(labels.max())+1) for gold in masks], axis=1)
    overlap[0] = 0
    metadata = {'gold_areas': masks.sum(axis=(1, 2)), 'glyph_names': record['glyph_names'],
                'overlap': overlap, 'components': np.asarray([[g.ids[0], *g.box] for g in raw],
                                                            dtype=np.int32).reshape(-1, 5)}
    return gray, mask, labels, raw, masks, metadata


def evaluate(args):
    from ..merge import Merger

    variants = check_variants(args)
    merger = Merger(args.merger)
    predictors = {}
    dataset_id = read(args.dataset / 'summary.json')['fingerprint']
    for directory in args.model:
        predictor = Predictor(directory)
        if predictor.info['dataset'] != dataset_id and not args.allow_external_model:
            raise ValueError(f'Model and evaluation dataset differ: {directory}')
        if predictor.mode in predictors:
            raise ValueError(f'Only one checkpoint per mode is allowed: {predictor.mode}')
        predictors[predictor.mode] = (directory, predictor)
    conditions = ('clean', 'degraded') if args.condition == 'both' else (args.condition,)
    cutoffs = list(dict.fromkeys([args.foreground] + args.compare_foreground))
    if args.split == 'test' and len(cutoffs) != 1:
        raise ValueError('Select the foreground threshold on validation before testing.')
    merger.threshold = args.merge_threshold if args.merge_threshold is not None else merger.threshold
    names = {(mode, cutoff): mode if len(cutoffs) == 1 else f'{mode}@{cutoff:g}'
             for mode in predictors for cutoff in cutoffs}
    report = {}
    for condition in conditions:
        scores = {mode: PixelMeasurement() for mode in ('otsu', *names.values())}
        for index, (entry, record) in enumerate(records(args.dataset, args.split, args.limit), 1):
            if condition == 'degraded' and variants is not None:
                gray, mask, labels, raw, masks, metadata = augmented_sample(args, entry, record)
            else:
                from .dataset import groups_from_record

                gray = evaluation_gray(record['image'], entry['id'], condition, args.seed)
                mask = record['labels'] != 0 if condition == 'clean' else otsu(gray)
                labels, raw = ((record['labels'], groups_from_record(record)) if condition == 'clean'
                               else raw_components(mask))
                masks, metadata = gold_masks(record), record
            diagnostics = diagnostic_targets(metadata, masks)
            scores['otsu'].add_result(metadata, masks, mask, labels, merger.groups(labels, raw), diagnostics)
            for mode, (_, predictor) in predictors.items():
                prediction = predictor(gray, mask)
                for cutoff in cutoffs:
                    params = argparse.Namespace(foreground=cutoff, center=args.center,
                                                vote_radius=args.vote_radius)
                    ink, labels, groups = decode(prediction, mode, merger, params)
                    scores[names[mode, cutoff]].add_result(metadata, masks, ink, labels, groups, diagnostics)
            if index % 50 == 0:
                print(f'[unet] {args.split}/{condition}: {index} formulas', flush=True)
        report[condition] = {mode: score.result() for mode, score in scores.items()}
        print(json.dumps({mode: {key: values[key] for key in ('both80_f1', 'strict_formula_rate', 'ink_f1')}
                          for mode, values in report[condition].items()}, indent=2), flush=True)
    save(args.output, {'split': args.split, 'dataset': dataset_id, 'limit': args.limit,
                       'seed': args.seed, 'foreground': args.foreground, 'foreground_thresholds': cutoffs,
                       'center': args.center, 'merge_threshold': merger.threshold,
                       'vote_radius': args.vote_radius, 'merger': digest(args.merger),
                       'source_variants': variants,
                       'models': {mode: digest(directory / 'model.keras')
                                  for mode, (directory, _) in predictors.items()},
                       'model_datasets': {mode: predictor.info['dataset']
                                          for mode, (_, predictor) in predictors.items()},
                       'conditions': report})


def segment(args):
    from ..merge import Merger
    from .handoff import save_batch

    predictor = Predictor(args.model)
    if predictor.mode != 'instance' and args.merger is None:
        raise ValueError('Foreground-only modes require --merger.')
    paths = [path.resolve() for path in (args.image, args.output, args.batch_output) if path is not None]
    if len(set(paths)) != len(paths):
        raise ValueError('Image, JSON and symbol batch require separate paths.')
    merger = Merger(args.merger) if args.merger and predictor.mode != 'instance' else None
    with Image.open(args.image) as image:
        prepared, binary = preprocess(image)
    gray = np.asarray(prepared)
    _, labels, groups = decode(predictor(gray, binary), predictor.mode, merger, args)
    if args.batch_output:
        save_batch(args.batch_output, prepared, labels, groups, preserve_extent=True)
    segments = []
    for index, group in enumerate(groups):
        row = {'box': list(group.box), 'component_ids': list(group.ids)}
        if args.crops:
            args.crops.mkdir(parents=True, exist_ok=True)
            row['file'] = f'glyph_{index:03d}.png'
            masked_patch(prepared, labels, group).save(args.crops / row['file'])
        segments.append(row)
    save(args.output, {'image_size': list(prepared.size), 'coordinates': 'preprocessed_image_xyxy',
                       'mode': predictor.mode, 'segments': segments})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--dataset', required=True, type=Path)
    training.add_argument('--mode', required=True, choices=MODES)
    training.add_argument('--epochs', type=int, default=20)
    training.add_argument('--width', type=int, default=16)
    training.add_argument('--batch', type=int, default=4)
    training.add_argument('--train-limit', type=int)
    training.add_argument('--val-limit', type=int)
    training.add_argument('--learning-rate', type=float, default=3e-4)
    training.add_argument('--augment', type=float, default=.5)
    training.add_argument('--init', type=Path)
    training.add_argument('--mixed-precision', action='store_true')
    training.add_argument('--validate-both', action='store_true')
    training.add_argument('--verbose', choices=(1, 2), type=int, default=1)
    training.add_argument('--source-variants', type=Path)
    training.add_argument('--tfrecord-cache', type=Path)
    training.add_argument('--progress-every', type=int, default=0)
    training.add_argument('--resume-weights', type=Path)
    assessment = commands.add_parser('evaluate')
    assessment.add_argument('--dataset', required=True, type=Path)
    assessment.add_argument('--model', action='append', default=[], type=Path)
    assessment.add_argument('--merger', required=True, type=Path)
    assessment.add_argument('--split', choices=('val', 'test'), default='val')
    assessment.add_argument('--condition', choices=('clean', 'degraded', 'both'), default='both')
    assessment.add_argument('--limit', type=int)
    assessment.add_argument('--compare-foreground', action='append', default=[], type=float)
    assessment.add_argument('--merge-threshold', type=float)
    assessment.add_argument('--source-variants', type=Path)
    assessment.add_argument('--allow-external-model', action='store_true',
                            help='explicitly evaluate a model trained on a different dataset')
    inference = commands.add_parser('segment')
    inference.add_argument('image', type=Path)
    inference.add_argument('--model', required=True, type=Path)
    inference.add_argument('--merger', type=Path)
    inference.add_argument('--crops', type=Path)
    inference.add_argument('--batch-output', type=Path)
    for command in (training, assessment, inference):
        command.add_argument('--output', required=True, type=Path)
        command.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
        command.add_argument('--seed', type=int, default=2026)
    for command in (assessment, inference):
        command.add_argument('--foreground', type=float, default=.5)
        command.add_argument('--center', type=float, default=.25)
        command.add_argument('--vote-radius', type=float, default=12.)
    args = parser.parse_args()
    for name in ('width', 'epochs', 'batch', 'limit', 'train_limit', 'val_limit',
                 'learning_rate', 'vote_radius'):
        value = getattr(args, name, None)
        if value is not None and not value > 0:
            parser.error(f'--{name.replace("_", "-")} must be positive.')
    for name in ('augment', 'foreground', 'center'):
        value = getattr(args, name, None)
        if value is not None and not 0 <= value <= 1:
            parser.error(f'--{name} must be in [0, 1].')
    if args.seed < 0:
        parser.error('--seed must not be negative.')
    if getattr(args, 'progress_every', 0) < 0:
        parser.error('--progress-every must not be negative.')
    if any(not 0 < value < 1 for value in getattr(args, 'compare_foreground', [])):
        parser.error('--compare-foreground values must be in (0, 1).')
    if getattr(args, 'merge_threshold', None) is not None and not 0 < args.merge_threshold <= 1:
        parser.error('--merge-threshold must be in (0, 1].')
    from ..device import configure_device

    configure_device(args.device)
    {'train': train, 'evaluate': evaluate, 'segment': segment}[args.command](args)


if __name__ == '__main__':
    main()
