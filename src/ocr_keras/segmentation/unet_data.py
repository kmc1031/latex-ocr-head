"""Pixel supervision and reproducible scan degradation for the U-Net experiment."""

from pathlib import Path
import zlib

import numpy as np
from scipy import ndimage

from ..components import otsu
from ..io import read, resolve

CONNECTIVITY = np.ones((3, 3), dtype=bool)
OFFSET_SCALE = 48.


def gold_masks(record):
    height, width = record['image'].shape
    return np.unpackbits(record['gold_mask_bits'], axis=1, count=height * width).reshape(
        -1, height, width).astype(bool)


def targets(gold, instance):
    height, width = gold.shape[1:]
    foreground = gold.any(axis=0)
    weights = np.ones((height, width), dtype=np.float32)
    if instance:
        heat = np.zeros((height, width), dtype=np.float32)
        dx = np.zeros_like(heat)
        dy = np.zeros_like(heat)
        single_owner = gold.sum(axis=0) == 1
    for mask in gold:
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
        region = np.s_[top:bottom + 1, left:right + 1]
        components, count = ndimage.label(mask[region], CONNECTIVITY)
        areas = np.bincount(components.ravel(), minlength=count + 1)
        boost = np.clip(np.sqrt(24 / np.maximum(1, areas)), 1, 4)
        boost[0] = 1
        np.maximum(weights[region], boost[components], out=weights[region])
        if instance:
            cy, cx = (top + bottom) // 2, (left + right) // 2
            ys, xs = np.nonzero(mask & single_owner)
            dx[ys, xs] = (cx - xs) / OFFSET_SCALE
            dy[ys, xs] = (cy - ys) / OFFSET_SCALE
            radius = min(6, max(2, round(.12 * max(bottom - top + 1, right - left + 1))))
            y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
            dist = (np.arange(y0, y1)[:, None] - cy)**2 + (np.arange(x0, x1) - cx)**2
            gaussian = np.exp(-dist / (2 * (radius / 2)**2))
            window = heat[y0:y1, x0:x1]
            np.maximum(window, gaussian, out=window)
    channels = (foreground, weights, heat, dx, dy, single_owner) if instance else (foreground, weights)
    return np.stack(channels, axis=-1).astype(np.float32)


def degrade(image, rng):
    ink = (255 - image.astype(np.float32)) / 255.
    ink = ndimage.gaussian_filter(ink, float(rng.uniform(.25, 1.3)))
    height, width = image.shape
    ys = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    shade = np.exp(-((ys - rng.uniform(-1, 1))**2 + (xs - rng.uniform(-1, 1))**2)
                   / rng.uniform(.15, 1.1))
    page = (rng.uniform(225, 254) + rng.uniform(-29, 29) * xs + rng.uniform(-20, 20) * ys
            - rng.uniform(10, 55) * shade)
    gray = page - rng.uniform(65, 208) * ink**rng.uniform(.75, 1.4)
    gray += rng.normal(0, rng.uniform(.4, 4.5), image.shape)
    return np.rint(np.clip(gray, 0, 255)).astype(np.uint8)


def evaluation_gray(image, item_id, condition, seed):
    if condition == 'clean':
        return image
    rng = np.random.default_rng([seed, zlib.crc32(item_id.encode('utf-8'))])
    return degrade(image, rng)


def model_input(gray, binary=None):
    ink = (255 - gray.astype(np.float32)) / 255 if binary is None else binary.astype(np.float32)
    return ink[..., None]


def pad(array):
    height, width = array.shape[:2]
    return np.pad(array, ((0, -height % 8), (0, -width % 8), (0, 0)))


def entries(dataset, split, limit):
    return read(Path(dataset) / f'{split}.json')[:limit]


def stream(dataset, split, rows, mode, seed, augment, shuffle, fixed_degradation=None,
           source_variants=None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows)) if shuffle else range(len(rows))
    for index in order:
        row = rows[index]
        use_augmentation = fixed_degradation is not None or rng.random() < augment
        root = source_variants if use_augmentation and source_variants is not None else dataset
        with np.load(resolve(root / split, row['id'] + '.npz'), allow_pickle=False) as record:
            gray = record['image']
            if root == dataset:
                if fixed_degradation is not None:
                    gray = evaluation_gray(gray, row['id'], 'degraded', fixed_degradation)
                elif use_augmentation:
                    gray = degrade(gray, rng)
            binary = (otsu(gray) if use_augmentation else record['labels'] != 0) if mode == 'binary' else None
            truth = targets(gold_masks(record), mode == 'instance')
        yield pad(model_input(gray, binary)), pad(truth)
