"""Paired source images for the illumination experiment, before any OCR preprocessing."""

import json

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage

from ..components import otsu
from ..io import read, resolve


def gray_image(source):
    source = ImageOps.exif_transpose(source).convert('RGBA')
    page = Image.new('RGBA', source.size, 'white')
    page.alpha_composite(source)
    result = page.convert('L')
    result.thumbnail((1600, 1600))
    return np.asarray(result)


def conventional(gray):
    image = Image.fromarray(gray)
    background = np.asarray(image.filter(ImageFilter.GaussianBlur(max(8, .05 * min(image.size)))),
                            dtype=np.float32)
    return np.clip(255 * gray.astype(np.float32) / (background + 1), 0, 255).astype(np.uint8)


def size_and_mask(gray):
    labels, _ = ndimage.label(otsu(gray), np.ones((3, 3)))
    heights = [region[0].stop - region[0].start for region in ndimage.find_objects(labels)
               if region and region[0].stop - region[0].start > 3
               and region[1].stop - region[1].start < 4 * (region[0].stop - region[0].start)]
    height = float(np.percentile(heights, 70)) if heights else max(1, gray.shape[0])
    width, rows = gray.shape[1], gray.shape[0]
    scale = min(48 / height, 1024 / width, 512 / rows)
    image = Image.fromarray(gray).resize((max(1, round(width * scale)), max(1, round(rows * scale))),
                                         Image.Resampling.LANCZOS)
    pixels = np.asarray(image)
    return image, otsu(pixels) if pixels.min() != pixels.max() else np.zeros(pixels.shape, dtype=bool)


def source_rows(sources, dataset, split, limit=None):
    directory = sources / 'crohme'
    rows = {row['id']: row for row in map(json.loads, (directory / f'{split}.jsonl').read_text(
        encoding='utf-8-sig').splitlines())}
    for entry in read(dataset / f'{split}.json')[:limit]:
        row = rows[entry['id']]
        if row['ink_hash'] != entry['ink_hash'] or row['source'] != entry['source']:
            raise ValueError(f'Prepared source mismatch: {entry["id"]}')
        yield row, resolve(directory, row['image'])


def patches(sources, dataset, split, count, seed, side=96):
    rng = np.random.default_rng(seed)
    source = list(source_rows(sources, dataset, split))
    if not source:
        raise ValueError(f'Empty source split: {split}')
    indexes = []
    while len(indexes) < count:
        indexes.extend(rng.permutation(len(source)).tolist())
    result = np.full((count, 1, side, side), 255, dtype=np.uint8)
    for out, index in enumerate(indexes[:count]):
        with Image.open(source[index][1]) as image:
            gray = gray_image(image)
        pad_y, pad_x = max(0, side-gray.shape[0]), max(0, side-gray.shape[1])
        gray = np.pad(gray, ((pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2)),
                      constant_values=255)
        height, width = gray.shape
        ys, xs = np.nonzero(gray < 150)
        if len(xs) and rng.random() < .90:
            pick = rng.integers(len(xs))
            top = int(np.clip(ys[pick]-rng.integers(side//4, 3*side//4), 0, height-side))
            left = int(np.clip(xs[pick]-rng.integers(side//4, 3*side//4), 0, width-side))
        else:
            top = rng.integers(height-side+1)
            left = rng.integers(width-side+1)
        result[out, 0] = gray[top:top+side, left:left+side]
    return result
