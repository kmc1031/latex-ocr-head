"""AI-assisted export of segmented symbols for the team classification CNN."""

import numpy as np
from PIL import Image

from ..components import masked_patch
from ..config import CLASSES
from ..data import crop, geometry


def _crop_owned_extent(image):
    image.thumbnail((42, 42), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (48, 48), 255)
    canvas.paste(image, ((48 - image.width) // 2, (48 - image.height) // 2))
    return ((255 - np.asarray(canvas, dtype=np.float32)) / 255)[..., None]


def symbol_batch(image, labels, groups, *, preserve_extent=False):
    boxes = np.asarray([group.box for group in groups], dtype=np.int32).reshape(-1, 4)
    if len(boxes):
        normalize = _crop_owned_extent if preserve_extent else crop
        patches = [normalize(masked_patch(image, labels, group)) for group in groups]
        images = np.ascontiguousarray(np.stack(patches).transpose(0, 3, 1, 2))
        positions = geometry(boxes, image.size)
    else:
        images = np.empty((0, 1, 48, 48), dtype=np.float32)
        positions = np.empty((0, 8), dtype=np.float32)
    return {
        'schema_version': np.array('ocr-symbols-v1'),
        'images': images,
        'boxes': boxes,
        'geometry': positions,
        'image_size': np.asarray(image.size, dtype=np.int32),
        'class_names': np.asarray(CLASSES),
    }


def save_batch(path, image, labels, groups, *, preserve_extent=False):
    if path.suffix.lower() != '.npz':
        raise ValueError('Symbol batch output must be an .npz file.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, **symbol_batch(image, labels, groups,
                                                  preserve_extent=preserve_extent))
