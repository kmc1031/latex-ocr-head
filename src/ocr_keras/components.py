"""Connected components retain their IDs through every merge.

AI-assisted implementation. A bounding box is used for geometry, never to
discover which other components belong to a symbol. Touching symbols within
one component still require a separate split algorithm; this does not add one.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


@dataclass(frozen=True)
class Group:
    ids: tuple
    box: tuple


def combine(a, b):
    return Group(tuple(sorted(a.ids + b.ids)), (
        min(a.box[0], b.box[0]), min(a.box[1], b.box[1]),
        max(a.box[2], b.box[2]), max(a.box[3], b.box[3])))


def otsu(a):
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    w = hist.cumsum()
    s = (hist * np.arange(256)).cumsum()
    variance = (s[-1] * w - s * w[-1]) ** 2 / np.maximum(1, w * (w[-1] - w))
    return a <= int(variance.argmax())


def preprocess(source):
    source = ImageOps.exif_transpose(source).convert('RGBA')
    background = Image.new('RGBA', source.size, 'white')
    background.alpha_composite(source)
    image = background.convert('L')
    image.thumbnail((1600, 1600))
    pixels = np.asarray(image, dtype=np.float32)
    bg = np.asarray(image.filter(ImageFilter.GaussianBlur(
        max(8, .05 * min(image.size)))), dtype=np.float32)
    pixels = np.clip(255 * pixels / (bg + 1), 0, 255).astype('uint8')
    labels, _ = ndimage.label(otsu(pixels), np.ones((3, 3)))
    heights = [s[0].stop - s[0].start for s in ndimage.find_objects(labels)
               if s and s[0].stop - s[0].start > 3
               and s[1].stop - s[1].start < 4 * (s[0].stop - s[0].start)]
    height = float(np.percentile(heights, 70)) if heights else max(1, image.height)
    scale = min(48 / height, 1024 / image.width, 512 / image.height)
    image = Image.fromarray(pixels).resize((max(1, round(image.width * scale)),
                                          max(1, round(image.height * scale))),
                                         Image.Resampling.LANCZOS)
    return image, otsu(np.asarray(image))


def group_components(mask):
    labels, _ = ndimage.label(mask, np.ones((3, 3)))
    groups = []
    for index, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds and np.count_nonzero(labels[bounds] == index) >= 3:
            y, x = bounds
            groups.append(Group((index,), (x.start, y.start, x.stop, y.stop)))
    heights = [g.box[3] - g.box[1] for g in groups
               if g.box[3] - g.box[1] > 4
               and g.box[2] - g.box[0] < 4 * (g.box[3] - g.box[1])]
    height = float(np.median(heights)) if heights else 32.0
    # '=' candidates keep the original IDs of the two bars.
    candidates = []
    for i, ga in enumerate(groups):
        a = ga.box
        aw, ah = a[2] - a[0], a[3] - a[1]
        if aw < 3 * ah:
            continue
        for j in range(i + 1, len(groups)):
            b = groups[j].box
            bw, bh = b[2] - b[0], b[3] - b[1]
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) / min(aw, bw)
            gap = max(a[1], b[1]) - min(a[3], b[3])
            if (bw >= 3 * bh and overlap >= .8 and min(aw, bw) / max(aw, bw) >= .7
                    and 0 < gap <= .5 * height):
                candidates.append((overlap - gap / height, i, j))
    used, merged = set(), []
    for _, i, j in sorted(candidates, reverse=True):
        if i in used or j in used:
            continue
        a, b = groups[i].box, groups[j].box
        between = any(k not in (i, j) and max(a[1], b[1]) > g.box[3]
                      and min(a[3], b[3]) < g.box[1]
                      and min(a[2], b[2]) > g.box[0]
                      and max(a[0], b[0]) < g.box[2] for k, g in enumerate(groups))
        if not between:
            used.update((i, j))
            merged.append(combine(groups[i], groups[j]))
    groups = [g for i, g in enumerate(groups) if i not in used] + merged
    heights = [g.box[3] - g.box[1] for g in groups
               if g.box[3] - g.box[1] > 3
               and g.box[2] - g.box[0] < 4 * (g.box[3] - g.box[1])]
    height = max(1., float(np.median(heights))) if heights else 48.
    # Dot + stem (i/j): the ownership set is merged as well as the box.
    used, merged = set(), []
    for i, dot in enumerate(groups):
        p = dot.box
        pw, ph = p[2] - p[0], p[3] - p[1]
        if pw > .25 * height or ph > .25 * height:
            continue
        choices = []
        for j, stem in enumerate(groups):
            b = stem.box
            bw, bh = b[2] - b[0], b[3] - b[1]
            dx = abs((p[0] + p[2] - b[0] - b[2]) / 2)
            gap = b[1] - p[3]
            if (i != j and bh >= .5 * height and bw <= .6 * height
                    and pw * ph <= .2 * bw * bh and dx <= .25 * height
                    and 0 <= gap <= .4 * height):
                choices.append((dx / height + gap / height, j))
        choices.sort()
        if choices and (len(choices) == 1 or choices[1][0] - choices[0][0] >= .1):
            j = choices[0][1]
            if i not in used and j not in used:
                used.update((i, j))
                merged.append(combine(dot, groups[j]))
    groups = [g for i, g in enumerate(groups) if i not in used] + merged
    groups.sort(key=lambda g: ((g.box[0] + g.box[2]) / 2,
                               (g.box[1] + g.box[3]) / 2))
    return labels, groups


def masked_patch(image, labels, group):
    x0, y0, x1, y1 = group.box
    own_pixels = np.isin(labels[y0:y1, x0:x1], group.ids)
    return Image.fromarray(np.where(own_pixels,
        np.asarray(image)[y0:y1, x0:x1], 255).astype('uint8'))


def crop_symbol(image):
    image = image.convert('L')
    bounds = Image.fromarray((np.asarray(image) < 180).astype('uint8') * 255).getbbox()
    if bounds:
        image = image.crop(bounds)
    image.thumbnail((28, 28))
    canvas = Image.new('L', (32, 32), 255)
    canvas.paste(image, ((32 - image.width) // 2, (32 - image.height) // 2))
    return (255 - np.asarray(canvas).astype('float32'))[None] / 255


def segment(source, max_segments=64):
    image, mask = preprocess(source)
    labels, groups = group_components(mask)
    if not groups or len(groups) > max_segments:
        raise ValueError('empty_or_excess_segments')
    boxes = [list(g.box) for g in groups]
    crops = np.stack([crop_symbol(masked_patch(image, labels, g)) for g in groups])
    b = np.asarray(boxes, dtype=np.float32)
    heights = [v[3] - v[1] for v in boxes
               if v[3] - v[1] > 3 and v[2] - v[0] < 4 * (v[3] - v[1])]
    h = max(1., float(np.median(heights))) if heights else 48.
    centers = (b[:, :2] + b[:, 2:]) / 2
    delta = np.vstack((np.zeros((1, 2), np.float32), np.diff(centers, axis=0))) / h
    geometry = np.column_stack((centers[:, 0] / image.width, centers[:, 1] / image.height,
        (b[:, 2] - b[:, 0]) / image.width, (b[:, 3] - b[:, 1]) / image.height,
        delta, (b[:, 2] - b[:, 0]) / h, (b[:, 3] - b[:, 1]) / h)).astype('float32')
    geometry[:, 4:] = geometry[:, 4:].clip(-4, 4)
    return crops, geometry, boxes
