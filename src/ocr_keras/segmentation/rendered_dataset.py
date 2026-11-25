"""AI-assisted font-rendered slash equations with audited per-glyph ownership."""

import argparse
from itertools import combinations
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from ..components import otsu, preprocess
from ..config import Settings
from ..io import identity, save
from ..notation import content_key, parse, tokenize
from ..raw import raw_components
from ..rendering import Renderer
from .dataset import candidates
from .restore_data import conventional
from .unet_data import evaluation_gray


def expression(split, rng, index):
    left, a, b, c, d = rng.sample('FEmatvPIQVrcnijk', 5)
    n, k = rng.randint(2, 37), rng.randint(2, 13)
    families = {
        'train': [f'{left}={n}{a}/{b}', f'{left}_{k}=({a}+{b})/{c}^{n}',
                  f'{left}={a}^{k}+{b}/{c}', f'{left}=({a}_{k}-{b})/({c}+{d})',
                  f'{left}={n}.{k}{a}/{b}', f'{left}=({a}+{b})/{k}+{c}'],
        'val': [f'{left}=({n}{a}-{b})/({c}_{k}+{d})', f'{left}^{{{k}}}={a}/{b}+{c}/{d}',
                f'{left}={n}.{k}({a}+{b})/{c}'],
        'test': [f'{left}_{k}=({a}^{n}+{b})/({c}-{d})', f'{left}=({a}/{b})/({c}+{k}{d})',
                 f'{left}={a}-{n}.{k}{b}/({c}+{d})'],
    }
    return families[split][index % len(families[split])]


def audit(gray, owner, count, mask):
    if not np.array_equal(np.unique(owner[owner > 0]), np.arange(1, count+1)):
        return False
    # Distinct glyphs may not touch even diagonally. Internal glyph strokes are unrestricted.
    for first, second in ((owner[:, 1:], owner[:, :-1]), (owner[1:], owner[:-1]),
                          (owner[1:, 1:], owner[:-1, :-1]), (owner[1:, :-1], owner[:-1, 1:])):
        if np.any((first > 0) & (second > 0) & (first != second)):
            return False
    labels, _ = ndimage.label((gray < 170) | mask, np.ones((3, 3)))
    hits = np.unique(np.stack((labels[owner > 0], owner[owner > 0]), axis=-1), axis=0)
    ids = hits[hits[:, 0] != 0, 0]
    return len(ids) == len(np.unique(ids))


def arrays(gray, owner, names, use_clean_mask=None):
    mask = otsu(gray) if use_clean_mask is None else use_clean_mask
    labels, groups = raw_components(mask)
    gold = owner[None] == np.arange(1, len(names)+1)[:, None, None]
    areas = gold.sum((1, 2))
    counts = np.bincount(labels.ravel())
    overlap = np.stack([np.bincount(labels[m], minlength=len(counts)) for m in gold], axis=1)
    overlap[0] = 0
    belonging = np.full(len(counts), -1, dtype=np.int32)
    for group in groups:
        idx = group.ids[0]
        order = np.argsort(overlap[idx])
        top = order[-1]
        other = overlap[idx, order[-2]] if len(names) > 1 else 0
        if overlap[idx, top] >= .8*counts[idx] and other <= .1*counts[idx]:
            belonging[idx] = top
    boxes = []
    for m in gold:
        y, x = np.nonzero(m)
        boxes.append((x.min(), y.min(), x.max()+1, y.max()+1))
    return dict(image=gray, labels=labels.astype(np.uint32), components=np.asarray(
        [(g.ids[0], *g.box) for g in groups], dtype=np.int32).reshape(-1, 5),
        glyph_names=np.asarray(names), glyph_boxes=np.asarray(boxes, dtype=np.float32),
        gold_mask_bits=np.packbits(gold.reshape(len(names), -1), axis=1),
        gold_areas=areas.astype(np.int32), component_areas=counts.astype(np.int32),
        overlap=overlap.astype(np.int32), owner=belonging, pairs=candidates(groups, belonging))


def prepare(output, count, seed):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a fresh output directory.')
    variants = output/'degraded'
    rng = random.Random(seed)
    settings = Settings(handwritten_probability=0)
    bank = SimpleNamespace(pools={split: {} for split in count})
    seen = set()
    totals = {}
    for split, amount in count.items():
        dest, noisy = output/split, variants/split
        dest.mkdir(parents=True, exist_ok=True)
        noisy.mkdir(parents=True, exist_ok=True)
        rows = []
        rejects = 0
        for attempt in range(amount*15):
            if len(rows) == amount:
                break
            tokens = tokenize(expression(split, rng, attempt))
            key = content_key(tokens)
            if key in seen:
                continue
            layout = Renderer(bank, settings, rng, split, 'slash').draw(parse(tokens), rng.randint(28, 48))
            names = [glyph['label'] for glyph in layout.glyphs]
            assert '/' in names and r'\frac' not in layout.tokens
            raw = ImageOps.expand(layout.image, border=8, fill=255)
            ownership = ImageOps.expand(layout.owner, border=8, fill=0)
            image, baseline = preprocess(raw)
            gray = np.asarray(image)
            owner = np.asarray(ownership.resize(image.size, Image.Resampling.NEAREST))
            if not audit(gray, owner, len(names), baseline):
                rejects += 1
                continue
            uid = identity([split, key, seed])[:16]
            np.savez_compressed(dest/(uid+'.npz'), **arrays(gray, owner, names, baseline))
            degraded = conventional(evaluation_gray(np.asarray(raw), uid, 'degraded', seed+1))
            degraded = np.asarray(Image.fromarray(degraded).resize(image.size, Image.Resampling.LANCZOS))
            np.savez_compressed(noisy/(uid+'.npz'), **arrays(degraded, owner, names))
            rows.append({'id': uid, 'tokens': layout.tokens, 'identity': key, 'source': 'mathtext-font'})
            seen.add(key)
            if len(rows) % 100 == 0 or len(rows) == amount:
                print(f'[rendered] {split} {len(rows)}/{amount}; rejected {rejects}', flush=True)
        if len(rows) != amount:
            raise RuntimeError(f'Not enough valid formulas for {split}: {len(rows)}/{amount}')
        save(output/f'{split}.json', rows)
        totals[split] = {'formulas': len(rows), 'contact_or_empty_rejected': rejects}
    fingerprint = identity({'seed': seed, 'count': count, 'rendered': sorted(seen), 'version': 1})
    save(output/'summary.json', {'fingerprint': fingerprint, 'seed': seed,
                                 'source': 'LaTeX MathText fonts', 'notation': 'slash', 'splits': totals})
    save(variants/'complete.json', {'dataset': fingerprint, 'seed': seed+1,
                                    'degradation': 'before_background_correction_fixed_geometry',
                                    'splits': count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    for split, default in (('train', 900), ('val', 180), ('test', 180)):
        parser.add_argument('--'+split, default=default, type=int)
    parser.add_argument('--seed', default=2026, type=int)
    args = parser.parse_args()
    count = {split: getattr(args, split) for split in ('train', 'val', 'test')}
    if min(count.values()) < 1:
        parser.error('Split sizes must be positive.')
    prepare(args.output, count, args.seed)


if __name__ == '__main__':
    main()
