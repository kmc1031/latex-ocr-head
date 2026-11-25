"""Segmentation-only dataset from local CROHME InkML annotations."""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from ..annotations import stroke_masks
from ..components import Group, preprocess
from ..io import digest, identity, read, resolve, save
from ..raw import nearby, raw_components, separate_script

SPLITS = ('train', 'val', 'test')
FORMAT = 1


def split_rows(source):
    held_formula, held_ink = set(), set()
    rows, excluded = {}, {}
    for split in reversed(SPLITS):
        original = [json.loads(line) for line in (source / f'{split}.jsonl').read_text(
            encoding='utf-8-sig').splitlines() if line.strip()]
        ids = set()
        rows[split] = []
        excluded[split] = Counter()
        for row in original:
            formula, ink = row['formula_identity'], row['ink_hash']
            if formula in held_formula or ink in held_ink:
                excluded[split]['higher_priority_split'] += 1
            elif row['id'] in ids:
                excluded[split]['duplicate_id'] += 1
            else:
                ids.add(row['id'])
                rows[split].append(row)
        held_formula.update(row['formula_identity'] for row in original)
        held_ink.update(row['ink_hash'] for row in original)
    return rows, excluded


def component_height(groups):
    return max(1., float(np.median([g.box[3] - g.box[1] for g in groups]))) if groups else 48.


def candidates(groups, owner):
    height = component_height(groups)
    pairs = []
    for first, second in combinations(groups, 2):
        if not nearby(first, second, height):
            continue
        a, b = first.ids[0], second.ids[0]
        label = int(owner[a] == owner[b]) if min(owner[a], owner[b]) >= 0 else -1
        pairs.append((a, b, label, int(not separate_script(first, second))))
    return np.array(pairs, dtype=np.int32).reshape(-1, 4)


def raster_record(original, ink, row):
    image, mask = preprocess(original)
    labels, groups = raw_components(mask)
    glyphs = row['glyphs']
    if not glyphs:
        raise ValueError('no_gold_glyphs')
    masks = stroke_masks(ink, len(glyphs), original.size)
    gold = np.stack([np.asarray(m.resize(image.size, Image.Resampling.LANCZOS)) > 100
                     for m in masks])
    flat_labels = labels.ravel()
    area = np.bincount(flat_labels)
    sizes = gold.sum(axis=(1, 2))
    if np.any(sizes == 0):
        raise ValueError('empty_gold_after_resize')
    overlap = np.stack([np.bincount(flat_labels[g.ravel()], minlength=len(area))
                        for g in gold], axis=1)
    overlap[0] = 0
    owners = np.full(len(area), -1, dtype=np.int32)
    for group in groups:
        idx = group.ids[0]
        order = np.argsort(overlap[idx])
        best = order[-1]
        runner_up = overlap[idx, order[-2]] if len(order) > 1 else 0
        if overlap[idx, best] >= .8 * area[idx] and runner_up <= .1 * area[idx]:
            owners[idx] = best
    box_scale = np.array([image.width / original.width, image.height / original.height] * 2)
    return dict(image=np.asarray(image), labels=labels.astype(np.uint32),
                components=np.array([(g.ids[0], *g.box) for g in groups], dtype=np.int32).reshape(-1, 5),
                glyph_names=np.array([g['label'] for g in glyphs]),
                glyph_boxes=np.asarray([g['box'] for g in glyphs], dtype=np.float32) * box_scale,
                gold_mask_bits=np.packbits(gold.reshape(len(glyphs), -1), axis=1),
                gold_areas=sizes.astype(np.int32), component_areas=area.astype(np.int32),
                overlap=overlap.astype(np.int32), owner=owners, pairs=candidates(groups, owners))


def groups_from_record(record):
    return [Group((int(row[0]),), tuple(int(value) for value in row[1:]))
            for row in record['components']]


def _add_stats(stats, record):
    stats['formulas'] += 1
    stats['gold_glyphs'] += len(record['gold_areas'])
    stats['components'] += len(record['components'])
    stats['unowned_retained_components'] += int(np.count_nonzero(record['owner'][
        record['components'][:, 0]] < 0))
    pairs = record['pairs']
    for name, label in (('positive', 1), ('negative', 0), ('ambiguous', -1)):
        selected = pairs[:, 2] == label
        stats[f'{name}_pairs'] += int(selected.sum())
        stats[f'{name}_pairs_vetoed'] += int((selected & (pairs[:, 3] == 0)).sum())


def source_fingerprint(source, limit=None):
    paths = [source / 'crohme2014.zip', *(source / f'{s}.jsonl' for s in SPLITS)]
    project = source.parent / 'sources_manifest.json'
    if project.is_file():
        paths.append(project)
    code = [Path(__file__), *(Path(__file__).parent.parent / name for name in
                              ('annotations.py', 'components.py', 'raw.py'))]
    return identity({'format': FORMAT, 'limit': limit,
                     'source': {p.name: digest(p) for p in paths},
                     'code': [digest(p) for p in code]})


def prepare(sources, output, limit=None):
    source = sources / 'crohme'
    archive_path = source / 'crohme2014.zip'
    fingerprint = source_fingerprint(source, limit)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        summary = read(summary_path)
        if summary.get('fingerprint') != fingerprint:
            raise ValueError('Choose a new dataset directory for a different source, code or limit.')
        return summary
    rows, excluded = split_rows(source)
    summary = {'format': FORMAT, 'fingerprint': fingerprint, 'limit_per_split': limit,
               'source': 'Locally prepared CROHME, using original InkML stroke ownership',
               'archive_sha256': digest(archive_path), 'splits': {}}
    with ZipFile(archive_path) as archive:
        for split in SPLITS:
            dest = output / split
            dest.mkdir(exist_ok=True)
            selected = rows[split][:limit]
            entries, errors, stats = [], Counter(), Counter()
            for index, row in enumerate(selected, 1):
                item = resolve(dest, row['id'] + '.npz')
                try:
                    with Image.open(resolve(source, row['image'])) as original:
                        data = raster_record(original, archive.read(row['source']), row)
                except (KeyError, ValueError) as exc:
                    errors[str(exc)] += 1
                else:
                    np.savez_compressed(item, **data)
                    _add_stats(stats, data)
                    entries.append({key: row[key] for key in
                                    ('id', 'source', 'ink_hash', 'formula_identity')})
                if index % 200 == 0 or index == len(selected):
                    print(f'[prepare] {split} {index}/{len(selected)} ({len(entries)} kept)', flush=True)
            if not entries:
                raise ValueError(f'{split}: no usable formulas')
            save(output / f'{split}.json', entries)
            summary['splits'][split] = dict(stats, excluded=dict(excluded[split]),
                                             annotation_errors=dict(errors))
    save(summary_path, summary)
    return summary


def records(dataset, split, limit=None):
    entries = read(dataset / f'{split}.json')[:limit]
    for entry in entries:
        path = resolve(dataset / split, entry['id'] + '.npz')
        with np.load(path, allow_pickle=False) as record:
            yield entry, record
