"""Class-agnostic segmentation metrics in the preprocessed pixel frame."""

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..components import combine


def owner_oracle(record, raw):
    owners = record['owner']
    buckets, result = {}, []
    for group in raw:
        owner = int(owners[group.ids[0]])
        if owner < 0:
            result.append(group)
        elif owner in buckets:
            buckets[owner] = combine(buckets[owner], group)
        else:
            buckets[owner] = group
    return result + list(buckets.values())


def _ratio(num, den):
    return num / den if den else None


def _match(valid, iou):
    if not valid.size:
        return set()
    merit = np.where(valid, 1 + iou / (1 + min(valid.shape)), 0)
    preds, gold = linear_sum_assignment(merit, maximize=True)
    return {int(g) for p, g in zip(preds, gold) if valid[p, g]}


class Measurement:
    def __init__(self):
        self.counts = Counter()
        self.by_symbol = defaultdict(Counter)

    def add(self, record, groups):
        overlap = record['overlap']
        areas = record['component_areas']
        gold = record['gold_areas']
        names = record['glyph_names']
        intersections = np.asarray([overlap[list(g.ids)].sum(axis=0) for g in groups],
                                   dtype=int).reshape(len(groups), len(gold))
        area = np.asarray([areas[list(g.ids)].sum() for g in groups], dtype=int)[:, None]
        iou = intersections / np.maximum(1, area + gold - intersections)
        basic = _match(iou >= .5, iou)
        tight = _match((intersections >= .8 * area) & (intersections >= .8 * gold), iou)
        substantial = (intersections >= .2 * gold) & (intersections >= 3)
        parts_per_glyph = substantial.sum(axis=0)
        mixed_groups = substantial.sum(axis=1) >= 2
        raw_hits = overlap[record['components'][:, 0]]
        raw_mixed = np.count_nonzero(((raw_hits >= .2 * gold) & (raw_hits >= 3)).sum(axis=1) >= 2)
        self.counts.update(formulas=1, predicted=len(groups), gold=len(gold), matched_iou50=len(basic),
                           matched_both80=len(tight), all_glyphs_both80=int(len(tight) == len(gold) == len(groups)),
                           fragmented_glyphs=int(np.count_nonzero(parts_per_glyph >= 2)),
                           mixed_groups=int(mixed_groups.sum()), mixed_raw_components=int(raw_mixed),
                           formulas_with_mixed_raw=int(raw_mixed > 0))
        for index, name in enumerate(names.tolist()):
            self.by_symbol[name].update(gold=1, matched_both80=int(index in tight),
                                        fragmented=int(parts_per_glyph[index] >= 2))
        return dict(gold=len(gold), predicted=len(groups), matched_both80=len(tight))

    def result(self):
        counts = self.counts
        return dict(counts) | {
            'iou50_f1': _ratio(2 * counts['matched_iou50'], counts['gold'] + counts['predicted']),
            'both80_precision': _ratio(counts['matched_both80'], counts['predicted']),
            'both80_recall': _ratio(counts['matched_both80'], counts['gold']),
            'both80_f1': _ratio(2 * counts['matched_both80'], counts['gold'] + counts['predicted']),
            'strict_formula_rate': _ratio(counts['all_glyphs_both80'], counts['formulas']),
            'by_symbol': {name: dict(row) | {'both80_recall': _ratio(row['matched_both80'], row['gold'])}
                          for name, row in sorted(self.by_symbol.items())},
        }
