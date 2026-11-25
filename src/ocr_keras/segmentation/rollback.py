"""Apply the saved v6 merge review policy without loading the OCR model."""

import numpy as np
from scipy.special import expit

from ..components import combine


def union(groups):
    merged = groups[0]
    for group in groups[1:]:
        merged = combine(merged, group)
    return merged


def candidate_groups(group, raw_by_id):
    if len(group.ids) <= 1:
        return
    parts = [raw_by_id[index] for index in group.ids]
    seen = set()
    for axis in (0, 1):
        ordered = sorted(parts, key=lambda x: x.box[axis] + x.box[axis + 2])
        for index in range(1, len(ordered)):
            a, b = union(ordered[:index]), union(ordered[index:])
            key = tuple(sorted((a.ids, b.ids)))
            if key not in seen:
                seen.add(key)
                yield a, b


def features(group, a, b, height, area):
    parent = np.asarray(group.box, dtype=float)
    a0, b0 = np.asarray(a.box, dtype=float), np.asarray(b.box, dtype=float)
    sa, sb = a0[2:]-a0[:2], b0[2:]-b0[:2]
    pa, pb = sum(area[i] for i in a.ids), sum(area[i] for i in b.ids)
    offset = np.abs(a0[:2]+a0[2:]-b0[:2]-b0[2:])/(2*height)
    overlap = (np.minimum(a0[2:],b0[2:])-np.maximum(a0[:2],b0[:2]))/height
    return np.clip(np.r_[np.minimum(sa,sb)/height, np.maximum(sa,sb)/height,
        (parent[2:]-parent[:2])/height, offset, overlap,
        min(pa,pb)/max(pa,pb), min(len(a.ids),len(b.ids)), max(len(a.ids),len(b.ids)),
        pa/np.prod(sa)+pb/np.prod(sb), abs(pa/np.prod(sa)-pb/np.prod(sb))], -5, 8)


def examples(raw, groups, labels):
    by_id = {g.ids[0]: g for g in raw}
    height = max(1., float(np.median([g.box[3]-g.box[1] for g in raw]))) if raw else 1.
    areas = np.bincount(labels.ravel())
    for group in groups:
        candidates = list(candidate_groups(group, by_id))
        if candidates:
            yield group, candidates, np.stack([features(group,a,b,height,areas) for a,b in candidates])


def probabilities(policy, x):
    weights = np.asarray(policy['weights'])
    return expit(((x-policy['mean'])/policy['scale'])@weights[:-1]+weights[-1])


def repair_groups(raw, groups, labels, policy):
    replacements = {}
    for group, parts, inputs in examples(raw, groups, labels):
        scores = probabilities(policy, inputs)
        index = int(np.argmax(scores))
        if scores[index] >= policy['threshold']:
            replacements[group.ids] = parts[index]
    result = [part for group in groups for part in replacements.get(group.ids, (group,))]
    result.sort(key=lambda group: (group.box[0]+group.box[2], group.box[1]+group.box[3]))
    return result, len(replacements)


class ReviewedMerger:
    def __init__(self, merger, policy, max_symbols):
        self.merger, self.policy, self.max_symbols = merger, policy, max_symbols

    def review(self, labels, raw, groups):
        reviewed, _ = repair_groups(raw, groups, labels, self.policy)
        return reviewed if len(reviewed) <= self.max_symbols else groups

    def groups(self, labels, raw):
        return self.review(labels, raw, self.merger.groups(labels, raw))
