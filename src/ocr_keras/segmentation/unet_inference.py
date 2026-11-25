"""Decode U-Net pixel predictions without changing the classification interface."""

from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.special import expit

from ..components import Group
from .metrics import Measurement, _match
from .unet_data import CONNECTIVITY, OFFSET_SCALE


def _peaks(probability, threshold):
    maximum = ndimage.maximum_filter(probability, size=5)
    regions, count = ndimage.label((probability >= threshold) & (probability == maximum), CONNECTIVITY)
    if not count:
        return np.empty((0, 2), dtype=int)
    positions = ndimage.maximum_position(probability, regions, np.arange(1, count + 1))
    positions = np.asarray(positions, dtype=int).reshape(-1, 2)
    order = np.argsort(-probability[positions[:, 0], positions[:, 1]], kind='stable')[:256]
    return positions[order]


def decode_instances(prediction, foreground, center, radius):
    ink = expit(prediction[..., 0]) >= foreground
    labels = np.zeros(ink.shape, dtype=np.int32)
    locations = _peaks(expit(prediction[..., 1]), center)
    if len(locations) and ink.any():
        ys, xs = np.nonzero(ink)
        offsets = prediction[ys, xs, 2:4] * OFFSET_SCALE
        votes = np.column_stack((ys + offsets[:, 1], xs + offsets[:, 0]))
        distances, indices = cKDTree(locations).query(votes, distance_upper_bound=radius)
        accepted = np.isfinite(distances)
        labels[ys[accepted], xs[accepted]] = indices[accepted] + 1
    residual, _ = ndimage.label(ink & (labels == 0), CONNECTIVITY)
    labels[residual > 0] = len(locations) + residual[residual > 0]
    groups = []
    for index, region in enumerate(ndimage.find_objects(labels), 1):
        if region:
            ys, xs = region
            groups.append(Group((index,), (xs.start, ys.start, xs.stop, ys.stop)))
    groups.sort(key=lambda group: (group.box[0] + group.box[2], group.box[1] + group.box[3]))
    return labels, groups


def _counts(labels, masks):
    base = np.bincount(labels.ravel())
    overlaps = np.stack([np.bincount(labels[mask], minlength=len(base)) for mask in masks], axis=1)
    overlaps[0] = 0
    return base, overlaps


def _parts(mask, kind):
    labels, count = ndimage.label(mask, CONNECTIVITY)
    if count < 2:
        return None
    areas = np.bincount(labels.ravel())
    order = np.argsort(-areas[1:]) + 1
    first = order[0]
    ranges = ndimage.find_objects(labels)
    body_y, body_x = ranges[first - 1]
    height = body_y.stop - body_y.start
    if kind == 'i':
        eligible = []
        for index in order[1:]:
            ys, xs = ranges[index - 1]
            center = (xs.start + xs.stop) / 2
            if (2 <= areas[index] <= .55 * areas[first] and ys.stop <= body_y.start
                    and ys.stop - ys.start <= .55 * height
                    and body_x.start - .25 * height <= center <= body_x.stop + .25 * height):
                eligible.append(index)
        if not eligible:
            return None
        second = eligible[0]
    else:
        second = order[1]
        other_y, other_x = ranges[second - 1]
        if (areas[second] < .2 * areas[first]
                or max(body_x.start, other_x.start) >= min(body_x.stop, other_x.stop)
                or max(body_y.start, other_y.start) < min(body_y.stop, other_y.stop)):
            return None
    return labels == first, labels == second


def diagnostic_targets(record, masks):
    parts = []
    for kind, name in (('i', 'i_dot'), ('=', 'equals_bars')):
        for index in np.flatnonzero(record['glyph_names'] == kind):
            pair = _parts(masks[index], kind)
            if pair is not None:
                parts.append((name, pair))
    touch_pairs = set()
    for hits in record['overlap'][record['components'][:, 0]]:
        substantial = np.flatnonzero((hits >= .2 * record['gold_areas']) & (hits >= 3))
        touch_pairs.update(combinations(substantial, 2))
    return parts, touch_pairs


class PixelMeasurement(Measurement):
    def add_result(self, record, masks, ink, labels, groups, diagnostics):
        areas, overlap = _counts(labels, masks)
        used_ids = np.unique(np.concatenate([np.asarray(g.ids) for g in groups])) if groups else []
        components = np.asarray(used_ids, dtype=int).reshape(-1, 1)
        self.add({'overlap': overlap, 'component_areas': areas, 'gold_areas': record['gold_areas'],
                  'glyph_names': record['glyph_names'], 'components': components}, groups)
        parts, touching = diagnostics
        gold_ink = masks.any(axis=0)
        self.counts.update(ink_true=int((ink & gold_ink).sum()), ink_predicted=int(ink.sum()),
                           ink_gold=int(gold_ink.sum()), touching_pairs=len(touching))
        by_id = np.zeros(len(areas), dtype=int)
        for index, group in enumerate(groups, 1):
            by_id[list(group.ids)] = index
        for name, (first, second) in parts:
            first_hits = np.bincount(by_id[labels[first]], minlength=len(groups) + 1)[1:]
            second_hits = np.bincount(by_id[labels[second]], minlength=len(groups) + 1)[1:]
            together = ((first_hits >= .5 * first.sum()) & (second_hits >= .5 * second.sum())).any()
            self.counts.update({name: 1, name + '_preserved': int(together)})
        if touching and groups:
            intersections = np.stack([overlap[list(group.ids)].sum(axis=0) for group in groups])
            area = np.asarray([areas[list(group.ids)].sum() for group in groups])[:, None]
            gold = record['gold_areas']
            iou = intersections / np.maximum(1, area + gold - intersections)
            matched = _match((intersections >= .8 * area) & (intersections >= .8 * gold), iou)
            self.counts['touching_pairs_split'] += sum(a in matched and b in matched for a, b in touching)

    def result(self):
        result = super().result()
        for total, hit in (('touching_pairs', 'touching_pairs_split'), ('i_dot', 'i_dot_preserved'),
                           ('equals_bars', 'equals_bars_preserved')):
            result[total] = self.counts[total]
            result[hit] = self.counts[hit]
            result[total + '_rate'] = self.counts[hit] / self.counts[total] if self.counts[total] else None
        den = self.counts['ink_predicted'] + self.counts['ink_gold']
        result['ink_f1'] = 2 * self.counts['ink_true'] / den if den else None
        return result
