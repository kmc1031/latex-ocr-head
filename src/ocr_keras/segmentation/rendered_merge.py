"""Geometry-based merge experiment for non-touching, font-rendered equations."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..components import combine, otsu
from ..io import read, resolve, save
from ..raw import raw_components
from .dataset import candidates, records
from .unet_data import gold_masks
from .unet_inference import PixelMeasurement, diagnostic_targets


def describe(a, b, area):
    upper, lower = sorted((a, b), key=lambda group: group.box[1]+group.box[3])
    x, y, r, s = upper.box
    u, v, w, z = lower.box
    aw, bw, ah, bh = r-x, w-u, s-y, z-v
    scale = max(aw, bw, ah, bh)
    return [(v-s)/scale, abs(r+x-w-u)/(2*scale), (min(r,w)-max(x,u))/scale,
            aw/scale, bw/scale, ah/scale, bh/scale,
            area[upper.ids[0]]/scale**2, area[lower.ids[0]]/scale**2]


def examples(row, fixed):
    gray, gold = row['image'], gold_masks(row)
    ink = gray < 180 if fixed else otsu(gray)
    labels, groups = raw_components(ink)
    area = np.bincount(labels.ravel())
    hits = np.stack([np.bincount(labels[m], minlength=len(area)) for m in gold], axis=1)
    hits[0] = 0
    owners = np.where(hits.max(1) >= .8*area, hits.argmax(1), -1)
    pairs = candidates(groups, owners)
    by_id = {group.ids[0]: group for group in groups}
    features = np.asarray([describe(by_id[a], by_id[b], area) for a,b,_,_ in pairs]).reshape(-1,9)
    return gray, gold, ink, labels, groups, by_id, pairs, features


def transform(features):
    i,j = np.triu_indices(features.shape[1])
    return np.c_[features, features[:,i]*features[:,j]]


def fit(rows):
    pairs, features = [], []
    for _, row in rows:
        for fixed in (True,False):
            *_, info,x = examples(row,fixed)
            valid = info[:,2] >= 0
            pairs.extend(info[valid,2])
            features.extend(x[valid])
    labels = np.array(pairs)
    design = transform(np.asarray(features))
    mean, scale = design.mean(0), np.maximum(design.std(0), .01)
    design = np.c_[np.clip((design-mean)/scale,-8,8), np.ones(len(labels))]
    weights = np.where(labels, 4., 1.)
    def loss(coef):
        score = design @ coef
        risk = (weights*(np.logaddexp(0,score)-labels*score)).mean() + .005*(coef@coef)
        grad = design.T@(weights*(expit(score)-labels))/len(labels) + .01*coef
        return risk, grad
    fit_result = minimize(loss,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B')
    if not fit_result.success:
        raise RuntimeError(str(fit_result.message))
    return mean, scale, fit_result.x


def evaluate(dataset, split, condition, cuts, model):
    root = dataset if condition=='clean' else dataset/'degraded'
    mean, scale, coef = model
    meters = {cut:PixelMeasurement() for cut in cuts}
    for entry in read(dataset/f'{split}.json'):
        with np.load(resolve(root/split, entry['id']+'.npz'), allow_pickle=False) as stored:
            row = {key: stored[key] for key in ('image', 'gold_mask_bits', 'glyph_names',
                                               'gold_areas', 'overlap', 'components')}
        _,gold,ink,labels,groups,by_id,pairs,x = examples(row, condition=='clean')
        dx = np.c_[np.clip((transform(x)-mean)/scale,-8,8), np.ones(len(x))]
        scores = expit(dx@coef)
        order = scores.argsort()[::-1]
        diag = diagnostic_targets(row,gold)
        for cut,meter in meters.items():
            used,merged = set(),[]
            for index in order:
                if scores[index] < cut: break
                a,b = pairs[index,:2]
                if a not in used and b not in used:
                    used.update((a,b))
                    merged.append(combine(by_id[a],by_id[b]))
            meter.add_result(row,gold,ink,labels,
                             merged+[g for g in groups if g.ids[0] not in used],diag)
    return {cut: meter.result() for cut,meter in meters.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    model = fit(records(args.dataset,'train'))
    cuts = (.4,.6,.8,.9)
    val = {c:evaluate(args.dataset,'val',c,cuts,model) for c in ('clean','degraded')}
    cut = max(cuts,key=lambda p:sum(out[p]['both80_f1'] for out in val.values()))
    test = {c:evaluate(args.dataset,'test',c,(cut,),model)[cut] for c in val}
    save(args.dataset/'geometry_merge.json',dict(cut=cut, coefficients=[v.tolist() for v in model],
                                                  val=val,test=test))
    print(cut,[(c,r['both80_f1'],r['all_glyphs_both80']) for c,r in test.items()],flush=True)


if __name__ == '__main__':
    main()
