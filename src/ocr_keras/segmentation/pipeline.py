"""AI-assisted final segmentation pipelines: classical or TensorFlow U-Net."""

from itertools import combinations

import numpy as np
from PIL import Image
import tensorflow as tf

from ..components import combine, otsu, preprocess
from ..io import read
from ..raw import nearby, raw_components
from .dataset import component_height
from .handoff import symbol_batch
from .rendered_merge import describe
from .unet import Predictor


class GeometryMerger:
    """Apply the fitted geometry model without requiring symbol labels."""

    def __init__(self, path):
        settings = read(path)
        self.threshold = float(settings['cut'])
        mean, scale, coef = (np.asarray(values, dtype=np.float32) for values in settings['coefficients'])
        if (mean.shape != (54,) or scale.shape != mean.shape or coef.shape != (55,)
                or not np.all(np.isfinite(np.r_[mean, scale, coef]))
                or np.any(scale <= 0) or not 0 < self.threshold <= 1):
            raise ValueError('Invalid geometry merge settings.')
        left, right = np.triu_indices(9)
        mean, scale, coef = map(tf.constant, (mean, scale, coef))

        @tf.function(input_signature=[tf.TensorSpec([None, 9], tf.float32)])
        def score(features):
            products = tf.gather(features, left, axis=1) * tf.gather(features, right, axis=1)
            design = tf.concat([features, products], axis=1)
            design = tf.clip_by_value((design - mean) / scale, -8., 8.)
            return tf.sigmoid(tf.linalg.matvec(design, coef[:-1]) + coef[-1])

        self.score = score

    def groups(self, labels, groups):
        height = component_height(groups)
        pairs = [(a, b) for a, b in combinations(groups, 2) if nearby(a, b, height)]
        if not pairs:
            return sorted_groups(groups)
        areas = np.bincount(labels.ravel())
        values = self.score(np.asarray([describe(a, b, areas) for a, b in pairs], np.float32)).numpy()
        used, merged = set(), []
        for index in values.argsort()[::-1]:
            if values[index] < self.threshold:
                break
            a, b = pairs[index]
            if not used.intersection(a.ids+b.ids):
                used.update(a.ids+b.ids)
                merged.append(combine(a, b))
        return sorted_groups(merged + [g for g in groups if g.ids[0] not in used])


def sorted_groups(groups):
    return sorted(groups, key=lambda g: (g.box[0]+g.box[2], g.box[1]+g.box[3]))


class SegmentationPipeline:
    """Only foreground extraction changes between methods with the same merger."""

    def __init__(self, method, merger, *, unet=None, foreground=.65, gray_threshold=None):
        if method not in ('classical', 'unet'):
            raise ValueError(f'Unsupported segmentation method: {method}')
        if not 0 < foreground < 1 or (gray_threshold is not None and not 0 < gray_threshold <= 255):
            raise ValueError('Invalid foreground threshold.')
        self.method = method
        self.merger = merger
        self.foreground_logit = float(np.log(foreground / (1-foreground)))
        self.gray_threshold = gray_threshold
        self.predict = Predictor(unet) if method == 'unet' and unet is not None else None
        if method == 'unet' and (self.predict is None or self.predict.mode != 'gray'):
            raise ValueError('U-Net mode requires a gray foreground checkpoint.')

    def on_gray(self, gray, *, classical_mask=None):
        if gray.min() == gray.max():
            ink = np.zeros(gray.shape, dtype=bool)
        elif self.method == 'unet':
            ink = self.predict(gray, None)[..., 0] >= self.foreground_logit
        elif self.gray_threshold is not None:
            ink = gray < self.gray_threshold
        else:
            ink = otsu(gray) if classical_mask is None else classical_mask
        labels, raw = raw_components(ink)
        return ink, labels, sorted_groups(self.merger.groups(labels, raw))

    def __call__(self, source: Image.Image):
        image, mask = preprocess(source)
        ink, labels, groups = self.on_gray(np.asarray(image), classical_mask=mask)
        return symbol_batch(image, labels, groups, preserve_extent=True), ink, labels, groups
