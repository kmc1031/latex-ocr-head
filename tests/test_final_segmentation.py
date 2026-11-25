import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.special import expit

from ocr_keras.io import save
from ocr_keras.segmentation.rendered_merge import transform


class FinalSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
        from ocr_keras.segmentation.pipeline import GeometryMerger, SegmentationPipeline

        cls.geometry = GeometryMerger
        cls.pipeline = SegmentationPipeline

    def test_tensorflow_geometry_matches_original_fit(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(12, 9)).astype(np.float32)
        mean, scale = rng.normal(size=54), rng.uniform(.25, 2, size=54)
        coefficients = rng.normal(scale=.1, size=55)
        design = np.clip((transform(features)-mean)/scale, -8, 8)
        expected = expit(design @ coefficients[:-1] + coefficients[-1])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'geometry.json'
            save(path, {'cut': .4, 'coefficients': [x.tolist() for x in (mean, scale, coefficients)]})
            merger = self.geometry(path)
            np.testing.assert_allclose(merger.score(features).numpy(), expected, atol=1e-6)

    def test_classical_blank_emits_no_symbols(self):
        class IdentityMerger:
            def groups(self, labels, groups):
                return groups

        model = self.pipeline('classical', IdentityMerger())
        batch, ink, _, groups = model(Image.new('L', (80, 20), 255))
        self.assertFalse(ink.any())
        self.assertFalse(groups)
        self.assertEqual(batch['images'].shape, (0, 1, 48, 48))
        self.assertEqual(batch['geometry'].shape, (0, 8))
