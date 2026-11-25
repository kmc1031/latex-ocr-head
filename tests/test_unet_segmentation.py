import unittest

import numpy as np
from PIL import Image

from ocr_keras.components import Group
from ocr_keras.segmentation.handoff import symbol_batch
from ocr_keras.segmentation.unet_data import evaluation_gray, pad, targets
from ocr_keras.segmentation.unet_inference import PixelMeasurement, decode_instances


class UNetSegmentationTests(unittest.TestCase):
    def test_center_votes_rejoin_detached_parts_and_split_touching_glyphs(self):
        masks = np.zeros((2, 25, 32), dtype=bool)
        masks[0, 2:4, 7:9] = True
        masks[0, 8:20, 7:12] = True
        masks[1, 12:20, 12:19] = True
        truth = targets(masks, instance=True)
        prediction = np.zeros((25, 32, 4), dtype=np.float32)
        prediction[..., 0] = np.where(truth[..., 0] > 0, 10, -10)
        prediction[..., 1] = np.where(truth[..., 2] == 1, 10, -10)
        prediction[..., 2:] = truth[..., 3:5]
        labels, groups = decode_instances(prediction, .5, .25, 4.)
        self.assertEqual(len(groups), 2)
        self.assertEqual(labels[2, 7], labels[9, 7])
        self.assertNotEqual(labels[12, 11], labels[12, 12])
        measurement = PixelMeasurement()
        measurement.add_result({'glyph_names': np.array(['i', 'l']),
                                'gold_areas': masks.sum(axis=(1, 2))},
                               masks, labels != 0, labels, groups,
                               ([('i_dot', (masks[0] & (np.arange(25)[:, None] < 5),
                                            masks[0] & (np.arange(25)[:, None] >= 5)))], {(0, 1)}))
        self.assertEqual(measurement.result()['i_dot_rate'], 1.)
        self.assertEqual(measurement.result()['touching_pairs_rate'], 1.)

    def test_overlaps_are_ignored_for_offsets_and_degradation_is_repeatable(self):
        masks = np.zeros((2, 17, 23), dtype=bool)
        masks[0, 1:6, 5:10] = True
        masks[1, 5:9, 7:17] = True
        truth = targets(masks, instance=True)
        self.assertEqual(truth[5, 8, 0], 1)
        self.assertEqual(truth[5, 8, 5], 0)
        self.assertEqual(pad(truth).shape, (24, 24, 6))
        self.assertTrue(np.all(pad(truth)[17:, :, 1] == 0))
        gray = np.where(masks.any(0), 20, 255).astype(np.uint8)
        np.testing.assert_array_equal(evaluation_gray(gray, 'sample', 'degraded', 42),
                                      evaluation_gray(gray, 'sample', 'degraded', 42))

    def test_cnn_handoff_does_not_crop_out_a_pale_dot(self):
        pixels = np.full((16, 9), 255, dtype=np.uint8)
        labels = np.zeros(pixels.shape, dtype=np.int32)
        pixels[2, 4] = 220
        pixels[7:14, 4] = 0
        labels[pixels < 255] = 1
        group = Group((1,), (4, 2, 5, 14))
        old = symbol_batch(Image.fromarray(pixels), labels, [group])['images']
        new = symbol_batch(Image.fromarray(pixels), labels, [group], preserve_extent=True)['images']
        self.assertEqual(old.shape, new.shape)
        self.assertEqual(np.count_nonzero((old > 0) & (old < .2)), 0)
        self.assertEqual(np.count_nonzero((new > 0) & (new < .2)), 1)
