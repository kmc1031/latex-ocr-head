import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from ocr_keras.components import Group
from ocr_keras.segmentation.__main__ import calibration
from ocr_keras.segmentation.dataset import split_rows
from ocr_keras.segmentation.handoff import symbol_batch
from ocr_keras.segmentation.metrics import Measurement, owner_oracle


class IsolatedSegmentationTests(unittest.TestCase):
    def test_handoff_batches_aligned_boxes_and_masks_out_neighbor_pixels(self):
        labels = np.zeros((12, 14), dtype=np.int32)
        labels[3, 2:12] = 1
        labels[8, 2:12] = 2
        labels[5:7, 6:8] = 3
        image = Image.fromarray(np.where(labels > 0, 0, 255).astype(np.uint8))
        groups = [Group((1, 2), (2, 3, 12, 9)), Group((3,), (6, 5, 8, 7))]
        batch = symbol_batch(image, labels, groups)
        self.assertEqual(batch['images'].shape, (2, 1, 48, 48))
        self.assertEqual(batch['images'].dtype, np.float32)
        self.assertEqual(batch['geometry'].shape, (2, 8))
        self.assertEqual(batch['boxes'].tolist(), [[2, 3, 12, 9], [6, 5, 8, 7]])
        self.assertEqual(batch['images'][:, 0, 23, 23].tolist(), [0., 1.])
        empty = symbol_batch(image, labels, [])
        self.assertEqual(empty['images'].shape, (0, 1, 48, 48))
        self.assertEqual(empty['boxes'].shape, (0, 4))

    def test_grouping_recovers_fragments_but_cannot_unmix_existing_component(self):
        raw = [Group((idx,), (idx * 10, 0, idx * 10 + 5, 8)) for idx in range(1, 4)]
        record = {
            'overlap': np.array([[0, 0, 0], [12, 0, 0], [12, 0, 0], [0, 10, 10]]),
            'component_areas': np.array([0, 12, 12, 20]), 'gold_areas': np.array([24, 10, 10]),
            'glyph_names': np.array(['=', 'x', '2']), 'owner': np.array([-1, 0, 0, -1]),
            'components': np.array([[g.ids[0], *g.box] for g in raw]),
        }
        measure = Measurement()
        measure.add(record, raw)
        self.assertEqual(measure.counts['matched_both80'], 0)
        self.assertEqual(measure.counts['fragmented_glyphs'], 1)
        repaired = Measurement()
        repaired.add(record, owner_oracle(record, raw))
        self.assertEqual(repaired.counts['matched_both80'], 1)
        self.assertEqual(repaired.counts['mixed_groups'], 1)
        self.assertEqual(repaired.counts['all_glyphs_both80'], 0)

    def test_no_formula_or_ink_reused_across_splits(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            for name, samples in {
                'test': [('t', 'f0', 'ink-1')], 'val': [('v0', 'f0', 'ink-2'), ('v1', 'f1', 'ink-3')],
                'train': [('r0', 'f9', 'ink-3'), ('r1', 'f1', 'ink-4'), ('r2', 'f2', 'ink-5')],
            }.items():
                rows = [dict(id=uid, formula_identity=formula, ink_hash=ink) for uid, formula, ink in samples]
                (path / f'{name}.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf8')
            splits, excluded = split_rows(path)
            self.assertEqual([r['id'] for r in splits['train']], ['r2'])
            self.assertEqual([r['id'] for r in splits['val']], ['v1'])
            self.assertEqual(excluded['train']['higher_priority_split'], 2)

    def test_calibration_has_explicit_fallback_if_precision_never_reached(self):
        result = calibration(np.array([.9, .95]), np.array([1, 0]), .9)
        self.assertGreater(result['threshold'], 1)
        self.assertIsNone(result['val_precision'])
        self.assertFalse(result['test_used_for_selection'])
