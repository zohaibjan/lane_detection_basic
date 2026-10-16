"""Synthetic geometry checks, independent of a display or OpenCV installation."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

spec = importlib.util.spec_from_file_location('lane_detection', Path(__file__).resolve().parents[1] / 'main.py')
lane = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'cv2': Mock()}):
    spec.loader.exec_module(lane)


class LaneGeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_missing_detections(self):
        self.assertEqual(lane.avgLines(self.image, None).shape, (0, 4))
        self.assertEqual(lane.avgLines(self.image, np.empty((0, 1, 4))).shape, (0, 4))

    def test_vertical_and_horizontal_segments_are_skipped(self):
        segments = np.array([[[20, 10, 20, 90]], [[10, 20, 90, 20]]])
        self.assertEqual(lane.avgLines(self.image, segments).shape, (0, 4))

    def test_one_sided_detection(self):
        actual = lane.avgLines(self.image, np.array([[[20, 80, 40, 60]]]))
        np.testing.assert_array_equal(actual, [[0, 100, 40, 60]])

    def test_both_sides(self):
        actual = lane.avgLines(self.image, np.array([[[20, 80, 40, 60]], [[120, 60, 140, 80]]]))
        np.testing.assert_array_equal(actual, [[0, 100, 40, 60], [160, 100, 120, 60]])

    def test_unreadable_image(self):
        with self.assertRaises(ValueError):
            lane.mainProgram(None)


if __name__ == '__main__':
    unittest.main()
