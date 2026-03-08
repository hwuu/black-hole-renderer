#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from disk_v2._array_utils import _restore_bool, _restore_shape, _to_array


class DiskV2ArrayUtilsTest(unittest.TestCase):
    def test_to_array_converts_scalar_to_float64_array(self):
        value = _to_array(1.25)

        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.dtype, np.float64)
        self.assertEqual(value.shape, ())
        self.assertEqual(float(value), 1.25)

    def test_restore_shape_returns_scalar_for_scalar_original(self):
        restored = _restore_shape(np.asarray(2.5, dtype=np.float64), 1.0)

        self.assertIsInstance(restored, float)
        self.assertEqual(restored, 2.5)

    def test_restore_shape_returns_array_for_array_original(self):
        restored = _restore_shape(np.asarray([1.0, 2.0], dtype=np.float64), np.asarray([0.0, 0.0]))

        self.assertIsInstance(restored, np.ndarray)
        self.assertTrue(np.allclose(restored, np.asarray([1.0, 2.0], dtype=np.float64)))

    def test_restore_bool_returns_scalar_for_scalar_original(self):
        restored = _restore_bool(np.asarray(True), 0.0)

        self.assertIsInstance(restored, bool)
        self.assertTrue(restored)

    def test_restore_bool_returns_array_for_array_original(self):
        restored = _restore_bool(np.asarray([True, False]), np.asarray([0.0, 0.0]))

        self.assertIsInstance(restored, np.ndarray)
        self.assertTrue(np.array_equal(restored, np.asarray([True, False])))


if __name__ == "__main__":
    unittest.main()
