"""V2 命令行：优化级别与超采样倍率的默认值解析。"""

import sys
import unittest
from unittest.mock import patch

from src import cli


def _args(*argv):
    with patch.object(sys, "argv", ["render.py", "--disk_model", "v2", *argv]):
        return cli.parse_args()


class ResolveV2QualityTest(unittest.TestCase):
    def test_mode_defaults(self):
        self.assertEqual(cli.resolve_v2_quality(_args(), video=False), (1, 2))
        self.assertEqual(cli.resolve_v2_quality(_args(), video=True), (2, 1))

    def test_explicit_values_win(self):
        a = _args("--v2_opt", "0", "--v2_supersample", "3")
        self.assertEqual(cli.resolve_v2_quality(a, video=True), (0, 3))

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            cli.resolve_v2_quality(_args("--v2_supersample", "0"), video=False)
        with self.assertRaises(SystemExit):
            _args("--v2_opt", "4")

    def test_old_ss_flag_removed(self):
        with self.assertRaises(SystemExit):
            _args("--v2_ss", "2")


class V2VolumeOverridesTest(unittest.TestCase):
    """`v2_volume_overrides`：只收集显式传入的体积参数。"""

    def test_unset_flags_keep_defaults(self):
        self.assertEqual(cli.v2_volume_overrides(_args()), {})
        self.assertIsNone(_args().v2_doppler_lum)

    def test_explicit_flags_collected(self):
        a = _args("--v2_az_stretch", "1.0", "--v2_core_contrast", "0.6", "--v2_doppler_lum", "0.45")
        self.assertEqual(cli.v2_volume_overrides(a), {"core_az_stretch": 1.0, "core_contrast": 0.6})
        self.assertEqual(a.v2_doppler_lum, 0.45)


if __name__ == "__main__":
    unittest.main()
