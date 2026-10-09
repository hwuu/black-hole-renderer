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
        a = _args("--v2_thickness_scale", "1.0", "--v2_core_contrast", "0.6", "--v2_doppler_lum", "0.45")
        self.assertEqual(cli.v2_volume_overrides(a), {"thickness_scale": 1.0, "core_contrast": 0.6})
        self.assertEqual(a.v2_doppler_lum, 0.45)

    def test_atmosphere_flags_collected(self):
        a = _args("--v2_atm_frac", "0.2", "--v2_atm_height", "0.015", "--v2_atm_fine", "0")
        self.assertEqual(cli.v2_volume_overrides(a),
                         {"atm_frac": 0.2, "atm_height": 0.015, "atm_fine_sigma": 0.0})

    def test_core_floor_and_coupling_flags(self):
        """`--v2_core_floor` → core_floor，`--v2_temp_density_coupling` → dt_i；未传时保持默认 0.15 / 0.05。"""
        from src.v2.params import DiskV2VolumeParams
        self.assertEqual(cli.v2_volume_overrides(_args()), {})
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(
            _args("--v2_core_floor", "0", "--v2_temp_density_coupling", "0.25")))
        self.assertEqual((vp.core_floor, vp.dt_i), (0.0, 0.25))
        self.assertEqual((DiskV2VolumeParams().core_floor, DiskV2VolumeParams().dt_i), (0.15, 0.05))
        for kv in (("--v2_core_floor", "1.2"), ("--v2_temp_density_coupling", "-0.1"),
                   ("--v2_temp_density_coupling", "1.5")):
            with self.assertRaises(ValueError):
                DiskV2VolumeParams(**cli.v2_volume_overrides(_args(*kv)))

    def test_atmosphere_overrides_build_params(self):
        """覆盖项可直接构造 DiskV2VolumeParams（字段名有效、取值通过校验）。"""
        from src.v2.params import DiskV2VolumeParams
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(
            _args("--v2_atm_frac", "0.3", "--v2_atm_height", "0.02", "--v2_atm_fine", "0.7")))
        self.assertEqual((vp.atm_frac, vp.atm_height, vp.atm_fine_sigma), (0.3, 0.02, 0.7))

    def test_grey_mix_flag(self):
        """`--v2_grey_mix` 写入 `grey_mix`；未传时不出现在覆盖项（保持默认 0.5）；不在 [0, 1] 时构造参数报错。"""
        from src.v2.params import DiskV2VolumeParams
        self.assertNotIn("grey_mix", cli.v2_volume_overrides(_args()))
        self.assertEqual(DiskV2VolumeParams().grey_mix, 0.5)
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(_args("--v2_grey_mix", "1")))
        self.assertEqual(vp.grey_mix, 1.0)
        for bad in ("-0.1", "1.2"):
            with self.assertRaises(ValueError):
                DiskV2VolumeParams(**cli.v2_volume_overrides(_args("--v2_grey_mix", bad)))

    def test_temp_turb_flag(self):
        """`--v2_temp_turb` 写入 `temp_turb_sigma`；未传时保持默认（关闭）。"""
        from src.v2.params import DiskV2VolumeParams
        self.assertNotIn("temp_turb_sigma", cli.v2_volume_overrides(_args()))
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(_args("--v2_temp_turb", "0.1")))
        self.assertEqual(vp.temp_turb_sigma, 0.1)
        with self.assertRaises(ValueError):
            DiskV2VolumeParams(**cli.v2_volume_overrides(_args("--v2_temp_turb", "0.6")))

    def test_temp_turb_fractal_flags(self):
        """`--v2_temp_turb_coarse/gain/clamp_px` 写入对应字段；未传时保持默认 0 / 0.69 / 3；越界报错。"""
        from src.v2.params import DiskV2VolumeParams
        self.assertEqual(cli.v2_volume_overrides(_args()), {})
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(_args(
            "--v2_temp_turb_coarse", "3", "--v2_temp_turb_gain", "1", "--v2_temp_turb_clamp_px", "1")))
        self.assertEqual((vp.temp_turb_coarse, vp.temp_turb_gain, vp.temp_turb_clamp_px), (3, 1.0, 1.0))
        self.assertIsInstance(vp.temp_turb_coarse, int)
        for kv in (("--v2_temp_turb_coarse", "5"), ("--v2_temp_turb_gain", "0"), ("--v2_temp_turb_clamp_px", "0")):
            with self.assertRaises(ValueError):
                DiskV2VolumeParams(**cli.v2_volume_overrides(_args(*kv)))

    def test_temp_turb_lens_deg_flag(self):
        """`--v2_temp_turb_lens_deg` → temp_turb_lens_deg；未传时保持默认 10；360 = 不淡出（上限）。"""
        from src.v2.params import DiskV2VolumeParams
        self.assertNotIn("temp_turb_lens_deg", cli.v2_volume_overrides(_args()))
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(_args("--v2_temp_turb_lens_deg", "360")))
        self.assertEqual(vp.temp_turb_lens_deg, 360.0)
        with self.assertRaises(ValueError):
            DiskV2VolumeParams(**cli.v2_volume_overrides(_args("--v2_temp_turb_lens_deg", "361")))

    def test_exposure_ev_flag(self):
        self.assertIsNone(_args().v2_exposure_ev)
        self.assertEqual(_args("--v2_exposure_ev", "0.5").v2_exposure_ev, 0.5)

    def test_film_response_flag(self):
        """`--v2_film_response`：未传为 None（渲染器取默认 0）；超出 [0, 1] 在校验阶段报错。"""
        self.assertIsNone(_args().v2_film_response)
        self.assertEqual(_args("--v2_film_response", "1").v2_film_response, 1.0)
        for bad in ("-0.1", "1.5"):
            with self.assertRaises(ValueError):
                cli.validate_args(_args("--v2_film_response", bad))

    def test_saturation_flag(self):
        """`--v2_saturation`：未传为 None（渲染器取默认 1）；不在 [0, 3] 在校验阶段报错。"""
        self.assertIsNone(_args().v2_saturation)
        self.assertEqual(_args("--v2_saturation", "1.6").v2_saturation, 1.6)
        for bad in ("-0.1", "3.5", "nan"):
            with self.assertRaises(ValueError):
                cli.validate_args(_args("--v2_saturation", bad))

    def test_az_stretch_flag_removed(self):
        """旧主云级联已删除，`--v2_az_stretch` 不再接受。"""
        with self.assertRaises(SystemExit):
            _args("--v2_az_stretch", "1.0")


class CameraPathArgsTest(unittest.TestCase):
    """运镜参数：组合校验与帧数确定。"""

    PATH = "scenes/v2_arts/interstellar_skim.json"

    def test_invalid_combinations_rejected(self):
        bad = [("--v2_camera_path", self.PATH, "--video", "--orbit"),
               ("--v2_camera_path_time", "3"),
               ("--v2_camera_path", self.PATH, "--video", "--v2_camera_path_time", "3"),
               ("--v2_camera_path", self.PATH),
               ("--v2_camera_path", self.PATH, "--video", "--interactive"),
               ("--v2_camera_path", self.PATH, "--interactive", "--v2_camera_path_time", "3"),
               ("--v2_camera_path", self.PATH, "--v2_camera_path_time", "nan"),
               ("--video", "--v2_orbit_seconds", "0"),
               ("--video", "--v2_orbit_seconds", "inf")]
        for argv in bad:
            with self.assertRaises(ValueError, msg=str(argv)):
                cli.validate_args(_args(*argv))
        with patch.object(sys, "argv", ["render.py", "--v2_camera_path", self.PATH, "--video"]):
            with self.assertRaises(ValueError):
                cli.validate_args(cli.parse_args())

    def test_valid_combinations_accepted(self):
        cli.validate_args(_args("--v2_camera_path", self.PATH, "--video"))
        cli.validate_args(_args("--v2_camera_path", self.PATH, "--v2_camera_path_time", "3"))

    def test_n_frames_resolution(self):
        self.assertEqual(cli.resolve_n_frames(_args(), None), cli.N_FRAMES_DEFAULT)
        self.assertEqual(cli.resolve_n_frames(_args("--n_frames", "10"), None), 10)
        self.assertEqual(cli.resolve_n_frames(_args("--video", "--fps", "60"), 87.5), 5250)
        self.assertEqual(cli.resolve_n_frames(_args("--video", "--fps", "60", "--n_frames", "5250"), 87.5), 5250)
        with self.assertRaises(ValueError):
            cli.resolve_n_frames(_args("--video", "--fps", "60", "--n_frames", "100"), 87.5)
        with self.assertRaises(ValueError):
            cli.resolve_n_frames(_args("--video", "--fps", "10"), 0.01)


if __name__ == "__main__":
    unittest.main()
