#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 视觉验收脚本配置回归测试。

D1：主验收输出关闭 visual atlas，走 V2 体积主路径。
D2：主验收使用 v2.2 推荐参数 (ar1=3, ar2=50, pov="30 0 10")，
    小盘 (ar1=2, ar2=15) 作为独立的 v1 兼容对照。
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "v2_visual_acceptance.sh"


def _command_for_output(script_text: str, output_name: str) -> str:
    """提取生成指定输出文件的 render.py 命令片段。

    Args:
        script_text: 脚本文本。
        output_name: `output/` 下的文件名，如 `v2_acceptance_bloom.png`。

    Returns:
        从 `$PYTHON render.py` 起、到 `-o output/<name>` 止之间的命令子串。
    """
    output_marker = f"-o output/{output_name}"
    output_idx = script_text.find(output_marker)
    if output_idx < 0:
        raise AssertionError(f"未找到生成 {output_name} 的命令")
    command_start = script_text.rfind("$PYTHON render.py", 0, output_idx)
    if command_start < 0:
        raise AssertionError(f"未找到生成 {output_name} 的 render.py 命令")
    return script_text[command_start:output_idx]


# v2.2 推荐主验收参数：r_out=50 让标准薄盘温度跨度达到 4.32 倍。
# 相机距离 ~100 r_s 让 r_out=50 的盘完整可见，留出黑洞透镜余地。
# 见 docs/design_ad_v2.md §2.4 与 v2.1 变更记录。
_EXPECTED_ACCEPT_AR1 = "3"
_EXPECTED_ACCEPT_AR2 = "50"
_EXPECTED_ACCEPT_POV = "95 0 32"

# v1 默认行为对应的小盘参数，留作对照。
_EXPECTED_COMPAT_AR1 = "2"
_EXPECTED_COMPAT_AR2 = "15"
_EXPECTED_COMPAT_POV = "24 0 8"


class V2VisualAcceptanceScriptTest(unittest.TestCase):
    def setUp(self):
        self.script = SCRIPT.read_text(encoding="utf-8")

    # --- D1：主验收必须走 volume 主路径 ---

    def test_acceptance_default_uses_volume_path(self):
        """D1：主验收输出默认关闭 visual atlas，走 V2 体积主路径。"""
        for output in (
            "v2_step0_baseline.png",
            "v2_acceptance_no_bloom.png",
            "v2_acceptance_bloom.png",
        ):
            command = _command_for_output(self.script, output)
            self.assertIn("--v2_disable_visual_atlas", command, msg=output)

    def test_atlas_output_is_only_comparison_path(self):
        """D1：atlas thin-layer 只作为额外对照图，不是主验收。"""
        command = _command_for_output(self.script, "v2_acceptance_atlas.png")
        self.assertNotIn("--v2_disable_visual_atlas", command)

    # --- D2：主验收使用 v2.2 推荐参数 ---

    def test_main_acceptance_params_use_v2_recommended_disk_radius(self):
        """D2：主验收脚本定义 `ACCEPT_AR1=3` 与 `ACCEPT_AR2=50`，对应 v2.2 推荐范围。"""
        m_ar1 = re.search(r"^ACCEPT_AR1\s*=\s*(\d+)\s*$", self.script, re.MULTILINE)
        m_ar2 = re.search(r"^ACCEPT_AR2\s*=\s*(\d+)\s*$", self.script, re.MULTILINE)
        self.assertIsNotNone(m_ar1, "脚本缺少 ACCEPT_AR1 定义")
        self.assertIsNotNone(m_ar2, "脚本缺少 ACCEPT_AR2 定义")
        self.assertEqual(m_ar1.group(1), _EXPECTED_ACCEPT_AR1)
        self.assertEqual(m_ar2.group(1), _EXPECTED_ACCEPT_AR2)

    def test_main_acceptance_pov_matches_v2_recommended_camera(self):
        """D2：主验收脚本 `ACCEPT_POV` 与 v2.2 推荐相机一致。"""
        m_pov = re.search(r'^ACCEPT_POV\s*=\s*"([^"]+)"\s*$', self.script, re.MULTILINE)
        self.assertIsNotNone(m_pov, "脚本缺少 ACCEPT_POV 定义")
        # POV 用空格分隔的三个浮点，比较时按空白归一化。
        self.assertEqual(
            re.sub(r"\s+", " ", m_pov.group(1).strip()),
            _EXPECTED_ACCEPT_POV,
        )

    def test_main_acceptance_commands_reference_accept_common(self):
        """D2：所有主验收命令引用 `ACCEPT_COMMON` 数组，而不是写死的小盘参数。"""
        for output in (
            "v2_step0_baseline.png",
            "v2_acceptance_no_bloom.png",
            "v2_acceptance_bloom.png",
            "v2_acceptance_atlas.png",
        ):
            command = _command_for_output(self.script, output)
            self.assertIn('"${ACCEPT_COMMON[@]}"', command, msg=output)
            self.assertNotIn('"${COMPAT_COMMON[@]}"', command, msg=output)

    # --- D2：小盘对照参数定义存在 ---

    def test_compat_params_preserve_v1_default_disk_radius(self):
        """D2：脚本保留 `COMPAT_AR1=2` 与 `COMPAT_AR2=15`，作为 v1 兼容对照。"""
        m_ar1 = re.search(r"^COMPAT_AR1\s*=\s*(\d+)\s*$", self.script, re.MULTILINE)
        m_ar2 = re.search(r"^COMPAT_AR2\s*=\s*(\d+)\s*$", self.script, re.MULTILINE)
        self.assertIsNotNone(m_ar1, "脚本缺少 COMPAT_AR1 定义")
        self.assertIsNotNone(m_ar2, "脚本缺少 COMPAT_AR2 定义")
        self.assertEqual(m_ar1.group(1), _EXPECTED_COMPAT_AR1)
        self.assertEqual(m_ar2.group(1), _EXPECTED_COMPAT_AR2)

    def test_v1_classic_uses_compat_small_disk(self):
        """D2：V1 classic 技术对照使用小盘参数，明确不与 V2 主验收混在一起。"""
        command = _command_for_output(self.script, "v1_classic_darksky.png")
        self.assertIn('"${COMPAT_COMMON[@]}"', command)
        self.assertNotIn('"${ACCEPT_COMMON[@]}"', command)

    def test_compat_small_disk_v2_output_exists(self):
        """D2：脚本产出独立的 `v2_compat_small_disk.png`，用作小盘对照。"""
        command = _command_for_output(self.script, "v2_compat_small_disk.png")
        self.assertIn('"${COMPAT_COMMON[@]}"', command)
        self.assertIn("--disk_model v2", command)
        # 小盘对照同样走 volume 路径，便于与主验收同条件横向比较。
        self.assertIn("--v2_disable_visual_atlas", command)


if __name__ == "__main__":
    unittest.main()
