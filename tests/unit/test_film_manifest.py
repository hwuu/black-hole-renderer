#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层清单（film.json）读取与校验的单元测试。

覆盖：合法清单的字段解析（含镜头级 padding 合并、分辨率预设映射）；
封闭 schema 与各类取值/引用错误的报错（信息须给出字段名）。
"""

import os
import tempfile
import unittest

from src.film.manifest import Asset, Padding, Shot, Transition, load_manifest
from tests.unit._film_fixture import valid_spec, write_manifest


class LoadValidManifestTest(unittest.TestCase):
    def _load(self, tmp, **kwargs):
        return load_manifest(write_manifest(tmp, **kwargs))

    def test_fields_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load(tmp)
            self.assertEqual(manifest.title, "测试片")
            self.assertEqual(manifest.size, (1920, 1080))
            self.assertEqual(manifest.fps, 30)
            self.assertEqual(manifest.voice.engine, "say")
            self.assertEqual(manifest.padding, Padding(lead=0.4, gap=0.25, tail=0.6))
            self.assertEqual(manifest.render_defaults, ("--ar1", "3", "--ar2", "30", "--device", "gpu"))
            self.assertEqual(manifest.assets["img1"], Asset(url="https://example.com/a.png",
                                                            credit="EHT Collaboration",
                                                            license="CC BY 4.0"))

    def test_resolution_key_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load(tmp)
            self.assertEqual(manifest.resolution_key, "fhd")
            self.assertEqual(manifest.film_id, os.path.basename(tmp))
            self.assertEqual(manifest.output_dir, os.path.join("output", "scenes", manifest.film_id))

    def test_shot_kinds_and_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load(tmp)
            s01, s02 = manifest.shots
            self.assertEqual((s01.kind, s02.kind), ("render", "diagram"))
            self.assertIsNone(s01.transition)
            self.assertEqual(s02.transition, Transition(type="fade", duration=0.5))
            self.assertFalse(s01.hud)
            self.assertEqual(s01.narration, ("第一句话。", "第二句话。"))

    def test_shot_padding_overrides_global(self):
        spec = valid_spec()
        spec["shots"][0]["padding"] = {"lead": 6.0}
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load(tmp, spec=spec)
            self.assertEqual(manifest.shots[0].padding, Padding(lead=6.0, gap=0.25, tail=0.6))
            self.assertEqual(manifest.shots[1].padding, Padding(lead=0.4, gap=0.25, tail=0.6))

    def test_image_shot_layers(self):
        spec = valid_spec()
        spec["shots"].append({"id": "s03", "kind": "image",
                              "layers": [{"asset": "img1", "box": [0.1, 0.2, 0.5, 0.4]}],
                              "min_duration": 4.0})
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load(tmp, spec=spec)
            shot = manifest.shot("s03")
            self.assertEqual(shot.layers[0].asset, "img1")
            self.assertEqual(shot.layers[0].box, (0.1, 0.2, 0.5, 0.4))

    def test_voxcpm_engine_accepted(self):
        spec = valid_spec()
        spec["voice"] = {"engine": "voxcpm",
                         "name": "沉稳亲切的年轻男性，语速偏慢，抑扬顿挫分明", "rate": "+0%"}
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load(tmp, spec=spec)
            self.assertEqual(manifest.voice.engine, "voxcpm")
            self.assertIn("语速偏慢", manifest.voice.name)


class ManifestValidationTest(unittest.TestCase):
    def _load(self, spec):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(tmp, spec=spec)
            with self.assertRaises(ValueError) as ctx:
                load_manifest(path)
            return str(ctx.exception)

    def _assert_rejects(self, mutate, *field_fragments):
        spec = valid_spec()
        mutate(spec)
        message = self._load(spec)
        for fragment in field_fragments:
            self.assertIn(fragment, message)

    def test_unknown_top_level_field(self):
        self._assert_rejects(lambda s: s.update(extra=1), "film.json", "extra")

    def test_unknown_shot_field(self):
        self._assert_rejects(lambda s: s["shots"][0].update(foo=1), "shots[0]", "foo")

    def test_missing_title(self):
        self._assert_rejects(lambda s: s.pop("title"), "title")

    def test_size_must_match_preset(self):
        self._assert_rejects(lambda s: s.update(size=[1000, 500]), "size")

    def test_fps_must_be_positive_int(self):
        for bad in (0, "30", 2.5, True):
            self._assert_rejects(lambda s, b=bad: s.update(fps=b), "fps")

    def test_unknown_voice_engine(self):
        self._assert_rejects(lambda s: s["voice"].update(engine="piper"), "voice.engine")

    def test_reserved_render_args_rejected(self):
        for reserved in ("--video", "-r", "--disk_model"):
            def mutate(s, r=reserved):
                s["render_defaults"].append(r)
            self._assert_rejects(mutate, "render_defaults", reserved)

    def test_reserved_render_args_with_equals_syntax(self):
        """`--参数=值` 写法同样拒绝：argparse 接受该写法，只比对整串会漏防。"""
        for reserved in ("--video=1", "--disk_model=v1", "--n_frames=1"):
            def mutate(s, r=reserved):
                s["render_defaults"].append(r)
            self._assert_rejects(mutate, "render_defaults", reserved)

    def test_reserved_shot_render_args_with_equals_syntax(self):
        def mutate(s):
            s["shots"][0]["render_args"] = ["--disk_model=v1"]
        self._assert_rejects(mutate, "render_args")

    def test_duplicate_shot_id(self):
        self._assert_rejects(lambda s: s["shots"][1].update(id="s01"), "s01")

    def test_invalid_shot_id(self):
        self._assert_rejects(lambda s: s["shots"][0].update(id="S-1"), "shots[0].id")

    def test_render_shot_requires_path(self):
        self._assert_rejects(lambda s: s["shots"][0].pop("path"), "path")

    def test_diagram_shot_requires_scene(self):
        self._assert_rejects(lambda s: s["shots"][1].pop("scene"), "scene")

    def test_diagram_scene_format(self):
        for bad in ("light.py", "a:b:c", ":Light"):
            def mutate(s, b=bad):
                s["shots"][1]["scene"] = b
            self._assert_rejects(mutate, "scene")

    def test_image_shot_requires_layers(self):
        def mutate(s):
            s["shots"].append({"id": "s03", "kind": "image", "min_duration": 4.0})
        self._assert_rejects(mutate, "layers")

    def test_kind_fields_are_exclusive(self):
        def mutate(s):
            s["shots"][1]["path"] = "shots/s02.json"
            s["shots"][1]["hud"] = True
        self._assert_rejects(mutate, "path")

    def test_layer_box_must_stay_in_frame(self):
        def mutate(s):
            s["shots"].append({"id": "s03", "kind": "image",
                               "layers": [{"asset": "img1", "box": [0.6, 0.0, 0.6, 0.5]}]})
        self._assert_rejects(mutate, "box")

    def test_layer_box_rejects_non_finite(self):
        for bad in (float("nan"), float("inf")):
            def mutate(s, b=bad):
                s["shots"].append({"id": "s03", "kind": "image",
                                   "layers": [{"asset": "img1", "box": [b, 0.0, 0.5, 0.5]}]})
            self._assert_rejects(mutate, "box")

    def test_layer_asset_must_be_defined(self):
        def mutate(s):
            s["shots"].append({"id": "s03", "kind": "image",
                               "layers": [{"asset": "nope", "box": [0, 0, 0.5, 0.5]}]})
        self._assert_rejects(mutate, "nope")

    def test_sfx_shot_must_exist(self):
        self._assert_rejects(lambda s: s["sfx"][0].update(shot="s99"), "sfx", "s99")

    def test_fade_duration_must_be_positive(self):
        def mutate(s):
            s["shots"][1]["transition"] = {"type": "fade", "duration": 0.0}
        self._assert_rejects(mutate, "transition.duration")

    def test_first_shot_cannot_fade(self):
        def mutate(s):
            s["shots"][0]["transition"] = {"type": "fade", "duration": 0.5}
        self._assert_rejects(mutate, "transition", "首个镜头")

    def test_padding_must_be_nonnegative(self):
        self._assert_rejects(lambda s: s["padding"].update(gap=-0.1), "padding.gap")

    def test_music_must_be_path_or_null(self):
        self._assert_rejects(lambda s: s.update(music=5), "music")

    def test_shot_lookup_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp))
            self.assertIsInstance(manifest.shot("s01"), Shot)
            with self.assertRaises(ValueError):
                manifest.shot("s99")


if __name__ == "__main__":
    unittest.main()
