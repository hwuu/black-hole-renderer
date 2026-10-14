"""制片层单测的共享清单构造器（文件名不以 test 开头，unittest 不会收集本模块）。"""

import json
import os


def valid_spec():
    """合法清单（docs/design_film.md §3.1 示例结构：render + diagram 两个镜头）。"""
    return {
        "title": "测试片",
        "size": [1920, 1080],
        "fps": 30,
        "voice": {"engine": "say", "name": "Flo", "rate": "+0%"},
        "music": None,
        "padding": {"lead": 0.4, "gap": 0.25, "tail": 0.6},
        "render_defaults": ["--ar1", "3", "--ar2", "30", "--device", "gpu"],
        "shots": [
            {"id": "s01", "kind": "render", "path": "shots/s01.json",
             "narration": ["第一句话。", "第二句话。"], "min_duration": 5.0},
            {"id": "s02", "kind": "diagram", "scene": "diagrams/light.py:Light",
             "narration": ["图解的一句。"],
             "transition": {"type": "fade", "duration": 0.5}},
        ],
        "assets": {"img1": {"url": "https://example.com/a.png",
                            "credit": "EHT Collaboration", "license": "CC BY 4.0"}},
        "sfx": [{"shot": "s02", "kind": "siren_doppler", "at": 1.0,
                 "params": {"f0": 700.0, "v": 25.0}}],
    }


def write_manifest(tmp_dir, spec=None, shot_paths=None):
    """把清单（及可选的运镜路径文件）写进 `tmp_dir/film.json`，返回清单路径。

    Args:
        tmp_dir: 目标目录（不存在则创建）。
        spec: 清单内容；None = `valid_spec()`。
        shot_paths: `{相对路径: 时长秒}`，逐个写出只含 `duration` 的路径文件
            （timeline 的路径时长检查只读该字段）。
    """
    spec = valid_spec() if spec is None else spec
    os.makedirs(tmp_dir, exist_ok=True)
    for rel, duration in (shot_paths or {}).items():
        full = os.path.join(tmp_dir, rel)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            json.dump({"duration": duration}, f)
    path = os.path.join(tmp_dir, "film.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    return path
