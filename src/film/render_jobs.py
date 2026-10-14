"""渲染任务：为每个 render 镜头生成 `render.py` 命令（本机 / 远端 4090 两种目标）。

设计见 `docs/design_film.md` §4.3。命令假定在仓库根目录执行（`render.py` 与
`scenes/<film_id>/` 相对可见）；远端目标按 AGENTS.md "远程渲染" 约定替换解释器与输出根目录。
"""

from __future__ import annotations

import json
import os
import shlex
from typing import List, Optional

from src.v2.exposure_curve import FadeSettings

from .manifest import FilmManifest, Shot

REMOTE_PYTHON = "/root/miniconda3/bin/python"
"""远端（AutoDL 4090）的 Python 解释器路径（AGENTS.md 远程渲染）。"""

REMOTE_OUTPUT_ROOT = "/root/autodl-tmp/out"
"""远端产物根目录（数据盘，关机保留 15 天）。"""

REMOTE_REPO_DIR = "/root/autodl-tmp/bhr"
"""远端代码目录（render_jobs.sh 在此目录下执行）。"""


def build_command(manifest: FilmManifest, shot: Shot, output_path: str,
                  python_bin: str = "python") -> str:
    """拼一条 render 镜头的 `render.py` 视频命令。

    Args:
        manifest: 清单（取 `render_defaults`、`resolution_key`、`fps` 与 `film_id`）。
        shot: render 镜头（取 `path` 与 `render_args`）。
        output_path: 输出 mp4 路径（`<输出根>/shots/<id>.mp4`）。
        python_bin: Python 解释器（本机 `python`，远端 `REMOTE_PYTHON`）。

    Returns:
        单行 shell 命令（参数以 `shlex.quote` 防御性加引号）。

    Notes:
        运镜路径按仓库约定写成 `scenes/<film_id>/<shot.path>`（清单位于
        `scenes/<film_id>/film.json`，命令在仓库根目录执行）。
    """
    camera_path = os.path.join("scenes", manifest.film_id, shot.path)
    parts = [python_bin, "render.py", "--disk_model", "v2", "--video",
             *manifest.render_defaults, *shot.render_args,
             "--v2_camera_path", camera_path,
             "-r", manifest.resolution_key,
             "--fps", str(manifest.fps),
             "-o", output_path]
    return " ".join(shlex.quote(p) for p in parts)


def _fade_comment(manifest: FilmManifest, shot: Shot) -> Optional[str]:
    """检查镜头路径文件的 `fade`：会与清单 transition 叠加时给出警告注释。

    引擎在路径省略 `fade` 块（或其中单个字段）时取默认淡入 2 s、淡出 3 s
    （`exposure_curve.FadeSettings`），因此缺失与显式非 0 同样会叠加淡化，一律警告。

    Args:
        manifest: 清单（定位路径文件）。
        shot: render 镜头。

    Returns:
        警告注释行；路径文件不可读 / `fade` 块结构非法时返回提示注释；
        两个字段都为 0 时返回 None（无淡化，与成片镜头的约定一致）。
    """
    path_file = os.path.join(manifest.base_dir, shot.path)
    try:
        with open(path_file, encoding="utf-8") as f:
            fade = json.load(f).get("fade")
    except (OSError, ValueError):
        return f"# {shot.id}: 路径文件不可读，无法检查 fade（timeline 步骤会给出详细错误）"
    defaults = FadeSettings()
    if fade is None:
        fade_in, fade_out = defaults.fade_in, defaults.fade_out
    elif isinstance(fade, dict):
        try:
            fade_in = float(fade.get("fade_in", defaults.fade_in))
            fade_out = float(fade.get("fade_out", defaults.fade_out))
        except (TypeError, ValueError):
            return f"# {shot.id}: fade 块字段不是数字，请检查路径文件"
    else:
        return f"# {shot.id}: fade 块不是对象，请检查路径文件"
    if fade_in > 0 or fade_out > 0:
        return (f"# {shot.id}: 路径 fade 为 {fade_in:g}s / {fade_out:g}s"
                f"（省略时引擎默认 {defaults.fade_in:g}s / {defaults.fade_out:g}s），"
                f"会与清单 transition 叠加；请把路径文件的 fade 块显式设为 0")
    return None


def build_jobs_script(manifest: FilmManifest, remote: bool = False, log=print) -> str:
    """生成 `render_jobs.sh` 的完整内容（本机或远端目标）。

    Args:
        manifest: 清单。
        remote: True = 生成 4090 远端命令（替换解释器与输出根目录、先 `cd` 到远端代码目录）。
        log: 警告输出函数（fade 非 0 等问题同时打到控制台）。

    Returns:
        shell 脚本文本（`#!/bin/sh`、`set -e`、`mkdir -p` 输出目录、逐镜头命令）。

    Notes:
        续传：某镜头中断后，把该行命令复制出来加 `--resume` 重跑（最多重渲一个分段）；
        分段临时目录 `.v2seg_*` 建在输出文件同目录，跨机器续传需一并同步。
    """
    python_bin = REMOTE_PYTHON if remote else "python"
    output_root = os.path.join(REMOTE_OUTPUT_ROOT, manifest.film_id) if remote else manifest.output_dir
    shots_dir = os.path.join(output_root, "shots")
    lines = ["#!/bin/sh",
             "# 由 python -m src.film <film.json> jobs 生成；在仓库根目录执行"
             if not remote else f"# 由 python -m src.film <film.json> jobs --remote 生成",
             "# 逐条执行；中断后把该行命令加 --resume 重跑续传（分段目录 .v2seg_* 与输出文件同目录）"]
    if remote:
        lines.append(f"cd {REMOTE_REPO_DIR}")
    lines += ["set -e", f"mkdir -p {shlex.quote(shots_dir)}"]
    for shot in manifest.shots:
        if shot.kind != "render":
            continue
        comment = _fade_comment(manifest, shot)
        if comment:
            lines.append(comment)
            log(comment)
        output_path = os.path.join(shots_dir, f"{shot.id}.mp4")
        lines.append(build_command(manifest, shot, output_path, python_bin))
    return "\n".join(lines) + "\n"


def write_jobs_script(manifest: FilmManifest, path: str, remote: bool = False,
                      log=print) -> str:
    """把 `render_jobs.sh` 写到 `path` 并加执行权限。

    Args:
        manifest: 清单。
        path: 脚本输出路径（设计约定为 `<产物目录>/render_jobs.sh`）。
        remote: 是否生成远端目标。
        log: 警告输出函数。

    Returns:
        脚本路径（同 `path`）。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_jobs_script(manifest, remote=remote, log=log))
    os.chmod(path, 0o755)
    return path


def render_shot_paths(manifest: FilmManifest) -> List[str]:
    """清单中全部 render 镜头的运镜路径文件。

    Args:
        manifest: 清单。

    Returns:
        绝对路径列表（按镜头顺序）；没有 render 镜头时为空。
    """
    return [os.path.join(manifest.base_dir, shot.path)
            for shot in manifest.shots if shot.kind == "render"]
