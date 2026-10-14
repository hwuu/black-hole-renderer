"""制片层命令行入口：`python -m src.film <film.json> <子命令> [--remote]`。

子命令定义见 `docs/design_film.md` §4：narrate / timeline / jobs / sfx / subtitles /
diagrams / hud / assemble。
产物统一写到 `output/scenes/<film_id>/`（`film_id` = 清单所在目录名）。
"""

import argparse
import os
import shutil
import subprocess
import sys

from .manifest import load_manifest
from .narration import narrate_manifest
from .render_jobs import write_jobs_script
from .sfx import render_sfx
from .subtitles import build_ass
from .timeline import build_timeline, load_timeline

FUTURE_COMMANDS = ("assets", "all")


def _render_diagram(manifest, shot, diagrams_dir: str, log=print) -> str:
    """用 manim CLI 渲染一个 diagram 镜头并把产物移到目标位置。

    Args:
        manifest: 清单。
        shot: diagram 镜头（`scene` 字段为 `文件路径:类名`）。
        diagrams_dir: 产物目录（`<产物目录>/diagrams`）。
        log: 进度输出函数。

    Returns:
        最终 mp4 路径（`<diagrams_dir>/<shot_id>.mp4`）。

    Raises:
        subprocess.CalledProcessError: manim 渲染失败。
        FileNotFoundError: 场景文件不存在。
    """
    scene_rel, class_name = shot.scene.split(":")
    scene_file = os.path.join(manifest.base_dir, scene_rel)
    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"场景文件不存在：{scene_file}")

    os.makedirs(diagrams_dir, exist_ok=True)
    media_dir = os.path.join(diagrams_dir, "_manim")
    env = os.environ.copy()
    env["FILM_OUTPUT_DIR"] = manifest.output_dir
    # 场景文件 import src.film.diagram_kit，需要项目根目录在 PYTHONPATH 里
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(manifest.base_dir)))) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "manim", "render",
           "-r", f"{manifest.size[0]},{manifest.size[1]}",
           "--fps", str(manifest.fps),
           "--media_dir", media_dir,
           "--format", "mp4",
           scene_file, class_name]
    log(f"  渲染 {shot.id}（{class_name}）…")
    subprocess.run(cmd, check=True, env=env, cwd=manifest.base_dir)

    # 找到 manim 输出的 mp4 并移到目标位置
    src = None
    for root, _dirs, files in os.walk(media_dir):
        for f in files:
            if f.endswith(".mp4"):
                src = os.path.join(root, f)
                break
        if src:
            break
    if src is None:
        raise RuntimeError(f"manim 未产出 mp4（{media_dir}）")
    dst = os.path.join(diagrams_dir, f"{shot.id}.mp4")
    shutil.move(src, dst)
    shutil.rmtree(media_dir, ignore_errors=True)
    log(f"  完成 {shot.id} → {dst}")
    return dst


def main(argv=None) -> None:
    """命令行入口：读清单，按子命令分发到对应流水线步骤。

    Args:
        argv: 命令行参数；None = `sys.argv[1:]`（测试可显式传入）。

    Raises:
        NotImplementedError: `diagrams` / `hud` / `assets` / `assemble` / `all`（S6/S7 接入）。
    """
    parser = argparse.ArgumentParser(prog="python -m src.film",
                                     description="成片流水线（docs/design_film.md）")
    parser.add_argument("manifest", help="清单 film.json 路径")
    parser.add_argument("command",
                        choices=("narrate", "timeline", "jobs", "sfx", "subtitles",
                                  "diagrams", "hud", "assemble") + FUTURE_COMMANDS,
                        help="流水线子命令")
    parser.add_argument("--remote", action="store_true",
                        help="jobs：生成远端 4090 命令（替换解释器与输出根目录，见 AGENTS.md 远程渲染）")
    args = parser.parse_args(argv)
    if args.command in FUTURE_COMMANDS:
        raise NotImplementedError(f"子命令 {args.command!r} 在 S6/S7 接入")
    manifest = load_manifest(args.manifest)
    out_dir = manifest.output_dir
    if args.command == "narrate":
        durations = narrate_manifest(manifest, os.path.join(out_dir, "narration"))
        print(f"旁白 {len(durations)} 句 → {os.path.join(out_dir, 'narration')}")
    elif args.command == "timeline":
        durations = narrate_manifest(manifest, os.path.join(out_dir, "narration"))
        timeline = build_timeline(manifest, durations)
        path = os.path.join(out_dir, "timeline.json")
        timeline.save(path)
        print(f"总时长 {timeline.total_duration:.2f}s，镜头 {len(timeline.shots)} 个 → {path}")
    elif args.command == "jobs":
        path = write_jobs_script(manifest, os.path.join(out_dir, "render_jobs.sh"),
                                 remote=args.remote)
        print(f"渲染任务 → {path}（{'远端 4090' if args.remote else '本机'}）")
    elif args.command == "sfx":
        paths = render_sfx(manifest, os.path.join(out_dir, "sfx"))
        print(f"音效 {len(paths)} 条 → {os.path.join(out_dir, 'sfx')}")
    elif args.command == "subtitles":
        timeline = load_timeline(os.path.join(out_dir, "timeline.json"))
        path = os.path.join(out_dir, "subtitles.ass")
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_ass(manifest, timeline))
        print(f"字幕 → {path}")
    elif args.command == "diagrams":
        diagrams_dir = os.path.join(out_dir, "diagrams")
        os.makedirs(diagrams_dir, exist_ok=True)
        paths = []
        for shot in manifest.shots:
            if shot.kind == "diagram":
                paths.append(_render_diagram(manifest, shot, diagrams_dir))
        print(f"图解 {len(paths)} 段 → {diagrams_dir}")
    elif args.command == "hud":
        raise NotImplementedError("hud 子命令待实现（遥控器叠加层）")
    elif args.command == "assemble":
        from .assemble import assemble_film
        path = assemble_film(manifest, out_dir)
        print(f"成片 → {path}")


if __name__ == "__main__":
    main()
