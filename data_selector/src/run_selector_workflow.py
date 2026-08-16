#!/usr/bin/env python3
"""Run the LeRobot dataset selection interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = TOOL_ROOT / "src"
DEFAULT_HF_HOME = TOOL_ROOT.parent / "data" / "selected"


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resolve_dataset(target: str, hf_home: Path) -> Path:
    """Resolve an absolute path, a relative path, or a dataset name."""
    target_path = Path(target).expanduser()
    candidates = [target_path, Path.cwd() / target_path, hf_home / target_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到数据集 {target!r}，已检查：\n  - {checked}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互筛选 LeRobot 数据。"
    )
    parser.add_argument("target", help="数据集目录、相对路径，或 --hf-home 下的数据集名称")
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"本地 LeRobot 数据集根目录（默认：{DEFAULT_HF_HOME}）",
    )
    parser.add_argument(
        "--only",
        choices=("select",),
        default="select",
        help="打开数据筛选界面（仅支持：select）",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="selection JSON 路径；默认保存在 data_selector/selections/",
    )
    parser.add_argument(
        "--start-episode", type=int, default=0, help="从指定 episode 开始查看（默认：0）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_home = args.hf_home.expanduser().resolve()
    dataset_dir = resolve_dataset(args.target, hf_home)

    selections_dir = TOOL_ROOT / "selections"
    selections_dir.mkdir(parents=True, exist_ok=True)
    selection_path = (
        args.selection.expanduser().resolve()
        if args.selection
        else selections_dir / f"selection_{dataset_dir.name}.json"
    )
    print(f"数据集：{dataset_dir}")
    print(f"筛选记录：{selection_path}")

    python = sys.executable
    run_command(
        [
            python,
            str(SRC_DIR / "data_selector.py"),
            "--dataset",
            str(dataset_dir),
            "--output",
            str(selection_path),
            "--start",
            str(args.start_episode),
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
