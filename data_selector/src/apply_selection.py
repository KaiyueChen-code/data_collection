#!/usr/bin/env python3
"""
Apply DataSelector selection.json to a LeRobot v2 dataset.

This script creates a filtered copy of the dataset by dropping episodes
listed in selection.json["deleted"]["episodes"], then reindexes episodes
and frame indices to keep metadata consistent.

Layout: ``data/chunk-*/episode_*.parquet`` only (LeRobot v2.x).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _table_with_reindexed_episode(table: pa.Table, new_ep: int, frame_start: int) -> pa.Table:
    """Update episode_index / index in-place on column types; keeps Arrow schema (e.g. image types)."""
    out = table
    n = table.num_rows
    if "episode_index" in table.column_names:
        field = table.schema.field("episode_index")
        out = out.set_column(
            out.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array([new_ep] * n, type=field.type),
        )
    if "index" in out.column_names:
        field = out.schema.field("index")
        out = out.set_column(
            out.schema.get_field_index("index"),
            "index",
            pa.array(range(frame_start, frame_start + n), type=field.type),
        )
    return out


def collect_episode_parquet_paths(dataset_dir: Path) -> dict[int, Path]:
    """
    LeRobot v2 only: map episode_index -> parquet path (no full table load; avoids OOM).

    Episode id is taken from filename ``episode_<idx>.parquet``; if the file embeds a
    different ``episode_index`` column, the writer still rewrites indices consistently.
    """
    episodes: dict[int, Path] = {}

    v2_files = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not v2_files:
        raise FileNotFoundError(
            "No LeRobot v2 episode parquet files found under data/chunk-*/episode_*.parquet. "
            "This script only supports v2 layout (one file per episode)."
        )

    for p in v2_files:
        old_ep = int(p.stem.split("_")[-1])
        episodes[old_ep] = p

    return episodes


def parquet_num_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def rebuild_dataset(src_dir: Path, out_dir: Path, deleted_eps: set[int], dry_run: bool):
    meta_dir = src_dir / "meta"
    info = load_json(meta_dir / "info.json")
    tasks = load_jsonl(meta_dir / "tasks.jsonl")
    episodes_meta = load_jsonl(meta_dir / "episodes.jsonl")
    stats_meta = load_jsonl(meta_dir / "episodes_stats.jsonl")
    stats_by_old = {int(x["episode_index"]): x for x in stats_meta if "episode_index" in x}

    paths_by_ep = collect_episode_parquet_paths(src_dir)
    all_old_eps = sorted(paths_by_ep.keys())
    keep_old_eps = [ep for ep in all_old_eps if ep not in deleted_eps]

    if not keep_old_eps:
        raise RuntimeError("All episodes are filtered out. Nothing to write.")

    old_to_new = {old_ep: new_ep for new_ep, old_ep in enumerate(keep_old_eps)}
    used_task_indices: set[int] = set()
    new_episodes: list[dict] = []
    new_stats: list[dict] = []

    frame_offset = 0
    total_frames = 0

    for old_ep in keep_old_eps:
        new_ep = old_to_new[old_ep]
        ep_len = parquet_num_rows(paths_by_ep[old_ep])

        # Prefer original episode metadata when available.
        old_meta = next((x for x in episodes_meta if int(x["episode_index"]) == old_ep), None)
        if old_meta is not None:
            ep_tasks = old_meta.get("tasks", [])
            ep_length = int(old_meta.get("length", ep_len))
        else:
            ep_tasks = []
            ep_length = ep_len

        for t in ep_tasks:
            if isinstance(t, int):
                used_task_indices.add(t)

        new_episodes.append(
            {
                "episode_index": new_ep,
                "tasks": ep_tasks,
                "length": ep_length,
            }
        )

        old_stat = stats_by_old.get(old_ep)
        if old_stat is not None:
            stat = json.loads(json.dumps(old_stat))
            stat["episode_index"] = new_ep
            inner = stat.get("stats", {})
            if "episode_index" in inner:
                inner["episode_index"]["min"] = [new_ep]
                inner["episode_index"]["max"] = [new_ep]
                inner["episode_index"]["mean"] = [float(new_ep)]
            if "index" in inner:
                for k in ("min", "max", "mean"):
                    if k in inner["index"] and inner["index"][k]:
                        inner["index"][k] = [inner["index"][k][0] - int(inner["index"]["min"][0]) + frame_offset]
            new_stats.append(stat)

        frame_offset += ep_len
        total_frames += ep_len

    new_tasks = [t for t in tasks if int(t.get("task_index", -1)) in used_task_indices] if tasks else []

    new_info = json.loads(json.dumps(info))
    new_info["total_episodes"] = len(keep_old_eps)
    new_info["total_frames"] = total_frames
    if "total_tasks" in new_info:
        new_info["total_tasks"] = len(new_tasks) if new_tasks else len(tasks)
    if "splits" in new_info:
        new_info["splits"] = {"train": f"0:{len(keep_old_eps)}"}

    print(f"[INFO] Source dataset : {src_dir}")
    print(f"[INFO] Output dataset : {out_dir}")
    print(f"[INFO] Deleted episodes: {len(deleted_eps)}")
    print(f"[INFO] Kept episodes   : {len(keep_old_eps)}")
    print(f"[INFO] Total frames   : {total_frames}")

    if dry_run:
        print("[DRY-RUN] Skip writing files.")
        return

    # Copy all metadata and overwrite filtered ones.
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    shutil.copytree(src_dir, out_dir)
    shutil.rmtree(out_dir / "data", ignore_errors=True)

    save_json(out_dir / "meta" / "info.json", new_info)
    save_jsonl(out_dir / "meta" / "episodes.jsonl", new_episodes)
    if tasks:
        save_jsonl(out_dir / "meta" / "tasks.jsonl", new_tasks if new_tasks else tasks)
    if stats_meta:
        save_jsonl(out_dir / "meta" / "episodes_stats.jsonl", new_stats)

    # Re-write data after copy/reset to ensure only kept episodes remain.
    frame_offset = 0
    for old_ep in keep_old_eps:
        new_ep = old_to_new[old_ep]
        ep_path = paths_by_ep[old_ep]
        tbl = pq.read_table(ep_path)
        if "episode_index" not in tbl.column_names and tbl.num_rows > 0:
            old_from_name = int(ep_path.stem.split("_")[-1])
            tbl = tbl.append_column(
                "episode_index",
                pa.array([old_from_name] * tbl.num_rows, type=pa.int64()),
            )
        ep_len = tbl.num_rows
        new_tbl = _table_with_reindexed_episode(tbl, new_ep, frame_offset)
        chunk = new_ep // 1000
        out_file = out_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{new_ep:06d}.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_tbl, out_file, compression="snappy")
        frame_offset += ep_len
        del tbl, new_tbl

    print("[DONE] Filtered dataset written successfully.")


def main():
    parser = argparse.ArgumentParser(description="Apply DataSelector selection.json to LeRobot dataset.")
    parser.add_argument("--dataset", "-d", required=True, type=Path, help="Source LeRobot dataset path")
    parser.add_argument("--selection", "-s", required=True, type=Path, help="selection.json path")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output filtered dataset path")
    parser.add_argument("--dry-run", action="store_true", help="Only print summary, do not write files")
    args = parser.parse_args()

    selection = load_json(args.selection)
    deleted = selection.get("deleted", {}).get("episodes", [])
    deleted_eps = {int(x) for x in deleted}
    rebuild_dataset(args.dataset, args.output, deleted_eps, args.dry_run)


if __name__ == "__main__":
    main()
