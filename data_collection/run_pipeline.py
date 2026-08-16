import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

work_dir = Path(__file__).parent.absolute()
print(work_dir)
sys.path.append(str(work_dir))
os.chdir(work_dir)
print(f"Working directory: {work_dir}")

project_root = work_dir.parent  # VB-VLA 根目录

# 从 config.py 中读取所有参数（单一配置入口，替代 VB_task_config.yaml）
sys.path.insert(0, str(project_root))
from data_collection.config import DATA_CONVERT_CONFIG


def build_cmd_for_step(step_name, step_path, cfg):
    """为每个 pipeline 步骤构建 CLI 命令，参数全部来自 DataConvertConfig。"""
    if step_name == "01_crop_img.py":
        return [
            sys.executable, str(step_path),
            "--task_name", cfg.task_name,
            "--task_type", cfg.task_type,
            "--single_hand_side", cfg.single_hand_side,
            "--visual_out_res", str(cfg.visual_out_res[0]), str(cfg.visual_out_res[1]),
            "--tactile_out_res", str(cfg.tactile_out_res[0]), str(cfg.tactile_out_res[1]),
        ]

    elif step_name == "02_get_aruco_pos.py":
        marker_size_map_json = json.dumps(
            {str(k): v for k, v in cfg.marker_size_map.items()}
        )
        return [
            sys.executable, str(step_path),
            "--task_name", cfg.task_name,
            "--task_type", cfg.task_type,
            "--single_hand_side", cfg.single_hand_side,
            "--cam_intrinsic_json_path", cfg.cam_intrinsic_json_path,
            "--aruco_dict", cfg.aruco_dict,
            "--marker_size_map", marker_size_map_json,
            "--max_workers", str(cfg.aruco_max_workers),
        ]

    elif step_name == "03_get_width.py":
        return [
            sys.executable, str(step_path),
            "--task_name", cfg.task_name,
            "--task_type", cfg.task_type,
            "--single_hand_side", cfg.single_hand_side,
            "--left_aruco_left_id", str(cfg.left_aruco_left_id),
            "--left_aruco_right_id", str(cfg.left_aruco_right_id),
            "--right_aruco_left_id", str(cfg.right_aruco_left_id),
            "--right_aruco_right_id", str(cfg.right_aruco_right_id),
        ]

    elif step_name == "04_generate_dataset_plan.py":
        cmd = [
            sys.executable, str(step_path),
            "--task_name", cfg.task_name,
            "--min_episode_length", str(cfg.min_episode_length),
            "--visual_cam_latency", str(cfg.visual_cam_latency),
            "--pose_latency", str(cfg.pose_latency),
        ]
        if cfg.use_tactile_img:
            cmd.append("--use_tactile_img")
        return cmd

    elif step_name == "05_convert_raw_to_lerobot.py":
        cmd = [
            sys.executable, str(step_path),
            "--task_name", cfg.task_name,
            "--raw_data_dir", cfg.raw_data_dir,
            "--selected_data_dir", cfg.selected_data_dir,
            "--episodes_per_chunk", str(cfg.episodes_per_chunk),
            "--image_writer_threads", str(cfg.image_writer_threads),
            "--image_writer_processes", str(cfg.image_writer_processes),
            "--lerobot-jpeg-quality", str(cfg.lerobot_jpeg_quality),
            "--fps", str(cfg.fps),
            "--smooth_sigma", str(cfg.smooth_sigma),
            "--tag_scale", str(cfg.tag_scale),
            "--language_instruction", *cfg.language_instruction,
        ]
        if cfg.output_repo_id:
            cmd.extend(["--output_repo_id", cfg.output_repo_id])
        if cfg.single_arm:
            cmd.append("--single_arm")
        if cfg.no_state:
            cmd.append("--no_state")
        if cfg.use_tactile_img:
            cmd.append("--use_tactile_img")
        if cfg.use_inpaint_tag:
            cmd.append("--use_inpaint_tag")
        if cfg.use_mask:
            cmd.extend([
                "--use_mask",
                "--fisheye_radius", str(cfg.fisheye_mask_radius),
                "--fisheye_fill_color",
                str(cfg.fisheye_mask_fill_color[0]),
                str(cfg.fisheye_mask_fill_color[1]),
                str(cfg.fisheye_mask_fill_color[2]),
            ])
            if cfg.fisheye_mask_center is not None:
                cmd.extend(["--fisheye_center",
                             str(cfg.fisheye_mask_center[0]),
                             str(cfg.fisheye_mask_center[1])])
        return cmd

    raise ValueError(f"Unknown step: {step_name}")


def run_pipeline(only_step: str | None = None):
    cfg = DATA_CONVERT_CONFIG
    pipeline_dir = work_dir / "pipeline"

    print("Pipeline config:")
    print(f"  task_name      : {cfg.task_name}")
    print(f"  task_type      : {cfg.task_type}")
    print(f"  raw_data_dir   : {cfg.raw_data_dir}")
    print(f"  selected_dir   : {cfg.selected_data_dir}")
    print(f"  chunk_size     : {cfg.episodes_per_chunk} episodes")
    print(f"  visual_out_res : {cfg.visual_out_res}")
    print(f"  use_tactile_img: {cfg.use_tactile_img}")
    print(f"  use_mask       : {cfg.use_mask}")
    print(f"  use_inpaint_tag: {cfg.use_inpaint_tag}")
    print(f"  fps            : {cfg.fps}")
    print(f"  no_state       : {cfg.no_state}")
    input("\nPress Enter to continue...")

    pipeline_steps = [
        "01_crop_img.py",
        "02_get_aruco_pos.py",
        "03_get_width.py",
        "04_generate_dataset_plan.py",
        "05_convert_raw_to_lerobot.py",
    ]
    if only_step is not None:
        pipeline_steps = [only_step]

    for step_name in pipeline_steps:
        step_path = pipeline_dir / step_name

        print(f"\n{'='*50}")
        print(f"Running step: {step_name}")
        print(f"{'='*50}\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(project_root)
            + os.pathsep + str(work_dir)
            + os.pathsep + env.get("PYTHONPATH", "")
        )

        cmd = build_cmd_for_step(step_name, step_path, cfg)
        print(f"Executing: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running step {step_name}:")
            print(f"Exit code: {e.returncode}")
            print("\nPipeline failed. Stopping execution.")
            return False

        print(f"\nCompleted step: {step_name}\n")

    print("\nPipeline completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    parser.add_argument(
        "--only",
        choices=["01", "02", "03", "04", "05"],
        default=None,
        help="Run only one pipeline step, for example --only 05 to resume conversion",
    )
    cli_args = parser.parse_args()
    step_by_number = {
        "01": "01_crop_img.py",
        "02": "02_get_aruco_pos.py",
        "03": "03_get_width.py",
        "04": "04_generate_dataset_plan.py",
        "05": "05_convert_raw_to_lerobot.py",
    }
    run_pipeline(step_by_number.get(cli_args.only))
