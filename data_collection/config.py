#数据采集和数据处理的配置文件
from dataclasses import dataclass, field
from pathlib import Path

DATASET_RAW_TASK_NAME = "0407_test_raw"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
SELECTED_DATA_DIR = str(PROJECT_ROOT / "data" / "selected")

@dataclass
class CamConfig:
    # 01_crop_img.py 按 3840x800 三联图切分为三个 1280x800 区域。
    camera_format: str = "MJPG"
    camera_width: int = 3840
    camera_height: int = 800
    auto_exposure: int = 1
    exposure: int | None = None
    auto_white_balance: int = 1
    wb_temperature: int | None = None
    brightness: int = 0
    gain: int = 100
    gamma: int = 100
    capture_timestamp_delay: float = 0.101
    left_hand_path: str = "/dev/video0"
    right_hand_path: str = "/dev/video2"


@dataclass
class DataCollectionConfig(CamConfig):
    task_name: str = DATASET_RAW_TASK_NAME
    task_type: str = "bimanual"
    # 为兼容脚本参数保留；双臂模式下不会使用。
    single_hand_side: str = "left"
    output: str = RAW_DATA_DIR
    control_host: str = "localhost"
    control_port: int = 50010


@dataclass
class DataConvertConfig:
    task_name: str = DATASET_RAW_TASK_NAME
    task_type: str = "bimanual"
    single_hand_side: str = "left"
    raw_data_dir: str = RAW_DATA_DIR
    selected_data_dir: str = SELECTED_DATA_DIR
    episodes_per_chunk: int = 50

    visual_out_res: tuple[int, int] = (224, 224)
    tactile_out_res: tuple[int, int] = (224, 224)
    use_mask: bool = False
    fisheye_mask_radius: int = 390
    fisheye_mask_center: tuple[int, int] | None = None
    fisheye_mask_fill_color: tuple[int, int, int] = (0, 0, 0)

    cam_intrinsic_json_path: str = "./assets/intri_result/gopro_intrinsics_2_7k.json"
    aruco_dict: str = "DICT_4X4_50"
    marker_size_map: dict[int, float] = field(
        default_factory=lambda: {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02}
    )
    left_aruco_left_id: int = 0
    left_aruco_right_id: int = 1
    right_aruco_left_id: int = 2
    right_aruco_right_id: int = 3
    aruco_max_workers: int = 4

    min_episode_length: int = 10
    visual_cam_latency: float = 0.101
    pose_latency: float = 0.002
    use_tactile_img: bool = True

    output_repo_id: str | None = None
    fps: int = 30
    language_instruction: list[str] = field(
        default_factory=lambda: ["perform manipulation task"]
    )
    # 默认双臂，并保留完整双臂位姿 state。
    single_arm: bool = False
    no_state: bool = False
    smooth_sigma: float = 1.0
    use_inpaint_tag: bool = True
    tag_scale: float = 1.3


DATA_COLLECTION_CONFIG = DataCollectionConfig()
DATA_CONVERT_CONFIG = DataConvertConfig()
