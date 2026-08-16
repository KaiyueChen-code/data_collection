# 双臂数据采集与处理

1. 双臂拼接相机与 Quest 位姿采集；
2. 三联图裁剪；
3. ArUco 检测；
4. 双手夹爪宽度计算；
5. 位姿/相机时间对齐和 `dataset_plan.pkl` 生成；
6. Raw 数据直接转换成 LeRobot。

## 配置

统一修改 `data_collection/config.py`。采集和转换默认均为
`bimanual`，相机默认输入为供裁剪脚本使用的 3840x800 三联图。

## Pipeline 顺序

```text
data_collection/pipeline/
├── 00_get_data.py                 # 双臂数据采集
├── 01_crop_img.py                 # 三联图裁剪
├── 02_get_aruco_pos.py            # ArUco 检测
├── 03_get_width.py                # 双手夹爪宽度
├── 04_generate_dataset_plan.py    # 时间对齐和 episode plan
└── 05_convert_raw_to_lerobot.py   # Raw -> LeRobot
```
## 相机端口配置和左右检查
```bash
#配置环境
bash scripts/setup_visual_env.sh

#source


## 数据采集和处理

```bash
# 配置环境
bash scripts/setup_collection_env.sh

# 采集（Linux/V4L2）
bash scripts/get_data.sh

# Raw -> LeRobot 全流程
bash scripts/process_raw_data.sh
```

## 数据筛选

```bash
# 配置环境
bash scripts/setup_visual_env.sh

# 可视化筛选
python3 data_selector/src/run_selector_workflow.py task01 \
  --hf-home /home/sudo2/VB-VLA/select_data \
  --only select
```

界面使用说明：
- `X`：标记当前 episode 为删除，并前进到下一个 episode。
- `U`：撤销最近一次删除标记。
- `W` / `S`：切换 episode。
- `Space`：播放或暂停。
- `A` / `D`：播放时后退或前进一帧。
- `Q` 或 `Esc`：保存筛选记录并退出。

筛选完成后会得到 `-filtered` 文件（删除不合格的数据）：

```bash
python3 data_selector/src/run_selector_workflow.py task01 \
  --hf-home /home/sudo2/VB-VLA/select_data \
  --only apply
```

处理过程使用根目录下的 `data/{task_name}`，相机内参位于
`assets/intri_result/gopro_intrinsics_2_7k.json`。
