#!/usr/bin/env python3
"""
LeRobot 数据筛选可视化工具

布局：相机图像（放大）+ 时序曲线 + Save/Delete 按钮
无 3D 视图

用法:
  python src/data_selector.py --dataset /path/to/lerobot_dataset
  python src/data_selector.py --dataset /path/to/dataset --start 5 --output selection.json

键盘:
  A/D      前/后帧
  W/S      前/后 Episode
  Space    播放/暂停
  1–4      倍速 0.5× / 1× / 2× / 4×
  K/Enter  Save 当前 Episode
  X/Delete Delete 当前 Episode
  U        撤销上一次标记
  Q        退出并保存结果
"""

import os
import sys
import json
import argparse
import glob
import time
from pathlib import Path
from io import BytesIO

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import pandas as pd
except ImportError:
    print("缺少 pandas: pip install pandas pyarrow")
    sys.exit(1)

# ─────────────────── 颜色常量 ───────────────────
C_BG       = (18, 20, 28)
C_HEADER   = (28, 31, 42)
C_SEP      = (45, 50, 65)
C_SEP2     = (55, 60, 78)       # 稍亮分隔线
C_WHITE    = (235, 238, 248)
C_GRAY     = (130, 138, 158)
C_GREEN    = (72, 200, 110)
C_RED      = (72, 90, 218)
C_YELLOW   = (50, 195, 215)
C_SAVE_BG  = (38, 130, 55)
C_DEL_BG   = (45, 55, 185)
C_DEL_ACT  = (60, 70, 230)     # Delete 激活态
C_BTN      = (52, 57, 78)       # 普通按钮
C_BTN_ACT  = (70, 80, 108)      # 普通按钮激活
C_PROG     = (80, 150, 220)
C_PROG_BG  = (38, 42, 58)

FONT       = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────── 数据加载 ───────────────────

def find_parquet_files(dataset_path: str):
    """返回按 episode 编号排序的 parquet 文件路径列表"""
    base = Path(dataset_path)
    files = sorted(base.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        # 尝试扁平结构
        files = sorted(base.glob("data/episode_*.parquet"))
    return files


def load_meta(dataset_path: str):
    meta_path = Path(dataset_path) / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def decode_image(raw) -> np.ndarray | None:
    """从 bytes / PIL Image / dict / ndarray 解码为 HxWx3 RGB"""
    if raw is None:
        return None
    # PIL Image
    if hasattr(raw, 'convert'):
        arr = np.array(raw.convert('RGB'))
        return arr
    # dict with 'bytes'
    if isinstance(raw, dict):
        raw = raw.get('bytes') or raw.get('path')
        if raw is None:
            return None
    if isinstance(raw, bytes):
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            img = cv2.imdecode(raw.astype(np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return raw
    return None


IMAGE_COLS = [
    ('observation.images.camera0',        'R0-Cam',   (0, 120, 255)),
    ('observation.images.camera1',        'R1-Cam',   (0, 200, 80)),
    ('observation.images.tactile_left_0', 'R0-L-Tact',(0, 120, 255)),
    ('observation.images.tactile_right_0','R0-R-Tact',(0, 120, 255)),
    ('observation.images.tactile_left_1', 'R1-L-Tact',(0, 200, 80)),
    ('observation.images.tactile_right_1','R1-R-Tact',(0, 200, 80)),
]


class EpisodeLoader:
    """按 episode 懒加载 parquet（支持 LeRobot v2/v3 格式）

    v2: data/chunk-000/episode_000000.parquet  (每文件一个 episode)
    v3: data/chunk-000/file-000.parquet        (大文件多 episode, 按 episode_index 分组)
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._ep_to_loc: dict[int, tuple[Path, int, int]] = {}  # ep -> (file, row_start, row_end)
        self._file_cache: dict[Path, pd.DataFrame] = {}
        self._cache_order: list[Path] = []
        self._max_cache = 3

        self._scan(dataset_path)
        if not self._ep_to_loc:
            raise FileNotFoundError(f"找不到 parquet 数据: {dataset_path}")
        self.n_episodes = len(self._ep_to_loc)
        # sorted episode indices (may not be 0-based if dataset is partial)
        self._ep_ids: list[int] = sorted(self._ep_to_loc.keys())
        print(f"  Found {self.n_episodes} episodes in {dataset_path}")

    def _scan(self, dataset_path: str):
        base = Path(dataset_path)
        # v2 style: one episode per file
        v2_files = sorted(base.glob("data/chunk-*/episode_*.parquet"))
        if v2_files:
            for i, f in enumerate(v2_files):
                # peek to get episode_index if available
                try:
                    idx_df = pd.read_parquet(f, columns=['episode_index'])
                    ep_id = int(idx_df['episode_index'].iloc[0])
                except Exception:
                    ep_id = i
                n = len(pd.read_parquet(f, columns=['frame_index']))
                self._ep_to_loc[ep_id] = (f, 0, n)
            return

        # v3 style: big files with episode_index column
        v3_files = sorted(base.glob("data/chunk-*/file-*.parquet"))
        if not v3_files:
            v3_files = sorted(base.glob("data/*.parquet"))
        for f in v3_files:
            try:
                idx_df = pd.read_parquet(f, columns=['episode_index'])
            except Exception:
                continue
            # parquet read gives RangeIndex 0..n-1; groupby preserves it
            for ep_id, grp in idx_df.groupby('episode_index'):
                pos_start = int(grp.index[0])
                pos_end   = int(grp.index[-1]) + 1
                self._ep_to_loc[int(ep_id)] = (f, pos_start, pos_end)

    def _load_file(self, fpath: Path) -> pd.DataFrame:
        if fpath not in self._file_cache:
            df = pd.read_parquet(fpath)
            self._file_cache[fpath] = df
            self._cache_order.append(fpath)
            if len(self._cache_order) > self._max_cache:
                old = self._cache_order.pop(0)
                self._file_cache.pop(old, None)
        return self._file_cache[fpath]

    def get(self, ep_idx: int) -> pd.DataFrame:
        """ep_idx is the positional index (0..n_episodes-1)"""
        ep_id = self._ep_ids[ep_idx]
        fpath, row_start, row_end = self._ep_to_loc[ep_id]
        full_df = self._load_file(fpath)
        return full_df.iloc[row_start:row_end].reset_index(drop=True)

    def episode_length(self, ep_idx: int) -> int:
        ep_id = self._ep_ids[ep_idx]
        _, r0, r1 = self._ep_to_loc[ep_id]
        return r1 - r0


# ─────────────────── 时序绘图 ───────────────────

def render_timeseries(df: pd.DataFrame, frame_idx: int,
                      width: int, height: int) -> np.ndarray:
    """将 EEF 位置 + gripper 曲线渲染为 numpy BGR 图像"""
    state_col = None
    for c in ['observation.state', 'obs.state']:
        if c in df.columns:
            state_col = c
            break

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor('#14161E')
    n_frames = len(df)
    t = np.arange(n_frames)
    marker_x = frame_idx

    if state_col is not None:
        try:
            states = np.stack(df[state_col].tolist()).astype(np.float32)
        except Exception:
            states = None
    else:
        states = None

    # 列数：最多 3 列（R0 xyz、R1 xyz、gripper）
    # ── 提取 gripper 宽度（用于叠加到 EEF 图）
    grip0 = states[:, 6]  if (states is not None and states.shape[1] > 6)  else None
    grip1 = states[:, 13] if (states is not None and states.shape[1] > 13) else None

    # ── 主曲线列：R0 EEF | R1 EEF（不再单独放 Gripper 列）
    cols_available = []
    if states is not None and states.shape[1] >= 3:
        cols_available.append(('R0 EEF + Gripper', states[:, :3],
                                ['X', 'Y', 'Z'], ['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                grip0, 'R0 Grip', '#FFD93D'))
    if states is not None and states.shape[1] >= 17:
        cols_available.append(('R1 EEF + Gripper', states[:, 14:17],
                                ['X', 'Y', 'Z'], ['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                grip1, 'R1 Grip', '#6BCB77'))

    n_plots = max(len(cols_available), 1)
    gs = gridspec.GridSpec(1, n_plots, figure=fig,
                           left=0.05, right=0.97, top=0.88, bottom=0.15,
                           wspace=0.40)

    if not cols_available:
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor('#1E2030')
        ax.text(0.5, 0.5, 'No state data', transform=ax.transAxes,
                color='gray', ha='center', va='center', fontsize=10)
    else:
        for i, (title, eef_data, labels, colors, grip_data, grip_lbl, grip_clr) in enumerate(cols_available):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor('#1E2030')
            ax.tick_params(colors='#888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            ax.set_title(title, color='#CCC', fontsize=8, pad=4)

            # EEF xyz 曲线（左轴）
            for j, (lbl, clr) in enumerate(zip(labels, colors)):
                y = eef_data[:, j] if eef_data.ndim == 2 else eef_data
                ax.plot(t, y, color=clr, linewidth=0.9, label=lbl)

            ax.axvline(x=marker_x, color='#FFFFFF', linewidth=1.2,
                       linestyle='--', alpha=0.7)
            ax.set_xlim(0, max(n_frames - 1, 1))
            ax.set_xlabel('Frame', color='#888', fontsize=7)

            # ── Gripper 宽度叠加（右轴）
            if grip_data is not None:
                ax2 = ax.twinx()
                ax2.set_facecolor('none')
                ax2.plot(t, grip_data, color=grip_clr, linewidth=1.4,
                         linestyle='--', label=grip_lbl)
                ax2.tick_params(colors=grip_clr, labelsize=6)
                ax2.set_ylabel('Gripper (m)', color=grip_clr, fontsize=6)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#444')
                # 合并图例
                lines1, labs1 = ax.get_legend_handles_labels()
                lines2, labs2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labs1 + labs2,
                           loc='upper right', fontsize=6,
                           facecolor='#2A2C3E', edgecolor='#555',
                           labelcolor='#CCC', framealpha=0.7)
            else:
                ax.legend(loc='upper right', fontsize=6,
                          facecolor='#2A2C3E', edgecolor='#555',
                          labelcolor='#CCC', framealpha=0.7)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).copy()
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    img = arr[:, :, :3]  # RGB
    img = cv2.resize(img, (width, height))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ─────────────────── UI 绘制辅助 ───────────────────

def draw_text(img, text, pos, scale=0.55, color=C_WHITE, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_button(img, rect, label, bg_color, text_color=C_WHITE,
                border_color=None, scale=0.60, thickness=1):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    # 顶部高光线
    lighter = tuple(min(c + 30, 255) for c in bg_color)
    cv2.line(img, (x + 1, y + 1), (x + w - 1, y + 1), lighter, 1)
    border = border_color if border_color else tuple(min(c + 20, 255) for c in bg_color)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)
    tw, th = cv2.getTextSize(label, FONT, scale, thickness)[0]
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(img, label, (tx, ty), FONT, scale, text_color, thickness, cv2.LINE_AA)


def resize_img(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0:
        return img
    new_w = int(w * target_h / h)
    return cv2.resize(img, (new_w, target_h))


def label_img(img: np.ndarray, label: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.48, 1)
    # 半透明背景块
    overlay = out.copy()
    cv2.rectangle(overlay, (5, 5), (tw + 14, th + 12), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, label, (10, th + 8), FONT, 0.48, color, 1, cv2.LINE_AA)
    return out


def make_placeholder(w: int, h: int, text: str = "N/A") -> np.ndarray:
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    tw, th = cv2.getTextSize(text, FONT, 0.5, 1)[0]
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                FONT, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    return img


def hstack_fill(imgs: list[np.ndarray], height: int, gap: int = 4) -> np.ndarray:
    """水平拼接，统一高度，间隔 gap 像素"""
    resized = []
    for img in imgs:
        if img is None:
            continue
        resized.append(resize_img(img, height))
    if not resized:
        return make_placeholder(200, height, "No images")
    parts = []
    sep = np.full((height, gap, 3), C_SEP, dtype=np.uint8)
    for i, r in enumerate(resized):
        parts.append(r)
        if i < len(resized) - 1:
            parts.append(sep)
    return np.hstack(parts)


# ─────────────────── 主可视化器 ───────────────────

class DataSelector:
    WIN = "LeRobot Data Selector"

    # 窗口尺寸
    W            = 1680
    HEADER_H     = 64
    CAM_AREA_H   = 540   # 相机区域总高度（自动分配给有效行）
    PLOT_H       = 240
    PROGRESS_H   = 34
    CTRL_H       = 84
    SEP          = 2     # 细分隔线

    def __init__(self, dataset_path: str, start_ep: int = 0,
                 output_path: str = "selection.json"):
        self.loader = EpisodeLoader(dataset_path)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.n_ep = self.loader.n_episodes

        self.ep_idx   = max(0, min(start_ep, self.n_ep - 1))
        self.frame_idx = 0
        self.playing   = False
        self.play_spd  = 1.0           # fps multiplier
        self._last_play_t = 0.0

        self.selections: dict[int, str] = {}   # ep_idx -> 'save' | 'delete'
        self.undo_stack: list[tuple[int, str | None]] = []

        self._df: pd.DataFrame | None = None
        self._avail_img_cols: list[tuple[str, str, tuple]] = []
        # 按钮区域 (x, y, w, h)
        self._btn_save   = (0, 0, 0, 0)
        self._btn_delete = (0, 0, 0, 0)
        self._btn_prev_ep  = (0, 0, 0, 0)
        self._btn_next_ep  = (0, 0, 0, 0)
        self._btn_play     = (0, 0, 0, 0)
        self._progress_rect = (0, 0, 0, 0)
        self._btn_speeds: list[tuple[tuple, float]] = []

        self._frame_cache: dict[int, np.ndarray] = {}
        self._plot_cache: tuple[int, int, np.ndarray] | None = None  # (ep, frame, img)
        self._reset_mouse_zoom_requested = False
        self._stable_window_size = (self.W, self._total_height())
        self._window_rect_settle_frames = 2

        self._create_window(self.W, self._total_height())

        self._load_episode()

    def _total_height(self):
        return (self.HEADER_H + self.SEP
                + self.CAM_AREA_H + self.SEP
                + self.PLOT_H + self.SEP
                + self.PROGRESS_H + self.SEP
                + self.CTRL_H)

    def _create_window(self, width: int, height: int) -> None:
        # WINDOW_NORMAL lets HighGUI scale the full canvas with the window.
        # GUI_NORMAL removes Qt's toolbar/context menu zoom controls.
        window_flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        cv2.namedWindow(self.WIN, window_flags)
        cv2.resizeWindow(self.WIN, max(1, width), max(1, height))
        cv2.setMouseCallback(self.WIN, self._on_mouse)

    def _remember_stable_window_size(self) -> None:
        """Cache geometry before waitKey dispatches a possible wheel event."""
        if self._reset_mouse_zoom_requested:
            return
        if self._window_rect_settle_frames > 0:
            self._window_rect_settle_frames -= 1
            return
        try:
            _, _, width, height = cv2.getWindowImageRect(self.WIN)
            if width > 0 and height > 0:
                self._stable_window_size = (width, height)
        except cv2.error:
            # Wayland may not expose window geometry; retain the last size.
            pass

    def _restore_window_bound_view(self) -> None:
        """Discard mouse-wheel zoom using geometry captured before the event."""
        width, height = self._stable_window_size
        cv2.destroyWindow(self.WIN)
        cv2.waitKey(1)
        self._create_window(width, height)
        self._window_rect_settle_frames = 2
        self._reset_mouse_zoom_requested = False

    def _load_episode(self):
        self._df = self.loader.get(self.ep_idx)
        self.frame_idx = 0
        self._avail_img_cols = [
            (col, lbl, clr)
            for col, lbl, clr in IMAGE_COLS
            if col in self._df.columns
        ]
        # 检测哪些 robot 行有实际内容（非全黑）
        self._active_robots: set[int] = set()
        sample_row = self._df.iloc[min(5, len(self._df) - 1)]
        for col, lbl, _ in self._avail_img_cols:
            img = decode_image(sample_row[col])
            if img is not None and img.mean() > 5:
                self._active_robots.add(0 if lbl.startswith('R0') else 1)
        if not self._active_robots:
            self._active_robots = {0}  # 至少显示 robot0

        self._frame_cache.clear()
        self._plot_cache = None
        print(f"  Episode {self.ep_idx+1}/{self.n_ep}  "
              f"frames={len(self._df)}  "
              f"active_robots={sorted(self._active_robots)}  "
              f"status={self.selections.get(self.ep_idx, '—')}")

    # ── 渲染 ──────────────────────────────────────

    def _get_frame_img(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]
        row = self._df.iloc[frame_idx]
        imgs = {}
        for col, lbl, clr in self._avail_img_cols:
            raw = row[col]
            decoded = decode_image(raw)
            if decoded is not None:
                # BGR for OpenCV display
                imgs[lbl] = (cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR), clr)
        result = imgs if imgs else None
        self._frame_cache[frame_idx] = result
        return result

    def _build_camera_section(self, imgs_by_label: dict | None) -> np.ndarray:
        W = self.W
        all_rows_def = [
            (0, [('R0-Cam', 0), ('R0-L-Tact', 0), ('R0-R-Tact', 0)]),
            (1, [('R1-Cam', 1), ('R1-L-Tact', 1), ('R1-R-Tact', 1)]),
        ]
        robot_colors = {0: (80, 120, 255), 1: (60, 210, 90)}

        # 只保留 active robot 行
        active_rows = [(rid, cells) for rid, cells in all_rows_def
                       if rid in self._active_robots]
        n_rows = len(active_rows)
        if n_rows == 0:
            return make_placeholder(W, self.CAM_AREA_H, "No camera data")

        # 每行高度：平均分配，留细分隔
        row_h = (self.CAM_AREA_H - self.SEP * (n_rows - 1)) // n_rows

        section_rows = []
        for rid, cells_def in active_rows:
            cells = []
            for lbl, robot_id in cells_def:
                if imgs_by_label and lbl in imgs_by_label:
                    img, _ = imgs_by_label[lbl]
                    img = label_img(img, lbl, robot_colors[robot_id])
                    cells.append(img)
            valid = [c for c in cells if c is not None]
            if not valid:
                row_img = make_placeholder(W, row_h, f"Robot{rid} — no data")
            else:
                row_img = hstack_fill(valid, row_h, gap=3)
                if row_img.shape[1] < W:
                    pad = np.full((row_h, W - row_img.shape[1], 3), C_BG, dtype=np.uint8)
                    row_img = np.hstack([row_img, pad])
                else:
                    row_img = row_img[:, :W]
            # 左侧 robot 色条
            stripe_color = robot_colors[rid]
            row_img[:, :4] = stripe_color
            section_rows.append(row_img)

        sep = np.full((self.SEP, W, 3), C_SEP, dtype=np.uint8)
        parts = []
        for i, r in enumerate(section_rows):
            parts.append(r)
            if i < len(section_rows) - 1:
                parts.append(sep)

        out = np.vstack(parts)
        # 高度不足时补齐（保持总高度固定）
        if out.shape[0] < self.CAM_AREA_H:
            pad = np.full((self.CAM_AREA_H - out.shape[0], W, 3), C_BG, dtype=np.uint8)
            out = np.vstack([out, pad])
        umi_block = out[:self.CAM_AREA_H]

        if umi_block.shape[1] < self.W:
            pad = np.full((self.CAM_AREA_H, self.W - umi_block.shape[1], 3), C_BG, dtype=np.uint8)
            umi_block = np.hstack([umi_block, pad])
        return umi_block[:, :self.W]

    def _build_plot_section(self) -> np.ndarray:
        ep = self.ep_idx
        fr = self.frame_idx
        if (self._plot_cache is not None
                and self._plot_cache[0] == ep
                and self._plot_cache[1] == fr):
            return self._plot_cache[2]
        img = render_timeseries(self._df, fr, self.W, self.PLOT_H)
        self._plot_cache = (ep, fr, img)
        return img

    def _build_progress_bar(self) -> np.ndarray:
        bar = np.full((self.PROGRESS_H, self.W, 3), C_PROG_BG, dtype=np.uint8)
        n = len(self._df)
        ratio = self.frame_idx / max(n - 1, 1)
        filled = int(ratio * self.W)

        # 背景轨道
        track_y0, track_y1 = 12, self.PROGRESS_H - 12
        cv2.rectangle(bar, (0, track_y0), (self.W, track_y1), C_SEP2, -1)
        # 已播放部分
        if filled > 0:
            cv2.rectangle(bar, (0, track_y0), (filled, track_y1), C_PROG, -1)
        # 滑块
        cx = max(6, min(filled, self.W - 6))
        cy = self.PROGRESS_H // 2
        cv2.circle(bar, (cx, cy), 7, (220, 235, 255), -1)
        cv2.circle(bar, (cx, cy), 7, C_PROG, 1)

        # 帧号
        frame_txt = f"{self.frame_idx + 1} / {n}"
        tw, _ = cv2.getTextSize(frame_txt, FONT, 0.42, 1)[0]
        cv2.putText(bar, frame_txt, (self.W - tw - 10, cy + 5),
                    FONT, 0.42, C_GRAY, 1, cv2.LINE_AA)

        self._progress_rect = (0, 0, self.W, self.PROGRESS_H)
        return bar

    def _build_ctrl_bar(self) -> np.ndarray:
        bar = np.full((self.CTRL_H, self.W, 3), C_HEADER, dtype=np.uint8)
        cv2.line(bar, (0, 0), (self.W, 0), C_SEP2, 1)

        bh = 44
        by = (self.CTRL_H - bh) // 2

        # ── 导航按钮
        bx = 14
        bw = 116
        self._btn_prev_ep = (bx, by, bw, bh)
        draw_button(bar, self._btn_prev_ep, "< Prev Ep", C_BTN, scale=0.54)
        bx += bw + 6

        play_label = "|| Pause" if self.playing else "> Play"
        play_bg = C_BTN_ACT if self.playing else C_BTN
        self._btn_play = (bx, by, 116, bh)
        draw_button(bar, self._btn_play, play_label, play_bg, scale=0.54)
        bx += 122

        self._btn_next_ep = (bx, by, bw, bh)
        draw_button(bar, self._btn_next_ep, "Next Ep >", C_BTN, scale=0.54)
        bx += bw + 18

        # ── 倍速按钮
        speeds = [0.5, 1.0, 2.0, 4.0]
        spw = 52
        self._btn_speeds = []
        for spd in speeds:
            active = abs(self.play_spd - spd) < 0.01
            bg     = (88, 138, 210) if active else C_BTN
            border = (140, 190, 255) if active else None
            rect   = (bx, by, spw, bh)
            spd_lbl = f"{int(spd)}x" if spd == int(spd) else f"{spd}x"
            draw_button(bar, rect, spd_lbl, bg, border_color=border, scale=0.50)
            self._btn_speeds.append((rect, spd))
            bx += spw + 4

        # ── 键盘提示（底部）
        hints = (
            "A / D : frame    W / S : episode    Space : play    "
            "1–4 : speed    X : delete    U : undo    Q : quit"
        )
        draw_text(bar, hints, (14, self.CTRL_H - 10), scale=0.38, color=C_GRAY)

        # ── DELETE 按钮（右侧）
        dbw, dbh2 = 172, 52
        dby2 = (self.CTRL_H - dbh2) // 2
        dbx  = self.W - dbw - 14

        status = self.selections.get(self.ep_idx)
        is_del = status == 'delete'
        del_bg     = C_DEL_ACT if is_del else C_DEL_BG
        del_border = (120, 120, 255) if is_del else (80, 80, 200)

        self._btn_save   = (0, 0, 0, 0)
        self._btn_delete = (dbx, dby2, dbw, dbh2)
        draw_button(bar, self._btn_delete, "DELETE  [X]",
                    del_bg, scale=0.68, thickness=1, border_color=del_border)

        return bar

    def _build_header(self) -> np.ndarray:
        bar = np.full((self.HEADER_H, self.W, 3), C_HEADER, dtype=np.uint8)
        cv2.line(bar, (0, self.HEADER_H - 1), (self.W, self.HEADER_H - 1), C_SEP2, 1)

        status = self.selections.get(self.ep_idx, '—')
        status_color = {'delete': C_RED, '—': C_GRAY}.get(status, C_GRAY)

        # 左侧状态色条 (5px)
        stripe = C_RED if status == 'delete' else C_SEP2
        cv2.rectangle(bar, (0, 0), (5, self.HEADER_H), stripe, -1)

        n = len(self._df)
        deleted_n = sum(1 for v in self.selections.values() if v == 'delete')

        draw_text(bar, "LeRobot  Data Selector",
                  (16, 26), scale=0.68, color=C_WHITE, thickness=1)

        ep_info = (f"Ep {self.ep_idx + 1} / {self.n_ep}   "
                   f"Frame {self.frame_idx + 1} / {n}   "
                   f"Speed {self.play_spd:.1f}x")
        draw_text(bar, ep_info, (16, 50), scale=0.46, color=C_GRAY)

        stat_info = f"deleted: {deleted_n}    remaining: {self.n_ep - deleted_n}"
        tw = cv2.getTextSize(stat_info, FONT, 0.46, 1)[0][0]
        draw_text(bar, stat_info, (self.W // 2 - tw // 2, 38),
                  scale=0.46, color=C_GRAY)

        # 状态徽章（右）
        badge = "DELETED" if status == 'delete' else "UNREVIEWED"
        btw = cv2.getTextSize(badge, FONT, 0.62, 1)[0][0]
        draw_text(bar, badge, (self.W - btw - 20, 38),
                  scale=0.62, color=status_color, thickness=1)

        return bar

    def _build_frame(self) -> np.ndarray:
        imgs = self._get_frame_img(self.frame_idx)
        header  = self._build_header()
        cams    = self._build_camera_section(imgs)
        plots   = self._build_plot_section()
        prog    = self._build_progress_bar()
        ctrl    = self._build_ctrl_bar()

        sep = lambda: np.full((self.SEP, self.W, 3), C_SEP, dtype=np.uint8)
        parts = [header, sep(), cams, sep(), plots, sep(), prog, sep(), ctrl]
        out = np.vstack(parts)

        # clip/pad height
        total_h = self._total_height()
        if out.shape[0] > total_h:
            out = out[:total_h]
        elif out.shape[0] < total_h:
            pad = np.full((total_h - out.shape[0], self.W, 3), C_BG, dtype=np.uint8)
            out = np.vstack([out, pad])
        return out

    # ── 鼠标回调 ──────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        wheel_events = {
            getattr(cv2, "EVENT_MOUSEWHEEL", -1),
            getattr(cv2, "EVENT_MOUSEHWHEEL", -2),
        }
        if event in wheel_events:
            # Qt applies its own zoom after this callback returns. Recreate the
            # viewport in the main loop to restore a window-bound 100% view.
            self._reset_mouse_zoom_requested = True
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 累积 y 偏移到各 section
        y_offset = self.HEADER_H + self.SEP  # after header
        # camera section uses a fixed total area height
        y_offset += self.CAM_AREA_H + self.SEP
        y_offset += self.PLOT_H + self.SEP

        # Progress bar click → seek
        prog_y0 = y_offset
        prog_y1 = prog_y0 + self.PROGRESS_H
        if prog_y0 <= y < prog_y1:
            n = len(self._df)
            self.frame_idx = int(x / self.W * (n - 1))
            self.frame_idx = max(0, min(self.frame_idx, n - 1))
            self._plot_cache = None
            return

        y_offset += self.PROGRESS_H + self.SEP
        # Control bar click
        rel_y = y - y_offset
        if 0 <= rel_y < self.CTRL_H:
            self._handle_ctrl_click(x, rel_y)

    def _in_rect(self, x, y, rect):
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def _handle_ctrl_click(self, x, y):
        if self._in_rect(x, y, self._btn_delete):
            self._mark('delete')
        elif self._in_rect(x, y, self._btn_play):
            self.playing = not self.playing
            self._last_play_t = time.time()
        elif self._in_rect(x, y, self._btn_prev_ep):
            self._go_ep(-1)
        elif self._in_rect(x, y, self._btn_next_ep):
            self._go_ep(1)
        else:
            for rect, spd in self._btn_speeds:
                if self._in_rect(x, y, rect):
                    self.play_spd = spd
                    break

    # ── 标记 ──────────────────────────────────────

    def _mark(self, action: str):
        prev = self.selections.get(self.ep_idx)
        self.undo_stack.append((self.ep_idx, prev))
        self.selections[self.ep_idx] = action
        print(f"  Marked episode {self.ep_idx} as {action.upper()}")

    def _undo(self):
        if not self.undo_stack:
            return
        ep, prev = self.undo_stack.pop()
        if prev is None:
            self.selections.pop(ep, None)
        else:
            self.selections[ep] = prev
        print(f"  Undone: episode {ep} → {prev or '—'}")

    def _go_ep(self, delta: int):
        new_ep = self.ep_idx + delta
        if 0 <= new_ep < self.n_ep:
            self.ep_idx = new_ep
            self._load_episode()

    # ── 主循环 ────────────────────────────────────

    def run(self):
        meta = load_meta(self.dataset_path)
        fps_data = float(meta.get('fps', 30))
        print(f"\n  Dataset FPS: {fps_data}")
        print(f"  Episodes:   {self.n_ep}")
        print("  Controls:  A/D=frame  W/S=episode  Space=play  "
              "1–4=speed(0.5/1/2/4x)  K=save  X=delete  U=undo  Q=quit\n")

        try:
            while True:
                frame_img = self._build_frame()
                cv2.imshow(self.WIN, frame_img)
                self._remember_stable_window_size()

                # 自动播放
                if self.playing:
                    now = time.time()
                    dt  = now - self._last_play_t
                    target_fps = max(fps_data * self.play_spd, 1e-6)
                    # Advance multiple frames when speed is high, instead of
                    # capping to +1 frame per UI tick.
                    advance = int(dt * target_fps)
                    if advance > 0:
                        n = len(self._df)
                        self._last_play_t += advance / target_fps
                        next_idx = self.frame_idx + advance
                        if next_idx < n - 1:
                            self.frame_idx = next_idx
                            self._plot_cache = None
                        else:
                            self.frame_idx = n - 1
                            self._plot_cache = None
                            self.playing = False

                wait_ms = 1 if self.playing else 30
                key = cv2.waitKey(wait_ms) & 0xFF

                if self._reset_mouse_zoom_requested:
                    self._restore_window_bound_view()
                    continue

                if key == 255:
                    continue

                if key in (ord('q'), ord('Q'), 27):
                    break

                elif key in (ord('d'), ord('D')):
                    n = len(self._df)
                    if self.frame_idx < n - 1:
                        self.frame_idx += 1
                        self._plot_cache = None
                    else:
                        self._go_ep(1)

                elif key in (ord('a'), ord('A')):
                    if self.frame_idx > 0:
                        self.frame_idx -= 1
                        self._plot_cache = None

                elif key in (ord('w'), ord('W')):
                    self._go_ep(1)

                elif key in (ord('s'), ord('S')):
                    self._go_ep(-1)

                elif key in (ord(' '),):
                    self.playing = not self.playing
                    self._last_play_t = time.time()

                elif key in (ord('x'), ord('X')):   # X
                    self._mark('delete')
                    # auto-advance to next episode
                    self._go_ep(1)

                elif key in (ord('u'), ord('U')):
                    self._undo()

                elif key in (ord('1'),):
                    self.play_spd = 0.5
                elif key in (ord('2'),):
                    self.play_spd = 1.0
                elif key in (ord('3'),):
                    self.play_spd = 2.0
                elif key in (ord('4'),):
                    self.play_spd = 4.0
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt received, saving results before exit...")
        finally:
            cv2.destroyAllWindows()
            self._save_results()

    # ── 保存结果 ──────────────────────────────────

    def _save_results(self):
        deleted = sorted(ep for ep, v in self.selections.items() if v == 'delete')
        kept = sorted(ep for ep in range(self.n_ep)
                      if self.selections.get(ep) != 'delete')

        result = {
            "dataset": str(self.dataset_path),
            "total_episodes": self.n_ep,
            "deleted":    {"count": len(deleted), "episodes": deleted},
            # Semantics: only explicitly deleted episodes are dropped.
            # All other episodes are kept by default.
            "kept": {"count": len(kept), "episodes": kept},
            # Backward compatibility with earlier outputs/tools.
            "unreviewed": {"count": len(kept), "episodes": kept},
        }

        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Results saved: {out}")
        print(f"  Deleted:     {len(deleted)} episodes")
        print(f"  Kept:        {len(kept)} episodes")
        print(f"{'='*60}\n")

        if deleted:
            print(f"  Deleted episode indices: {deleted}")


# ─────────────────── 入口 ───────────────────

def resolve_short_paths(dataset_arg: str, output_arg: str) -> tuple[str, str]:
    """Resolve legacy repo-id/output shorthands without machine-specific paths."""
    tool_root = Path(__file__).resolve().parents[1]
    hf_home = Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            str(Path.home() / ".cache" / "huggingface" / "lerobot"),
        )
    ).expanduser()

    dataset = dataset_arg
    output = output_arg

    if dataset.startswith("/eric/") and not Path(dataset).exists():
        dataset = str(hf_home / dataset.lstrip("/"))

    if output.lower().startswith("/data_selector/"):
        output = str(tool_root / "selections" / Path(output).name)

    return dataset, output


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot 数据筛选可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/data_selector.py --dataset /path/to/lerobot_dataset
  python src/data_selector.py --dataset ~/data/my_dataset --start 10 --output my_selection.json
        """
    )
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='LeRobot 数据集路径')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='起始 episode 编号 (default: 0)')
    parser.add_argument('--output', '-o', type=str, default='selection.json',
                        help='输出 JSON 文件路径 (default: selection.json)')
    args = parser.parse_args()

    dataset_path, output_path = resolve_short_paths(args.dataset, args.output)
    try:
        sel = DataSelector(
            dataset_path=dataset_path,
            start_ep=args.start,
            output_path=output_path,
        )
        sel.run()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n中断，已退出（结果已在中断时自动保存）")
        sys.exit(0)


if __name__ == '__main__':
    main()
