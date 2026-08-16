import argparse
import json
import os
import select
import sys
import threading
import time
import tty
import termios
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from queue import Queue
import threading as _threading

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.camera_device import V4L2Camera
from data_collection.config import DATA_COLLECTION_CONFIG


def _build_omegaconf_from_collection_config(cfg) -> OmegaConf:
    """Convert flat DataCollectionConfig into the nested OmegaConf structure
    that the rest of this file expects (config.recorder.camera.width, etc.)."""
    nested = {
        "task": {
            "name": cfg.task_name,
            "type": cfg.task_type,
            "single_hand_side": cfg.single_hand_side,
        },
        "recorder": {
            "output": cfg.output,
            "control_host": cfg.control_host,
            "control_port": cfg.control_port,
            "camera": {
                "format": cfg.camera_format,
                "width": cfg.camera_width,
                "height": cfg.camera_height,
                "auto_exposure": cfg.auto_exposure,
                "exposure": cfg.exposure,
                "auto_white_balance": cfg.auto_white_balance,
                "wb_temperature": cfg.wb_temperature,
                "brightness": cfg.brightness,
                "gain": cfg.gain,
                "gamma": cfg.gamma,
            },
            "camera_paths": {
                "left_hand": cfg.left_hand_path,
                "right_hand": cfg.right_hand_path,
            },
        },
    }
    return OmegaConf.create(nested)


class StatusBoard:
    """Lightweight split-screen status printer for camera/pose."""

    def __init__(self):
        self.lock = _threading.Lock()
        self.camera_stats: Dict[str, Tuple[float, int]] = {}
        self.pose_stat: Tuple[float, int] = (0.0, 0)
        self.last_render = 0.0
        self.min_interval = 0.2  # seconds
        self.demo_status = "Idle"

    def update_camera(self, name: str, fps: float, total: int):
        with self.lock:
            self.camera_stats[name] = (fps, total)
            self._render_locked()

    def update_pose(self, fps: float, packets: int):
        with self.lock:
            self.pose_stat = (fps, packets)
            self._render_locked()

    def set_demo_status(self, text: str):
        with self.lock:
            self.demo_status = text
            self._render_locked()

    def _render_locked(self):
        now = time.time()
        if now - self.last_render < self.min_interval:
            return
        self.last_render = now

        # Build left column (cameras) and right column (pose)
        cam_lines = ["Camera FPS"]
        for name in sorted(self.camera_stats.keys()):
            fps, total = self.camera_stats[name]
            cam_lines.append(f"{name:>10}: {fps:5.1f}  (total {total})")
        if len(cam_lines) == 1:
            cam_lines.append("  waiting...")

        pose_fps, pose_packets = self.pose_stat
        pose_lines = ["Pose", f"fps: {pose_fps:5.1f} Hz", f"packets: {pose_packets}"]

        # Pad to equal height
        height = max(len(cam_lines), len(pose_lines))
        cam_lines += [""] * (height - len(cam_lines))
        pose_lines += [""] * (height - len(pose_lines))

        # Compose two-column layout
        combined = ["\033[2J\033[H"]  # clear screen & move cursor home
        for cl, pl in zip(cam_lines, pose_lines):
            combined.append(f"{cl:<32} | {pl}")
        combined.append("")
        combined.append(f"State: {self.demo_status}")
        combined.append("[S] start/stop  [Q] quit")

        sys.stdout.write("\n".join(combined) + "\n")
        sys.stdout.flush()

 
class BatchSaver:
    """Batch writer for Quest pose data."""

    def __init__(self, save_dir: str, frames_per_file: int = 5000):
        self.save_dir = Path(save_dir)
        self.frames_per_file = frames_per_file
        self.session_timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
        self.file_counter = 1
        self.current_batch = []
        self.total_saved_frames = 0
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, pose_data, receive_timestamp: float):
        pose_data["timestamp_unix"] = receive_timestamp
        pose_data["timestamp_readable"] = datetime.fromtimestamp(receive_timestamp).strftime(
            "%Y.%m.%d_%H.%M.%S.%f"
        )
        self.current_batch.append(pose_data)
        self.total_saved_frames += 1

        if len(self.current_batch) >= self.frames_per_file:
            self._save_batch()

    def _save_batch(self):
        if not self.current_batch:
            return

        filename = f"quest_poses_{self.session_timestamp}_part{self.file_counter:03d}.json"
        with open(self.save_dir / filename, "w") as f:
            json.dump(self.current_batch, f, indent=2)

        print(f"\nSaved {filename} ({len(self.current_batch)} frames)")
        self.current_batch = []
        self.file_counter += 1

    def finalize(self):
        if self.current_batch:
            self._save_batch()

        summary = {
            "session_timestamp": self.session_timestamp,
            "total_frames": self.total_saved_frames,
            "total_files": self.file_counter - 1,
            "frames_per_file": self.frames_per_file,
        }

        with open(self.save_dir / f"session_summary_{self.session_timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)


class ControlClient:
    """Lightweight TCP client to notify Quest of demo start/stop and FPS updates."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.last_fps_send_time = 0.0
        self.fps_send_interval = 0.5  # Send FPS every 0.5 seconds

    def _ensure_connected(self) -> bool:
        if self.sock:
            return True
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=1.0)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return True
        except Exception as e:
            print(f"[CTRL][ERROR] connect failed: {e}")
            self.sock = None
            return False

    def send_demo_control(self, demo_id: int, action: str):
        """Send demo start/stop control message."""
        if action not in ("start", "stop"):
            return
        if not self._ensure_connected():
            return
        payload = {
            "type": "demo_control",
            "demo_id": demo_id,
            "action": action,
            "timestamp": time.time(),
        }
        try:
            msg = json.dumps(payload) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
            print(f"[CTRL] sent {action} for demo #{demo_id}")
        except Exception as e:
            print(f"[CTRL][ERROR] send failed: {e}")
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

    def send_fps_update(self, camera_fps: Dict[str, float], pose_fps: float):
        """Send FPS update message (throttled)."""
        now = time.time()
        if now - self.last_fps_send_time < self.fps_send_interval:
            return  # Throttle FPS updates
        
        if not self._ensure_connected():
            return
        
        payload = {
            "type": "fps_update",
            "camera_fps": camera_fps,
            "pose_fps": pose_fps,
            "timestamp": now,
        }
        try:
            msg = json.dumps(payload) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
            self.last_fps_send_time = now
        except Exception as e:
            # Silently fail for FPS updates to avoid spam
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None


class QuestPoseReceiver(threading.Thread):
    """Background receiver for Quest pose data with on/off recording."""

    def __init__(
        self,
        host: str,
        port: int,
        base_output_dir: Path,
        frames_per_file: int = 5000,
        stop_event: Optional[threading.Event] = None,
        status_board: Optional[StatusBoard] = None,
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.base_output_dir = base_output_dir
        self.frames_per_file = frames_per_file
        self.stop_event = stop_event or threading.Event()
        self.status_board = status_board

        self.socket = None
        self.buffer = ""
        self.buffer_max_size = 10 * 1024 * 1024
        self.recording = False
        self.saver: Optional[BatchSaver] = None
        self.pose_count = 0
        self.window = []
        self.window_size = 30
        self.last_display_time = 0.0
        self.lock = threading.Lock()

    def connect(self) -> bool:
        try:
            import socket

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1.0)
            self.socket.connect((self.host, self.port))
            print(f"[POSE] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[POSE][ERROR] Failed to connect: {e}")
            print("        Check: adb forward tcp:7777 tcp:7777")
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def start_recording(self, demo_dir: Optional[Path] = None):
        with self.lock:
            target_dir = demo_dir if demo_dir else self.base_output_dir
            pose_dir = Path(target_dir)
            pose_dir.mkdir(parents=True, exist_ok=True)
            self.saver = BatchSaver(str(pose_dir), frames_per_file=self.frames_per_file)
            self.recording = True
            print(f"[POSE] Recording to {pose_dir}")

    def stop_recording(self):
        with self.lock:
            if self.saver:
                self.saver.finalize()
                print(
                    f"[POSE] Saved {self.saver.total_saved_frames} frames "
                    f"in {self.saver.file_counter - 1} file(s)"
                )
            self.saver = None
            self.recording = False

    def parse_pose_data(self, json_str):
        try:
            data = json.loads(json_str.strip())
            if all(k in data for k in ["head_pose", "left_wrist", "right_wrist", "timestamp"]):
                return data
            return None
        except Exception:
            return None

    def _update_fps_display(self):
        now = time.time()
        self.pose_count += 1
        self.window.append(now)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size :]

        if len(self.window) >= 2:
            dur = self.window[-1] - self.window[0]
            window_fps = (len(self.window) - 1) / dur if dur > 0 else 0
        else:
            window_fps = 0

        if now - self.last_display_time >= 0.5:
            self.last_display_time = now
            if self.status_board:
                self.status_board.update_pose(window_fps, self.pose_count)
            else:
                print(f"\r[POSE] FPS: {window_fps:.1f}Hz | Packets: {self.pose_count}", end="", flush=True)

    def run(self):
        if not self.connect():
            return

        import socket

        print("[POSE] Receiving Quest data (Ctrl+C handled by main)...")
        # Always record pose as soon as connected (matches original standalone behavior)
        self.start_recording()
        try:
            while not self.stop_event.is_set():
                try:
                    data = self.socket.recv(1024).decode("utf-8")
                    if not data:
                        print("\n[POSE] No more data, closing...")
                        break

                    self.buffer += data
                    if len(self.buffer) > self.buffer_max_size:
                        print(f"\n[POSE][WARN] Buffer too large ({len(self.buffer)} bytes), clearing...")
                        self.buffer = ""
                        continue

                    while "\n" in self.buffer:
                        line, self.buffer = self.buffer.split("\n", 1)
                        if not line.strip():
                            continue

                        receive_ts = time.time()
                        pose_data = self.parse_pose_data(line)
                        if pose_data:
                            self._update_fps_display()
                            with self.lock:
                                if self.recording and self.saver:
                                    self.saver.add(pose_data, receive_ts)

                except socket.timeout:
                    continue
                except UnicodeDecodeError:
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    print("\n[POSE] Connection lost")
                    break
        finally:
            print("\n[POSE] Closing connection...")
            self.disconnect()
            with self.lock:
                if self.saver:
                    self.saver.finalize()
                    print(
                        f"[POSE] Saved {self.saver.total_saved_frames} frames "
                        f"in {self.saver.file_counter - 1} file(s)"
                    )
                self.saver = None
                self.recording = False


class CameraConfig:
    def __init__(
        self,
        name: str,
        path: str,
        format: str,
        width: int,
        height: int,
        auto_exposure: int,
        brightness: int,
        gain: int,
        gamma: int,
        exposure: Optional[int] = None,
        auto_white_balance: int = 1,
        wb_temperature: Optional[int] = None,
    ):
        self.name = name
        self.path = path
        self.format = format
        self.width = width
        self.height = height
        self.auto_exposure = auto_exposure
        self.brightness = brightness
        self.gain = gain
        self.gamma = gamma
        self.exposure = exposure
        self.auto_white_balance = auto_white_balance
        self.wb_temperature = wb_temperature


class RecordingSession:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.recording = False
        self.start_time: Optional[float] = None
        self.frame_counts: Dict[str, int] = {}
        self.timestamps: Dict[str, list] = {}
        self.lock = threading.Lock()

    def start(self, demo_name: str, cameras: Dict[str, CameraConfig]):
        with self.lock:
            self.recording = True
            self.start_time = time.time()
            demo_dir = os.path.join(self.output_dir, demo_name)
            os.makedirs(demo_dir, exist_ok=True)

            for cam_name in cameras:
                img_dir = os.path.join(demo_dir, f"{cam_name}_img")
                os.makedirs(img_dir, exist_ok=True)
                self.frame_counts[cam_name] = 0
                self.timestamps[cam_name] = []

            print(f"[START] {demo_name}")
            return demo_dir

    def stop(self, demo_dir: str) -> Dict:
        with self.lock:
            if not self.recording:
                return {}

            self.recording = False
            duration = time.time() - self.start_time
            stats = {}

            for cam_name, count in self.frame_counts.items():
                df = pd.DataFrame(self.timestamps[cam_name])
                csv_path = os.path.join(demo_dir, f"{cam_name}_timestamps.csv")
                df.to_csv(csv_path, index=False)

                fps = count / duration if duration > 0 else 0
                stats[cam_name] = {"frames": count, "fps": round(fps, 2)}
                print(f"[STOP] {cam_name}: {count} frames, {fps:.1f} fps")

            self.frame_counts.clear()
            self.timestamps.clear()
            return stats

    def save_frame(self, demo_dir: str, cam_name: str, frame: np.ndarray, ram_time: str):
        with self.lock:
            if not self.recording:
                return

            frame_id = self.frame_counts[cam_name]
            filename = f"{cam_name}_{frame_id}.jpg"
            filepath = os.path.join(demo_dir, f"{cam_name}_img", filename)
            cv2.imwrite(filepath, frame)

            self.timestamps[cam_name].append(
                {"frame_id": frame_id, "ram_time": ram_time, "filename": filename}
            )
            self.frame_counts[cam_name] += 1


class CameraWorker(threading.Thread):
    def __init__(self, config: CameraConfig, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.queue: Queue[Tuple[np.ndarray, str]] = Queue(maxsize=64)
        self.camera = None

    def run(self):
        try:
            self.camera = V4L2Camera(
                device_path=self.config.path,
                format=self.config.format,
                width=self.config.width,
                height=self.config.height,
            )
            self.camera.set_white_balance(
                auto=(self.config.auto_white_balance == 1), temperature=self.config.wb_temperature
            )
            self.camera.set_exposure(auto=(self.config.auto_exposure == 1), exposure_time=self.config.exposure)
            self.camera.set_brightness(brightness=self.config.brightness)
            self.camera.set_gain(self.config.gain)
            self.camera.set_gamma(gamma=self.config.gamma)

            print(f"[CAM] {self.config.name} ready: {self.config.width}x{self.config.height}")

            while not self.stop_event.is_set():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    _, ram_time = ret
                    try:
                        self.queue.put((frame, ram_time), block=False)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ERROR] {self.config.name}: {e}")
        finally:
            if self.camera:
                self.camera.release()

    def get_frame(self) -> Optional[Tuple[np.ndarray, str]]:
        try:
            return self.queue.get(block=False)
        except Exception:
            return None


class DataRecorder:
    """Image recorder (unchanged behavior)."""

    def __init__(self, config: OmegaConf, status_board: Optional[StatusBoard] = None):
        self.config = config
        self.cameras = self._load_cameras()
        self.workers = {}
        self.stop_event = threading.Event()
        self.session = RecordingSession(
            output_dir=os.path.join(config.recorder.output, config.task.name, "demos")
        )
        self.current_demo_dir = None
        self.demo_count = 0
        self.status_board = status_board
        self.last_totals: Dict[str, int] = {name: 0 for name in self.cameras}
        self._init_fps_monitoring()

    def _load_cameras(self) -> Dict[str, CameraConfig]:
        cam_cfg = self.config.recorder.camera
        paths_cfg = self.config.recorder.camera_paths
        task_type = self.config.task.type
        single_side = getattr(self.config.task, "single_hand_side", "left")

        settings = {
            "format": getattr(cam_cfg, "format", "MJPG"),
            "width": cam_cfg.width,
            "height": cam_cfg.height,
            "auto_exposure": getattr(cam_cfg, "auto_exposure", 1),
            "exposure": getattr(cam_cfg, "exposure", None),
            "auto_white_balance": getattr(cam_cfg, "auto_white_balance", 1),
            "wb_temperature": getattr(cam_cfg, "wb_temperature", None),
            "brightness": getattr(cam_cfg, "brightness", 0),
            "gain": getattr(cam_cfg, "gain", 100),
            "gamma": getattr(cam_cfg, "gamma", 100),
        }

        if task_type == "single":
            paths = {f"{single_side}_hand": paths_cfg[f"{single_side}_hand"]}
        else:
            paths = {"left_hand": paths_cfg.left_hand, "right_hand": paths_cfg.right_hand}

        cameras = {}
        for name, path in paths.items():
            cameras[name] = CameraConfig(name=name, path=path, **settings)
            print(f"[CONFIG] {name}: {path}")

        return cameras

    def _init_fps_monitoring(self):
        self.fps_counters = {name: 0 for name in self.cameras}
        self.fps_timers = {name: time.time() for name in self.cameras}
        self.last_fps = {name: 0.0 for name in self.cameras}  # Store last calculated FPS

    def start_workers(self):
        for name, config in self.cameras.items():
            worker = CameraWorker(config, self.stop_event)
            self.workers[name] = worker
            worker.start()

    def start_recording(self):
        if self.session.recording:
            return

        self.demo_count += 1
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")
        task_type = self.config.task.type
        demo_name = f"demo_{task_type}_{timestamp}"

        self.current_demo_dir = self.session.start(demo_name, self.cameras)
        print(f"[REC] Demo #{self.demo_count} started")
        if self.status_board:
            self.status_board.set_demo_status(f"Recording demo #{self.demo_count}")

    def stop_recording(self):
        if not self.session.recording:
            return

        stats = self.session.stop(self.current_demo_dir)
        print(f"[REC] Demo #{self.demo_count} completed\n")
        if self.status_board:
            # Persist totals for display after stop
            for cam_name, cam_stats in stats.items():
                self.last_totals[cam_name] = cam_stats["frames"]
            totals_str = ", ".join(
                f"{k}:{v['frames']}" for k, v in stats.items()
            ) if stats else "no frames"
            self.status_board.set_demo_status(
                f"Stopped demo #{self.demo_count} ({totals_str})"
            )
        self.current_demo_dir = None
        return stats

    def process_frames(self):
        frames_processed = 0
        temp_frame = {}
        for name, worker in self.workers.items():
            frame_data = worker.get_frame()
            if frame_data:
                frame, ram_time = frame_data
        #         temp_frame[name] = {}
        #         temp_frame[name]["frame"] = frame
        #         temp_frame[name]["ram_time"] = ram_time
        #         time.sleep(0.00001)

        # all_read = True
        # for name, worker in self.workers.items():
        #     if name not in temp_frame.keys():
        #         all_read = False
        #         return

        # for name, worker in self.workers.items():
                if self.session.recording and self.current_demo_dir:
                    # pass
                    # self.session.save_frame(self.current_demo_dir, name, temp_frame[name]["frame"], temp_frame[name]["ram_time"])
                    self.session.save_frame(self.current_demo_dir, name, frame, ram_time)

                self.fps_counters[name] += 1
                now = time.time()
                elapsed = now - self.fps_timers[name]
                if elapsed >= 1.0:
                    fps = self.fps_counters[name] / elapsed
                    self.last_fps[name] = fps  # Store the calculated FPS
                    if self.session.recording:
                        total = self.session.frame_counts.get(name, 0)
                    else:
                        total = self.last_totals.get(name, 0)
                    if self.status_board:
                        self.status_board.update_camera(name, fps, total)
                    else:
                        if self.session.recording:
                            print(f"[{name}] FPS: {fps:.1f}, Total: {total}")
                    self.fps_counters[name] = 0
                    self.fps_timers[name] = now

                frames_processed += 1

        if frames_processed == 0:
            time.sleep(0.001)

    def cleanup(self):
        if self.session.recording:
            self.stop_recording()

        self.stop_event.set()
        for worker in self.workers.values():
            worker.join(timeout=1)

        print(f"[EXIT] Recorded {self.demo_count} demos")


class CombinedRunner:
    """Combines image capture and pose capture with unified start/stop."""

    def __init__(self, config: OmegaConf, enable_pose: bool = True):
        self.config = config
        self.status_board = StatusBoard()
        self.status_board.set_demo_status("Idle")
        self.image_recorder = DataRecorder(config, status_board=self.status_board)
        self.enable_pose = enable_pose
        output_root = Path(config.recorder.output) / config.task.name
        self.pose_receiver = (
            QuestPoseReceiver(
                host="localhost",
                port=7777,
                base_output_dir=output_root / "all_trajectory",
                frames_per_file=5000,
                status_board=self.status_board,
            )
            if enable_pose
            else None
        )
        ctrl_host = getattr(config.recorder, "control_host", "localhost")
        ctrl_port = getattr(config.recorder, "control_port", 50010)
        self.control_client = ControlClient(ctrl_host, ctrl_port)
        self.should_exit = False
        self.tty_enabled = sys.stdin.isatty()
        self.old_settings = None
        self._setup_terminal()

    def _setup_terminal(self):
        if self.tty_enabled:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.tty_enabled = False

    def _restore_terminal(self):
        if self.old_settings and self.tty_enabled:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def _handle_input(self):
        if not self.tty_enabled:
            return True

        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q"):
                    self.should_exit = True
                    return False
                elif ch in ("s", "S"):
                    if self.image_recorder.session.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
        except Exception:
            pass
        return True

    def start_recording(self):
        self.image_recorder.start_recording()
        if self.control_client:
            self.control_client.send_demo_control(self.image_recorder.demo_count, "start")

    def stop_recording(self):
        stats = self.image_recorder.stop_recording()
        if self.control_client:
            self.control_client.send_demo_control(self.image_recorder.demo_count, "stop")
        return stats
    
    def _send_fps_updates(self):
        """Collect and send FPS updates to Quest."""
        if not self.control_client:
            return
        
        # Collect camera FPS (use last calculated FPS values)
        camera_fps = {}
        for name in self.image_recorder.cameras:
            fps = self.image_recorder.last_fps.get(name, 0.0)
            camera_fps[name] = round(fps, 1)
        
        # Collect pose FPS
        pose_fps = 0.0
        if self.enable_pose and self.pose_receiver:
            if len(self.pose_receiver.window) >= 2:
                dur = self.pose_receiver.window[-1] - self.pose_receiver.window[0]
                pose_fps = round((len(self.pose_receiver.window) - 1) / dur if dur > 0 else 0, 1)
        
        self.control_client.send_fps_update(camera_fps, pose_fps)

    def run(self):
        print("\n" + "=" * 60)
        print("  ViTaMIn-B Image + Pose Recorder")
        print("=" * 60)
        print("  [S] - Start/Stop recording both image & pose")
        print("  [Q] - Quit")
        print("=" * 60 + "\n")

        self.image_recorder.start_workers()
        if self.enable_pose and self.pose_receiver:
            self.pose_receiver.start()

        try:
            while not self.should_exit:
                self.image_recorder.process_frames()
                self._send_fps_updates()  # Send FPS updates to Quest
                if not self._handle_input():
                    break
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        self.image_recorder.cleanup()
        if self.enable_pose and self.pose_receiver:
            self.pose_receiver.stop_event.set()
            self.pose_receiver.join(timeout=2)
        if self.control_client:
            self.control_client.close()
        self._restore_terminal()
        print("[EXIT] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Combined image + pose recorder")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose capture")
    parser.add_argument("--control-host", type=str, default=None, help="Override control host")
    parser.add_argument("--control-port", type=int, default=None, help="Override control port")
    args = parser.parse_args()

    try:
        config = _build_omegaconf_from_collection_config(DATA_COLLECTION_CONFIG)

        # allow CLI override for control channel
        if args.control_host is not None:
            config.recorder.control_host = args.control_host
        if args.control_port is not None:
            config.recorder.control_port = args.control_port

        task_dir = os.path.join(config.recorder.output, config.task.name)
        os.makedirs(os.path.join(task_dir, "demos"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "all_trajectory"), exist_ok=True)

        runner = CombinedRunner(config, enable_pose=not args.no_pose)
        runner.run()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
