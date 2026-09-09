import csv
import os
import time

import cv2
import numpy as np


COMPATIBLE_AVI_CODECS = ("MJPG", "XVID")


def open_compatible_avi_writer(path, fps, size):
    """Open a full-resolution AVI writer that standard Windows players support."""
    width, height = (int(size[0]), int(size[1]))
    if width < 2 or height < 2:
        raise ValueError(f"Invalid video frame size: {width}x{height}")
    if width % 2 or height % 2:
        raise ValueError(f"Video frame size must be even: {width}x{height}")

    for codec in COMPATIBLE_AVI_CODECS:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Unable to open a compatible AVI writer: {path}")


def inspect_video_file(path):
    """Return basic decode information after a writer has finalized its file."""
    if not path or not os.path.isfile(path):
        return {"readable": False, "frames": 0, "fps": 0.0, "size": (0, 0)}
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    readable, _ = capture.read() if opened else (False, None)
    info = {
        "readable": bool(opened and readable),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0,
        "fps": float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
        "size": (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0,
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0,
        ),
    }
    capture.release()
    return info


class DualCameraRecorder:
    """Write synchronized camera and simulation views to one session."""

    def __init__(self, save_dir, timestamp_str, fps=20.0):
        os.makedirs(save_dir, exist_ok=True)
        self.fps = float(fps)
        self.frame_count = 0
        self.real_path = os.path.join(save_dir, f"RealCamera_{timestamp_str}.avi")
        self.virtual_path = os.path.join(save_dir, f"VirtualCamera_{timestamp_str}.avi")
        self.mujoco_path = os.path.join(save_dir, f"MujocoView_{timestamp_str}.avi")
        self.control_path = os.path.join(save_dir, f"RemoteControl_{timestamp_str}.avi")
        self.manifest_path = os.path.join(save_dir, f"CameraFrames_{timestamp_str}.csv")
        self.keypoint_path = os.path.join(save_dir, f"KeyPointPoses_{timestamp_str}.csv")
        self.real_writer = None
        self.virtual_writer = None
        self.mujoco_writer = None
        self.control_writer = None
        self.real_size = None
        self.virtual_size = None
        self.mujoco_size = None
        self.control_size = None
        self.mujoco_frame_count = 0
        self.control_frame_count = 0
        self.codec_by_path = {}
        self._closed = False
        self._manifest_file = open(self.manifest_path, "w", newline="", encoding="utf-8")
        self._manifest_writer = csv.writer(self._manifest_file)
        self._manifest_writer.writerow(
            [
                "FrameIndex",
                "Time(s)",
                "RealWidth",
                "RealHeight",
                "VirtualWidth",
                "VirtualHeight",
                "MujocoFrameIndex",
                "MujocoWidth",
                "MujocoHeight",
                "RemoteControlFrameIndex",
                "RemoteControlWidth",
                "RemoteControlHeight",
            ]
        )
        self._keypoint_file = None
        self._keypoint_writer = None

    @staticmethod
    def _prepare_bgr(frame):
        if frame is None:
            return None
        image = np.asarray(frame)
        if image.size == 0:
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim != 3:
            return None
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3:
            return None
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return None
        # Common AVI codecs require even frame dimensions.
        if height % 2 or width % 2:
            image = image[: height - (height % 2), : width - (width % 2)]
        return np.ascontiguousarray(image)

    def _open_writer(self, path, size):
        writer, codec = open_compatible_avi_writer(path, self.fps, size)
        self.codec_by_path[path] = codec
        print(
            f">>> [Record] {os.path.basename(path)}: "
            f"{codec}, {size[0]}x{size[1]} @ {self.fps:g} FPS"
        )
        return writer

    def _write_optional_frame(
        self,
        frame,
        writer_attr,
        size_attr,
        path,
        count_attr,
    ):
        prepared = self._prepare_bgr(frame)
        if prepared is None:
            return -1
        writer = getattr(self, writer_attr)
        size = getattr(self, size_attr)
        if writer is None:
            size = (prepared.shape[1], prepared.shape[0])
            writer = self._open_writer(path, size)
            setattr(self, writer_attr, writer)
            setattr(self, size_attr, size)
        if (prepared.shape[1], prepared.shape[0]) != size:
            prepared = cv2.resize(prepared, size, interpolation=cv2.INTER_AREA)
        frame_index = int(getattr(self, count_attr))
        writer.write(prepared)
        setattr(self, count_attr, frame_index + 1)
        return frame_index

    def _write_keypoint_poses(self, frame_index, elapsed_s, keypoint_poses):
        if not keypoint_poses:
            return
        point_names = ("tip", "middle_platform", "master_slave_connection")
        if self._keypoint_writer is None:
            self._keypoint_file = open(
                self.keypoint_path,
                "w",
                newline="",
                encoding="utf-8",
            )
            self._keypoint_writer = csv.writer(self._keypoint_file)
            headers = ["FrameIndex", "Time(s)"]
            for point_name in point_names:
                headers.extend(
                    [
                        f"{point_name}_World_X_mm",
                        f"{point_name}_World_Y_mm",
                        f"{point_name}_World_Z_mm",
                        f"{point_name}_World_QX",
                        f"{point_name}_World_QY",
                        f"{point_name}_World_QZ",
                        f"{point_name}_World_QW",
                    ]
                )
            self._keypoint_writer.writerow(headers)

        row = [frame_index, f"{float(elapsed_s):.6f}"]
        for point_name in point_names:
            pose = keypoint_poses.get(point_name)
            if pose is None:
                row.extend([float("nan")] * 7)
                continue
            position = np.asarray(pose["position_mm"], dtype=float).reshape(3)
            quaternion = np.asarray(
                pose["quaternion_xyzw"],
                dtype=float,
            ).reshape(4)
            row.extend(position.tolist() + quaternion.tolist())
        self._keypoint_writer.writerow(row)

    def write_pair(
        self,
        real_frame_bgr,
        virtual_frame_bgr,
        elapsed_s=None,
        mujoco_frame_bgr=None,
        control_frame_bgr=None,
        keypoint_poses=None,
    ):
        if self._closed:
            return False
        real_frame = self._prepare_bgr(real_frame_bgr)
        virtual_frame = self._prepare_bgr(virtual_frame_bgr)
        if real_frame is None or virtual_frame is None:
            return False

        if self.real_writer is None:
            self.real_size = (real_frame.shape[1], real_frame.shape[0])
            self.real_writer = self._open_writer(self.real_path, self.real_size)
        if self.virtual_writer is None:
            self.virtual_size = (virtual_frame.shape[1], virtual_frame.shape[0])
            self.virtual_writer = self._open_writer(self.virtual_path, self.virtual_size)

        if (real_frame.shape[1], real_frame.shape[0]) != self.real_size:
            real_frame = cv2.resize(real_frame, self.real_size, interpolation=cv2.INTER_AREA)
        if (virtual_frame.shape[1], virtual_frame.shape[0]) != self.virtual_size:
            virtual_frame = cv2.resize(virtual_frame, self.virtual_size, interpolation=cv2.INTER_AREA)

        self.real_writer.write(real_frame)
        self.virtual_writer.write(virtual_frame)
        mujoco_frame_index = self._write_optional_frame(
            mujoco_frame_bgr,
            "mujoco_writer",
            "mujoco_size",
            self.mujoco_path,
            "mujoco_frame_count",
        )
        control_frame_index = self._write_optional_frame(
            control_frame_bgr,
            "control_writer",
            "control_size",
            self.control_path,
            "control_frame_count",
        )
        current_time = time.time() if elapsed_s is None else float(elapsed_s)
        self._write_keypoint_poses(
            self.frame_count,
            current_time,
            keypoint_poses,
        )
        self._manifest_writer.writerow(
            [
                self.frame_count,
                f"{current_time:.6f}",
                self.real_size[0],
                self.real_size[1],
                self.virtual_size[0],
                self.virtual_size[1],
                mujoco_frame_index,
                "" if self.mujoco_size is None else self.mujoco_size[0],
                "" if self.mujoco_size is None else self.mujoco_size[1],
                control_frame_index,
                "" if self.control_size is None else self.control_size[0],
                "" if self.control_size is None else self.control_size[1],
            ]
        )
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            self._manifest_file.flush()
            if self._keypoint_file is not None:
                self._keypoint_file.flush()
        return True

    def close(self):
        if self._closed:
            return self.frame_count
        self._closed = True
        for writer in (
            self.real_writer,
            self.virtual_writer,
            self.mujoco_writer,
            self.control_writer,
        ):
            if writer is not None:
                writer.release()
        self.real_writer = None
        self.virtual_writer = None
        self.mujoco_writer = None
        self.control_writer = None
        self._manifest_file.flush()
        self._manifest_file.close()
        if self._keypoint_file is not None:
            self._keypoint_file.flush()
            self._keypoint_file.close()
            self._keypoint_file = None
            self._keypoint_writer = None
        for path in self.codec_by_path:
            info = inspect_video_file(path)
            if not info["readable"]:
                print(f">>> [Record] WARNING: finalized video is not readable: {path}")
        return self.frame_count


class DualCameraRecordingMixin:
    """Capture the two camera panes and feed a synchronized recorder."""

    def initialize_dual_camera_recording(self):
        self.dual_camera_recorder = None
        self.recording_timestamp_str = None
        self.recording_session_dir = None
        self._camera_recording_last_error = None

    def prepare_recording_session(self, timestamp_str):
        self.recording_timestamp_str = str(timestamp_str)
        self.recording_session_dir = os.path.join(
            self.save_dir,
            f"Record_{self.recording_timestamp_str}",
        )
        os.makedirs(self.recording_session_dir, exist_ok=True)
        return self.recording_session_dir

    def start_dual_camera_recording(self, timestamp_str):
        self.stop_dual_camera_recording()
        timestamp_str = str(timestamp_str)
        if self.recording_timestamp_str != timestamp_str or not self.recording_session_dir:
            self.prepare_recording_session(timestamp_str)
        self._camera_recording_last_error = None
        self.dual_camera_recorder = DualCameraRecorder(
            self.recording_session_dir,
            self.recording_timestamp_str,
            fps=20.0,
        )
        simulator = getattr(self, "mujoco_simulator", None)
        if getattr(self, "is_simulation_mode", False) and simulator is not None:
            simulator.set_screen_recording_enabled(True)
        print(
            ">>> [Record] Camera and MuJoCo view synchronized recording started"
        )

    def _get_real_camera_recording_frame(self):
        if getattr(self, "is_simulation_mode", False):
            simulator = getattr(self, "mujoco_simulator", None)
            if simulator is not None:
                return getattr(simulator, "latest_tip_frame_bgr", None)
        return getattr(self, "latest_real_frame_bgr", None)

    def _get_virtual_camera_recording_frame(self):
        virtual_view = getattr(self, "virtual_view", None)
        if virtual_view is None or getattr(virtual_view, "_closed", False):
            return None
        rgb_frame = virtual_view.screenshot(return_img=True)
        if rgb_frame is None:
            return None
        rgb_frame = np.asarray(rgb_frame)
        if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
            rgb_frame = rgb_frame[:, :, :3]
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            return None
        return cv2.cvtColor(np.ascontiguousarray(rgb_frame), cv2.COLOR_RGB2BGR)

    def _get_mujoco_recording_frames(self):
        if not getattr(self, "is_simulation_mode", False):
            return None, None
        simulator = getattr(self, "mujoco_simulator", None)
        if simulator is None:
            return None, None
        mujoco_frame = getattr(simulator, "latest_viewer_frame_bgr", None)
        control_panel = getattr(simulator, "physics_panel", None)
        control_frame = (
            None
            if control_panel is None
            else getattr(control_panel, "latest_frame_bgr", None)
        )
        return mujoco_frame, control_frame

    def _get_mujoco_keypoint_poses(self):
        if not getattr(self, "is_simulation_mode", False):
            return None
        simulator = getattr(self, "mujoco_simulator", None)
        if simulator is None:
            return None
        return simulator.get_keypoint_poses()

    def record_dual_camera_frame_pair(self):
        recorder = getattr(self, "dual_camera_recorder", None)
        if not getattr(self, "is_recording_trajectory", False) or recorder is None:
            return False
        try:
            real_frame = self._get_real_camera_recording_frame()
            virtual_frame = self._get_virtual_camera_recording_frame()
            if real_frame is None or virtual_frame is None:
                return False
            mujoco_frame, control_frame = self._get_mujoco_recording_frames()
            keypoint_poses = self._get_mujoco_keypoint_poses()
            elapsed_s = max(0.0, time.time() - float(self.recording_start_time))
            return recorder.write_pair(
                real_frame,
                virtual_frame,
                elapsed_s,
                mujoco_frame_bgr=mujoco_frame,
                control_frame_bgr=control_frame,
                keypoint_poses=keypoint_poses,
            )
        except Exception as exc:
            error_text = str(exc)
            if error_text != self._camera_recording_last_error:
                print(f"Camera pair recording failed: {error_text}")
                self._camera_recording_last_error = error_text
            return False

    def stop_dual_camera_recording(self):
        recorder = getattr(self, "dual_camera_recorder", None)
        if recorder is None:
            return 0
        frame_count = recorder.close()
        self.dual_camera_recorder = None
        simulator = getattr(self, "mujoco_simulator", None)
        if simulator is not None:
            simulator.set_screen_recording_enabled(False)
        print(
            ">>> [Record] Recording stopped: "
            f"{frame_count} camera pairs, "
            f"{recorder.mujoco_frame_count} MuJoCo frames, "
            f"{recorder.control_frame_count} control frames"
        )
        return frame_count
