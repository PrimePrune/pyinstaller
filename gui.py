from __future__ import annotations

import os
import sys
import time
import tempfile
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QObject
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)


# -----------------------------
# Utils
# -----------------------------
def to_qimage_gray8(gray8: np.ndarray) -> QImage:
    if gray8.dtype != np.uint8:
        gray8 = gray8.astype(np.uint8, copy=False)
    h, w = gray8.shape[:2]
    return QImage(gray8.data, w, h, w, QImage.Format_Grayscale8).copy()


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return frame
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return frame


def as_u8(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        return gray
    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


def safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# -----------------------------
# Processing
# -----------------------------
@dataclass
class ProcParams:
    # Bilateral
    denoise_on: bool = True
    sigma_color: int = 10   # 0~150 (권장 10~60)
    sigma_space: int = 3    # 0~50  (권장 3~12)

    # CLAHE
    clahe_on: bool = True
    clahe_clip: int = 20    # 1~100 -> 0.5~6.0
    clahe_tile: int = 8     # 4~16

    # Sharpen (Unsharp)
    sharp_on: bool = True
    sharp_amount: int = 30  # 0~100 -> 0~2.0
    sharp_sigma: int = 10   # 1~30  -> 0.5~3.0


class Processor:
    def apply(self, gray_u8: np.ndarray, p: ProcParams) -> np.ndarray:
        if gray_u8 is None:
            return gray_u8

        x = gray_u8.astype(np.float32, copy=False)

        # 1) Bilateral
        if p.denoise_on:
            x = cv2.bilateralFilter(
                x,
                d=0,
                sigmaColor=float(p.sigma_color),
                sigmaSpace=float(p.sigma_space),
            )

        # 2) CLAHE
        if p.clahe_on:
            tmp = np.clip(x, 0, 255).astype(np.uint8)
            clip = float(np.interp(p.clahe_clip, [1, 100], [0.5, 6.0]))
            tile = int(np.clip(p.clahe_tile, 4, 16))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            tmp = clahe.apply(tmp)
            x = tmp.astype(np.float32)

        # 3) Sharpen
        if p.sharp_on:
            amount = float(np.interp(p.sharp_amount, [0, 100], [0.0, 2.0]))
            sigma = float(np.interp(p.sharp_sigma, [1, 30], [0.5, 3.0]))
            blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
            x = cv2.addWeighted(x, 1.0 + amount, blur, -amount, 0)

        return np.clip(x, 0, 255).astype(np.uint8)


# -----------------------------
# Workers
# -----------------------------
class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray, np.ndarray, float)  # orig_u8, proc_u8, proc_ms
    meta_ready = Signal(float, int, int)                 # fps, w, h
    finished = Signal(str)

    def __init__(self, path: str, params_getter):
        super().__init__()
        self.path = path
        self.params_getter = params_getter
        self._stop = False
        self._pause = False
        self.processor = Processor()

    def stop(self):
        self._stop = True

    def set_pause(self, pause: bool):
        self._pause = pause

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.finished.emit("영상 파일을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1e-6:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.meta_ready.emit(float(fps), w, h)

        frame_period = 1.0 / float(fps)

        while not self._stop:
            if self._pause:
                self.msleep(30)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            orig_u8 = as_u8(ensure_gray(frame))

            t0 = time.perf_counter()
            params = self.params_getter()
            proc_u8 = self.processor.apply(orig_u8, params)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            self.frame_ready.emit(orig_u8, proc_u8, dt_ms)

            sleep_ms = max(0, int(frame_period * 1000) - 1)
            self.msleep(sleep_ms)

        cap.release()
        self.finished.emit("재생 종료")


class PreprocessWorker(QThread):
    progress = Signal(int)
    done = Signal(bool, str, str)  # ok, message, processed_path

    def __init__(self, in_path: str, out_path: str, params: ProcParams):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.params = params
        self._stop = False
        self.processor = Processor()

    def stop(self):
        self._stop = True

    def run(self):
        cap = cv2.VideoCapture(self.in_path)
        if not cap.isOpened():
            self.done.emit(False, "입력 영상을 열 수 없습니다.", "")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1e-6:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.out_path, fourcc, float(fps), (w, h), isColor=False)
        if not writer.isOpened():
            cap.release()
            self.done.emit(False, "출력 VideoWriter를 열 수 없습니다. 코덱 또는 경로를 확인해 주세요.", "")
            return

        idx = 0
        while True:
            if self._stop:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            orig_u8 = as_u8(ensure_gray(frame))
            out_u8 = self.processor.apply(orig_u8, self.params)
            writer.write(out_u8)

            idx += 1
            if total > 0:
                self.progress.emit(min(100, int((idx / total) * 100)))
            elif idx % 30 == 0:
                self.progress.emit(min(99, idx % 100))

        cap.release()
        writer.release()

        if self._stop:
            self.done.emit(False, "Preprocessing canceled", self.out_path)
        else:
            self.progress.emit(100)
            self.done.emit(True, "Preprocessing complete", self.out_path)


# -----------------------------
# Compare Player (Non-Realtime) + Seek
# -----------------------------
class ComparePlayer(QObject):
    frame_pair = Signal(np.ndarray, np.ndarray)      # orig_u8, proc_u8
    status = Signal(str)
    position_changed = Signal(int, int, float)       # cur_frame, total_frames, fps

    def __init__(self):
        super().__init__()
        self.cap_a: Optional[cv2.VideoCapture] = None
        self.cap_b: Optional[cv2.VideoCapture] = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self.fps: float = 30.0
        self.total_frames: int = 0
        self.cur_frame: int = -1  # 현재 표시된 프레임 index

    def is_running(self) -> bool:
        return self.timer.isActive()

    def start(self, orig_path: str, proc_path: str):
        self.stop()

        self.cap_a = cv2.VideoCapture(orig_path)
        self.cap_b = cv2.VideoCapture(proc_path)
        if not (self.cap_a.isOpened() and self.cap_b.isOpened()):
            self.stop()
            self.status.emit("비교 재생을 시작할 수 없습니다. 파일을 확인해 주세요.")
            return

        fps = self.cap_a.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1e-6:
            fps = 30.0
        self.fps = float(fps)

        fa = int(self.cap_a.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fb = int(self.cap_b.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.total_frames = min(fa, fb) if (fa > 0 and fb > 0) else max(fa, fb)

        # 첫 프레임 표시
        self.seek(0, emit_status=False)

        # 재생 시작
        self.timer.start(int(1000 / self.fps))
        self.status.emit("비교 재생을 시작합니다.")

    def pause(self, pause: bool):
        if pause:
            if self.timer.isActive():
                self.timer.stop()
                self.status.emit("비교 재생이 일시정지되었습니다.")
        else:
            if self.cap_a is not None and self.cap_b is not None:
                if not self.timer.isActive():
                    self.timer.start(int(1000 / self.fps))
                    self.status.emit("비교 재생을 재개합니다.")

    def stop(self):
        self.timer.stop()
        if self.cap_a is not None:
            self.cap_a.release()
        if self.cap_b is not None:
            self.cap_b.release()
        self.cap_a = None
        self.cap_b = None
        self.total_frames = 0
        self.cur_frame = -1

    def seek(self, frame_idx: int, emit_status: bool = True):
        """두 영상 모두 같은 프레임으로 이동해서 즉시 1프레임 표시."""
        if self.cap_a is None or self.cap_b is None:
            return
        if self.total_frames > 0:
            frame_idx = int(np.clip(frame_idx, 0, self.total_frames - 1))
        else:
            frame_idx = max(0, int(frame_idx))

        # codec 특성상 정확히 안 맞을 수 있음(키프레임), 그래도 일반적으로 충분
        self.cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ok1, f1 = self.cap_a.read()
        ok2, f2 = self.cap_b.read()
        if not ok1 or not ok2 or f1 is None or f2 is None:
            # seek 실패하면 종료 처리
            self.stop()
            if emit_status:
                self.status.emit("비교 재생이 종료되었습니다.")
            return

        o = as_u8(ensure_gray(f1))
        p = as_u8(ensure_gray(f2))
        self.cur_frame = frame_idx
        self.frame_pair.emit(o, p)
        self.position_changed.emit(self.cur_frame, self.total_frames, self.fps)

    def _tick(self):
        if self.cap_a is None or self.cap_b is None:
            self.stop()
            return

        ok1, f1 = self.cap_a.read()
        ok2, f2 = self.cap_b.read()
        if not ok1 or not ok2 or f1 is None or f2 is None:
            self.stop()
            self.status.emit("비교 재생이 종료되었습니다.")
            return

        self.cur_frame += 1
        o = as_u8(ensure_gray(f1))
        p = as_u8(ensure_gray(f2))
        self.frame_pair.emit(o, p)
        self.position_changed.emit(self.cur_frame, self.total_frames, self.fps)

        # total_frames가 있으면 끝에서 stop
        if self.total_frames > 0 and self.cur_frame >= self.total_frames - 1:
            self.stop()
            self.status.emit("비교 재생이 종료되었습니다.")


# -----------------------------
# MainWindow
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR Processing GUI (PySide6)")

        self.params = ProcParams()
        self.current_path: Optional[str] = None
        self.is_video: bool = False

        self.video_worker: Optional[VideoWorker] = None
        self.prep_worker: Optional[PreprocessWorker] = None

        self.compare_player = ComparePlayer()
        self.compare_player.frame_pair.connect(self._on_compare_frame)
        self.compare_player.status.connect(self._set_status)
        self.compare_player.position_changed.connect(self._on_compare_position)

        self.orig_u8: Optional[np.ndarray] = None
        self.proc_u8: Optional[np.ndarray] = None

        # ✅ 항상 같은 임시 캐시 파일(덮어쓰기)
        self.cache_path = os.path.join(tempfile.gettempdir(), "ir_gui_preprocessed_cache.mp4")
        self.preprocessed_path: Optional[str] = None  # 성공 시 cache_path로 설정
        self.preprocessed_is_temp: bool = True        # 항상 temp 캐시를 사용

        # slider 드래그 상태
        self._slider_dragging = False
        self._compare_was_running = False

        self._build_ui()
        self._bind_actions()
        self._apply_tooltips()
        self._set_compare_slider_enabled(False)

    # ---------- UI ----------
    def _build_ui(self):
        self.act_open = QAction("Open", self)
        self.act_save_img = QAction("Save Image As...", self)
        self.act_play = QAction("Play (Realtime)", self)
        self.act_pause = QAction("Pause", self)
        self.act_stop = QAction("Stop", self)

        tb = self.addToolBar("Main")
        tb.addAction(self.act_open)
        tb.addAction(self.act_save_img)
        tb.addSeparator()
        tb.addAction(self.act_play)
        tb.addAction(self.act_pause)
        tb.addAction(self.act_stop)

        # Left controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # Denoise
        self.grp_denoise = QGroupBox("1) Denoise (Bilateral)")
        gl = QVBoxLayout(self.grp_denoise)

        self.chk_denoise = QCheckBox("Enable")
        self.chk_denoise.setChecked(self.params.denoise_on)
        gl.addWidget(self.chk_denoise)

        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("sigmaColor"))
        self.sld_sigmac = QSlider(Qt.Horizontal)
        self.sld_sigmac.setRange(0, 60)
        self.sld_sigmac.setValue(self.params.sigma_color)
        self.spn_sigmac = QSpinBox()
        self.spn_sigmac.setRange(0, 60)
        self.spn_sigmac.setValue(self.params.sigma_color)
        row_c.addWidget(self.sld_sigmac)
        row_c.addWidget(self.spn_sigmac)

        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("sigmaSpace"))
        self.sld_sigmas = QSlider(Qt.Horizontal)
        self.sld_sigmas.setRange(0, 15)
        self.sld_sigmas.setValue(self.params.sigma_space)
        self.spn_sigmas = QSpinBox()
        self.spn_sigmas.setRange(0, 15)
        self.spn_sigmas.setValue(self.params.sigma_space)
        row_s.addWidget(self.sld_sigmas)
        row_s.addWidget(self.spn_sigmas)

        gl.addLayout(row_c)
        gl.addLayout(row_s)

        # CLAHE
        self.grp_clahe = QGroupBox("2) Contrast (CLAHE)")
        cl = QVBoxLayout(self.grp_clahe)
        self.chk_clahe = QCheckBox("Enable")
        self.chk_clahe.setChecked(self.params.clahe_on)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("clipLimit"))
        self.sld_clip = QSlider(Qt.Horizontal)
        self.sld_clip.setRange(1, 100)
        self.sld_clip.setValue(self.params.clahe_clip)
        self.spn_clip = QSpinBox()
        self.spn_clip.setRange(1, 100)
        self.spn_clip.setValue(self.params.clahe_clip)
        row1.addWidget(self.sld_clip)
        row1.addWidget(self.spn_clip)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("tile"))
        self.sld_tile = QSlider(Qt.Horizontal)
        self.sld_tile.setRange(4, 16)
        self.sld_tile.setValue(self.params.clahe_tile)
        self.spn_tile = QSpinBox()
        self.spn_tile.setRange(4, 16)
        self.spn_tile.setValue(self.params.clahe_tile)
        row2.addWidget(self.sld_tile)
        row2.addWidget(self.spn_tile)

        cl.addWidget(self.chk_clahe)
        cl.addLayout(row1)
        cl.addLayout(row2)

        # Sharpen
        self.grp_sharp = QGroupBox("3) Sharpen (Unsharp)")
        sl = QVBoxLayout(self.grp_sharp)
        self.chk_sharp = QCheckBox("Enable")
        self.chk_sharp.setChecked(self.params.sharp_on)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("amount"))
        self.sld_amount = QSlider(Qt.Horizontal)
        self.sld_amount.setRange(0, 100)
        self.sld_amount.setValue(self.params.sharp_amount)
        self.spn_amount = QSpinBox()
        self.spn_amount.setRange(0, 100)
        self.spn_amount.setValue(self.params.sharp_amount)
        row3.addWidget(self.sld_amount)
        row3.addWidget(self.spn_amount)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("sigma"))
        self.sld_sigma = QSlider(Qt.Horizontal)
        self.sld_sigma.setRange(1, 30)
        self.sld_sigma.setValue(self.params.sharp_sigma)
        self.spn_sigma = QSpinBox()
        self.spn_sigma.setRange(1, 30)
        self.spn_sigma.setValue(self.params.sharp_sigma)
        row4.addWidget(self.sld_sigma)
        row4.addWidget(self.spn_sigma)

        sl.addWidget(self.chk_sharp)
        sl.addLayout(row3)
        sl.addLayout(row4)

        # Buttons
        self.btn_preview = QPushButton("Preview (1-Frame)")
        self.btn_preview.setEnabled(False)

        self.btn_preprocess = QPushButton("Preprocess (Temp Cache)")
        self.btn_preprocess.setEnabled(False)

        self.btn_preprocess_cancel = QPushButton("Cancel Preprocess")
        self.btn_preprocess_cancel.setEnabled(False)

        self.btn_compare_play = QPushButton("Compare Play (Non-Realtime)")
        self.btn_compare_play.setEnabled(False)

        self.btn_save_preprocessed = QPushButton("Save Preprocessed As...")
        self.btn_save_preprocessed.setEnabled(False)

        left_layout.addWidget(self.grp_denoise)
        left_layout.addWidget(self.grp_clahe)
        left_layout.addWidget(self.grp_sharp)
        left_layout.addWidget(self.btn_preview)
        left_layout.addWidget(self.btn_preprocess)
        left_layout.addWidget(self.btn_preprocess_cancel)
        left_layout.addWidget(self.btn_compare_play)
        left_layout.addWidget(self.btn_save_preprocessed)
        left_layout.addStretch(1)

        # Preview panels
        self.lbl_orig = QLabel("Original")
        self.lbl_proc = QLabel("Processed")
        for lbl in (self.lbl_orig, self.lbl_proc):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(480, 360)
            lbl.setStyleSheet("QLabel { background: #111; color: #eee; }")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._wrap_panel("Original", self.lbl_orig))
        splitter.addWidget(self._wrap_panel("Processed", self.lbl_proc))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # ✅ Compare playback slider row
        self.sld_playback = QSlider(Qt.Horizontal)
        self.sld_playback.setRange(0, 0)
        self.lbl_playback = QLabel("--:-- / --:--   (0 / 0)")
        self.lbl_playback.setAlignment(Qt.AlignRight)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.sld_playback, 1)
        slider_row.addWidget(self.lbl_playback, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(splitter, 1)
        right_layout.addLayout(slider_row, 0)

        central = QWidget()
        main = QHBoxLayout(central)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(10)
        main.addWidget(left, 0)
        main.addWidget(right, 1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.status.addPermanentWidget(self.progress)

    def _wrap_panel(self, title: str, content: QLabel) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(6, 6, 6, 6)
        t = QLabel(title)
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("QLabel { color: #bbb; }")
        l.addWidget(t)
        l.addWidget(content, 1)
        return w

    # ---------- Binds ----------
    def _bind_actions(self):
        self.act_open.triggered.connect(self.open_file)
        self.act_save_img.triggered.connect(self.save_image_as)
        self.act_play.triggered.connect(self.play_realtime)
        self.act_pause.triggered.connect(self.pause_all)
        self.act_stop.triggered.connect(self.stop_all)

        def bind_slider_spin(slider: QSlider, spin: QSpinBox):
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)

        bind_slider_spin(self.sld_sigmac, self.spn_sigmac)
        bind_slider_spin(self.sld_sigmas, self.spn_sigmas)
        bind_slider_spin(self.sld_clip, self.spn_clip)
        bind_slider_spin(self.sld_tile, self.spn_tile)
        bind_slider_spin(self.sld_amount, self.spn_amount)
        bind_slider_spin(self.sld_sigma, self.spn_sigma)

        def on_param_changed():
            self._pull_params_from_ui()

        for w in [
            self.chk_denoise, self.sld_sigmac, self.sld_sigmas,
            self.chk_clahe, self.sld_clip, self.sld_tile,
            self.chk_sharp, self.sld_amount, self.sld_sigma,
        ]:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(on_param_changed)
            else:
                w.valueChanged.connect(on_param_changed)

        self.btn_preview.clicked.connect(self.preview_one_frame)
        self.btn_preprocess.clicked.connect(self.preprocess_temp_cache)
        self.btn_preprocess_cancel.clicked.connect(self.cancel_preprocess)
        self.btn_compare_play.clicked.connect(self.compare_play_non_realtime)
        self.btn_save_preprocessed.clicked.connect(self.save_preprocessed_as)

        # ✅ Playback slider events
        self.sld_playback.sliderPressed.connect(self._on_slider_pressed)
        self.sld_playback.sliderReleased.connect(self._on_slider_released)
        self.sld_playback.sliderMoved.connect(self._on_slider_moved)

    def _apply_tooltips(self):
        self.btn_preview.setToolTip(
            "현재 프레임에 영상처리 적용"
        )
        self.btn_preprocess.setToolTip(
            "영상 전체를 전처리"
        )
        self.btn_preprocess_cancel.setToolTip(
            "진행 중인 전처리 중단"
        )
        self.btn_compare_play.setToolTip(
            "원본/처리 영상 동시 재생"
        )
        self.btn_save_preprocessed.setToolTip(
            "전처리 결과를 원하는 경로에 저장"
        )

        self.sld_sigmac.setToolTip(
            "값이 클수록 밝기 차이가 큰 픽셀도 함께 섞여 더 부드러워질 수 있습니다."
        )
        self.sld_sigmas.setToolTip(
            "값이 클수록 더 먼 픽셀도 함께 섞이며 처리 속도가 느려질 수 있습니다."
        )

        self.sld_clip.setToolTip("clipLimit을 조절합니다. 값이 클수록 대비가 증가할 수 있습니다.")
        self.sld_tile.setToolTip("tileGridSize(블록 크기)를 조절합니다. 값이 작을수록 더 국부적으로 동작합니다.")

        self.sld_amount.setToolTip("샤프닝 강도를 조절합니다. 값이 클수록 경계가 더 강조됩니다.")
        self.sld_sigma.setToolTip("블러 반경(sigma)을 조절합니다. 값이 작으면 미세 경계를 더 강조합니다.")

    # ---------- Params ----------
    def _pull_params_from_ui(self):
        self.params.denoise_on = self.chk_denoise.isChecked()
        self.params.sigma_color = int(self.sld_sigmac.value())
        self.params.sigma_space = int(self.sld_sigmas.value())

        self.params.clahe_on = self.chk_clahe.isChecked()
        self.params.clahe_clip = int(self.sld_clip.value())
        self.params.clahe_tile = int(self.sld_tile.value())

        self.params.sharp_on = self.chk_sharp.isChecked()
        self.params.sharp_amount = int(self.sld_amount.value())
        self.params.sharp_sigma = int(self.sld_sigma.value())

    def _get_params_snapshot(self) -> ProcParams:
        return ProcParams(**self.params.__dict__)

    # ---------- Compare slider helpers ----------
    def _set_compare_slider_enabled(self, enabled: bool):
        self.sld_playback.setEnabled(enabled)
        if not enabled:
            self.sld_playback.blockSignals(True)
            self.sld_playback.setRange(0, 0)
            self.sld_playback.setValue(0)
            self.sld_playback.blockSignals(False)
            self.lbl_playback.setText("--:-- / --:--   (0 / 0)")

    def _on_slider_pressed(self):
        if not self.sld_playback.isEnabled():
            return
        self._slider_dragging = True
        self._compare_was_running = self.compare_player.is_running()
        if self._compare_was_running:
            self.compare_player.pause(True)

    def _on_slider_moved(self, value: int):
        # 드래그 중엔 표시만 갱신(실제 seek는 release에서)
        total = max(1, self.compare_player.total_frames)
        fps = self.compare_player.fps if self.compare_player.fps > 0 else 30.0
        cur_sec = value / fps
        tot_sec = (total - 1) / fps if total > 1 else 0.0
        self.lbl_playback.setText(
            f"{format_time(cur_sec)} / {format_time(tot_sec)}   ({value} / {max(0, total-1)})"
        )

    def _on_slider_released(self):
        if not self.sld_playback.isEnabled():
            return
        value = int(self.sld_playback.value())
        self.compare_player.seek(value)
        if self._compare_was_running:
            self.compare_player.pause(False)
        self._slider_dragging = False

    def _on_compare_position(self, cur: int, total: int, fps: float):
        if total <= 0:
            self._set_compare_slider_enabled(False)
            return

        # slider range 설정은 start 시점에 한 번만 해도 되지만, 안전하게 유지
        if self.sld_playback.maximum() != total - 1:
            self.sld_playback.blockSignals(True)
            self.sld_playback.setRange(0, total - 1)
            self.sld_playback.blockSignals(False)

        # 드래그 중이면 자동 업데이트 금지
        if not self._slider_dragging:
            self.sld_playback.blockSignals(True)
            self.sld_playback.setValue(int(np.clip(cur, 0, total - 1)))
            self.sld_playback.blockSignals(False)

        fps = fps if fps > 0 else 30.0
        cur_sec = max(0, cur) / fps
        tot_sec = (total - 1) / fps if total > 1 else 0.0
        self.lbl_playback.setText(
            f"{format_time(cur_sec)} / {format_time(tot_sec)}   ({max(0, cur)} / {max(0, total-1)})"
        )

    # ---------- File ----------
    def _is_video(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"]

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image/video",
            "",
            "Media (*.mp4 *.avi *.mkv *.mov *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        self.stop_all()
        self._reset_preprocess_state()

        self.current_path = path
        self.is_video = self._is_video(path)

        self.btn_preview.setEnabled(True)
        self.btn_compare_play.setEnabled(False)
        self.btn_save_preprocessed.setEnabled(False)
        self.btn_preprocess_cancel.setEnabled(False)

        if self.is_video:
            self._load_video_first_frame(path)
            self.btn_preprocess.setEnabled(True)
        else:
            self._load_image(path)
            self.btn_preprocess.setEnabled(False)

        self._set_status(f"Loaded: {os.path.basename(path)}")

    def _load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Error", "이미지를 읽을 수 없습니다.")
            return
        self.orig_u8 = as_u8(ensure_gray(img))
        self._update_orig(self.orig_u8)

    def _load_video_first_frame(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "영상을 열 수 없습니다.")
            return
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.critical(self, "Error", "첫 프레임을 읽을 수 없습니다.")
            return
        self.orig_u8 = as_u8(ensure_gray(frame))
        self._update_orig(self.orig_u8)

    # ---------- Preview ----------
    def preview_one_frame(self):
        if self.orig_u8 is None:
            return
        self._pull_params_from_ui()
        proc = Processor()
        t0 = time.perf_counter()
        out = proc.apply(self.orig_u8, self.params)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.proc_u8 = out
        self._update_proc(out)
        self._set_status(f"Preview (1-frame): {dt_ms:.2f} ms")

    # ---------- Realtime ----------
    def play_realtime(self):
        if not self.is_video or not self.current_path:
            return

        # compare 중이면 stop
        self.compare_player.stop()
        self._set_compare_slider_enabled(False)

        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.set_pause(False)
            return

        self._pull_params_from_ui()
        self.video_worker = VideoWorker(self.current_path, self._get_params_snapshot)
        self.video_worker.frame_ready.connect(self._on_realtime_frame)
        self.video_worker.finished.connect(self._set_status)
        self.video_worker.start()
        self._set_status("Realtime play started.")

    def _on_realtime_frame(self, orig_u8: np.ndarray, proc_u8: np.ndarray, ms: float):
        self.orig_u8 = orig_u8
        self.proc_u8 = proc_u8
        self._update_orig(orig_u8)
        self._update_proc(proc_u8)
        self._set_status(f"Realtime proc: {ms:.2f} ms")

    # ---------- Preprocess: 항상 같은 캐시 파일에 덮어쓰기 ----------
    def preprocess_temp_cache(self):
        if not self.is_video or not self.current_path:
            return
        if self.prep_worker is not None and self.prep_worker.isRunning():
            QMessageBox.information(self, "Info", "전처리가 이미 진행 중입니다.")
            return

        # 재생/비교 중이면 중지
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(800)
            self.video_worker = None
        self.compare_player.stop()
        self._set_compare_slider_enabled(False)

        # ✅ 항상 같은 캐시 파일: 시작 전에 삭제(덮어쓰기 보장)
        safe_remove(self.cache_path)

        self._pull_params_from_ui()
        params_snapshot = self._get_params_snapshot()

        # UI 상태
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_preprocess.setEnabled(False)
        self.btn_preprocess_cancel.setEnabled(True)
        self.btn_compare_play.setEnabled(False)
        self.btn_save_preprocessed.setEnabled(False)

        self.prep_worker = PreprocessWorker(self.current_path, self.cache_path, params_snapshot)
        self.prep_worker.progress.connect(self.progress.setValue)
        self.prep_worker.done.connect(self._on_preprocess_done_cache)
        self.prep_worker.start()
        self._set_status("Start preprocessing...")

    def cancel_preprocess(self):
        if self.prep_worker is not None and self.prep_worker.isRunning():
            self.prep_worker.stop()
            self._set_status("Cancel preprocessing...")

    def _on_preprocess_done_cache(self, ok: bool, msg: str, processed_path: str):
        self.progress.setVisible(False)
        self._set_status(msg)

        # UI 복구
        self.btn_preprocess.setEnabled(True)
        self.btn_preprocess_cancel.setEnabled(False)

        if ok and os.path.exists(self.cache_path):
            self.preprocessed_path = self.cache_path
            self.btn_compare_play.setEnabled(True)
            self.btn_save_preprocessed.setEnabled(True)
        else:
            # 중단/실패 시 캐시 정리
            safe_remove(self.cache_path)
            self.preprocessed_path = None
            self.btn_compare_play.setEnabled(False)
            self.btn_save_preprocessed.setEnabled(False)

    def _reset_preprocess_state(self):
        # 캐시 파일은 “항상 하나”이므로, 새 파일 열 때는 상태만 초기화
        self.preprocessed_path = None
        self.btn_compare_play.setEnabled(False)
        self.btn_save_preprocessed.setEnabled(False)
        self.btn_preprocess_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self._set_compare_slider_enabled(False)

    # ---------- Compare Play ----------
    def compare_play_non_realtime(self):
        if not self.is_video or not self.current_path or not self.preprocessed_path:
            QMessageBox.information(self, "Info", "전처리 결과가 없습니다. Preprocess를 먼저 실행해 주세요.")
            return

        # 실시간 재생 중이면 중지
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(800)
            self.video_worker = None

        # 비교 재생 시작 (start에서 seek(0) + timer start)
        self.compare_player.start(self.current_path, self.preprocessed_path)

        # total_frames 기준으로 슬라이더 활성화
        if self.compare_player.total_frames > 0:
            self._set_compare_slider_enabled(True)
            self.sld_playback.blockSignals(True)
            self.sld_playback.setRange(0, self.compare_player.total_frames - 1)
            self.sld_playback.setValue(0)
            self.sld_playback.blockSignals(False)
        else:
            # frame_count를 못 얻는 코덱이면 슬라이더 비활성
            self._set_compare_slider_enabled(False)

    def _on_compare_frame(self, orig_u8: np.ndarray, proc_u8: np.ndarray):
        self.orig_u8 = orig_u8
        self.proc_u8 = proc_u8
        self._update_orig(orig_u8)
        self._update_proc(proc_u8)

    # ---------- Save cached result ----------
    def save_preprocessed_as(self):
        if not self.preprocessed_path or not os.path.exists(self.preprocessed_path):
            QMessageBox.information(self, "Info", "저장할 전처리 결과가 없습니다. Preprocess를 먼저 실행해 주세요.")
            return

        base = os.path.splitext(os.path.basename(self.current_path or "video"))[0]
        out_default = os.path.join(os.path.dirname(self.current_path or ""), f"{base}_processed.mp4")

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save preprocessed video as",
            out_default,
            "MP4 (*.mp4);;AVI (*.avi);;All files (*.*)",
        )
        if not out_path:
            return

        try:
            shutil.copy2(self.preprocessed_path, out_path)
            self._set_status(f"Saved: {out_path}")
            QMessageBox.information(self, "Saved", "저장이 완료되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"저장에 실패했습니다.\n{e}")

    # ---------- Pause/Stop ----------
    def pause_all(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.set_pause(True)
        self.compare_player.pause(True)

    def stop_all(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(800)
        self.video_worker = None

        if self.prep_worker is not None and self.prep_worker.isRunning():
            self.prep_worker.stop()
            self.prep_worker.wait(800)
        self.prep_worker = None

        self.compare_player.stop()
        self._set_compare_slider_enabled(False)

        self.btn_preprocess_cancel.setEnabled(False)
        if self.is_video and self.current_path:
            self.btn_preprocess.setEnabled(True)

    # ---------- Save image ----------
    def save_image_as(self):
        if self.proc_u8 is None:
            QMessageBox.information(self, "Info", "저장할 처리 결과가 없습니다. Preview 후 저장해 주세요.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save processed image", "", "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)"
        )
        if not out_path:
            return
        ok = cv2.imwrite(out_path, self.proc_u8)
        if ok:
            self._set_status(f"Saved: {out_path}")
        else:
            QMessageBox.critical(self, "Error", "저장에 실패했습니다.")

    # ---------- Display ----------
    def _fit_pixmap(self, label: QLabel, pix: QPixmap) -> QPixmap:
        if pix.isNull():
            return pix
        w = max(1, label.width() - 6)
        h = max(1, label.height() - 6)
        return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _update_orig(self, orig_u8: np.ndarray):
        pix = QPixmap.fromImage(to_qimage_gray8(orig_u8))
        self.lbl_orig.setPixmap(self._fit_pixmap(self.lbl_orig, pix))

    def _update_proc(self, proc_u8: np.ndarray):
        pix = QPixmap.fromImage(to_qimage_gray8(proc_u8))
        self.lbl_proc.setPixmap(self._fit_pixmap(self.lbl_proc, pix))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.orig_u8 is not None:
            self._update_orig(self.orig_u8)
        if self.proc_u8 is not None:
            self._update_proc(self.proc_u8)

    def _set_status(self, text: str):
        self.status.showMessage(text)

    def closeEvent(self, event):
        self.stop_all()
        # ✅ 앱 종료 시 캐시 파일 정리
        safe_remove(self.cache_path)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1320, 720)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

