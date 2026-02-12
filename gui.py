"""
PySide6 + OpenCV IR 영상처리 GUI (요구사항 반영 최종)
- 입력: mp4/avi 등 일반 영상 포맷(이미지도 지원)
- 화면: 좌측 원본 / 우측 처리 결과
- 영상: Preview(1-Frame)로 1프레임 처리 미리보기 + Play로 실시간 처리
- 이미지: Preview(1-Frame)로 처리 결과 갱신
- 처리: Bilateral Denoise + CLAHE + Unsharp Sharpen
- Normalize 옵션 제거 (입력이 이미 8-bit 영상인 전제)
- 툴팁 문장/말투/형식 통일

설치:
  pip install pyside6 opencv-python numpy

실행:
  python ir_gui.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
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
    """gray8: HxW uint8"""
    if gray8.dtype != np.uint8:
        gray8 = gray8.astype(np.uint8, copy=False)
    h, w = gray8.shape[:2]
    return QImage(gray8.data, w, h, w, QImage.Format_Grayscale8).copy()


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    """입력이 컬러로 들어와도 그레이로 변환합니다."""
    if frame is None:
        return frame
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return frame


def as_u8_for_display(gray: np.ndarray) -> np.ndarray:
    """
    일반 mp4/avi 입력(대개 uint8)을 전제로 표시용 변환만 합니다.
    - uint8이면 그대로 사용
    - 그 외 타입이면 안전하게 clip 후 uint8로 변환
    """
    if gray is None:
        return gray
    if gray.dtype == np.uint8:
        return gray
    x = np.clip(gray, 0, 255).astype(np.uint8)
    return x


# -----------------------------
# Processing
# -----------------------------
@dataclass
class ProcParams:
    # Bilateral Denoise
    denoise_on: bool = True
    denoise_strength: int = 20  # 0~100 -> sigma 5~75

    # CLAHE
    clahe_on: bool = True
    clahe_clip: int = 20        # 1~100 -> 0.5~6.0
    clahe_tile: int = 8         # 4~16

    # Sharpen (Unsharp)
    sharp_on: bool = True
    sharp_amount: int = 30      # 0~100 -> 0~2.0
    sharp_sigma: int = 10       # 1~30 -> 0.5~3.0


class Processor:
    """상태 없는(Stateless) 처리 파이프라인입니다."""

    def apply(self, gray_u8: np.ndarray, p: ProcParams) -> np.ndarray:
        """
        입력/출력: uint8 그레이 이미지
        """
        if gray_u8 is None:
            return gray_u8

        # 내부 계산은 float32로 하고 마지막에 uint8로 돌아옵니다.
        x = gray_u8.astype(np.float32, copy=False)

        # 1) Bilateral Denoise
        if p.denoise_on:
            sigma = float(np.interp(p.denoise_strength, [0, 100], [5, 75]))
            # d=0이면 sigma 기반으로 자동 결정
            x = cv2.bilateralFilter(x, d=0, sigmaColor=sigma, sigmaSpace=sigma)

        # 2) CLAHE (uint8 기반이 가장 안정적)
        if p.clahe_on:
            tmp_u8 = np.clip(x, 0, 255).astype(np.uint8)
            clip = float(np.interp(p.clahe_clip, [1, 100], [0.5, 6.0]))
            tile = int(np.clip(p.clahe_tile, 4, 16))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            tmp_u8 = clahe.apply(tmp_u8)
            x = tmp_u8.astype(np.float32)

        # 3) Sharpen (Unsharp)
        if p.sharp_on:
            amount = float(np.interp(p.sharp_amount, [0, 100], [0.0, 2.0]))
            sigma = float(np.interp(p.sharp_sigma, [1, 30], [0.5, 3.0]))
            blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
            x = cv2.addWeighted(x, 1.0 + amount, blur, -amount, 0)

        return np.clip(x, 0, 255).astype(np.uint8)


# -----------------------------
# Video Worker
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

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-6:
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

            gray = ensure_gray(frame)
            orig_u8 = as_u8_for_display(gray)

            t0 = time.perf_counter()
            params = self.params_getter()
            proc_u8 = self.processor.apply(orig_u8, params)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            self.frame_ready.emit(orig_u8, proc_u8, dt_ms)

            sleep_ms = max(0, int(frame_period * 1000) - 1)
            self.msleep(sleep_ms)

        cap.release()
        self.finished.emit("재생 종료")


# -----------------------------
# Video Export Worker
# -----------------------------
class ExportWorker(QThread):
    progress = Signal(int)
    done = Signal(bool, str)

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
            self.done.emit(False, "입력 영상을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-6:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.out_path, fourcc, float(fps), (w, h), isColor=False)
        if not writer.isOpened():
            cap.release()
            self.done.emit(False, "출력 VideoWriter를 열 수 없습니다. 코덱 또는 경로를 확인해 주세요.")
            return

        idx = 0
        while not self._stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            gray = ensure_gray(frame)
            orig_u8 = as_u8_for_display(gray)
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
            self.done.emit(False, "사용자에 의해 중단되었습니다.")
        else:
            self.progress.emit(100)
            self.done.emit(True, "저장이 완료되었습니다.")


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR Processing GUI (PySide6)")

        self.params = ProcParams()
        self.current_path: Optional[str] = None
        self.is_video: bool = False

        self.video_worker: Optional[VideoWorker] = None
        self.export_worker: Optional[ExportWorker] = None

        self.orig_u8: Optional[np.ndarray] = None
        self.proc_u8: Optional[np.ndarray] = None

        self._build_ui()
        self._bind_actions()
        self._apply_tooltips()

    # ---------- UI ----------
    def _build_ui(self):
        self.act_open = QAction("Open", self)
        self.act_save_img = QAction("Save Image As...", self)
        self.act_export_vid = QAction("Export Video...", self)
        self.act_play = QAction("Play", self)
        self.act_pause = QAction("Pause", self)
        self.act_stop = QAction("Stop", self)

        tb = self.addToolBar("Main")
        tb.addAction(self.act_open)
        tb.addAction(self.act_save_img)
        tb.addAction(self.act_export_vid)
        tb.addSeparator()
        tb.addAction(self.act_play)
        tb.addAction(self.act_pause)
        tb.addAction(self.act_stop)

        # Controls (left)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # Denoise (Bilateral only)
        self.grp_denoise = QGroupBox("1) Denoise (Bilateral)")
        gl = QVBoxLayout(self.grp_denoise)

        self.chk_denoise = QCheckBox("Enable")
        self.chk_denoise.setChecked(self.params.denoise_on)

        den_row = QHBoxLayout()
        den_row.addWidget(QLabel("Strength"))
        self.sld_denoise = QSlider(Qt.Horizontal)
        self.sld_denoise.setRange(0, 100)
        self.sld_denoise.setValue(self.params.denoise_strength)
        self.spn_denoise = QSpinBox()
        self.spn_denoise.setRange(0, 100)
        self.spn_denoise.setValue(self.params.denoise_strength)
        den_row.addWidget(self.sld_denoise)
        den_row.addWidget(self.spn_denoise)

        gl.addWidget(self.chk_denoise)
        gl.addLayout(den_row)

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

        # Preview button
        self.btn_preview = QPushButton("Preview (1-Frame)")
        self.btn_preview.setEnabled(False)

        left_layout.addWidget(self.grp_denoise)
        left_layout.addWidget(self.grp_clahe)
        left_layout.addWidget(self.grp_sharp)
        left_layout.addWidget(self.btn_preview)
        left_layout.addStretch(1)

        # Side-by-side previews
        self.lbl_orig = QLabel("Left: Original")
        self.lbl_proc = QLabel("Right: Processed")
        for lbl in (self.lbl_orig, self.lbl_proc):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(480, 360)
            lbl.setStyleSheet("QLabel { background: #111; color: #ddd; }")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._wrap_panel("Original", self.lbl_orig))
        splitter.addWidget(self._wrap_panel("Processed", self.lbl_proc))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        main = QHBoxLayout(central)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(10)
        main.addWidget(left, 0)
        main.addWidget(splitter, 1)
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

    def _bind_actions(self):
        self.act_open.triggered.connect(self.open_file)
        self.act_save_img.triggered.connect(self.save_image_as)
        self.act_export_vid.triggered.connect(self.export_video)
        self.act_play.triggered.connect(self.play_video)
        self.act_pause.triggered.connect(self.pause_video)
        self.act_stop.triggered.connect(self.stop_video)

        # slider <-> spin
        def bind_slider_spin(slider: QSlider, spin: QSpinBox):
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)

        bind_slider_spin(self.sld_denoise, self.spn_denoise)
        bind_slider_spin(self.sld_clip, self.spn_clip)
        bind_slider_spin(self.sld_tile, self.spn_tile)
        bind_slider_spin(self.sld_amount, self.spn_amount)
        bind_slider_spin(self.sld_sigma, self.spn_sigma)

        # 파라미터 변경 시: 내부 파라미터만 갱신합니다.
        def on_param_changed():
            self._pull_params_from_ui()

        for w in [
            self.chk_denoise, self.sld_denoise,
            self.chk_clahe, self.sld_clip, self.sld_tile,
            self.chk_sharp, self.sld_amount, self.sld_sigma,
        ]:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(on_param_changed)
            else:
                w.valueChanged.connect(on_param_changed)

        self.btn_preview.clicked.connect(self.preview_one_frame)

    def _apply_tooltips(self):
        # Tooltip 스타일 통일: "~합니다/~해 주세요" 형태로 맞춤
        self.btn_preview.setToolTip(
            "현재 프레임 1장에 파라미터를 적용하여 우측 결과를 갱신합니다.\n"
            "Play를 누르기 전에 결과를 빠르게 확인할 때 사용해 주세요."
        )

        self.grp_denoise.setToolTip("Bilateral 필터로 노이즈를 완화합니다. 경계는 비교적 보존됩니다.")
        self.chk_denoise.setToolTip("노이즈 제거 기능을 켜거나 끕니다.")
        self.sld_denoise.setToolTip(
            "노이즈 제거 강도를 조절합니다.\n"
            "값이 클수록 더 부드러워지지만 세부 디테일이 줄어들 수 있습니다."
        )
        self.spn_denoise.setToolTip(self.sld_denoise.toolTip())

        self.grp_clahe.setToolTip("CLAHE로 국부 대비를 향상하여 디테일을 강조합니다.")
        self.chk_clahe.setToolTip("대비 향상 기능을 켜거나 끕니다.")
        self.sld_clip.setToolTip(
            "clipLimit을 조절합니다.\n"
            "값이 클수록 대비가 증가하지만 노이즈가 함께 강조될 수 있습니다."
        )
        self.spn_clip.setToolTip(self.sld_clip.toolTip())
        self.sld_tile.setToolTip(
            "tileGridSize(블록 크기)를 조절합니다.\n"
            "값이 작을수록 더 국부적으로 동작하지만 인공적인 패턴이 생길 수 있습니다."
        )
        self.spn_tile.setToolTip(self.sld_tile.toolTip())

        self.grp_sharp.setToolTip("Unsharp Mask로 선명도를 강화합니다. 노이즈도 함께 커질 수 있습니다.")
        self.chk_sharp.setToolTip("샤프닝 기능을 켜거나 끕니다.")
        self.sld_amount.setToolTip(
            "샤프닝 강도를 조절합니다.\n"
            "값이 클수록 경계가 더 강조됩니다."
        )
        self.spn_amount.setToolTip(self.sld_amount.toolTip())
        self.sld_sigma.setToolTip(
            "블러 반경(sigma)을 조절합니다.\n"
            "값이 작으면 미세한 경계를, 값이 크면 큰 구조를 더 강조합니다."
        )
        self.spn_sigma.setToolTip(self.sld_sigma.toolTip())

    # ---------- Params ----------
    def _pull_params_from_ui(self):
        self.params.denoise_on = self.chk_denoise.isChecked()
        self.params.denoise_strength = int(self.sld_denoise.value())

        self.params.clahe_on = self.chk_clahe.isChecked()
        self.params.clahe_clip = int(self.sld_clip.value())
        self.params.clahe_tile = int(self.sld_tile.value())

        self.params.sharp_on = self.chk_sharp.isChecked()
        self.params.sharp_amount = int(self.sld_amount.value())
        self.params.sharp_sigma = int(self.sld_sigma.value())

    def _get_params_snapshot(self) -> ProcParams:
        return ProcParams(**self.params.__dict__)

    # ---------- Open ----------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image/video",
            "",
            "Media (*.mp4 *.avi *.mkv *.mov *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        self.stop_video()
        self.current_path = path
        self.is_video = self._is_video_file(path)

        self.orig_u8 = None
        self.proc_u8 = None
        self.lbl_proc.setText("Right: Processed")
        self.btn_preview.setEnabled(True)

        if self.is_video:
            self.load_video_first_frame(path)
            self.status.showMessage(f"Video loaded: {os.path.basename(path)} (Preview 또는 Play를 눌러 주세요.)")
        else:
            self.load_image(path)
            self.status.showMessage(f"Image loaded: {os.path.basename(path)} (Preview를 눌러 주세요.)")

    def _is_video_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"]

    def load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Error", "이미지를 읽을 수 없습니다.")
            return
        gray = ensure_gray(img)
        self.orig_u8 = as_u8_for_display(gray)
        self._update_orig(self.orig_u8)

    def load_video_first_frame(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "영상을 열 수 없습니다.")
            return
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.critical(self, "Error", "첫 프레임을 읽을 수 없습니다.")
            return
        gray = ensure_gray(frame)
        self.orig_u8 = as_u8_for_display(gray)
        self._update_orig(self.orig_u8)

    # ---------- Preview ----------
    def preview_one_frame(self):
        if self.orig_u8 is None:
            return

        self._pull_params_from_ui()

        proc = Processor()
        t0 = time.perf_counter()
        out_u8 = proc.apply(self.orig_u8, self.params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.proc_u8 = out_u8
        self._update_proc(out_u8)
        self.status.showMessage(f"Preview (1-frame): {dt_ms:.2f} ms   |   {os.path.basename(self.current_path or '')}")

    # ---------- Video controls ----------
    def play_video(self):
        if not self.is_video or not self.current_path:
            return

        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.set_pause(False)
            return

        self._pull_params_from_ui()
        self.video_worker = VideoWorker(self.current_path, self._get_params_snapshot)
        self.video_worker.meta_ready.connect(self._on_video_meta)
        self.video_worker.frame_ready.connect(self._on_video_frame)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.start()
        self.status.showMessage("Playing...")

    def pause_video(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.set_pause(True)
            self.status.showMessage("Paused")

    def stop_video(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(800)
        self.video_worker = None

    def _on_video_meta(self, fps: float, w: int, h: int):
        self.status.showMessage(f"Video: {os.path.basename(self.current_path or '')}  |  {w}x{h} @ {fps:.2f}fps")

    def _on_video_frame(self, orig_u8: np.ndarray, proc_u8: np.ndarray, proc_ms: float):
        self.orig_u8 = orig_u8
        self.proc_u8 = proc_u8
        self._update_orig(orig_u8)
        self._update_proc(proc_u8)
        self.status.showMessage(f"Proc: {proc_ms:.2f} ms   |   {os.path.basename(self.current_path or '')}")

    def _on_video_finished(self, msg: str):
        self.status.showMessage(msg)

    # ---------- Preview update ----------
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

    # ---------- Save / Export ----------
    def save_image_as(self):
        if self.proc_u8 is None:
            QMessageBox.information(self, "Info", "저장할 처리 결과가 없습니다. Preview 후 저장해 주세요.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed image",
            "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not out_path:
            return

        ok = cv2.imwrite(out_path, self.proc_u8)
        if ok:
            self.status.showMessage(f"Saved: {out_path}")
        else:
            QMessageBox.critical(self, "Error", "저장에 실패했습니다.")

    def export_video(self):
        if not self.is_video or not self.current_path:
            QMessageBox.information(self, "Info", "영상 파일을 먼저 열어 주세요.")
            return
        if self.export_worker is not None and self.export_worker.isRunning():
            QMessageBox.information(self, "Info", "이미 Export 중입니다.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export processed video",
            "",
            "MP4 (*.mp4);;AVI (*.avi);;All files (*.*)",
        )
        if not out_path:
            return

        self._pull_params_from_ui()
        params_snapshot = self._get_params_snapshot()

        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.export_worker = ExportWorker(self.current_path, out_path, params_snapshot)
        self.export_worker.progress.connect(self.progress.setValue)
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.start()
        self.status.showMessage("Exporting...")

    def _on_export_done(self, ok: bool, msg: str):
        self.progress.setVisible(False)
        self.status.showMessage(msg)
        if not ok:
            QMessageBox.warning(self, "Export", msg)

    def closeEvent(self, event):
        try:
            self.stop_video()
            if self.export_worker is not None and self.export_worker.isRunning():
                self.export_worker.stop()
                self.export_worker.wait(800)
        finally:
            super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1250, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

