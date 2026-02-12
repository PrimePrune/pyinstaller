"""
PySide6 + OpenCV IR(그레이스케일) 영상처리 GUI 템플릿
- 파일 열기(이미지/영상)
- 3가지 처리(Denoise / CLAHE / Sharpen) On/Off + 강도 조절
- 원본/처리 결과 탭 프리뷰
- 이미지 저장, 영상 Export(현재 설정으로 전체 저장)
- 영상 재생은 QThread로 UI 프리징 방지

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
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)


# -----------------------------
# Utils: ndarray -> QImage
# -----------------------------
def to_qimage_gray8(gray8: np.ndarray) -> QImage:
    """gray8: HxW uint8"""
    if gray8.dtype != np.uint8:
        gray8 = gray8.astype(np.uint8, copy=False)
    h, w = gray8.shape[:2]
    # bytesPerLine = w
    return QImage(gray8.data, w, h, w, QImage.Format_Grayscale8).copy()


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    """Force grayscale."""
    if frame is None:
        return frame
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return frame


def normalize_to_u8(gray: np.ndarray, mode: str = "minmax", p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """
    Display normalization for preview.
    - mode="minmax": min~max
    - mode="percentile": p_lo~p_hi
    Returns uint8.
    """
    g = gray.astype(np.float32, copy=False)

    if mode == "percentile":
        lo = np.percentile(g, p_lo)
        hi = np.percentile(g, p_hi)
    else:
        lo = float(np.min(g))
        hi = float(np.max(g))

    if hi <= lo + 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)

    x = (g - lo) * (255.0 / (hi - lo))
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


# -----------------------------
# Processing params + pipeline
# -----------------------------
@dataclass
class ProcParams:
    # Denoise
    denoise_on: bool = True
    denoise_mode: str = "Bilateral"  # Bilateral | NLMeans | Temporal(EMA)
    denoise_strength: int = 20       # 0~100 (mapped)

    # CLAHE
    clahe_on: bool = True
    clahe_clip: int = 20            # 1~100 -> 0.5~6.0
    clahe_tile: int = 8             # 4~16

    # Sharpen
    sharp_on: bool = True
    sharp_amount: int = 30          # 0~100 -> 0~2.0
    sharp_sigma: int = 10           # 1~30 -> 0.5~3.0

    # Preview normalization (IR often benefits)
    preview_norm: str = "percentile"  # minmax | percentile


class Processor:
    """
    Stateless-ish processor. For Temporal(EMA), you can feed prev state.
    """
    def __init__(self):
        self._ema_prev: Optional[np.ndarray] = None

    def reset_state(self):
        self._ema_prev = None

    def apply(self, gray: np.ndarray, p: ProcParams) -> np.ndarray:
        """
        gray: uint8/uint16/float. returns same dtype-ish (we keep float32 internally).
        """
        if gray is None:
            return gray

        x = gray.astype(np.float32, copy=False)

        # 1) Denoise
        if p.denoise_on:
            if p.denoise_mode == "Bilateral":
                # Map strength 0~100 -> sigmaColor 5~75, sigmaSpace 5~75
                s = float(np.interp(p.denoise_strength, [0, 100], [5, 75]))
                # bilateralFilter expects uint8/float, ok on float32
                x = cv2.bilateralFilter(x, d=0, sigmaColor=s, sigmaSpace=s)
            elif p.denoise_mode == "NLMeans":
                # NLMeans expects uint8; use preview-normalized uint8 then re-scale back (simple version)
                u8 = normalize_to_u8(x, mode="percentile")
                h = float(np.interp(p.denoise_strength, [0, 100], [3, 30]))
                u8 = cv2.fastNlMeansDenoising(u8, None, h=h, templateWindowSize=7, searchWindowSize=21)
                x = u8.astype(np.float32)
            elif p.denoise_mode == "Temporal(EMA)":
                # alpha: 0.05~0.6
                alpha = float(np.interp(p.denoise_strength, [0, 100], [0.05, 0.6]))
                if self._ema_prev is None:
                    self._ema_prev = x.copy()
                else:
                    self._ema_prev = alpha * x + (1.0 - alpha) * self._ema_prev
                x = self._ema_prev

        # 2) CLAHE
        if p.clahe_on:
            # CLAHE expects uint8. We'll map via display normalization then apply CLAHE.
            u8 = normalize_to_u8(x, mode="percentile")
            clip = float(np.interp(p.clahe_clip, [1, 100], [0.5, 6.0]))
            tile = int(np.clip(p.clahe_tile, 4, 16))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            u8 = clahe.apply(u8)
            x = u8.astype(np.float32)

        # 3) Sharpen (Unsharp mask)
        if p.sharp_on:
            amount = float(np.interp(p.sharp_amount, [0, 100], [0.0, 2.0]))
            sigma = float(np.interp(p.sharp_sigma, [1, 30], [0.5, 3.0]))
            blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
            x = cv2.addWeighted(x, 1.0 + amount, blur, -amount, 0)

        # keep as float32; caller will normalize for preview / convert for saving
        return x


# -----------------------------
# Video Player Worker (QThread)
# -----------------------------
class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray, np.ndarray, float)  # (orig_gray, proc_float, proc_ms)
    meta_ready = Signal(float, int, int)                 # (fps, w, h)
    finished = Signal(str)

    def __init__(self, path: str, processor: Processor, params_getter):
        super().__init__()
        self.path = path
        self.processor = processor
        self.params_getter = params_getter

        self._stop = False
        self._pause = False

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
        self.processor.reset_state()

        while not self._stop:
            if self._pause:
                self.msleep(30)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            gray = ensure_gray(frame)

            t0 = time.perf_counter()
            params = self.params_getter()
            proc = self.processor.apply(gray, params)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            self.frame_ready.emit(gray, proc, dt_ms)

            # crude timing
            sleep_ms = max(0, int((frame_period * 1000) - 1))
            self.msleep(sleep_ms)

        cap.release()
        self.finished.emit("재생 종료")


# -----------------------------
# Video Export Worker
# -----------------------------
class ExportWorker(QThread):
    progress = Signal(int)         # 0~100
    done = Signal(bool, str)       # success, message

    def __init__(self, in_path: str, out_path: str, processor: Processor, params: ProcParams):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.processor = processor
        self.params = params
        self._stop = False

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

        # Output codec: mp4v (works widely). If .avi -> XVID might be better.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.out_path, fourcc, float(fps), (w, h), isColor=False)
        if not writer.isOpened():
            cap.release()
            self.done.emit(False, "출력 VideoWriter를 열 수 없습니다. (코덱/경로 확인)")
            return

        self.processor.reset_state()
        idx = 0

        while not self._stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = ensure_gray(frame)

            proc = self.processor.apply(gray, self.params)
            # Export는 8-bit로 저장(간단 버전). 필요하면 16-bit/RAW 저장은 별도 설계.
            out_u8 = normalize_to_u8(proc, mode="percentile")
            writer.write(out_u8)

            idx += 1
            if total > 0:
                pct = int((idx / total) * 100)
                self.progress.emit(min(100, max(0, pct)))
            else:
                # unknown total
                if idx % 30 == 0:
                    self.progress.emit(min(99, idx % 100))

            if self._stop:
                break

        cap.release()
        writer.release()

        if self._stop:
            self.done.emit(False, "사용자에 의해 중단됨")
        else:
            self.progress.emit(100)
            self.done.emit(True, "저장 완료")


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR Processing GUI (PySide6)")

        self.params = ProcParams()
        self.processor = Processor()

        self.current_path: Optional[str] = None
        self.is_video: bool = False

        self.video_worker: Optional[VideoWorker] = None
        self.export_worker: Optional[ExportWorker] = None

        self.orig_gray: Optional[np.ndarray] = None
        self.proc_float: Optional[np.ndarray] = None

        self._build_ui()
        self._bind_actions()

        # 디바운스(슬라이더 드래그 중 과도한 처리 방지)
        self.apply_timer = QTimer(self)
        self.apply_timer.setSingleShot(True)
        self.apply_timer.timeout.connect(self.apply_once)

    # ---------- UI ----------
    def _build_ui(self):
        # Toolbar actions
        self.act_open = QAction("Open", self)
        self.act_save = QAction("Save As...", self)
        self.act_export = QAction("Export Video...", self)
        self.act_play = QAction("Play", self)
        self.act_pause = QAction("Pause", self)
        self.act_stop = QAction("Stop", self)

        tb = self.addToolBar("Main")
        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addSeparator()
        tb.addAction(self.act_play)
        tb.addAction(self.act_pause)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_export)

        # Left panel: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # Denoise group
        self.grp_denoise = QGroupBox("1) Denoise")
        gl = QVBoxLayout(self.grp_denoise)

        self.chk_denoise = QCheckBox("Enable")
        self.chk_denoise.setChecked(self.params.denoise_on)

        self.cmb_denoise = QComboBox()
        self.cmb_denoise.addItems(["Bilateral", "NLMeans", "Temporal(EMA)"])
        self.cmb_denoise.setCurrentText(self.params.denoise_mode)

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
        gl.addWidget(QLabel("Mode"))
        gl.addWidget(self.cmb_denoise)
        gl.addLayout(den_row)

        # CLAHE group
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

        # Sharpen group
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

        # Preview norm
        self.grp_prev = QGroupBox("Preview")
        pl = QVBoxLayout(self.grp_prev)
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["percentile", "minmax"])
        self.cmb_norm.setCurrentText(self.params.preview_norm)
        pl.addWidget(QLabel("Normalize (for display/export-u8)"))
        pl.addWidget(self.cmb_norm)

        # Buttons
        self.btn_apply = QPushButton("Apply (Image only)")
        self.btn_apply.setEnabled(False)

        left_layout.addWidget(self.grp_denoise)
        left_layout.addWidget(self.grp_clahe)
        left_layout.addWidget(self.grp_sharp)
        left_layout.addWidget(self.grp_prev)
        left_layout.addWidget(self.btn_apply)
        left_layout.addStretch(1)

        # Right panel: previews
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        self.lbl_proc = QLabel("Open an image/video…")
        self.lbl_orig = QLabel("")

        for lbl in (self.lbl_proc, self.lbl_orig):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(640, 360)
            lbl.setStyleSheet("QLabel { background: #111; color: #ddd; }")
            lbl.setScaledContents(False)

        proc_page = QWidget()
        proc_l = QVBoxLayout(proc_page)
        proc_l.addWidget(self.lbl_proc)

        orig_page = QWidget()
        orig_l = QVBoxLayout(orig_page)
        orig_l.addWidget(self.lbl_orig)

        self.tabs.addTab(proc_page, "Processed")
        self.tabs.addTab(orig_page, "Original")

        right_layout.addWidget(self.tabs)

        # Main layout
        central = QWidget()
        main = QHBoxLayout(central)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(10)
        main.addWidget(left, 0)
        main.addWidget(right, 1)
        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.status.addPermanentWidget(self.progress)

    def _bind_actions(self):
        self.act_open.triggered.connect(self.open_file)
        self.act_save.triggered.connect(self.save_as)
        self.act_export.triggered.connect(self.export_video)

        self.act_play.triggered.connect(self.play_video)
        self.act_pause.triggered.connect(self.pause_video)
        self.act_stop.triggered.connect(self.stop_video)

        # control bindings (slider <-> spin)
        def bind_slider_spin(slider: QSlider, spin: QSpinBox):
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)

        bind_slider_spin(self.sld_denoise, self.spn_denoise)
        bind_slider_spin(self.sld_clip, self.spn_clip)
        bind_slider_spin(self.sld_tile, self.spn_tile)
        bind_slider_spin(self.sld_amount, self.spn_amount)
        bind_slider_spin(self.sld_sigma, self.spn_sigma)

        # any param change -> schedule apply (image) or just affect video worker processing
        def schedule():
            self._pull_params_from_ui()
            if not self.is_video:
                self.apply_timer.start(80)  # debounce

        for w in [
            self.chk_denoise, self.cmb_denoise, self.sld_denoise,
            self.chk_clahe, self.sld_clip, self.sld_tile,
            self.chk_sharp, self.sld_amount, self.sld_sigma,
            self.cmb_norm
        ]:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(schedule)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(schedule)
            else:
                w.valueChanged.connect(schedule)

        self.btn_apply.clicked.connect(self.apply_once)

    # ---------- Params ----------
    def _pull_params_from_ui(self):
        self.params.denoise_on = self.chk_denoise.isChecked()
        self.params.denoise_mode = self.cmb_denoise.currentText()
        self.params.denoise_strength = int(self.sld_denoise.value())

        self.params.clahe_on = self.chk_clahe.isChecked()
        self.params.clahe_clip = int(self.sld_clip.value())
        self.params.clahe_tile = int(self.sld_tile.value())

        self.params.sharp_on = self.chk_sharp.isChecked()
        self.params.sharp_amount = int(self.sld_amount.value())
        self.params.sharp_sigma = int(self.sld_sigma.value())

        self.params.preview_norm = self.cmb_norm.currentText()

    def _get_params_snapshot(self) -> ProcParams:
        # dataclass copy (cheap)
        return ProcParams(**self.params.__dict__)

    # ---------- File open ----------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image/video",
            "",
            "Media (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.avi *.mkv *.mov);;All files (*.*)",
        )
        if not path:
            return

        self.stop_video()
        self.current_path = path
        self.is_video = self._is_video_file(path)

        self.status.showMessage(f"Opened: {os.path.basename(path)}")

        if self.is_video:
            self.btn_apply.setEnabled(False)
            self.start_video_worker(path)
        else:
            self.btn_apply.setEnabled(True)
            self.load_image(path)
            self.apply_once()

    def _is_video_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"]

    def load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Error", "이미지를 읽을 수 없습니다.")
            return

        gray = ensure_gray(img)
        self.orig_gray = gray
        self.processor.reset_state()

        self._update_preview_labels(gray, None, proc_ms=None)

    # ---------- Apply (image) ----------
    def apply_once(self):
        if self.is_video:
            return
        if self.orig_gray is None:
            return

        self._pull_params_from_ui()

        t0 = time.perf_counter()
        proc = self.processor.apply(self.orig_gray, self.params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.proc_float = proc
        self._update_preview_labels(self.orig_gray, proc, proc_ms=dt_ms)

    # ---------- Preview update ----------
    def _fit_pixmap(self, label: QLabel, pix: QPixmap) -> QPixmap:
        if pix.isNull():
            return pix
        w = max(1, label.width() - 4)
        h = max(1, label.height() - 4)
        return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _update_preview_labels(self, orig_gray: Optional[np.ndarray], proc_float: Optional[np.ndarray], proc_ms: Optional[float]):
        if orig_gray is not None:
            u8o = normalize_to_u8(orig_gray, mode=self.params.preview_norm)
            qio = to_qimage_gray8(u8o)
            pixo = QPixmap.fromImage(qio)
            self.lbl_orig.setPixmap(self._fit_pixmap(self.lbl_orig, pixo))

        if proc_float is not None:
            u8p = normalize_to_u8(proc_float, mode=self.params.preview_norm)
            qip = to_qimage_gray8(u8p)
            pixp = QPixmap.fromImage(qip)
            self.lbl_proc.setPixmap(self._fit_pixmap(self.lbl_proc, pixp))

        if proc_ms is not None:
            self.status.showMessage(f"Proc: {proc_ms:.2f} ms   |   {os.path.basename(self.current_path or '')}")

    def resizeEvent(self, event):
        # refit current pixmaps on resize
        super().resizeEvent(event)
        if self.orig_gray is not None:
            u8o = normalize_to_u8(self.orig_gray, mode=self.params.preview_norm)
            self.lbl_orig.setPixmap(self._fit_pixmap(self.lbl_orig, QPixmap.fromImage(to_qimage_gray8(u8o))))
        if self.proc_float is not None:
            u8p = normalize_to_u8(self.proc_float, mode=self.params.preview_norm)
            self.lbl_proc.setPixmap(self._fit_pixmap(self.lbl_proc, QPixmap.fromImage(to_qimage_gray8(u8p))))

    # ---------- Video controls ----------
    def start_video_worker(self, path: str):
        self.video_worker = VideoWorker(path, self.processor, self._get_params_snapshot)
        self.video_worker.meta_ready.connect(self._on_video_meta)
        self.video_worker.frame_ready.connect(self._on_video_frame)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.start()

    def _on_video_meta(self, fps: float, w: int, h: int):
        self.status.showMessage(f"Video: {os.path.basename(self.current_path or '')}  |  {w}x{h} @ {fps:.2f}fps")

    def _on_video_frame(self, orig_gray: np.ndarray, proc_float: np.ndarray, proc_ms: float):
        self.orig_gray = orig_gray
        self.proc_float = proc_float
        self._update_preview_labels(orig_gray, proc_float, proc_ms=proc_ms)

    def _on_video_finished(self, msg: str):
        self.status.showMessage(msg)

    def play_video(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.set_pause(False)

    def pause_video(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.set_pause(True)

    def stop_video(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(800)
        self.video_worker = None

    # ---------- Save ----------
    def save_as(self):
        if self.proc_float is None:
            QMessageBox.information(self, "Info", "저장할 결과가 없습니다.")
            return

        # default extension: png
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save output image",
            "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not out_path:
            return

        out_u8 = normalize_to_u8(self.proc_float, mode=self.params.preview_norm)
        ok = cv2.imwrite(out_path, out_u8)
        if ok:
            self.status.showMessage(f"Saved: {out_path}")
        else:
            QMessageBox.critical(self, "Error", "저장 실패")

    def export_video(self):
        if not self.is_video or not self.current_path:
            QMessageBox.information(self, "Info", "영상 파일을 먼저 열어주세요.")
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

        # Export 동안 재생을 멈추는 게 안정적(간단/실전)
        self.pause_video()

        self._pull_params_from_ui()
        params_snapshot = self._get_params_snapshot()

        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.export_worker = ExportWorker(self.current_path, out_path, Processor(), params_snapshot)
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
    w.resize(1100, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
