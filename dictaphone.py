"""GUI диктофон: записывает с микрофона, выбирает формат и сохраняет аудио."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from tkinter import BOTH, DISABLED, NORMAL, Button, Entry, Frame, Label, OptionMenu, StringVar, Tk, filedialog

import lameenc
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 44_100
CHANNELS = 1
DTYPE = "int16"
# If folders are empty, numbering starts at 1
DEFAULT_START = 1
SUPPORTED_FORMATS = ("mp3", "wav", "flac", "ogg")
SCAN_DIR = Path(r"C:\Users\Иван\Desktop\хрустел")
OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"
BG = "#0b1220"
CARD_BG = "#111c2f"
TEXT = "#e8edf5"
MUTED = "#8aa0bf"
ACCENT = "#5cd4c4"
INPUT_BG = "#0f2438"
INPUT_BORDER = "#1f3651"
FONT = ("Segoe UI", 10)


class DictaphoneApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Диктофон")
        self.root.configure(bg=BG)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.scan_dir = SCAN_DIR
        self.output_dir = OUTPUT_DIR
        self.scan_dir_var = StringVar(value=str(self.scan_dir))
        self.output_dir_var = StringVar(value=str(self.output_dir))

        self.format_var = StringVar(value=SUPPORTED_FORMATS[0])
        self.last_auto_name = self._default_name()  # base name without extension
        self.filename_var = StringVar(value=self.last_auto_name)
        self.status_var = StringVar(value="Готов")
        self._updating_name = False

        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.recording = False
        self.start_time: float | None = None
        self.timer_job: str | None = None
        self.scan_job: str | None = None

        self._build_ui()
        self._center_window()
        self._schedule_scan()
        self.filename_var.trace_add("write", self._on_name_change)

    def _build_ui(self) -> None:
        wrapper = Frame(self.root, bg=BG, padx=14, pady=14)
        wrapper.pack(fill=BOTH)

        card = Frame(wrapper, bg=CARD_BG, padx=14, pady=14)
        card.pack(fill=BOTH)

        Label(card, text="Диктофон", bg=CARD_BG, fg=TEXT, font=("Segoe UI Semibold", 14)).pack(anchor="w", pady=(0, 10))

        # Директория сканирования
        scan_frame = Frame(card, bg=CARD_BG)
        scan_frame.pack(fill="x", pady=(0, 8))
        Label(scan_frame, text="Папка для нумерации:", bg=CARD_BG, fg=TEXT, font=FONT).pack(anchor="w")
        scan_row = Frame(scan_frame, bg=CARD_BG)
        scan_row.pack(fill="x", pady=(2, 0))
        self.scan_entry = Entry(scan_row, textvariable=self.scan_dir_var, bg=INPUT_BG, fg=TEXT, relief="flat", font=FONT, highlightthickness=1, highlightbackground=INPUT_BORDER, highlightcolor=ACCENT)
        self.scan_entry.pack(side="left", fill="x", expand=True)
        Button(scan_row, text="Выбрать", command=self._choose_scan_dir, bg=ACCENT, fg=BG, relief="flat", bd=0, width=10, font=FONT, activebackground="#6fe2d6", activeforeground=BG).pack(side="left", padx=(8, 0))

        # Директория сохранения
        out_frame = Frame(card, bg=CARD_BG)
        out_frame.pack(fill="x", pady=(0, 10))
        Label(out_frame, text="Папка сохранения:", bg=CARD_BG, fg=TEXT, font=FONT).pack(anchor="w")
        out_row = Frame(out_frame, bg=CARD_BG)
        out_row.pack(fill="x", pady=(2, 0))
        self.out_entry = Entry(out_row, textvariable=self.output_dir_var, bg=INPUT_BG, fg=TEXT, relief="flat", font=FONT, highlightthickness=1, highlightbackground=INPUT_BORDER, highlightcolor=ACCENT)
        self.out_entry.pack(side="left", fill="x", expand=True)
        Button(out_row, text="Выбрать", command=self._choose_output_dir, bg=ACCENT, fg=BG, relief="flat", bd=0, width=10, font=FONT, activebackground="#6fe2d6", activeforeground=BG).pack(side="left", padx=(8, 0))

        Label(card, text="Имя файла:", bg=CARD_BG, fg=TEXT, font=FONT).pack(anchor="w")
        self.filename_entry = Entry(
            card,
            textvariable=self.filename_var,
            width=40,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=ACCENT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=INPUT_BORDER,
            highlightcolor=ACCENT,
            font=FONT,
        )
        self.filename_entry.pack(fill="x", pady=(0, 10))

        fmt_frame = Frame(card, bg=CARD_BG)
        fmt_frame.pack(fill="x", pady=(0, 10))
        Label(fmt_frame, text="Формат:", bg=CARD_BG, fg=MUTED, font=FONT).pack(side="left")
        self.format_menu = OptionMenu(fmt_frame, self.format_var, *SUPPORTED_FORMATS, command=self._on_format_change)
        self.format_menu.config(
            bg=INPUT_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground=BG,
            highlightthickness=0,
            relief="flat",
            font=FONT,
        )
        self.format_menu["menu"].config(bg=INPUT_BG, fg=TEXT, activebackground=ACCENT, activeforeground=BG, font=FONT)
        self.format_menu.pack(side="left", padx=(8, 0))

        btn_frame = Frame(card, bg=CARD_BG)
        btn_frame.pack(fill="x", pady=(0, 10))

        btn_opts = dict(
            width=12,
            relief="flat",
            bd=0,
            font=FONT,
            activeforeground=BG,
        )
        self.start_btn = Button(btn_frame, text="Запись", command=self.start_recording, bg=ACCENT, fg=BG, activebackground="#6fe2d6", **btn_opts)
        self.start_btn.pack(side="left")

        self.stop_btn = Button(btn_frame, text="Стоп", command=self.stop_recording, state=DISABLED, bg="#25344a", fg=MUTED, activebackground="#2f496b", **btn_opts)
        self.stop_btn.pack(side="left", padx=6)

        self.save_btn = Button(btn_frame, text="Сохранить", command=self.save_audio, state=DISABLED, bg="#25344a", fg=MUTED, activebackground="#2f496b", **btn_opts)
        self.save_btn.pack(side="left")

        self.status_label = Label(card, textvariable=self.status_var, anchor="w", bg=CARD_BG, fg=MUTED, font=FONT)
        self.status_label.pack(fill="x")

    def start_recording(self) -> None:
        if self.stream is not None:
            return
        self.frames.clear()
        self.recording = True
        self.start_time = time.time()
        self.status_var.set("Идёт запись... Остановить (00:00)")
        self._set_buttons(recording=True)
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._audio_callback,
            )
            self.stream.start()
            self._tick_timer()
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Ошибка аудио: {exc}")
            self._set_buttons(recording=False)
            self.stream = None
            self.recording = False
            self.start_time = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"Предупреждение аудиоустройства: {status}", file=sys.stderr)
        self.frames.append(indata.copy())

    def stop_recording(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None
        self.recording = False
        self.start_time = None
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        if not self.frames:
            self.status_var.set("Звук не записан. Попробуйте снова.")
            self._set_buttons(recording=False, has_audio=False)
            return
        self.status_var.set("Запись остановлена. Можно сохранять.")
        self._set_buttons(recording=False, has_audio=True)

    def save_audio(self) -> None:
        if not self.frames:
            self.status_var.set("Нечего сохранять — сделайте запись.")
            return
        target = self._make_target_path()
        self.status_var.set(f"Сохраняю в {target.name} ...")
        self._set_buttons(enabled=False)
        threading.Thread(target=self._save_worker, args=(target,), daemon=True).start()

    def _save_worker(self, target: Path) -> None:
        audio = np.concatenate(self.frames, axis=0)
        ext = target.suffix.lstrip(".").lower() or self._ext()
        try:
            if ext == "mp3":
                encoder = lameenc.Encoder()
                encoder.set_bit_rate(128)
                encoder.set_in_sample_rate(SAMPLE_RATE)
                encoder.set_channels(CHANNELS)
                encoder.set_quality(2)
                mp3_data = encoder.encode(audio.tobytes()) + encoder.flush()
                target.write_bytes(mp3_data)
            else:
                sf.write(target, audio, SAMPLE_RATE, subtype="PCM_16")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self.status_var.set, f"Ошибка сохранения: {exc}")
        else:
            self.root.after(0, self.status_var.set, f"Сохранено: {target.name}")
            self.frames.clear()
            next_name = self._default_name()
            self.last_auto_name = next_name
            self.root.after(0, self._set_filename, next_name)
        finally:
            self.root.after(0, self._set_buttons, False, False, True)

    def _set_buttons(self, recording: bool = False, has_audio: bool | None = None, enabled: bool = True) -> None:
        if not enabled:
            self.start_btn.config(state=DISABLED)
            self.stop_btn.config(state=DISABLED)
            self.save_btn.config(state=DISABLED)
            return
        self.start_btn.config(state=DISABLED if recording else NORMAL, bg=ACCENT if not recording else "#1f9f91", fg=BG if not recording else BG)
        self.stop_btn.config(state=NORMAL if recording else DISABLED, bg="#f25f5c" if recording else "#25344a", fg=BG if recording else MUTED)
        if has_audio is None:
            has_audio = bool(self.frames)
        self.save_btn.config(state=NORMAL if has_audio and not recording else DISABLED, bg=ACCENT if (has_audio and not recording) else "#25344a", fg=BG if (has_audio and not recording) else MUTED)

    def _make_target_path(self) -> Path:
        base_dir = self.output_dir
        ext = self._ext()
        base_name = self._safe_name(self.filename_var.get()) or self._default_name()
        candidate = base_dir / f"{base_name}.{ext}"
        counter = 1
        while candidate.exists():
            candidate = base_dir / f"{candidate.stem}_{counter}.{ext}"
            counter += 1
        return candidate

    def _default_name(self) -> str:
        max_num = DEFAULT_START - 1
        if self.scan_dir.exists():
            for fmt in SUPPORTED_FORMATS:
                for path in self.scan_dir.glob(f"*.{fmt}"):
                    stem = path.stem
                    if stem.isdigit():
                        try:
                            max_num = max(max_num, int(stem))
                        except ValueError:
                            continue
        return f"{max_num + 1}"

    def _ext(self) -> str:
        ext = (self.format_var.get() or SUPPORTED_FORMATS[0]).lower()
        if ext not in SUPPORTED_FORMATS:
            ext = SUPPORTED_FORMATS[0]
        return ext

    @staticmethod
    def _safe_name(raw: str) -> str:
        allowed = []
        for ch in raw:
            if ch.isalnum() or ch in ("_", "-", " "):
                allowed.append(ch)
        return "".join(allowed).strip()

    def _tick_timer(self) -> None:
        if not self.recording or self.start_time is None:
            return
        elapsed = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        self.status_var.set(f"Идёт запись... Остановить ({minutes:02d}:{seconds:02d})")
        self.timer_job = self.root.after(200, self._tick_timer)

    def _schedule_scan(self) -> None:
        self.scan_job = self.root.after(1000, self._scan_for_update)

    def _scan_for_update(self) -> None:
        try:
            new_default = self._default_name()
            current = self.filename_var.get().strip()
            if (
                not self.recording
                and current == self.last_auto_name
                and new_default != self.last_auto_name
            ):
                self._set_filename(new_default)
                self.last_auto_name = new_default
        finally:
            self._schedule_scan()

    def _on_format_change(self, *_: object) -> None:
        new_default = self._default_name()
        if self.filename_var.get().strip() == self.last_auto_name:
            self._set_filename(new_default)
        self.last_auto_name = new_default
        self._set_dir_vars()

    def _on_name_change(self, *_: object) -> None:
        if self._updating_name:
            return
        name = self.filename_var.get().strip()
        if not name:
            return
        sanitized = self._safe_name(name)
        if not sanitized:
            sanitized = self.last_auto_name or self._default_name()
        if sanitized != name:
            self._set_filename(sanitized)

    def _set_filename(self, value: str) -> None:
        self._updating_name = True
        try:
            self.filename_var.set(value)
        finally:
            self._updating_name = False

    def _choose_scan_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Выберите папку для нумерации")
        if not chosen:
            return
        self.scan_dir = Path(chosen)
        self._set_dir_vars()
        new_default = self._default_name()
        if self.filename_var.get().strip() == self.last_auto_name:
            self._set_filename(new_default)
        self.last_auto_name = new_default

    def _choose_output_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Выберите папку для сохранения")
        if not chosen:
            return
        self.output_dir = Path(chosen)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_dir_vars()

    def _set_dir_vars(self) -> None:
        self.scan_dir_var.set(str(self.scan_dir))
        self.output_dir_var.set(str(self.output_dir))

    def _center_window(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")


def main() -> None:
    root = Tk()
    app = DictaphoneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
