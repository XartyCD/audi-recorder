"""GUI dictaphone: record from mic, name the file, and save to MP3."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from tkinter import BOTH, DISABLED, NORMAL, Button, Entry, Frame, Label, StringVar, Tk

import lameenc
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44_100
CHANNELS = 1
DTYPE = "int16"
DEFAULT_START = 555
SCAN_DIR = Path(r"C:\Users\Иван\Desktop\хрустел")
OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"


class DictaphoneApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Dictaphone")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.last_auto_name = self._default_name()
        self.filename_var = StringVar(value=self.last_auto_name)
        self.status_var = StringVar(value="Ready")

        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.recording = False
        self.start_time: float | None = None
        self.timer_job: str | None = None
        self.scan_job: str | None = None

        self._build_ui()
        self._schedule_scan()

    def _build_ui(self) -> None:
        frame = Frame(self.root, padx=10, pady=10)
        frame.pack(fill=BOTH)

        Label(frame, text="File name (mp3):").pack(anchor="w")
        self.filename_entry = Entry(frame, textvariable=self.filename_var, width=40)
        self.filename_entry.pack(fill="x", pady=(0, 8))

        btn_frame = Frame(frame)
        btn_frame.pack(fill="x", pady=(0, 8))

        self.start_btn = Button(btn_frame, text="Start", command=self.start_recording, width=10)
        self.start_btn.pack(side="left")

        self.stop_btn = Button(btn_frame, text="Stop", command=self.stop_recording, width=10, state=DISABLED)
        self.stop_btn.pack(side="left", padx=5)

        self.save_btn = Button(btn_frame, text="Save MP3", command=self.save_mp3, width=10, state=DISABLED)
        self.save_btn.pack(side="left")

        self.status_label = Label(frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x")

    def start_recording(self) -> None:
        if self.stream is not None:
            return
        self.frames.clear()
        self.recording = True
        self.start_time = time.time()
        self.status_var.set("Recording... Press Stop (00:00)")
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
            self.status_var.set(f"Audio error: {exc}")
            self._set_buttons(recording=False)
            self.stream = None
            self.recording = False
            self.start_time = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio device warning: {status}", file=sys.stderr)
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
            self.status_var.set("No audio captured. Try again.")
            self._set_buttons(recording=False, has_audio=False)
            return
        self.status_var.set("Recording stopped. Ready to save.")
        self._set_buttons(recording=False, has_audio=True)

    def save_mp3(self) -> None:
        if not self.frames:
            self.status_var.set("Nothing to save. Record first.")
            return
        target = self._make_target_path()
        self.status_var.set(f"Saving to {target.name} ...")
        self._set_buttons(enabled=False)
        threading.Thread(target=self._save_worker, args=(target,), daemon=True).start()

    def _save_worker(self, target: Path) -> None:
        audio = np.concatenate(self.frames, axis=0)
        try:
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(SAMPLE_RATE)
            encoder.set_channels(CHANNELS)
            encoder.set_quality(2)

            mp3_data = encoder.encode(audio.tobytes()) + encoder.flush()
            target.write_bytes(mp3_data)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self.status_var.set, f"Save failed: {exc}")
        else:
            self.root.after(0, self.status_var.set, f"Saved: {target.name}")
            self.frames.clear()
            next_name = self._default_name()
            self.last_auto_name = next_name
            self.root.after(0, self.filename_var.set, next_name)
        finally:
            self.root.after(0, self._set_buttons, False, False, True)

    def _set_buttons(self, recording: bool = False, has_audio: bool | None = None, enabled: bool = True) -> None:
        if not enabled:
            self.start_btn.config(state=DISABLED)
            self.stop_btn.config(state=DISABLED)
            self.save_btn.config(state=DISABLED)
            return
        self.start_btn.config(state=DISABLED if recording else NORMAL)
        self.stop_btn.config(state=NORMAL if recording else DISABLED)
        if has_audio is None:
            has_audio = bool(self.frames)
        self.save_btn.config(state=NORMAL if has_audio and not recording else DISABLED)

    def _make_target_path(self) -> Path:
        base_dir = OUTPUT_DIR
        name = self.filename_var.get().strip() or self._default_name()
        if not name.lower().endswith(".mp3"):
            name += ".mp3"
        candidate = base_dir / name
        counter = 1
        while candidate.exists():
            candidate = base_dir / f"{candidate.stem}_{counter}.mp3"
            counter += 1
        return candidate

    @staticmethod
    def _default_name() -> str:
        max_num = DEFAULT_START - 1
        if SCAN_DIR.exists():
            for path in SCAN_DIR.glob("*.mp3"):
                stem = path.stem
                if stem.isdigit():
                    try:
                        max_num = max(max_num, int(stem))
                    except ValueError:
                        continue
        return f"{max_num + 1}.mp3"

    def _tick_timer(self) -> None:
        if not self.recording or self.start_time is None:
            return
        elapsed = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        self.status_var.set(f"Recording... Press Stop ({minutes:02d}:{seconds:02d})")
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
                self.filename_var.set(new_default)
                self.last_auto_name = new_default
        finally:
            self._schedule_scan()


def main() -> None:
    root = Tk()
    app = DictaphoneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
