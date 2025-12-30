#!/usr/bin/env python3
"""Soniox batch transcription GUI for audio -> SRT (with optional translation)."""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DND_FILES = None
    TkinterDnD = None


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    secs = (milliseconds % 60_000) // 1000
    millis = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _is_cjk_language(language: str) -> bool:
    return language.lower().startswith(("ja", "zh", "ko"))


def _token_text(token: dict[str, Any]) -> str:
    for key in ("text", "token", "word", "value"):
        if key in token and token[key] is not None:
            return str(token[key])
    return ""


def _token_start_end(token: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    for start_key, end_key in (
        ("start_time", "end_time"),
        ("start", "end"),
        ("start_ts", "end_ts"),
        ("startTime", "endTime"),
    ):
        if start_key in token or end_key in token:
            start = token.get(start_key)
            end = token.get(end_key)
            return (float(start) if start is not None else None, float(end) if end is not None else None)
    return (None, None)


def _iter_tokens(payload: dict[str, Any], translation: bool) -> Iterable[dict[str, Any]]:
    result = payload.get("result") or payload
    if translation:
        candidates = []
        for key in ("translation", "translations", "translated"):
            value = result.get(key)
            if value:
                candidates.append(value)
        for candidate in candidates:
            if isinstance(candidate, list):
                for item in candidate:
                    tokens = item.get("tokens") if isinstance(item, dict) else None
                    if tokens:
                        return tokens
            if isinstance(candidate, dict):
                tokens = candidate.get("tokens")
                if tokens:
                    return tokens
        return []
    tokens = result.get("tokens") or payload.get("tokens")
    if tokens:
        return tokens
    return []


def _iter_segments(payload: dict[str, Any], translation: bool) -> Iterable[dict[str, Any]]:
    result = payload.get("result") or payload
    if translation:
        for key in ("translation", "translations", "translated"):
            value = result.get(key)
            if isinstance(value, dict) and value.get("segments"):
                return value["segments"]
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("segments"):
                        return item["segments"]
        return []
    return result.get("segments") or payload.get("segments") or []


def _join_tokens(tokens: list[str], language: str) -> str:
    if not tokens:
        return ""
    if _is_cjk_language(language):
        return "".join(tokens).replace("  ", " ").strip()
    text = ""
    for token in tokens:
        if not text:
            text = token
            continue
        if token in {".", ",", "!", "?", ":", ";", "%", "…", "。", "！", "？"}:
            text += token
        elif token.startswith("'"):
            text += token
        else:
            text += f" {token}"
    return text.strip()


def build_srt(
    payload: dict[str, Any],
    language: str,
    use_word_level: bool,
    max_chars_per_line: int,
    max_duration: float,
    max_silence_gap: float,
    translation: bool,
) -> str:
    segments: list[SubtitleSegment] = []
    if not use_word_level:
        for segment in _iter_segments(payload, translation):
            text = segment.get("text") or segment.get("transcript") or ""
            start = segment.get("start_time") or segment.get("start") or segment.get("startTime")
            end = segment.get("end_time") or segment.get("end") or segment.get("endTime")
            if text and start is not None and end is not None:
                segments.append(SubtitleSegment(float(start), float(end), str(text)))
        if segments:
            return _segments_to_srt(segments)

    tokens = list(_iter_tokens(payload, translation))
    if not tokens:
        return ""

    current_tokens: list[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def flush_segment() -> None:
        nonlocal current_tokens, current_start, current_end
        if not current_tokens or current_start is None or current_end is None:
            current_tokens = []
            current_start = None
            current_end = None
            return
        text = _join_tokens(current_tokens, language)
        if text:
            segments.append(SubtitleSegment(current_start, current_end, text))
        current_tokens = []
        current_start = None
        current_end = None

    punctuation_breaks = {".", "!", "?", "。", "！", "？", "…"}

    for token in tokens:
        token_text = _token_text(token)
        start, end = _token_start_end(token)
        if start is None or end is None:
            continue
        if current_start is None:
            current_start = start
        if current_end is not None and start - current_end > max_silence_gap:
            flush_segment()
            current_start = start
        current_tokens.append(token_text)
        current_end = end
        duration = (current_end - current_start) if current_start is not None else 0
        text_length = len(_join_tokens(current_tokens, language))
        if (
            duration >= max_duration
            or text_length >= max_chars_per_line
            or token_text in punctuation_breaks
        ):
            flush_segment()

    flush_segment()
    return _segments_to_srt(segments)


def _segments_to_srt(segments: list[SubtitleSegment]) -> str:
    lines: list[str] = []
    for index, seg in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = f"----soniox-form-{int(time.time() * 1000)}"
    lines: list[bytes] = []
    for key, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f"Content-Disposition: form-data; name=\"{key}\"\r\n\r\n".encode())
        lines.append(value.encode())
        lines.append(b"\r\n")
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        f"Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{os.path.basename(filename)}\"\r\n".encode()
    )
    lines.append(f"Content-Type: {mime_type}\r\n\r\n".encode())
    lines.append(data)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines), boundary


def soniox_transcribe(
    api_host: str,
    api_key: str,
    audio_path: str,
    model: str,
    language: str,
    enable_language_identification: bool,
    enable_word_level: bool,
    translation_target: Optional[str],
) -> dict[str, Any]:
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()

    config: dict[str, Any] = {
        "model": model,
        "language": language or None,
        "enable_language_identification": enable_language_identification,
        "enable_word_level": enable_word_level,
    }
    if translation_target:
        config["translation"] = {
            "type": "one_way",
            "target_language": translation_target,
        }

    fields = {"config": json.dumps(config)}
    body, boundary = _build_multipart(fields, "audio", audio_path, audio_data)

    api_host = api_host.rstrip("/")
    url = f"{api_host}/v1/transcribe"
    request = Request(url, data=body)
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("X-API-Key", api_key)

    with urlopen(request, timeout=300) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _safe_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(content)


class TranscriptionGUI:
    def __init__(self) -> None:
        self.root = TkinterDnD.Tk() if TkinterDnD else tk.Tk()
        self.root.title("Soniox Audio → SRT GUI")
        self.root.geometry("760x560")

        self.audio_path = tk.StringVar()
        self.api_key = tk.StringVar()
        self.api_host = tk.StringVar(value="https://api.soniox.com")
        self.model = tk.StringVar(value="stt-async")
        self.language = tk.StringVar(value="ja")
        self.translation_target = tk.StringVar(value="en")
        self.enable_language_id = tk.BooleanVar(value=False)
        self.enable_word_level = tk.BooleanVar(value=True)
        self.enable_transcription = tk.BooleanVar(value=True)
        self.enable_translation = tk.BooleanVar(value=True)
        self.max_chars = tk.IntVar(value=16)
        self.max_duration = tk.DoubleVar(value=6.0)
        self.max_silence = tk.DoubleVar(value=1.0)

        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self._poll_log()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main, text="音频文件")
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.audio_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(file_frame, text="选择文件", command=self._select_file).pack(side=tk.LEFT, padx=5)

        drop_label = ttk.Label(file_frame, text="拖动音频文件到这里")
        drop_label.pack(side=tk.LEFT, padx=5)
        if TkinterDnD and DND_FILES:
            drop_label.drop_target_register(DND_FILES)
            drop_label.dnd_bind("<<Drop>>", self._on_drop)

        config_frame = ttk.LabelFrame(main, text="API 配置")
        config_frame.pack(fill=tk.X, pady=5)
        self._add_labeled_entry(config_frame, "API Key", self.api_key, show="*")
        self._add_labeled_entry(config_frame, "API Host", self.api_host)
        self._add_labeled_entry(config_frame, "Model", self.model)
        self._add_labeled_entry(config_frame, "识别语言 (如 ja/en)", self.language)

        options_frame = ttk.LabelFrame(main, text="识别与翻译")
        options_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(options_frame, text="输出转录字幕", variable=self.enable_transcription).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="输出翻译字幕", variable=self.enable_translation).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="开启语言自动识别", variable=self.enable_language_id).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="字词级自动分段", variable=self.enable_word_level).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self._add_labeled_entry(options_frame, "翻译目标语言", self.translation_target, row=2)

        srt_frame = ttk.LabelFrame(main, text="SRT 输出设置")
        srt_frame.pack(fill=tk.X, pady=5)
        ttk.Label(srt_frame, text="每条最大字符数").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(srt_frame, textvariable=self.max_chars, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(srt_frame, text="最大时长 (秒)").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(srt_frame, textvariable=self.max_duration, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        ttk.Label(srt_frame, text="最大停顿 (秒)").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(srt_frame, textvariable=self.max_silence, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)

        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=8)
        ttk.Button(action_frame, text="开始转录", command=self._start_transcription).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int = 0, show: str | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Entry(parent, textvariable=variable, show=show)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        parent.columnconfigure(1, weight=1)

    def _select_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg")])
        if path:
            self.audio_path.set(path)

    def _on_drop(self, event: tk.Event) -> None:
        if event.data:
            path = event.data.strip("{}")
            self.audio_path.set(path)

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _poll_log(self) -> None:
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        self.root.after(200, self._poll_log)

    def _start_transcription(self) -> None:
        if not self.audio_path.get():
            messagebox.showerror("提示", "请选择或拖动音频文件")
            return
        if not self.api_key.get():
            messagebox.showerror("提示", "请输入 Soniox API Key")
            return
        if not (self.enable_transcription.get() or self.enable_translation.get()):
            messagebox.showerror("提示", "请至少选择一个输出选项")
            return

        thread = threading.Thread(target=self._run_transcription, daemon=True)
        thread.start()

    def _run_transcription(self) -> None:
        self._log("开始上传并转录...")
        try:
            payload = soniox_transcribe(
                api_host=self.api_host.get(),
                api_key=self.api_key.get(),
                audio_path=self.audio_path.get(),
                model=self.model.get(),
                language=self.language.get(),
                enable_language_identification=self.enable_language_id.get(),
                enable_word_level=self.enable_word_level.get(),
                translation_target=self.translation_target.get() if self.enable_translation.get() else None,
            )
        except HTTPError as error:
            self._log(f"HTTP 错误: {error.code} {error.reason}")
            return
        except URLError as error:
            self._log(f"连接错误: {error.reason}")
            return
        except Exception as error:  # pragma: no cover - unexpected
            self._log(f"未知错误: {error}")
            return

        base, _ = os.path.splitext(self.audio_path.get())
        if self.enable_transcription.get():
            transcription_srt = build_srt(
                payload,
                language=self.language.get(),
                use_word_level=self.enable_word_level.get(),
                max_chars_per_line=self.max_chars.get(),
                max_duration=self.max_duration.get(),
                max_silence_gap=self.max_silence.get(),
                translation=False,
            )
            if transcription_srt:
                output_path = f"{base}.srt"
                _safe_write(output_path, transcription_srt)
                self._log(f"已输出转录字幕: {output_path}")
            else:
                self._log("未生成转录字幕，请检查返回内容")

        if self.enable_translation.get():
            translation_srt = build_srt(
                payload,
                language=self.translation_target.get(),
                use_word_level=self.enable_word_level.get(),
                max_chars_per_line=self.max_chars.get(),
                max_duration=self.max_duration.get(),
                max_silence_gap=self.max_silence.get(),
                translation=True,
            )
            if translation_srt:
                output_path = f"{base}.translated.srt"
                _safe_write(output_path, translation_srt)
                self._log(f"已输出翻译字幕: {output_path}")
            else:
                self._log("未生成翻译字幕，请检查返回内容")

        self._log("处理完成。")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TranscriptionGUI().run()
