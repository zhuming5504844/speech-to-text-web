import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Optional

import importlib.util
import requests
from requests import Session

SONIOX_API_BASE_URL = "https://api.soniox.com"
if getattr(sys, "frozen", False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".soniox_gui_settings.json")
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".wma",
}

TK_DND_AVAILABLE = importlib.util.find_spec("tkinterdnd2") is not None
if TK_DND_AVAILABLE:
    from tkinterdnd2 import DND_FILES, TkinterDnD
else:
    DND_FILES = None
    TkinterDnD = None


@dataclass
class TranscriptionResult:
    transcript_srt_path: Optional[str]
    translation_srt_path: Optional[str]


BATCH_PARALLEL_WORKERS = 3
TRANSCRIPTION_POLL_INTERVAL_S = 1.0
TRANSCRIPTION_TIMEOUT_S = 1800.0


@dataclass
class SrtSettings:
    max_chars_per_segment: int
    max_duration_s: float
    max_pause_s: float
    word_level_segmentation: bool


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str


def format_timestamp(ms: int) -> str:
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_segments(tokens: Iterable[dict], settings: SrtSettings) -> list[Segment]:
    segments: list[Segment] = []
    current_tokens: list[dict] = []
    start_ms: Optional[int] = None
    last_end: Optional[int] = None

    max_pause_ms = int(settings.max_pause_s * 1000)
    max_duration_ms = int(settings.max_duration_s * 1000)

    def flush_segment() -> None:
        nonlocal current_tokens, start_ms, last_end
        if not current_tokens or start_ms is None or last_end is None:
            return
        text = "".join(token.get("text", "") for token in current_tokens).strip()
        segments.append(Segment(start_ms=start_ms, end_ms=last_end, text=text))
        current_tokens = []
        start_ms = None
        last_end = None

    for token in tokens:
        token_start = token.get("start_ms")
        token_end = token.get("end_ms")
        if token_start is None or token_end is None:
            continue

        if start_ms is None:
            start_ms = token_start

        if last_end is not None and token_start - last_end > max_pause_ms:
            flush_segment()
            start_ms = token_start

        current_text = "".join(item.get("text", "") for item in current_tokens)
        incoming_text = token.get("text", "")

        if (
            settings.word_level_segmentation
            and current_tokens
            and settings.max_chars_per_segment > 0
            and len(current_text) + len(incoming_text) > settings.max_chars_per_segment
        ):
            flush_segment()
            start_ms = token_start

        if (
            settings.word_level_segmentation
            and current_tokens
            and start_ms is not None
            and token_end - start_ms > max_duration_ms
        ):
            flush_segment()
            start_ms = token_start

        current_tokens.append(token)
        last_end = token_end

        if not settings.word_level_segmentation:
            text = token.get("text", "")
            if text.strip().endswith((".", "?", "!", "。", "？", "！")):
                flush_segment()

    flush_segment()
    return segments


def segments_to_srt(segments: Iterable[Segment]) -> str:
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(segment.start_ms)} --> {format_timestamp(segment.end_ms)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def is_supported_audio_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_AUDIO_EXTENSIONS


def format_supported_audio_types() -> str:
    return ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_AUDIO_EXTENSIONS))


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def parse_translation_terms(value: str) -> list[dict]:
    terms: list[dict] = []
    for item in parse_comma_list(value):
        if "=" not in item:
            raise ValueError(f"翻译术语格式错误: {item}")
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"翻译术语格式错误: {item}")
        terms.append({"source": source, "target": target})
    return terms



def _get_token_language(token: dict) -> Optional[str]:
    return token.get("language") or token.get("source_language")


def filter_tokens(tokens: Iterable[dict], translation_only: bool, target_language: Optional[str]) -> list[dict]:
    filtered: list[dict] = []
    for token in tokens:
        status = token.get("translation_status")
        is_translation = status == "translation"
        if translation_only:
            if is_translation:
                filtered.append(token)
                continue
            token_language = _get_token_language(token)
            if token_language and target_language and token_language == target_language:
                filtered.append(token)
        elif not is_translation:
            filtered.append(token)
    return filtered


def get_config(
    audio_url: Optional[str],
    file_id: Optional[str],
    model: str,
    language: Optional[str],
    enable_language_identification: bool,
    enable_speaker_diarization: bool,
    target_language: Optional[str],
    context_text: Optional[str] = None,
    context_terms: Optional[list[str]] = None,
    translation_terms: Optional[list[dict]] = None,
    client_reference_id: Optional[str] = None,
) -> dict:
    config = {
        "model": model,
        "enable_language_identification": enable_language_identification,
        "enable_speaker_diarization": enable_speaker_diarization,
        "audio_url": audio_url,
        "file_id": file_id,
    }

    if language:
        config["language_hints"] = [language]

    if target_language:
        config["translation"] = {
            "type": "one_way",
            "target_language": target_language,
        }

    context = {}
    if context_text:
        context["text"] = context_text
    if context_terms:
        context["terms"] = context_terms
    if translation_terms:
        context["translation_terms"] = translation_terms
    if context:
        config["context"] = context

    if client_reference_id:
        config["client_reference_id"] = client_reference_id

    return config


def upload_audio(session: Session, audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        res = session.post(
            f"{SONIOX_API_BASE_URL}/v1/files",
            files={"file": audio_file},
        )
    res.raise_for_status()
    return res.json()["id"]


def create_transcription(session: Session, config: dict) -> str:
    res = session.post(f"{SONIOX_API_BASE_URL}/v1/transcriptions", json=config)
    res.raise_for_status()
    return res.json()["id"]


def wait_until_completed(
    session: Session,
    transcription_id: str,
    poll_interval_s: float = TRANSCRIPTION_POLL_INTERVAL_S,
    timeout_s: float = TRANSCRIPTION_TIMEOUT_S,
) -> None:
    start_time = time.monotonic()
    while True:
        res = session.get(f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}")
        res.raise_for_status()
        data = res.json()
        status = data.get("status")
        if status == "completed":
            return
        if status in {"error", "canceled", "cancelled", "deleted"}:
            raise RuntimeError(data.get("error_message", "Unknown error"))

        elapsed_s = time.monotonic() - start_time
        if elapsed_s >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for transcription {transcription_id} after {timeout_s:.1f}s "
                f"(last status: {status})"
            )

        time.sleep(poll_interval_s)


def get_transcription(session: Session, transcription_id: str) -> dict:
    res = session.get(
        f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}/transcript"
    )
    res.raise_for_status()
    return res.json()


def delete_transcription(session: Session, transcription_id: str) -> None:
    res = session.delete(f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}")
    res.raise_for_status()


def delete_file(session: Session, file_id: str) -> None:
    res = session.delete(f"{SONIOX_API_BASE_URL}/v1/files/{file_id}")
    res.raise_for_status()


def transcribe_file(
    session: Session,
    audio_path: str,
    output_dir: str,
    model: str,
    language: Optional[str],
    enable_language_identification: bool,
    enable_speaker_diarization: bool,
    target_language: Optional[str],
    srt_settings: SrtSettings,
    output_transcript: bool,
    output_translation: bool,
    context_text: Optional[str] = None,
    context_terms: Optional[list[str]] = None,
    translation_terms: Optional[list[dict]] = None,
    client_reference_id: Optional[str] = None,
) -> TranscriptionResult:
    file_id: Optional[str] = None
    transcription_id: Optional[str] = None
    try:
        file_id = upload_audio(session, audio_path)
        config = get_config(
            None,
            file_id,
            model,
            language,
            enable_language_identification,
            enable_speaker_diarization,
            target_language if output_translation else None,
            context_text=context_text,
            context_terms=context_terms,
            translation_terms=translation_terms,
            client_reference_id=client_reference_id,
        )

        transcription_id = create_transcription(session, config)
        wait_until_completed(session, transcription_id)
        result = get_transcription(session, transcription_id)

        tokens = result.get("tokens", [])
        transcript_tokens = filter_tokens(
            tokens, translation_only=False, target_language=target_language
        )
        translation_tokens = filter_tokens(
            tokens, translation_only=True, target_language=target_language
        )

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_srt_path = os.path.join(output_dir, f"{base_name}.srt")
        translation_srt_path = os.path.join(output_dir, f"{base_name}.translation.srt")

        transcript_segments = build_segments(transcript_tokens, srt_settings)
        transcript_path: Optional[str] = None
        if output_transcript:
            with open(transcript_srt_path, "w", encoding="utf-8") as handle:
                handle.write(segments_to_srt(transcript_segments))
            transcript_path = transcript_srt_path

        translation_path: Optional[str] = None
        translation_segments: list[Segment] = []
        if output_translation:
            translation_segments = build_segments(translation_tokens, srt_settings)
            with open(translation_srt_path, "w", encoding="utf-8") as handle:
                handle.write(segments_to_srt(translation_segments))
            translation_path = translation_srt_path

        return TranscriptionResult(
            transcript_srt_path=transcript_path,
            translation_srt_path=translation_path,
        )
    finally:
        if transcription_id:
            try:
                delete_transcription(session, transcription_id)
            except requests.RequestException:
                pass
        if file_id:
            try:
                delete_file(session, file_id)
            except requests.RequestException:
                pass


class SonioxGui:
    def __init__(self) -> None:
        settings = self._load_settings()
        if TK_DND_AVAILABLE:
            self.root = TkinterDnD.Tk()  # type: ignore[misc]
        else:
            self.root = tk.Tk()
        self.root.title("Soniox 智能字幕助手")
        self.root.geometry("920x820")
        self.root.minsize(860, 760)
        self.root.configure(bg="#f4f7fb")
        self._configure_style()

        self.file_paths: list[str] = []
        self.api_key_var = tk.StringVar(
            value=settings.get("api_key") or os.environ.get("SONIOX_API_KEY", "")
        )
        self.model_var = tk.StringVar(value=settings.get("model", "stt-async-v5"))
        self.language_var = tk.StringVar(value=settings.get("language", "日语 ja"))
        self.output_transcript_var = tk.BooleanVar(value=settings.get("output_transcript", True))
        self.output_translation_var = tk.BooleanVar(value=settings.get("output_translation", True))
        self.enable_language_id_var = tk.BooleanVar(
            value=settings.get("enable_language_identification", False)
        )
        self.enable_speaker_diarization_var = tk.BooleanVar(
            value=settings.get("enable_speaker_diarization", False)
        )
        self.word_segmentation_var = tk.BooleanVar(value=settings.get("word_segmentation", False))
        self.target_language_var = tk.StringVar(value=settings.get("target_language", "中文 zh"))
        self.max_chars_var = tk.StringVar(value=str(settings.get("max_chars", "16")))
        self.max_duration_var = tk.StringVar(value=str(settings.get("max_duration", "6.0")))
        self.max_pause_var = tk.StringVar(value=str(settings.get("max_pause", "1.0")))
        self.context_text_var = tk.StringVar(value=settings.get("context_text", ""))
        self.context_terms_var = tk.StringVar(value=settings.get("context_terms", ""))
        self.translation_terms_var = tk.StringVar(value=settings.get("translation_terms", ""))
        self.client_reference_id_var = tk.StringVar(value=settings.get("client_reference_id", ""))
        self.status_var = tk.StringVar(value="准备就绪：添加音频后点击开始转录")
        self.last_output_dir: Optional[str] = None

        self._log_lock = threading.Lock()

        self.language_options = [
            ("自动识别", ""),
            ("日语 ja", "ja"),
            ("中文 zh", "zh"),
            ("英语 en", "en"),
            ("韩语 ko", "ko"),
            ("法语 fr", "fr"),
            ("德语 de", "de"),
            ("西班牙语 es", "es"),
        ]
        self.language_map = {label: code for label, code in self.language_options}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f4f7fb")
        style.configure("Card.TLabelframe", background="#ffffff", bordercolor="#d9e2ef", relief="solid")
        style.configure("Card.TLabelframe.Label", background="#f4f7fb", foreground="#1f3b57", font=("Segoe UI", 11, "bold"))
        style.configure("TLabel", background="#ffffff", foreground="#263238", font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background="#f4f7fb", foreground="#5c6f82", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#1677ff")
        style.map("Accent.TButton", background=[("active", "#0958d9"), ("disabled", "#9bbcff")])

    def _load_settings(self) -> dict:
        if not os.path.exists(SETTINGS_PATH):
            return {}
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_settings(self) -> None:
        settings = {
            "api_key": self.api_key_var.get().strip(),
            "model": self.model_var.get().strip(),
            "language": self.language_var.get().strip(),
            "output_transcript": self.output_transcript_var.get(),
            "output_translation": self.output_translation_var.get(),
            "enable_language_identification": self.enable_language_id_var.get(),
            "enable_speaker_diarization": self.enable_speaker_diarization_var.get(),
            "word_segmentation": self.word_segmentation_var.get(),
            "target_language": self.target_language_var.get().strip(),
            "max_chars": self.max_chars_var.get().strip(),
            "max_duration": self.max_duration_var.get().strip(),
            "max_pause": self.max_pause_var.get().strip(),
            "context_text": self.context_text_var.get().strip(),
            "context_terms": self.context_terms_var.get().strip(),
            "translation_terms": self.translation_terms_var.get().strip(),
            "client_reference_id": self.client_reference_id_var.get().strip(),
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#1677ff", height=74)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Soniox 智能字幕助手",
            bg="#1677ff",
            fg="#ffffff",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor=tk.W, padx=18, pady=(12, 0))
        tk.Label(
            header,
            text="批量音频转 SRT · 自动识别 · 翻译 · 上下文术语增强",
            bg="#1677ff",
            fg="#d8e8ff",
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W, padx=20, pady=(2, 0))

        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Hint.TLabel")
        status_bar.pack(fill=tk.X, padx=16, pady=(10, 0))

        file_frame = ttk.LabelFrame(self.root, text="1. 音频文件", style="Card.TLabelframe")
        file_frame.pack(fill=tk.X, padx=12, pady=8)

        self.file_listbox = tk.Listbox(file_frame, height=4, selectmode=tk.EXTENDED)
        self.file_listbox.grid(row=0, column=0, rowspan=3, padx=8, pady=8, sticky=tk.NSEW)
        file_scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        file_scrollbar.grid(row=0, column=1, rowspan=3, sticky=tk.NS, pady=8)
        self.file_listbox.configure(yscrollcommand=file_scrollbar.set)

        add_button = ttk.Button(file_frame, text="添加文件", command=self._select_files)
        add_button.grid(row=0, column=2, padx=8, pady=4, sticky=tk.EW)
        remove_button = ttk.Button(file_frame, text="移除选中", command=self._remove_selected_files)
        remove_button.grid(row=1, column=2, padx=8, pady=4, sticky=tk.EW)
        clear_button = ttk.Button(file_frame, text="清空队列", command=self._clear_files)
        clear_button.grid(row=2, column=2, padx=8, pady=4, sticky=tk.EW)

        if TK_DND_AVAILABLE:
            self.file_listbox.drop_target_register(DND_FILES)
            self.file_listbox.dnd_bind("<<Drop>>", self._on_drop)
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)

        api_frame = ttk.LabelFrame(self.root, text="2. API 配置", style="Card.TLabelframe")
        api_frame.pack(fill=tk.X, padx=12, pady=8)

        api_key_label = ttk.Label(api_frame, text="API Key")
        api_key_label.grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=70, show="*")
        api_key_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=8, pady=6)

        model_label = ttk.Label(api_frame, text="转录模型 (model)")
        model_label.grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        model_combo = ttk.Combobox(
            api_frame,
            textvariable=self.model_var,
            values=["stt-async-v5", "stt-async-v4", "stt-async-v3", "stt-async-v2"],
            state="normal",
            width=24,
        )
        model_combo.grid(row=1, column=1, sticky=tk.W, padx=8, pady=6)

        language_label = ttk.Label(api_frame, text="识别语言 (language)")
        language_label.grid(row=2, column=0, sticky=tk.W, padx=8, pady=6)
        language_combo = ttk.Combobox(
            api_frame,
            textvariable=self.language_var,
            values=[label for label, _ in self.language_options],
            state="readonly",
            width=24,
        )
        language_combo.grid(row=2, column=1, sticky=tk.W, padx=8, pady=6)

        feature_frame = ttk.LabelFrame(self.root, text="3. 识别与翻译", style="Card.TLabelframe")
        feature_frame.pack(fill=tk.X, padx=12, pady=8)

        output_transcript = ttk.Checkbutton(
            feature_frame, text="输出转录字幕", variable=self.output_transcript_var
        )
        output_transcript.grid(row=0, column=0, sticky=tk.W, padx=8, pady=4)
        output_translation = ttk.Checkbutton(
            feature_frame, text="输出翻译字幕", variable=self.output_translation_var
        )
        output_translation.grid(row=0, column=1, sticky=tk.W, padx=8, pady=4)

        enable_language_id = ttk.Checkbutton(
            feature_frame, text="开启语言自动识别", variable=self.enable_language_id_var
        )
        enable_language_id.grid(row=1, column=0, sticky=tk.W, padx=8, pady=4)
        enable_speaker_diarization = ttk.Checkbutton(
            feature_frame, text="开启说话人区分", variable=self.enable_speaker_diarization_var
        )
        enable_speaker_diarization.grid(row=1, column=1, sticky=tk.W, padx=8, pady=4)

        enable_word_seg = ttk.Checkbutton(
            feature_frame, text="字词级自动分段", variable=self.word_segmentation_var
        )
        enable_word_seg.grid(row=2, column=0, sticky=tk.W, padx=8, pady=4)

        translation_label = ttk.Label(feature_frame, text="翻译目标语言")
        translation_label.grid(row=3, column=0, sticky=tk.W, padx=8, pady=6)
        translation_combo = ttk.Combobox(
            feature_frame,
            textvariable=self.target_language_var,
            values=["中文 zh", "英语 en", "日语 ja", "韩语 ko", "法语 fr", "德语 de", "西班牙语 es"],
            state="readonly",
            width=20,
        )
        translation_combo.grid(row=3, column=1, sticky=tk.W, padx=8, pady=6)

        context_frame = ttk.LabelFrame(self.root, text="4. 上下文增强（可选）", style="Card.TLabelframe")
        context_frame.pack(fill=tk.X, padx=12, pady=8)

        ttk.Label(context_frame, text="场景描述").grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(context_frame, textvariable=self.context_text_var, width=80).grid(
            row=0, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=6
        )
        ttk.Label(context_frame, text="专有名词").grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(context_frame, textvariable=self.context_terms_var, width=80).grid(
            row=1, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=6
        )
        ttk.Label(context_frame, text="翻译术语").grid(row=2, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(context_frame, textvariable=self.translation_terms_var, width=80).grid(
            row=2, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=6
        )
        ttk.Label(context_frame, text="追踪 ID").grid(row=3, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(context_frame, textvariable=self.client_reference_id_var, width=32).grid(
            row=3, column=1, sticky=tk.W, padx=8, pady=6
        )
        ttk.Label(
            context_frame,
            text="提示：专有名词用逗号分隔；翻译术语用“原文=译文”分隔，多组用逗号。",
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=8, pady=(0, 8))
        context_frame.columnconfigure(1, weight=1)

        srt_frame = ttk.LabelFrame(self.root, text="5. SRT 输出设置", style="Card.TLabelframe")
        srt_frame.pack(fill=tk.X, padx=12, pady=8)

        max_chars_label = ttk.Label(srt_frame, text="每条最大字符数")
        max_chars_label.grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        max_chars_entry = ttk.Entry(srt_frame, textvariable=self.max_chars_var, width=8)
        max_chars_entry.grid(row=0, column=1, sticky=tk.W, padx=8, pady=6)

        max_duration_label = ttk.Label(srt_frame, text="最大时长 (秒)")
        max_duration_label.grid(row=0, column=2, sticky=tk.W, padx=8, pady=6)
        max_duration_entry = ttk.Entry(srt_frame, textvariable=self.max_duration_var, width=8)
        max_duration_entry.grid(row=0, column=3, sticky=tk.W, padx=8, pady=6)

        max_pause_label = ttk.Label(srt_frame, text="最大停顿 (秒)")
        max_pause_label.grid(row=0, column=4, sticky=tk.W, padx=8, pady=6)
        max_pause_entry = ttk.Entry(srt_frame, textvariable=self.max_pause_var, width=8)
        max_pause_entry.grid(row=0, column=5, sticky=tk.W, padx=8, pady=6)

        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=12, pady=6)
        self.start_button = ttk.Button(action_frame, text="开始转录", command=self._start, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT)
        ttk.Button(action_frame, text="打开输出目录", command=self._open_last_output_dir).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        log_frame = ttk.LabelFrame(self.root, text="日志", style="Card.TLabelframe")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.log_text = tk.Text(log_frame, height=14, wrap=tk.WORD)
        self.log_text.configure(bg="#0f172a", fg="#dbeafe", insertbackground="#ffffff", relief=tk.FLAT)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _select_files(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_AUDIO_EXTENSIONS))
        file_paths = filedialog.askopenfilenames(
            title="选择音频文件", filetypes=[("Audio Files", extensions)]
        )
        if file_paths:
            self._add_files(list(file_paths))

    def _on_drop(self, event: tk.Event) -> None:
        if not event.data:
            return
        try:
            paths = list(self.root.tk.splitlist(event.data))
        except tk.TclError:
            paths = [event.data]
        self._add_files(paths)

    def _add_files(self, paths: list[str]) -> None:
        added = False
        unsupported: list[str] = []
        for path in paths:
            normalized = path.strip()
            if not normalized:
                continue
            if not is_supported_audio_file(normalized):
                unsupported.append(normalized)
                continue
            if normalized not in self.file_paths:
                self.file_paths.append(normalized)
                self.file_listbox.insert(tk.END, normalized)
                added = True
        if added:
            self.file_listbox.see(tk.END)
        if unsupported:
            self._log(
                "以下文件格式暂不支持，已跳过:\n"
                + "\n".join(unsupported)
                + f"\n支持格式: {format_supported_audio_types()}"
            )

    def _remove_selected_files(self) -> None:
        selected = list(self.file_listbox.curselection())
        for index in reversed(selected):
            self.file_listbox.delete(index)
            del self.file_paths[index]

    def _clear_files(self) -> None:
        self.file_listbox.delete(0, tk.END)
        self.file_paths.clear()

    def _start(self) -> None:
        file_paths = [path for path in self.file_paths if path.strip()]
        if not file_paths:
            messagebox.showwarning("提示", "请先添加要处理的音频文件。")
            return
        missing_files = [path for path in file_paths if not os.path.exists(path)]
        if missing_files:
            messagebox.showerror("错误", f"以下文件不存在:\n" + "\n".join(missing_files))
            return
        unsupported_files = [path for path in file_paths if not is_supported_audio_file(path)]
        if unsupported_files:
            messagebox.showerror(
                "错误",
                "以下文件格式不支持:\n"
                + "\n".join(unsupported_files)
                + f"\n\n支持格式: {format_supported_audio_types()}",
            )
            return

        api_key = self.api_key_var.get().strip() or os.environ.get("SONIOX_API_KEY")
        if not api_key:
            messagebox.showerror("错误", "请先设置环境变量 SONIOX_API_KEY 或填写 API Key。")
            return

        model = self.model_var.get().strip() or "stt-async-v5"
        language = self.language_map.get(self.language_var.get().strip(), "")
        target_language = self._get_language_code(self.target_language_var.get().strip())
        enable_language_identification = self.enable_language_id_var.get()
        enable_speaker_diarization = self.enable_speaker_diarization_var.get()
        output_transcript = self.output_transcript_var.get()
        output_translation = self.output_translation_var.get()
        word_segmentation = self.word_segmentation_var.get()

        try:
            max_chars = int(self.max_chars_var.get().strip())
            max_duration = float(self.max_duration_var.get().strip())
            max_pause = float(self.max_pause_var.get().strip())
        except ValueError:
            messagebox.showerror("错误", "SRT 输出设置格式不正确，请检查输入。")
            return
        try:
            context_terms = parse_comma_list(self.context_terms_var.get().strip())
            translation_terms = parse_translation_terms(self.translation_terms_var.get().strip())
        except ValueError as exc:
            messagebox.showerror("错误", str(exc))
            return

        if not output_transcript and not output_translation:
            messagebox.showwarning("提示", "请至少选择输出转录字幕或翻译字幕。")
            return

        if output_translation and not target_language:
            messagebox.showwarning("提示", "请选择翻译目标语言。")
            return

        self._save_settings()
        srt_settings = SrtSettings(
            max_chars_per_segment=max_chars,
            max_duration_s=max_duration,
            max_pause_s=max_pause,
            word_level_segmentation=word_segmentation,
        )
        self.start_button.config(state=tk.DISABLED)
        self._log("正在上传并转录，请稍候...")

        thread = threading.Thread(
            target=self._run_transcription,
            args=(
                file_paths,
                model,
                language or None,
                enable_language_identification,
                enable_speaker_diarization,
                target_language if output_translation else None,
                srt_settings,
                output_transcript,
                output_translation,
                api_key,
                self.context_text_var.get().strip() or None,
                context_terms,
                translation_terms,
                self.client_reference_id_var.get().strip() or None,
            ),
            daemon=True,
        )
        thread.start()

    def _run_transcription(
        self,
        file_paths: list[str],
        model: str,
        language: Optional[str],
        enable_language_identification: bool,
        enable_speaker_diarization: bool,
        target_language: Optional[str],
        srt_settings: SrtSettings,
        output_transcript: bool,
        output_translation: bool,
        api_key: str,
        context_text: Optional[str],
        context_terms: list[str],
        translation_terms: list[dict],
        client_reference_id: Optional[str],
    ) -> None:
        def process_audio(audio_path: str) -> TranscriptionResult:
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {api_key}"
            output_dir = os.path.dirname(audio_path) or os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            return transcribe_file(
                session=session,
                audio_path=audio_path,
                output_dir=output_dir,
                model=model,
                language=language,
                enable_language_identification=enable_language_identification,
                enable_speaker_diarization=enable_speaker_diarization,
                target_language=target_language,
                srt_settings=srt_settings,
                output_transcript=output_transcript,
                output_translation=output_translation,
                context_text=context_text,
                context_terms=context_terms,
                translation_terms=translation_terms,
                client_reference_id=client_reference_id,
            )

        try:
            self._set_status(f"处理中：共 {len(file_paths)} 个文件，并行 {BATCH_PARALLEL_WORKERS} 路")
            self._log(f"批量模式已开启并行线路: {BATCH_PARALLEL_WORKERS}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_PARALLEL_WORKERS) as executor:
                future_map = {}
                for audio_path in file_paths:
                    self._log(f"开始处理: {audio_path}")
                    future = executor.submit(process_audio, audio_path)
                    future_map[future] = audio_path

                for future in concurrent.futures.as_completed(future_map):
                    audio_path = future_map[future]
                    try:
                        result = future.result()
                        if result.transcript_srt_path:
                            self.last_output_dir = os.path.dirname(result.transcript_srt_path)
                            self._log(f"转录完成: {result.transcript_srt_path}")
                        if result.translation_srt_path:
                            self.last_output_dir = os.path.dirname(result.translation_srt_path)
                            self._log(f"翻译完成: {result.translation_srt_path}")
                    except Exception as exc:
                        self._log(f"处理失败: {audio_path} -> {exc}")
        except Exception as exc:
            self._log(f"转录失败: {exc}")
        finally:
            self._set_status("处理结束：可查看日志或打开输出目录")
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def _log(self, text: str) -> None:
        def append() -> None:
            with self._log_lock:
                self.log_text.insert(tk.END, text + "\n")
                self.log_text.see(tk.END)

        self.root.after(0, append)

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _open_last_output_dir(self) -> None:
        output_dir = self.last_output_dir
        if not output_dir and self.file_paths:
            output_dir = os.path.dirname(self.file_paths[0]) or os.getcwd()
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showinfo("提示", "暂无可打开的输出目录。")
            return
        if sys.platform.startswith("win"):
            os.startfile(output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f"open {json.dumps(output_dir)}")
        else:
            os.system(f"xdg-open {json.dumps(output_dir)}")

    def _get_language_code(self, label: str) -> Optional[str]:
        label = label.strip()
        if not label:
            return None
        return label.split()[-1] if " " in label else label

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Soniox GUI for SRT generation")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--audio_path", help="Audio file path for CLI mode")
    parser.add_argument("--output_dir", default=base_dir)
    parser.add_argument("--model", default="stt-async-v5")
    parser.add_argument("--language", default="ja")
    parser.add_argument(
        "--enable_language_identification", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--enable_speaker_diarization", action="store_true")
    parser.add_argument("--target_language", default="zh")
    parser.add_argument("--output_transcript", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_translation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_chars", type=int, default=16)
    parser.add_argument("--max_duration", type=float, default=6.0)
    parser.add_argument("--max_pause", type=float, default=1.0)
    parser.add_argument("--word_segmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api_key", help="Soniox API key (fallback to SONIOX_API_KEY)")
    parser.add_argument("--context_text", default="", help="Optional domain/context text")
    parser.add_argument("--context_terms", default="", help="Comma-separated custom vocabulary terms")
    parser.add_argument(
        "--translation_terms",
        default="",
        help='Comma-separated translation terms such as "source=target,foo=bar"',
    )
    parser.add_argument("--client_reference_id", default="", help="Optional request tracking id")
    args = parser.parse_args()

    if args.no_gui:
        if not args.audio_path:
            raise SystemExit("--audio_path required when using --no-gui")
        if not is_supported_audio_file(args.audio_path):
            raise SystemExit(
                "Unsupported audio format. Supported formats: "
                + format_supported_audio_types()
            )
        api_key = args.api_key or os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SONIOX_API_KEY")
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {api_key}"
        srt_settings = SrtSettings(
            max_chars_per_segment=args.max_chars,
            max_duration_s=args.max_duration,
            max_pause_s=args.max_pause,
            word_level_segmentation=args.word_segmentation,
        )
        result = transcribe_file(
            session=session,
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            model=args.model,
            language=args.language,
            enable_language_identification=args.enable_language_identification,
            enable_speaker_diarization=args.enable_speaker_diarization,
            target_language=args.target_language if args.output_translation else None,
            srt_settings=srt_settings,
            output_transcript=args.output_transcript,
            output_translation=args.output_translation,
            context_text=args.context_text or None,
            context_terms=parse_comma_list(args.context_terms),
            translation_terms=parse_translation_terms(args.translation_terms),
            client_reference_id=args.client_reference_id or None,
        )
        if result.transcript_srt_path:
            print(f"Transcript: {result.transcript_srt_path}")
        if result.translation_srt_path:
            print(f"Translation: {result.translation_srt_path}")
        return

    gui = SonioxGui()
    gui.run()


if __name__ == "__main__":
    main()
