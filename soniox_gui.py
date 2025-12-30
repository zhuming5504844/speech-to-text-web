import argparse
import os
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

TK_DND_AVAILABLE = importlib.util.find_spec("tkinterdnd2") is not None
if TK_DND_AVAILABLE:
    from tkinterdnd2 import DND_FILES, TkinterDnD
else:
    DND_FILES = None
    TkinterDnD = None


@dataclass
class TranscriptionResult:
    transcript_srt_path: str
    translation_srt_path: Optional[str]


def format_timestamp(ms: int) -> str:
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def tokens_to_srt(tokens: Iterable[dict]) -> str:
    segments: list[tuple[int, int, str]] = []
    current_tokens: list[dict] = []
    start_ms: Optional[int] = None
    last_end: Optional[int] = None

    def flush_segment() -> None:
        nonlocal current_tokens, start_ms, last_end
        if not current_tokens or start_ms is None or last_end is None:
            return
        text = "".join(token["text"] for token in current_tokens).strip()
        if text:
            segments.append((start_ms, last_end, text))
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
        if last_end is not None and token_start - last_end > 1000:
            flush_segment()
            start_ms = token_start
        current_tokens.append(token)
        last_end = token_end
        text = token.get("text", "")
        if text.strip().endswith((".", "?", "!")):
            flush_segment()

    flush_segment()

    lines: list[str] = []
    for idx, (seg_start, seg_end, seg_text) in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(seg_start)} --> {format_timestamp(seg_end)}")
        lines.append(seg_text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def filter_tokens(tokens: Iterable[dict], translation_only: bool) -> list[dict]:
    filtered: list[dict] = []
    for token in tokens:
        is_translation = token.get("translation_status") == "translation"
        if translation_only and is_translation:
            filtered.append(token)
        elif not translation_only and not is_translation:
            filtered.append(token)
    return filtered


def get_config(audio_url: Optional[str], file_id: Optional[str], target_language: str) -> dict:
    config = {
        "model": "stt-async-v3",
        "language_hints": ["en", "zh"],
        "enable_language_identification": True,
        "enable_speaker_diarization": False,
        "audio_url": audio_url,
        "file_id": file_id,
    }

    if target_language:
        config["translation"] = {
            "type": "one_way",
            "target_language": target_language,
        }

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


def wait_until_completed(session: Session, transcription_id: str) -> None:
    while True:
        res = session.get(f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}")
        res.raise_for_status()
        data = res.json()
        if data["status"] == "completed":
            return
        if data["status"] == "error":
            raise RuntimeError(data.get("error_message", "Unknown error"))
        time.sleep(1)


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
    session: Session, audio_path: str, output_dir: str, target_language: str
) -> TranscriptionResult:
    file_id = upload_audio(session, audio_path)
    config = get_config(None, file_id, target_language)

    transcription_id = create_transcription(session, config)
    wait_until_completed(session, transcription_id)
    result = get_transcription(session, transcription_id)

    tokens = result.get("tokens", [])
    transcript_tokens = filter_tokens(tokens, translation_only=False)
    translation_tokens = filter_tokens(tokens, translation_only=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_srt_path = os.path.join(output_dir, f"{base_name}.srt")
    translation_srt_path = os.path.join(output_dir, f"{base_name}.translation.srt")

    with open(transcript_srt_path, "w", encoding="utf-8") as handle:
        handle.write(tokens_to_srt(transcript_tokens))

    translation_path: Optional[str] = None
    if translation_tokens:
        with open(translation_srt_path, "w", encoding="utf-8") as handle:
            handle.write(tokens_to_srt(translation_tokens))
        translation_path = translation_srt_path

    delete_transcription(session, transcription_id)
    delete_file(session, file_id)

    return TranscriptionResult(
        transcript_srt_path=transcript_srt_path,
        translation_srt_path=translation_path,
    )


class SonioxGui:
    def __init__(self) -> None:
        if TK_DND_AVAILABLE:
            self.root = TkinterDnD.Tk()  # type: ignore[misc]
        else:
            self.root = tk.Tk()
        self.root.title("Soniox SRT Generator")
        self.root.geometry("560x320")

        self.file_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.target_language_var = tk.StringVar(value="zh")
        self.status_var = tk.StringVar(value="拖拽 mp3 文件或点击浏览选择文件。")

        self._build_ui()

    def _build_ui(self) -> None:
        header = ttk.Label(self.root, text="Soniox 转录 + 翻译 SRT", font=("Arial", 14))
        header.pack(pady=10)

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.X, padx=20)

        file_label = ttk.Label(frame, text="音频文件:")
        file_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        file_entry = ttk.Entry(frame, textvariable=self.file_path_var, width=50)
        file_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        browse_button = ttk.Button(frame, text="浏览", command=self._select_file)
        browse_button.grid(row=0, column=2, padx=5)

        output_label = ttk.Label(frame, text="输出目录:")
        output_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(frame, textvariable=self.output_dir_var, width=50)
        output_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        output_button = ttk.Button(frame, text="选择", command=self._select_output_dir)
        output_button.grid(row=1, column=2, padx=5)

        lang_label = ttk.Label(frame, text="翻译目标语言:")
        lang_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        lang_entry = ttk.Entry(frame, textvariable=self.target_language_var, width=10)
        lang_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        self.api_key_var = tk.StringVar(value=os.environ.get("SONIOX_API_KEY", ""))
        api_key_label = ttk.Label(frame, text="API Key:")
        api_key_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        api_key_entry = ttk.Entry(frame, textvariable=self.api_key_var, width=50, show="*")
        api_key_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        api_key_hint = ttk.Label(frame, text="可留空使用环境变量 SONIOX_API_KEY", foreground="#666")
        api_key_hint.grid(row=3, column=2, sticky=tk.W)

        self.drop_label = ttk.Label(
            self.root,
            text="拖拽 mp3 到此处",
            relief=tk.RIDGE,
            padding=20,
        )
        self.drop_label.pack(fill=tk.BOTH, padx=20, pady=10, expand=True)

        if TK_DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop_label.configure(text="未检测到拖拽插件，使用浏览按钮选择文件。")

        status_label = ttk.Label(self.root, textvariable=self.status_var, foreground="#444")
        status_label.pack(pady=5)

        self.start_button = ttk.Button(self.root, text="开始生成", command=self._start)
        self.start_button.pack(pady=5)

    def _select_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择 mp3 文件", filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def _select_output_dir(self) -> None:
        output_dir = filedialog.askdirectory(title="选择输出目录")
        if output_dir:
            self.output_dir_var.set(output_dir)

    def _on_drop(self, event: tk.Event) -> None:
        if not event.data:
            return
        file_path = event.data.strip("{}")
        self.file_path_var.set(file_path)

    def _start(self) -> None:
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showwarning("提示", "请选择要上传的音频文件。")
            return
        if not os.path.exists(file_path):
            messagebox.showerror("错误", "文件不存在。")
            return
        output_dir = self.output_dir_var.get().strip() or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        api_key = self.api_key_var.get().strip() or os.environ.get("SONIOX_API_KEY")
        if not api_key:
            messagebox.showerror("错误", "请先设置环境变量 SONIOX_API_KEY 或填写 API Key。")
            return

        target_language = self.target_language_var.get().strip()
        self.start_button.config(state=tk.DISABLED)
        self.status_var.set("正在上传并转录，请稍候...")

        thread = threading.Thread(
            target=self._run_transcription,
            args=(file_path, output_dir, target_language, api_key),
            daemon=True,
        )
        thread.start()

    def _run_transcription(
        self, file_path: str, output_dir: str, target_language: str, api_key: str
    ) -> None:
        try:
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {api_key}"
            result = transcribe_file(session, file_path, output_dir, target_language)
            message = f"转录完成: {result.transcript_srt_path}"
            if result.translation_srt_path:
                message += f"\n翻译完成: {result.translation_srt_path}"
            self._set_status(message)
        except Exception as exc:
            self._set_status(f"转录失败: {exc}")
        finally:
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Soniox GUI for SRT generation")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--audio_path", help="Audio file path for CLI mode")
    parser.add_argument("--output_dir", default=os.getcwd())
    parser.add_argument("--target_language", default="zh")
    parser.add_argument("--api_key", help="Soniox API key (fallback to SONIOX_API_KEY)")
    args = parser.parse_args()

    if args.no_gui:
        if not args.audio_path:
            raise SystemExit("--audio_path required when using --no-gui")
        api_key = args.api_key or os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SONIOX_API_KEY")
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {api_key}"
        result = transcribe_file(session, args.audio_path, args.output_dir, args.target_language)
        print(f"Transcript: {result.transcript_srt_path}")
        if result.translation_srt_path:
            print(f"Translation: {result.translation_srt_path}")
        return

    gui = SonioxGui()
    gui.run()


if __name__ == "__main__":
    main()
