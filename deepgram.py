import json
import os
import textwrap
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, scrolledtext, ttk
import importlib.util

from dotenv import load_dotenv

from deepgram import DeepgramClient
from deepgram.core import RequestOptions

load_dotenv()

TK_DND_AVAILABLE = importlib.util.find_spec("tkinterdnd2") is not None
if TK_DND_AVAILABLE:
    from tkinterdnd2 import DND_FILES, TkinterDnD
else:
    DND_FILES = None
    TkinterDnD = None


@dataclass
class TranscriptionOptions:
    api_key: str
    model: str
    language: str
    detect_language: bool
    word_timestamps: bool
    punctuate: bool
    smart_format: bool
    utterances: bool
    diarize: bool
    numerals: bool
    filler_words: bool
    profanity_filter: bool
    paragraphs: bool
    keywords: str
    search: str
    replace: str
    tag: str
    redact: str
    summarize: str
    extra_json: str
    line_width: int
    timeout_seconds: int
    max_retries: int


class DeepgramSubtitleGUI:
    def __init__(self, root: tk.Tk, dnd_available: bool = False) -> None:
        self.root = root
        self.dnd_available = dnd_available
        self.root.title("Deepgram 字幕转录 (拖放音频)")
        self.root.geometry("900x720")

        self.audio_path = tk.StringVar()
        self.status_text = tk.StringVar(value="准备就绪")

        self._build_ui()
        if self.dnd_available:
            self._setup_drag_and_drop()
        else:
            self.log("拖放功能需要 tkinterdnd2 (可选依赖)，当前将使用选择文件按钮。")

        self._load_settings()
        self._ensure_settings_file()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(top_frame, text="音频文件").pack(anchor="w")
        file_row = tk.Frame(top_frame)
        file_row.pack(fill=tk.X, pady=4)

        self.file_entry = tk.Entry(file_row, textvariable=self.audio_path)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(file_row, text="选择文件", command=self.select_file).pack(side=tk.LEFT, padx=6)
        tk.Button(file_row, text="开始转录", command=self.start_transcription).pack(side=tk.LEFT)

        options_frame = tk.LabelFrame(self.root, text="转录参数 (全参数)")
        options_frame.pack(fill=tk.BOTH, padx=12, pady=8, expand=True)

        self.api_key_var = tk.StringVar(value=os.getenv("DEEPGRAM_API_KEY", ""))
        self.model_var = tk.StringVar(value="nova-3")
        self.language_var = tk.StringVar(value="自动(检测)")
        self.detect_language_var = tk.BooleanVar(value=False)
        self.word_timestamps_var = tk.BooleanVar(value=False)
        self.punctuate_var = tk.BooleanVar(value=True)
        self.smart_format_var = tk.BooleanVar(value=False)
        self.utterances_var = tk.BooleanVar(value=True)
        self.diarize_var = tk.BooleanVar(value=False)
        self.numerals_var = tk.BooleanVar(value=True)
        self.filler_words_var = tk.BooleanVar(value=False)
        self.profanity_filter_var = tk.BooleanVar(value=False)
        self.paragraphs_var = tk.BooleanVar(value=False)
        self.keywords_var = tk.StringVar(value="")
        self.search_var = tk.StringVar(value="")
        self.replace_var = tk.StringVar(value="")
        self.tag_var = tk.StringVar(value="")
        self.redact_var = tk.StringVar(value="")
        self.summarize_var = tk.StringVar(value="")
        self.extra_json_var = tk.StringVar(value="{}")
        self.line_width_var = tk.IntVar(value=42)
        self.timeout_seconds_var = tk.IntVar(value=300)
        self.max_retries_var = tk.IntVar(value=2)
        self.api_key_history: list[str] = []

        row = 0
        row = self._add_api_key_row(options_frame, row)
        row = self._add_labeled_entry(options_frame, row, "模型(model)", self.model_var)
        row = self._add_language_dropdown(options_frame, row)

        checkbox_frame = tk.Frame(options_frame)
        checkbox_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1

        self._add_checkbox(checkbox_frame, "自动识别语言", self.detect_language_var)
        self._add_checkbox(checkbox_frame, "字词级时间戳", self.word_timestamps_var)
        self._add_checkbox(checkbox_frame, "标点", self.punctuate_var)
        self._add_checkbox(checkbox_frame, "智能格式化", self.smart_format_var)
        self._add_checkbox(checkbox_frame, "按话语分段(utterances)", self.utterances_var)
        self._add_checkbox(checkbox_frame, "说话人区分", self.diarize_var)
        self._add_checkbox(checkbox_frame, "数字转写", self.numerals_var)
        self._add_checkbox(checkbox_frame, "填充词(filler)", self.filler_words_var)
        self._add_checkbox(checkbox_frame, "敏感词过滤", self.profanity_filter_var)
        self._add_checkbox(checkbox_frame, "段落(paragraphs)", self.paragraphs_var)

        row = self._add_labeled_entry(options_frame, row, "关键词(keywords, 逗号分隔)", self.keywords_var)
        row = self._add_labeled_entry(options_frame, row, "搜索(search, 逗号分隔)", self.search_var)
        row = self._add_labeled_entry(options_frame, row, "替换(replace, 逗号分隔)", self.replace_var)
        row = self._add_labeled_entry(options_frame, row, "标签(tag)", self.tag_var)
        row = self._add_labeled_entry(options_frame, row, "敏感替换(redact)", self.redact_var)
        row = self._add_labeled_entry(options_frame, row, "摘要(summarize)", self.summarize_var)

        tk.Label(options_frame, text="额外参数(JSON, 覆盖上方设置)").grid(
            row=row, column=0, sticky="w", padx=4, pady=(6, 2)
        )
        row += 1
        extra_entry = tk.Entry(options_frame, textvariable=self.extra_json_var)
        extra_entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4)
        row += 1

        tk.Label(options_frame, text="字幕行宽(字符数)").grid(
            row=row, column=0, sticky="w", padx=4, pady=(6, 2)
        )
        line_width_entry = tk.Entry(options_frame, textvariable=self.line_width_var)
        line_width_entry.grid(row=row, column=1, sticky="ew", padx=4)
        row += 1

        tk.Label(options_frame, text="请求超时(秒)").grid(
            row=row, column=0, sticky="w", padx=4, pady=(6, 2)
        )
        timeout_entry = tk.Entry(options_frame, textvariable=self.timeout_seconds_var)
        timeout_entry.grid(row=row, column=1, sticky="ew", padx=4)
        row += 1

        tk.Label(options_frame, text="重试次数").grid(
            row=row, column=0, sticky="w", padx=4, pady=(6, 2)
        )
        retries_entry = tk.Entry(options_frame, textvariable=self.max_retries_var)
        retries_entry.grid(row=row, column=1, sticky="ew", padx=4)
        row += 1

        options_frame.columnconfigure(1, weight=1)

        output_frame = tk.LabelFrame(self.root, text="日志")
        output_frame.pack(fill=tk.BOTH, padx=12, pady=8, expand=True)

        self.log_output = scrolledtext.ScrolledText(output_frame, height=10)
        self.log_output.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        status_bar = tk.Label(self.root, textvariable=self.status_text, anchor="w")
        status_bar.pack(fill=tk.X, padx=12, pady=(0, 8))

    def _add_labeled_entry(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar) -> int:
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        entry = tk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        return row + 1

    def _add_api_key_row(self, parent: tk.Widget, row: int) -> int:
        tk.Label(parent, text="API Key").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        api_frame = tk.Frame(parent)
        api_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        api_frame.columnconfigure(0, weight=1)

        self.api_key_combo = ttk.Combobox(
            api_frame,
            textvariable=self.api_key_var,
            values=self.api_key_history,
        )
        self.api_key_combo.grid(row=0, column=0, sticky="ew")
        self.api_key_combo.bind("<<ComboboxSelected>>", self._on_api_key_selected)
        tk.Button(api_frame, text="粘贴", command=self.paste_api_key).grid(row=0, column=1, padx=4)
        tk.Button(api_frame, text="导入", command=self.import_api_key).grid(row=0, column=2, padx=4)
        tk.Button(api_frame, text="清空", command=self.clear_api_key).grid(row=0, column=3, padx=4)
        tk.Button(api_frame, text="删除当前", command=self.remove_selected_api_key).grid(row=0, column=4, padx=4)
        return row + 1

    def _add_language_dropdown(self, parent: tk.Widget, row: int) -> int:
        tk.Label(parent, text="语言(language)").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        language_values = list(self._language_options().keys())
        combo = ttk.Combobox(parent, textvariable=self.language_var, values=language_values)
        combo.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        combo.set(self.language_var.get() or language_values[0])
        return row + 1

    def _add_checkbox(self, parent: tk.Widget, label: str, variable: tk.BooleanVar) -> None:
        tk.Checkbutton(parent, text=label, variable=variable).pack(side=tk.LEFT, padx=6)

    def _setup_drag_and_drop(self) -> None:
        if not DND_FILES:
            return
        self.file_entry.drop_target_register(DND_FILES)
        self.file_entry.dnd_bind("<<Drop>>", self.handle_drop)

    def select_file(self) -> None:
        filetypes = (
            ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.mp4"),
            ("All files", "*.*"),
        )
        filename = filedialog.askopenfilename(title="选择音频文件", filetypes=filetypes)
        if filename:
            self.audio_path.set(filename)
            self.log(f"已选择文件: {filename}")

    def handle_drop(self, event: tk.Event) -> None:
        path = event.data.strip("{}")
        if path:
            self.audio_path.set(path)
            self.log(f"已拖放文件: {path}")

    def paste_api_key(self) -> None:
        try:
            key = self.root.clipboard_get().strip()
        except tk.TclError:
            messagebox.showwarning("提示", "剪贴板为空或无法读取。")
            return
        if key:
            self._set_api_keys_from_text(key)

    def import_api_key(self) -> None:
        filename = filedialog.askopenfilename(
            title="导入 API Key",
            filetypes=(("Text", "*.txt *.env"), ("All files", "*.*")),
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as key_file:
                content = key_file.read().strip()
        except OSError as exc:
            messagebox.showerror("错误", f"读取失败: {exc}")
            return

        keys = self._extract_api_keys(content)
        if not keys:
            messagebox.showwarning("提示", "未找到有效的 API Key。")
            return
        self._set_api_keys(keys)

    def clear_api_key(self) -> None:
        self.api_key_var.set("")
        self.api_key_combo.set("")
        self.api_key_history.clear()
        self._update_api_key_combo()

    def remove_selected_api_key(self) -> None:
        selected = self.api_key_var.get().strip()
        if not selected:
            return
        if selected in self.api_key_history:
            self.api_key_history.remove(selected)
        self.api_key_var.set(self.api_key_history[0] if self.api_key_history else "")
        self._update_api_key_combo()

    def _extract_api_keys(self, content: str) -> list[str]:
        keys: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("DEEPGRAM_API_KEY"):
                _, value = line.split("=", 1)
                candidate = value.strip().strip("'").strip('"')
                if candidate:
                    keys.append(candidate)
                continue
            for token in line.replace(",", " ").split():
                if token:
                    keys.append(token)
        return list(dict.fromkeys(keys))

    def _set_api_keys_from_text(self, text: str) -> None:
        keys = self._extract_api_keys(text)
        if not keys:
            messagebox.showwarning("提示", "未找到有效的 API Key。")
            return
        self._set_api_keys(keys)

    def _set_api_keys(self, keys: list[str]) -> None:
        if not keys:
            return
        for key in keys:
            self._add_api_key_to_history(key)
        self.api_key_var.set(keys[0])
        self._update_api_key_combo()

    def _add_api_key_to_history(self, key: str) -> None:
        if not key:
            return
        if key not in self.api_key_history:
            self.api_key_history.append(key)

    def _update_api_key_combo(self) -> None:
        if hasattr(self, "api_key_combo"):
            self.api_key_combo["values"] = self.api_key_history

    def _on_api_key_selected(self, event: tk.Event) -> None:
        selected = self.api_key_var.get().strip()
        if selected:
            self._add_api_key_to_history(selected)
            self._update_api_key_combo()

    def start_transcription(self) -> None:
        path = self.audio_path.get().strip()
        if not path:
            messagebox.showwarning("提示", "请先选择或拖放音频文件。")
            return
        if not os.path.exists(path):
            messagebox.showerror("错误", f"文件不存在: {path}")
            return

        options = TranscriptionOptions(
            api_key=self.api_key_var.get().strip(),
            model=self.model_var.get().strip(),
            language=self._normalize_language(self.language_var.get().strip()),
            detect_language=self.detect_language_var.get(),
            word_timestamps=self.word_timestamps_var.get(),
            punctuate=self.punctuate_var.get(),
            smart_format=self.smart_format_var.get(),
            utterances=self.utterances_var.get(),
            diarize=self.diarize_var.get(),
            numerals=self.numerals_var.get(),
            filler_words=self.filler_words_var.get(),
            profanity_filter=self.profanity_filter_var.get(),
            paragraphs=self.paragraphs_var.get(),
            keywords=self.keywords_var.get().strip(),
            search=self.search_var.get().strip(),
            replace=self.replace_var.get().strip(),
            tag=self.tag_var.get().strip(),
            redact=self.redact_var.get().strip(),
            summarize=self.summarize_var.get().strip(),
            extra_json=self.extra_json_var.get().strip() or "{}",
            line_width=int(self.line_width_var.get()),
            timeout_seconds=int(self.timeout_seconds_var.get()),
            max_retries=int(self.max_retries_var.get()),
        )

        self.status_text.set("转录中，请稍候...")
        self.log("开始转录，请等待...")

        self._save_settings()

        threading.Thread(
            target=self.transcribe_file,
            args=(path, options),
            daemon=True,
        ).start()

    def transcribe_file(self, path: str, options: TranscriptionOptions) -> None:
        try:
            client = DeepgramClient(api_key=options.api_key or None)
            with open(path, "rb") as audio_file:
                audio_data = audio_file.read()

            params = self._build_params(options)
            self.log(f"请求参数: {json.dumps(params, ensure_ascii=False)}")

            response = client.listen.v1.media.transcribe_file(request=audio_data, **params)
            response_dict = self._response_to_dict(response)
            if response_dict.get("results") is None:
                raise ValueError("转录返回结果为空，请确认请求未被异步接收或参数设置正确。")
            srt_text = self._build_srt(response_dict, options.line_width)

            output_path = os.path.splitext(path)[0] + ".srt"
            with open(output_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(srt_text)

            self.status_text.set("完成")
            self.log(f"转录完成，已保存字幕: {output_path}")
        except Exception as exc:
            self.status_text.set("转录失败")
            self.log(f"转录失败: {exc}")

    def _build_params(self, options: TranscriptionOptions) -> dict:
        params = {
            "model": options.model or None,
            "language": options.language or None,
            "detect_language": options.detect_language,
            "punctuate": options.punctuate,
            "smart_format": options.smart_format,
            "utterances": options.utterances,
            "diarize": options.diarize,
            "numerals": options.numerals,
            "filler_words": options.filler_words,
            "profanity_filter": options.profanity_filter,
            "paragraphs": options.paragraphs,
        }

        request_options: RequestOptions = {
            "timeout_in_seconds": max(1, options.timeout_seconds),
            "max_retries": max(0, options.max_retries),
        }
        params["request_options"] = request_options

        if options.keywords:
            params["keywords"] = [item.strip() for item in options.keywords.split(",") if item.strip()]
        if options.search:
            params["search"] = [item.strip() for item in options.search.split(",") if item.strip()]
        if options.replace:
            params["replace"] = [item.strip() for item in options.replace.split(",") if item.strip()]
        if options.tag:
            params["tag"] = [item.strip() for item in options.tag.split(",") if item.strip()]
        if options.redact:
            params["redact"] = options.redact
        if options.summarize:
            params["summarize"] = options.summarize
        if options.word_timestamps:
            params["timestamps"] = "word"

        params = {key: value for key, value in params.items() if value is not None}

        extra_params = self._parse_extra_json(options.extra_json)
        params.update(extra_params)
        return params

    def _language_options(self) -> dict:
        return {
            "自动(检测)": "",
            "多语言(multi)": "multi",
            "英语(en)": "en",
            "日语(ja)": "ja",
            "中文(zh)": "zh",
            "韩语(ko)": "ko",
            "法语(fr)": "fr",
            "德语(de)": "de",
            "西班牙语(es)": "es",
            "葡萄牙语(pt)": "pt",
            "意大利语(it)": "it",
        }

    def _normalize_language(self, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        options = self._language_options()
        return options.get(value, value)

    def _settings_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

    def _load_settings(self) -> None:
        path = self._settings_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as settings_file:
                data = json.load(settings_file)
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"读取设置失败: {exc}")
            return

        self.api_key_var.set(data.get("api_key", self.api_key_var.get()))
        self.api_key_history = list(dict.fromkeys(data.get("api_keys", [])))
        if self.api_key_var.get():
            self._add_api_key_to_history(self.api_key_var.get())
        self._update_api_key_combo()
        self.model_var.set(data.get("model", self.model_var.get()))
        self.language_var.set(data.get("language", self.language_var.get()))
        self.detect_language_var.set(data.get("detect_language", self.detect_language_var.get()))
        self.word_timestamps_var.set(data.get("word_timestamps", self.word_timestamps_var.get()))
        self.punctuate_var.set(data.get("punctuate", self.punctuate_var.get()))
        self.smart_format_var.set(data.get("smart_format", self.smart_format_var.get()))
        self.utterances_var.set(data.get("utterances", self.utterances_var.get()))
        self.diarize_var.set(data.get("diarize", self.diarize_var.get()))
        self.numerals_var.set(data.get("numerals", self.numerals_var.get()))
        self.filler_words_var.set(data.get("filler_words", self.filler_words_var.get()))
        self.profanity_filter_var.set(data.get("profanity_filter", self.profanity_filter_var.get()))
        self.paragraphs_var.set(data.get("paragraphs", self.paragraphs_var.get()))
        self.keywords_var.set(data.get("keywords", self.keywords_var.get()))
        self.search_var.set(data.get("search", self.search_var.get()))
        self.replace_var.set(data.get("replace", self.replace_var.get()))
        self.tag_var.set(data.get("tag", self.tag_var.get()))
        self.redact_var.set(data.get("redact", self.redact_var.get()))
        self.summarize_var.set(data.get("summarize", self.summarize_var.get()))
        self.extra_json_var.set(data.get("extra_json", self.extra_json_var.get()))
        self.line_width_var.set(int(data.get("line_width", self.line_width_var.get())))
        self.timeout_seconds_var.set(int(data.get("timeout_seconds", self.timeout_seconds_var.get())))
        self.max_retries_var.set(int(data.get("max_retries", self.max_retries_var.get())))

    def _ensure_settings_file(self) -> None:
        if not os.path.exists(self._settings_path()):
            self._save_settings()

    def _save_settings(self) -> None:
        self._add_api_key_to_history(self.api_key_var.get().strip())
        data = {
            "api_key": self.api_key_var.get().strip(),
            "api_keys": self.api_key_history,
            "model": self.model_var.get().strip(),
            "language": self.language_var.get().strip(),
            "detect_language": self.detect_language_var.get(),
            "word_timestamps": self.word_timestamps_var.get(),
            "punctuate": self.punctuate_var.get(),
            "smart_format": self.smart_format_var.get(),
            "utterances": self.utterances_var.get(),
            "diarize": self.diarize_var.get(),
            "numerals": self.numerals_var.get(),
            "filler_words": self.filler_words_var.get(),
            "profanity_filter": self.profanity_filter_var.get(),
            "paragraphs": self.paragraphs_var.get(),
            "keywords": self.keywords_var.get().strip(),
            "search": self.search_var.get().strip(),
            "replace": self.replace_var.get().strip(),
            "tag": self.tag_var.get().strip(),
            "redact": self.redact_var.get().strip(),
            "summarize": self.summarize_var.get().strip(),
            "extra_json": self.extra_json_var.get().strip(),
            "line_width": int(self.line_width_var.get()),
            "timeout_seconds": int(self.timeout_seconds_var.get()),
            "max_retries": int(self.max_retries_var.get()),
        }
        try:
            with open(self._settings_path(), "w", encoding="utf-8") as settings_file:
                json.dump(data, settings_file, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.log(f"保存设置失败: {exc}")

    def _parse_extra_json(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("额外参数必须是 JSON 对象")
            return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"额外参数 JSON 解析失败: {exc}") from exc

    def _response_to_dict(self, response: object) -> dict:
        if hasattr(response, "dict"):
            return response.dict()  # type: ignore[no-any-return]
        if isinstance(response, dict):
            return response
        return {}

    def _build_srt(self, response: dict, line_width: int) -> str:
        results = response.get("results", {}) or {}
        utterances = results.get("utterances")
        channels = results.get("channels", [])
        if channels:
            alternatives = channels[0].get("alternatives", [])
            if alternatives:
                words = alternatives[0].get("words", [])
                if utterances:
                    utterance_end = max((item.get("end", 0) for item in utterances), default=0)
                    words_end = max((item.get("end", 0) for item in words), default=0)
                    if words_end > utterance_end + 0.5 or self._words_outside_utterances(words, utterances):
                        return self._srt_from_words(words, line_width)
                    return self._srt_from_utterances(utterances, line_width)
                return self._srt_from_words(words, line_width)

        if utterances:
            return self._srt_from_utterances(utterances, line_width)

        transcript = results.get("transcripts") or []
        if transcript:
            return self._srt_from_text(transcript[0].get("transcript", ""), line_width)

        return ""

    def _words_outside_utterances(self, words: list, utterances: list) -> bool:
        if not words or not utterances:
            return False
        sorted_utterances = sorted(utterances, key=lambda item: item.get("start", 0))
        utterance_index = 0
        current = sorted_utterances[utterance_index]
        for word in words:
            word_start = word.get("start", 0)
            while utterance_index < len(sorted_utterances) and word_start > current.get("end", 0):
                utterance_index += 1
                if utterance_index >= len(sorted_utterances):
                    return True
                current = sorted_utterances[utterance_index]
            if word_start < current.get("start", 0):
                return True
        return False

    def _srt_from_utterances(self, utterances: list, line_width: int) -> str:
        lines = []
        for index, utterance in enumerate(utterances, start=1):
            start = utterance.get("start", 0)
            end = utterance.get("end", 0)
            transcript = utterance.get("transcript", "").strip()
            transcript = textwrap.fill(transcript, width=line_width)
            lines.append(
                f"{index}\n{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n{transcript}\n"
            )
        return "\n".join(lines)

    def _srt_from_words(self, words: list, line_width: int) -> str:
        if not words:
            return ""
        segments = []
        current = {"start": words[0]["start"], "end": words[0]["end"], "text": words[0]["word"]}
        for word in words[1:]:
            text = current["text"]
            if word["start"] - current["end"] > 1.2 or len(text) > line_width:
                segments.append(current)
                current = {"start": word["start"], "end": word["end"], "text": word["word"]}
            else:
                current["text"] = f"{current['text']} {word['word']}"
                current["end"] = word["end"]
        segments.append(current)

        lines = []
        for index, seg in enumerate(segments, start=1):
            transcript = textwrap.fill(seg["text"].strip(), width=line_width)
            lines.append(
                f"{index}\n{self._format_timestamp(seg['start'])} --> {self._format_timestamp(seg['end'])}\n{transcript}\n"
            )
        return "\n".join(lines)

    def _srt_from_text(self, text: str, line_width: int) -> str:
        transcript = textwrap.fill(text.strip(), width=line_width)
        return f"1\n00:00:00,000 --> 00:00:10,000\n{transcript}\n"

    def _format_timestamp(self, seconds: float) -> str:
        millis = int(round(seconds * 1000))
        hours, remainder = divmod(millis, 3600 * 1000)
        minutes, remainder = divmod(remainder, 60 * 1000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def log(self, message: str) -> None:
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.see(tk.END)

    def on_close(self) -> None:
        self._save_settings()
        self.root.destroy()


if __name__ == "__main__":
    if TK_DND_AVAILABLE and TkinterDnD is not None:
        root = TkinterDnD.Tk()
        dnd_available = True
    else:
        root = tk.Tk()
        dnd_available = False

    app = DeepgramSubtitleGUI(root, dnd_available=dnd_available)
    root.mainloop()
