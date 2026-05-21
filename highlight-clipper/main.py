"""Highlight Clipper — Gradio UI и оркестрация пайплайна.

Пайплайн: скачивание VOD -> транскрипция -> LLM-анализ -> нарезка клипов.
Запуск: python main.py
"""

import os
import traceback

import gradio as gr

from analyzer import AnalysisError, analyze_transcript
from clipper import ClippingError, cut_clips
from downloader import DownloadError, download_vod
from transcriber import TranscriptionError, transcribe

# ============================================================
#  КОНФИГ — меняй параметры здесь
# ============================================================

CONFIG = {
    # --- ollama (LLM-анализ) ---
    "ollama_url": "http://localhost:11434",   # адрес локального ollama API
    # qwen2.5:32b полностью помещается в 32GB VRAM RTX 5090 (быстро).
    # qwen2.5:72b / llama3.3:70b точнее, но при Q4 уходят в offload на RAM
    # (медленнее). 96GB RAM позволяют их запускать как опцию.
    "ollama_model": "qwen2.5:32b",
    "top_highlights": 10,                     # сколько моментов выбирать

    # --- faster-whisper (транскрипция) ---
    # large-v3-turbo — быстрая дистилляция large-v3 (≈4-8x быстрее,
    # качество почти то же). По умолчанию выбран как лучший баланс.
    "whisper_model": "large-v3-turbo",
    "whisper_device": "cuda",                 # "cuda" для RTX 5090, иначе "cpu"
    "whisper_compute_type": "float16",        # "float16" для CUDA, "int8" для CPU
    "whisper_language": None,                 # код языка ("ru"/"en") или None — авто

    # --- скачивание / нарезка ---
    "cache_dir": "./vod_cache",               # кэш скачанных VOD (без перекачки)
    "clips_dir": "./clips",                   # корневая папка для готовых клипов
    "buffer_seconds": 15.0,                   # буфер до/после момента, секунды
}

# Модели Whisper для выпадающего списка: от самых быстрых к самым точным.
WHISPER_CHOICES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3-turbo",
    "large-v3",
]

# ============================================================


def process_vod(url: str, whisper_model: str, progress=gr.Progress()):
    """Прогоняет VOD через весь пайплайн и возвращает результат для UI.

    Возвращает кортеж (статусное_сообщение, список_файлов_для_gr.Files).
    """
    if not url or not url.strip():
        return "Ошибка: вставь ссылку на VOD.", None

    # Выбранная в UI модель имеет приоритет над значением из CONFIG.
    whisper_model = whisper_model or CONFIG["whisper_model"]

    try:
        # --- Шаг 1: скачивание VOD (с кэшем) ---
        progress(0.0, desc="Шаг 1/4 — Скачивание VOD")

        def dl_progress(pct, msg):
            # Скачивание занимает первые 25% общего прогресс-бара.
            progress(pct * 0.25, desc=f"Шаг 1/4 — {msg}")

        vod = download_vod(
            url,
            cache_dir=CONFIG["cache_dir"],
            progress_callback=dl_progress,
        )

        # --- Шаг 2: транскрипция ---
        progress(0.25, desc="Шаг 2/4 — Транскрипция аудио")

        def tr_progress(pct, msg):
            # Транскрипция — это 25%…65% прогресс-бара.
            progress(0.25 + pct * 0.40, desc=f"Шаг 2/4 — {msg}")

        segments = transcribe(
            vod["video_path"],
            model_size=whisper_model,
            device=CONFIG["whisper_device"],
            compute_type=CONFIG["whisper_compute_type"],
            language=CONFIG["whisper_language"],
            progress_callback=tr_progress,
        )

        # --- Шаг 3: LLM-анализ ---
        progress(0.65, desc="Шаг 3/4 — Анализ транскрипта (ollama)")

        def an_progress(pct, msg):
            # Анализ — это 65%…80% прогресс-бара.
            progress(0.65 + pct * 0.15, desc=f"Шаг 3/4 — {msg}")

        highlights = analyze_transcript(
            segments,
            ollama_url=CONFIG["ollama_url"],
            model=CONFIG["ollama_model"],
            top_n=CONFIG["top_highlights"],
            progress_callback=an_progress,
        )

        # --- Шаг 4: нарезка клипов ---
        progress(0.80, desc="Шаг 4/4 — Нарезка клипов (ffmpeg)")

        def cl_progress(pct, msg):
            # Нарезка — финальные 80%…100%.
            progress(0.80 + pct * 0.20, desc=f"Шаг 4/4 — {msg}")

        clips = cut_clips(
            vod["video_path"],
            highlights,
            output_dir=CONFIG["clips_dir"],
            vod_title=vod["title"],
            buffer_seconds=CONFIG["buffer_seconds"],
            video_duration=vod["duration"],
            progress_callback=cl_progress,
        )

        progress(1.0, desc="Готово!")

        # Формируем текстовый отчёт по клипам.
        cache_note = " (из кэша)" if vod.get("from_cache") else ""
        lines = [
            f"Готово! Нарезано клипов: {len(clips)}",
            "",
            f"VOD: {vod['title']}{cache_note}",
            f"Модель транскрипции: {whisper_model}",
            f"Папка: {os.path.abspath(os.path.join(CONFIG['clips_dir'], vod['title']))}",
            "",
        ]
        for i, clip in enumerate(clips, start=1):
            lines.append(
                f"{i}. [{clip['start']:.0f}s–{clip['end']:.0f}s] {clip['description']}"
            )

        clip_paths = [clip["path"] for clip in clips]
        return "\n".join(lines), clip_paths

    except DownloadError as exc:
        return f"Ошибка скачивания: {exc}", None
    except TranscriptionError as exc:
        return f"Ошибка транскрипции: {exc}", None
    except AnalysisError as exc:
        return f"Ошибка анализа: {exc}", None
    except ClippingError as exc:
        return f"Ошибка нарезки: {exc}", None
    except Exception as exc:  # noqa: BLE001 — финальная страховка для UI
        # Печатаем полный трейс в консоль, в UI — короткое сообщение.
        traceback.print_exc()
        return f"Непредвиденная ошибка: {exc}", None


def build_ui() -> gr.Blocks:
    """Собирает интерфейс Gradio."""
    with gr.Blocks(title="Highlight Clipper", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎬 Highlight Clipper
            Автоматическая нарезка highlights из стримов — **Twitch · YouTube · Kick**.

            Вставь ссылку на VOD, выбери модель транскрипции и нажми **Обработать**.
            """
        )

        with gr.Row():
            # Левая колонка — ввод и настройки.
            with gr.Column(scale=1):
                url_input = gr.Textbox(
                    label="Ссылка на VOD",
                    placeholder="https://www.twitch.tv/videos/...",
                )
                whisper_dropdown = gr.Dropdown(
                    choices=WHISPER_CHOICES,
                    value=CONFIG["whisper_model"],
                    label="Модель транскрипции (Whisper)",
                    info="turbo — быстро, large-v3 — точнее, tiny/base — очень быстро",
                )
                run_button = gr.Button(
                    "🚀 Обработать", variant="primary", size="lg"
                )

            # Правая колонка — статус и результат.
            with gr.Column(scale=2):
                status_output = gr.Textbox(
                    label="Статус / результат",
                    lines=14,
                    interactive=False,
                    show_copy_button=True,
                )
                clips_output = gr.Files(
                    label="Готовые клипы (нажми, чтобы скачать)",
                )

        run_button.click(
            fn=process_vod,
            inputs=[url_input, whisper_dropdown],
            outputs=[status_output, clips_output],
            # Показываем прогресс ТОЛЬКО на одном поле, иначе индикатор
            # дублируется поверх каждого выходного компонента.
            show_progress_on=[status_output],
        )

    return demo


if __name__ == "__main__":
    # Заранее создаём папки, чтобы не падать на первом запуске.
    os.makedirs(CONFIG["clips_dir"], exist_ok=True)
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    app = build_ui()
    app.launch()
