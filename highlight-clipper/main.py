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
    "ollama_model": "qwen2.5:72b",            # модель: qwen2.5:72b или llama3.3:70b
    "top_highlights": 10,                     # сколько моментов выбирать

    # --- faster-whisper (транскрипция) ---
    "whisper_model": "large-v3",              # размер модели Whisper
    "whisper_device": "cuda",                 # "cuda" для RTX 5090, иначе "cpu"
    "whisper_compute_type": "float16",        # "float16" для CUDA, "int8" для CPU
    "whisper_language": None,                 # код языка ("ru"/"en") или None — авто

    # --- ffmpeg (нарезка) ---
    "clips_dir": "./clips",                   # корневая папка для готовых клипов
    "buffer_seconds": 15.0,                   # буфер до/после момента, секунды
}

# ============================================================


def process_vod(url: str, progress=gr.Progress()):
    """Прогоняет VOD через весь пайплайн и возвращает результат для UI.

    Возвращает кортеж (статусное_сообщение, список_файлов_для_gr.Files).
    """
    if not url or not url.strip():
        return "Ошибка: вставь ссылку на VOD.", None

    try:
        # --- Шаг 1: скачивание VOD ---
        progress(0.0, desc="Шаг 1/4 — Скачивание VOD")

        def dl_progress(pct, msg):
            # Скачивание занимает первые 25% общего прогресс-бара.
            progress(pct * 0.25, desc=f"Шаг 1/4 — {msg}")

        vod = download_vod(url, progress_callback=dl_progress)

        # --- Шаг 2: транскрипция ---
        progress(0.25, desc="Шаг 2/4 — Транскрипция аудио")

        def tr_progress(pct, msg):
            # Транскрипция — это 25%…65% прогресс-бара.
            progress(0.25 + pct * 0.40, desc=f"Шаг 2/4 — {msg}")

        segments = transcribe(
            vod["video_path"],
            model_size=CONFIG["whisper_model"],
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
        lines = [f"Готово! Нарезано клипов: {len(clips)}", ""]
        lines.append(f"VOD: {vod['title']}")
        lines.append(f"Папка: {os.path.abspath(os.path.join(CONFIG['clips_dir'], vod['title']))}")
        lines.append("")
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
    with gr.Blocks(title="Highlight Clipper") as demo:
        gr.Markdown(
            "# Highlight Clipper\n"
            "Автоматическая нарезка highlights из стримов "
            "(Twitch, YouTube, Kick).\n\n"
            "Вставь ссылку на VOD и нажми **Обработать**."
        )

        with gr.Row():
            url_input = gr.Textbox(
                label="Ссылка на VOD",
                placeholder="https://www.twitch.tv/videos/...",
                scale=4,
            )
            run_button = gr.Button("Обработать", variant="primary", scale=1)

        status_output = gr.Textbox(
            label="Статус / результат",
            lines=14,
            interactive=False,
        )

        clips_output = gr.Files(
            label="Готовые клипы (можно скачать)",
        )

        run_button.click(
            fn=process_vod,
            inputs=[url_input],
            outputs=[status_output, clips_output],
        )

    return demo


if __name__ == "__main__":
    # Заранее создаём папку для клипов, чтобы не падать на первом запуске.
    os.makedirs(CONFIG["clips_dir"], exist_ok=True)
    app = build_ui()
    app.launch()
