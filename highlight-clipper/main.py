"""Highlight Clipper — Gradio UI и оркестрация пайплайна.

Пайплайн: скачивание VOD -> транскрипция -> [детекция боёв] -> LLM-анализ
-> нарезка клипов. Запуск: python main.py
"""

import os
import traceback

import gradio as gr

import combat_segmenter
from analyzer import (
    AnalysisError,
    analyze_combat_segments,
    analyze_transcript,
)
from clipper import ClippingError, cut_clips
from combat_segmenter import CombatDetectionError, segment_combat
from downloader import DownloadError, download_vod
from transcriber import TranscriptionError, transcribe

# ============================================================
#  КОНФИГ — меняй параметры здесь
# ============================================================

CONFIG = {
    # --- ollama (LLM-анализ) ---
    "ollama_url": "http://localhost:11434",   # адрес локального ollama API
    # qwen2.5:32b полностью помещается в 32GB VRAM RTX 5090 (быстро).
    # qwen2.5:72b / llama3.3:70b точнее, но при Q4 уходят в offload на RAM.
    "ollama_model": "qwen2.5:32b",
    "top_highlights": 10,                     # лимит моментов в режиме только LLM

    # --- faster-whisper (транскрипция) ---
    "whisper_model": "large-v3-turbo",        # модель по умолчанию (меняется в UI)
    "whisper_device": "cuda",                 # "cuda" для RTX 5090, иначе "cpu"
    "whisper_compute_type": "float16",        # "float16" для CUDA, "int8" для CPU
    "whisper_language": None,                 # код языка ("ru"/"en") или None — авто

    # --- combat_detector (детекция боёв) ---
    "combat_model_path": "./combat_detector.pt",  # путь к обученной ResNet18
    "combat_device": "cuda",                  # инференс на CUDA
    "combat_compute_type": "float16",         # fp16 на CUDA
    "combat_sample_step": 0.5,                # шаг семплирования кадров, секунды
    "combat_threshold": 0.7,                  # порог вероятности боя

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

# Модель детекции боя загружается ОДИН раз при старте и висит в памяти.
# None означает работу в режиме только LLM (модель не найдена/не загрузилась).
COMBAT_MODEL = None

# ============================================================


def _clamp_length(highlights: list[dict], max_len: float) -> list[dict]:
    """Обрезает слишком длинные клипы по центру боя до max_len секунд.

    Учитываем, что clipper добавит буфер ±buffer_seconds с каждой стороны,
    поэтому целевую длину боевого фрагмента уменьшаем на 2*буфер — так
    итоговый клип уложится примерно в выбранную максимальную длину.
    """
    buffer = CONFIG["buffer_seconds"]
    target = max(1.0, max_len - 2 * buffer)
    for h in highlights:
        length = h["end"] - h["start"]
        if length > target:
            center = (h["start"] + h["end"]) / 2.0
            h["start"] = center - target / 2.0
            h["end"] = center + target / 2.0
    return highlights


def process_vod(
    url: str,
    whisper_model: str,
    use_combat: bool,
    max_clip_len: float,
    min_score: float,
    progress=gr.Progress(),
):
    """Прогоняет VOD через весь пайплайн и возвращает результат для UI.

    Возвращает кортеж (статусное_сообщение, список_файлов_для_gr.Files).
    """
    if not url or not url.strip():
        return "Ошибка: вставь ссылку на VOD.", None

    whisper_model = whisper_model or CONFIG["whisper_model"]

    # Детекция боя возможна, только если она включена И модель загружена.
    combat_enabled = bool(use_combat) and COMBAT_MODEL is not None
    notes: list[str] = []
    if use_combat and COMBAT_MODEL is None:
        notes.append("⚠ Модель детекции боя не загружена — режим только LLM.")

    # Шаги пайплайна и их доли в общем прогресс-баре зависят от режима.
    if combat_enabled:
        bounds = {"dl": (0.0, 0.20), "tr": (0.20, 0.45),
                  "cb": (0.45, 0.65), "an": (0.65, 0.80), "cl": (0.80, 1.0)}
        total_steps = 5
    else:
        bounds = {"dl": (0.0, 0.25), "tr": (0.25, 0.65),
                  "an": (0.65, 0.80), "cl": (0.80, 1.0)}
        total_steps = 4

    def make_cb(key: str, step_no: int, label: str):
        """Создаёт колбэк, проецирующий локальный прогресс этапа в общий бар."""
        lo, hi = bounds[key]

        def _cb(pct, msg):
            progress(lo + pct * (hi - lo), desc=f"Шаг {step_no}/{total_steps} — {msg}")

        return _cb

    try:
        # --- Шаг 1: скачивание VOD (с кэшем) ---
        progress(0.0, desc=f"Шаг 1/{total_steps} — Скачивание VOD")
        vod = download_vod(
            url,
            cache_dir=CONFIG["cache_dir"],
            progress_callback=make_cb("dl", 1, "Скачивание VOD"),
        )

        # --- Шаг 2: транскрипция ---
        segments = transcribe(
            vod["video_path"],
            model_size=whisper_model,
            device=CONFIG["whisper_device"],
            compute_type=CONFIG["whisper_compute_type"],
            language=CONFIG["whisper_language"],
            progress_callback=make_cb("tr", 2, "Транскрипция"),
        )

        # --- Шаг 3 (опц.): детекция боёв ---
        combat_segments: list[dict] = []
        if combat_enabled:
            try:
                combat_segments = segment_combat(
                    vod["video_path"],
                    COMBAT_MODEL,
                    sample_step=CONFIG["combat_sample_step"],
                    threshold=CONFIG["combat_threshold"],
                    progress_callback=make_cb("cb", 3, "Детекция боёв"),
                )
            except CombatDetectionError as exc:
                # Не валим весь процесс — откатываемся на режим только LLM.
                notes.append(f"⚠ Детекция боёв не удалась ({exc}) — режим только LLM.")
                combat_enabled = False

            if combat_enabled and not combat_segments:
                notes.append("⚠ Бои не обнаружены — переключаюсь на режим только LLM.")
                combat_enabled = False

        # --- Шаг 4: LLM-анализ ---
        an_step = 4 if total_steps == 5 else 3
        if combat_enabled:
            highlights = analyze_combat_segments(
                segments,
                combat_segments,
                ollama_url=CONFIG["ollama_url"],
                model=CONFIG["ollama_model"],
                min_score=min_score,
                progress_callback=make_cb("an", an_step, "Оценка боёв (ollama)"),
            )
        else:
            highlights = analyze_transcript(
                segments,
                ollama_url=CONFIG["ollama_url"],
                model=CONFIG["ollama_model"],
                top_n=CONFIG["top_highlights"],
                progress_callback=make_cb("an", an_step, "Анализ транскрипта (ollama)"),
            )

        # Обрезаем слишком длинные клипы по центру боя.
        highlights = _clamp_length(highlights, max_clip_len)

        # --- Шаг 5: нарезка клипов ---
        cl_step = 5 if total_steps == 5 else 4
        clips = cut_clips(
            vod["video_path"],
            highlights,
            output_dir=CONFIG["clips_dir"],
            vod_title=vod["title"],
            buffer_seconds=CONFIG["buffer_seconds"],
            video_duration=vod["duration"],
            progress_callback=make_cb("cl", cl_step, "Нарезка клипов"),
        )

        progress(1.0, desc="Готово!")

        # Формируем текстовый отчёт по клипам.
        cache_note = " (из кэша)" if vod.get("from_cache") else ""
        mode = "детекция боёв + LLM" if combat_enabled else "только LLM"
        lines = list(notes)
        if notes:
            lines.append("")
        lines += [
            f"Готово! Нарезано клипов: {len(clips)}",
            "",
            f"VOD: {vod['title']}{cache_note}",
            f"Режим отбора: {mode}",
            f"Модель транскрипции: {whisper_model}",
            f"Папка: {os.path.abspath(os.path.join(CONFIG['clips_dir'], vod['title']))}",
            "",
        ]
        for i, clip in enumerate(clips, start=1):
            score = highlights[i - 1].get("score") if i - 1 < len(highlights) else None
            score_str = f" (оценка {score:.0f})" if score is not None else ""
            lines.append(
                f"{i}. [{clip['start']:.0f}s–{clip['end']:.0f}s]{score_str} "
                f"{clip['description']}"
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
        traceback.print_exc()
        return f"Непредвиденная ошибка: {exc}", None


def build_ui() -> gr.Blocks:
    """Собирает интерфейс Gradio."""
    with gr.Blocks(title="Highlight Clipper") as demo:
        gr.Markdown(
            """
            # 🎬 Highlight Clipper
            Автоматическая нарезка highlights из стримов — **Twitch · YouTube · Kick**.

            Вставь ссылку на VOD, настрой параметры и нажми **Обработать**.
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
                combat_checkbox = gr.Checkbox(
                    value=True,
                    label="Использовать детекцию боя",
                    info="Выключи для классического режима только через LLM",
                )
                max_len_slider = gr.Slider(
                    minimum=30, maximum=300, value=120, step=5,
                    label="Максимальная длина клипа, сек",
                    info="Длинные бои обрезаются по центру",
                )
                min_score_slider = gr.Slider(
                    minimum=1, maximum=10, value=7, step=1,
                    label="Минимальная оценка клипа",
                    info="Порог отбора боёв по оценке LLM (0-10)",
                )
                run_button = gr.Button(
                    "🚀 Обработать", variant="primary", size="lg"
                )

            # Правая колонка — статус и результат.
            with gr.Column(scale=2):
                status_output = gr.Textbox(
                    label="Статус / результат",
                    lines=16,
                    interactive=False,
                )
                clips_output = gr.Files(
                    label="Готовые клипы (нажми, чтобы скачать)",
                )

        run_button.click(
            fn=process_vod,
            inputs=[
                url_input,
                whisper_dropdown,
                combat_checkbox,
                max_len_slider,
                min_score_slider,
            ],
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

    # Загружаем модель детекции боя ОДИН раз при старте (висит в памяти).
    COMBAT_MODEL = combat_segmenter.load_model(
        CONFIG["combat_model_path"],
        device=CONFIG["combat_device"],
        compute_type=CONFIG["combat_compute_type"],
    )

    app = build_ui()
    # В Gradio 6 тема передаётся в launch(), а не в конструктор Blocks.
    app.launch(theme=gr.themes.Soft())
