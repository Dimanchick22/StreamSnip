"""Транскрипция аудио через faster-whisper с CUDA-ускорением.

Модель загружается один раз и кешируется, чтобы повторные запуски в рамках
одной сессии Gradio не перечитывали веса с диска.
"""

from faster_whisper import WhisperModel


class TranscriptionError(Exception):
    """Ошибка на этапе транскрипции."""


# Кеш загруженной модели: ключ — кортеж параметров, значение — объект модели.
_model_cache: dict = {}


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Возвращает модель из кеша или загружает её при первом обращении."""
    cache_key = (model_size, device, compute_type)
    if cache_key not in _model_cache:
        try:
            _model_cache[cache_key] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                f"Не удалось загрузить модель Whisper '{model_size}' "
                f"на устройстве '{device}': {exc}"
            ) from exc
    return _model_cache[cache_key]


def transcribe(
    video_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
    progress_callback=None,
) -> list[dict]:
    """Транскрибирует аудиодорожку видео в список сегментов с таймстампами.

    Параметры:
        video_path: путь к видеофайлу (faster-whisper сам извлечёт аудио).
        model_size: размер модели Whisper (tiny/base/small/medium/large-v3).
        device: "cuda" для RTX 5090 или "cpu".
        compute_type: тип вычислений ("float16" для CUDA, "int8" для CPU).
        language: код языка ("ru", "en", …) или None для автоопределения.
        progress_callback: функция callback(percent, message) для UI.

    Возвращает список словарей:
        [{"start": float, "end": float, "text": str}, …]

    Бросает TranscriptionError при проблемах.
    """
    model = _get_model(model_size, device, compute_type)

    if progress_callback:
        progress_callback(0.0, "Запуск транскрипции…")

    try:
        # vad_filter отсекает паузы/тишину и ускоряет обработку длинных VOD.
        segments_iter, info = model.transcribe(
            video_path,
            language=language,
            vad_filter=True,
            beam_size=5,
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Ошибка транскрипции аудио: {exc}") from exc

    # Общая длительность нужна для расчёта прогресса (segments_iter ленивый).
    total_duration = info.duration or 0.0

    segments: list[dict] = []
    for segment in segments_iter:
        segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
        )
        if progress_callback and total_duration > 0:
            percent = min(segment.end / total_duration, 1.0)
            progress_callback(percent, f"Транскрипция… {percent * 100:.0f}%")

    if not segments:
        raise TranscriptionError(
            "Транскрипт пуст — в видео не распознана речь."
        )

    if progress_callback:
        progress_callback(1.0, f"Транскрипция завершена ({len(segments)} сегментов).")

    return segments
