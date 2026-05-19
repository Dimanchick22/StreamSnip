"""Анализ транскрипта через локальный ollama: выбор лучших моментов.

Транскрипт сворачивается в компактный текст с таймстампами и отправляется
LLM-модели, которая возвращает JSON со списком highlights.
"""

import json
import re

import requests


class AnalysisError(Exception):
    """Ошибка на этапе LLM-анализа."""


# Системный промпт: задаёт модели роль и строгий формат ответа.
_SYSTEM_PROMPT = """Ты — редактор коротких видео. Твоя задача — найти в транскрипте \
стрима самые интересные, эмоциональные и динамичные моменты для нарезки highlights.

Правила:
- Выбери топ-N самых ярких моментов (смех, неожиданные повороты, накал, реакции, \
кульминационные фразы).
- Для каждого момента укажи start и end в СЕКУНДАХ (числа с точкой), исходя из \
таймстампов сегментов транскрипта.
- Длительность одного момента — от 15 до 90 секунд.
- Моменты не должны пересекаться по времени.
- description — короткое описание момента на русском (до 100 символов).

Верни ТОЛЬКО валидный JSON без markdown-обёртки, строго в формате:
{"highlights": [{"start": 123.4, "end": 156.7, "description": "текст"}]}"""


def _build_transcript_text(segments: list[dict], max_chars: int = 48000) -> str:
    """Сворачивает сегменты в текст вида "[начало-конец] реплика".

    Длинные транскрипты обрезаются до max_chars, чтобы уложиться в контекст
    модели (для qwen2.5:72b / llama3.3:70b с запасом).
    """
    lines = []
    for seg in segments:
        lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(транскрипт обрезан)"
    return text


def _extract_json(raw: str) -> dict:
    """Достаёт JSON-объект из ответа модели.

    Модель иногда оборачивает ответ в ```json … ``` или добавляет текст —
    поэтому берём подстроку от первой "{" до последней "}".
    """
    # Срезаем markdown-ограждения, если они есть.
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AnalysisError(f"Модель вернула ответ без JSON: {raw[:200]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Не удалось разобрать JSON из ответа модели: {exc}") from exc


def analyze_transcript(
    segments: list[dict],
    ollama_url: str,
    model: str,
    top_n: int = 10,
    progress_callback=None,
) -> list[dict]:
    """Отправляет транскрипт в ollama и возвращает список highlights.

    Параметры:
        segments: список сегментов из transcriber.transcribe().
        ollama_url: базовый URL ollama API (например, http://localhost:11434).
        model: имя модели (qwen2.5:72b, llama3.3:70b и т. п.).
        top_n: сколько моментов запросить.
        progress_callback: функция callback(percent, message) для UI.

    Возвращает список словарей:
        [{"start": float, "end": float, "description": str}, …]

    Бросает AnalysisError при проблемах.
    """
    if progress_callback:
        progress_callback(0.1, "Подготовка транскрипта для анализа…")

    transcript_text = _build_transcript_text(segments)

    user_prompt = (
        f"Выбери топ-{top_n} лучших моментов из этого транскрипта стрима.\n\n"
        f"Транскрипт:\n{transcript_text}"
    )

    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": _SYSTEM_PROMPT,
        "stream": False,
        # Просим ollama вернуть строго JSON.
        "format": "json",
        "options": {"temperature": 0.4},
    }

    if progress_callback:
        progress_callback(0.3, f"Запрос к ollama ({model})…")

    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise AnalysisError(
            f"Не удалось подключиться к ollama по адресу {ollama_url}. "
            f"Проверь, что ollama запущена (`ollama serve`). Детали: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise AnalysisError("Превышено время ожидания ответа ollama.") from exc
    except requests.exceptions.HTTPError as exc:
        raise AnalysisError(f"ollama вернула ошибку HTTP: {exc}") from exc

    if progress_callback:
        progress_callback(0.8, "Разбор ответа модели…")

    try:
        model_output = response.json().get("response", "")
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Некорректный ответ ollama API: {exc}") from exc

    if not model_output:
        raise AnalysisError("ollama вернула пустой ответ.")

    parsed = _extract_json(model_output)
    highlights = parsed.get("highlights")
    if not isinstance(highlights, list) or not highlights:
        raise AnalysisError("В ответе модели нет списка highlights.")

    # Валидируем и нормализуем каждый момент.
    result: list[dict] = []
    for item in highlights:
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            # Пропускаем некорректные элементы, но не падаем целиком.
            continue
        if end <= start:
            continue
        result.append(
            {
                "start": start,
                "end": end,
                "description": str(item.get("description", "")).strip()
                or "Без описания",
            }
        )

    if not result:
        raise AnalysisError("После валидации не осталось ни одного корректного момента.")

    # Сортируем по времени начала для предсказуемого порядка клипов.
    result.sort(key=lambda h: h["start"])

    if progress_callback:
        progress_callback(1.0, f"Анализ завершён: выбрано {len(result)} моментов.")

    return result
