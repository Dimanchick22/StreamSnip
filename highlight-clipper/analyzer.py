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


def _query_ollama(ollama_url: str, model: str, system: str, prompt: str) -> str:
    """Отправляет запрос в ollama /api/generate и возвращает текст ответа.

    Общий хелпер для обоих режимов анализа. Бросает AnalysisError при
    сетевых проблемах или пустом ответе.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",  # просим строго JSON
        "options": {"temperature": 0.4},
    }
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

    try:
        model_output = response.json().get("response", "")
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Некорректный ответ ollama API: {exc}") from exc

    if not model_output:
        raise AnalysisError("ollama вернула пустой ответ.")
    return model_output


def _transcript_window(
    segments: list[dict], start: float, end: float, pad: float = 2.0, max_chars: int = 1500
) -> str:
    """Собирает реплики транскрипта, попадающие в окно [start-pad, end+pad]."""
    parts = []
    for seg in segments:
        if seg["end"] >= start - pad and seg["start"] <= end + pad:
            parts.append(seg["text"])
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text or "(речь не распознана)"


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

    if progress_callback:
        progress_callback(0.3, f"Запрос к ollama ({model})…")

    model_output = _query_ollama(ollama_url, model, _SYSTEM_PROMPT, user_prompt)

    if progress_callback:
        progress_callback(0.8, "Разбор ответа модели…")

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


# Системный промпт для оценки боевых сегментов.
_COMBAT_SYSTEM_PROMPT = """Ты — редактор хайлайтов из игровых стримов. Тебе дают \
список боевых сегментов (их нашла модель компьютерного зрения) с их \
длительностью и фрагментом транскрипта речи стримера во время каждого боя.

Оцени КАЖДЫЙ сегмент по шкале от 0 до 10, учитывая совокупность:
- что происходит в речи стримера (эмоции, крики, реакции, напряжённые \
комментарии — повышают оценку; тишина/скука — понижают);
- длину боя (более длинные и насыщенные бои интереснее коротких).

Для каждого сегмента верни его index, оценку score (0-10), флаг take \
(стоит ли брать) и короткое description на русском (до 100 символов).

Верни ТОЛЬКО валидный JSON без markdown, строго в формате:
{"evaluations": [{"index": 0, "score": 8, "take": true, "description": "текст"}]}"""


def _select_by_score(scored: list[dict], min_score: float) -> list[dict]:
    """Отбор сегментов по оценке без жёсткого лимита.

    - Берём все с оценкой >= min_score.
    - Если таких меньше 3 — берём топ-3 по оценке.
    - Если таких больше 20 — берём топ-20 по оценке.
    """
    by_score = sorted(scored, key=lambda s: s["score"], reverse=True)
    qualified = [s for s in by_score if s["score"] >= min_score]
    if len(qualified) < 3:
        return by_score[:3]
    if len(qualified) > 20:
        return qualified[:20]
    return qualified


def analyze_combat_segments(
    segments: list[dict],
    combat_segments: list[dict],
    ollama_url: str,
    model: str,
    min_score: float = 7.0,
    progress_callback=None,
) -> list[dict]:
    """Оценивает боевые сегменты через LLM и отбирает лучшие.

    Параметры:
        segments: транскрипт (из transcriber.transcribe()).
        combat_segments: боевые сегменты (из combat_segmenter.segment_combat()).
        ollama_url: адрес ollama API.
        model: имя LLM-модели.
        min_score: минимальный порог оценки для отбора (по умолчанию 7).
        progress_callback: функция callback(percent, message) для UI.

    Возвращает список словарей:
        [{"start": float, "end": float, "description": str, "score": float}, …]

    Бросает AnalysisError при проблемах.
    """
    if not combat_segments:
        raise AnalysisError("Список боевых сегментов пуст — нечего оценивать.")

    if progress_callback:
        progress_callback(0.1, "Подготовка боевых сегментов для оценки…")

    # Собираем компактное описание каждого боя с фрагментом транскрипта.
    lines = []
    for idx, seg in enumerate(combat_segments):
        duration = seg["end"] - seg["start"]
        excerpt = _transcript_window(segments, seg["start"], seg["end"])
        lines.append(
            f"index={idx} | время {seg['start']:.0f}-{seg['end']:.0f}с "
            f"| длина {duration:.0f}с | речь: {excerpt}"
        )
    user_prompt = (
        "Оцени каждый боевой сегмент стрима.\n\n"
        "Сегменты:\n" + "\n".join(lines)
    )

    if progress_callback:
        progress_callback(0.3, f"Запрос к ollama ({model})…")

    model_output = _query_ollama(ollama_url, model, _COMBAT_SYSTEM_PROMPT, user_prompt)

    if progress_callback:
        progress_callback(0.8, "Разбор оценок модели…")

    parsed = _extract_json(model_output)
    evaluations = parsed.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise AnalysisError("В ответе модели нет списка evaluations.")

    # Сопоставляем оценки с боевыми сегментами по index.
    eval_by_index: dict[int, dict] = {}
    for item in evaluations:
        try:
            idx = int(item["index"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(combat_segments):
            eval_by_index[idx] = {
                "score": max(0.0, min(10.0, score)),
                "description": str(item.get("description", "")).strip(),
            }

    # Каждому боевому сегменту присваиваем оценку (0, если модель пропустила).
    scored: list[dict] = []
    for idx, seg in enumerate(combat_segments):
        ev = eval_by_index.get(idx, {"score": 0.0, "description": ""})
        scored.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "score": ev["score"],
                "description": ev["description"]
                or f"Бой {seg['start']:.0f}-{seg['end']:.0f}с",
            }
        )

    selected = _select_by_score(scored, min_score)
    if not selected:
        raise AnalysisError("После отбора не осталось ни одного сегмента.")

    # Сортируем по времени начала для предсказуемого порядка клипов.
    selected.sort(key=lambda h: h["start"])

    if progress_callback:
        progress_callback(
            1.0, f"Анализ завершён: отобрано боёв — {len(selected)}."
        )

    return selected
