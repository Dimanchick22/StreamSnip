"""Нарезка клипов через ffmpeg без перекодирования.

Каждый highlight вырезается из исходного видео потоковым копированием
(-c copy), что почти мгновенно и не теряет качество.
"""

import os
import re
import shutil
import subprocess


class ClippingError(Exception):
    """Ошибка на этапе нарезки клипов."""


def _check_ffmpeg() -> None:
    """Проверяет, что ffmpeg доступен в PATH."""
    if shutil.which("ffmpeg") is None:
        raise ClippingError(
            "ffmpeg не найден в PATH. Установи ffmpeg (см. README) и перезапусти."
        )


def _safe_name(text: str, index: int) -> str:
    """Формирует безопасное имя файла клипа из описания и порядкового номера."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned[:60] or "clip"
    return f"{index:02d}_{cleaned}.mp4"


def cut_clips(
    video_path: str,
    highlights: list[dict],
    output_dir: str,
    vod_title: str,
    buffer_seconds: float = 15.0,
    video_duration: float | None = None,
    progress_callback=None,
) -> list[dict]:
    """Нарезает клипы по таймстампам highlights.

    Параметры:
        video_path: путь к исходному видеофайлу.
        highlights: список словарей с ключами start/end/description.
        output_dir: корневая папка для клипов (например, ./clips).
        vod_title: название VOD — станет именем подпапки.
        buffer_seconds: буфер до/после момента (±15 c по умолчанию).
        video_duration: длительность видео в секундах для обрезки буфера.
        progress_callback: функция callback(percent, message) для UI.

    Возвращает список словарей:
        [{"path": str, "description": str, "start": float, "end": float}, …]

    Бросает ClippingError, если не удалось нарезать ни одного клипа.
    """
    _check_ffmpeg()

    if not os.path.exists(video_path):
        raise ClippingError(f"Исходное видео не найдено: {video_path}")

    # Папка вида ./clips/[название VOD]/
    target_dir = os.path.join(output_dir, vod_title)
    os.makedirs(target_dir, exist_ok=True)

    clips: list[dict] = []
    total = len(highlights)

    for index, hl in enumerate(highlights, start=1):
        # Расширяем границы момента буфером, не выходя за пределы видео.
        start = max(0.0, hl["start"] - buffer_seconds)
        end = hl["end"] + buffer_seconds
        if video_duration:
            end = min(end, video_duration)
        duration = end - start
        if duration <= 0:
            continue

        out_path = os.path.join(target_dir, _safe_name(hl["description"], index))

        # -ss до -i + -c copy: быстрый seek и копирование потоков без перекодирования.
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            # Сдвигаем таймстампы клипа к нулю, чтобы он корректно воспроизводился.
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                # Явно декодируем вывод ffmpeg как UTF-8 с заменой битых байт,
                # иначе на Windows используется локаль (cp1251) и чтение падает.
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            # Один проблемный клип не должен ронять весь процесс.
            continue

        if result.returncode != 0 or not os.path.exists(out_path):
            # Пропускаем неудачный клип, остальные продолжаем нарезать.
            continue

        clips.append(
            {
                "path": out_path,
                "description": hl["description"],
                "start": start,
                "end": end,
            }
        )

        if progress_callback:
            percent = index / total
            progress_callback(percent, f"Нарезка клипов… {index}/{total}")

    if not clips:
        raise ClippingError(
            "Не удалось нарезать ни одного клипа. Проверь видеофайл и ffmpeg."
        )

    return clips
