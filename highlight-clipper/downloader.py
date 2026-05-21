"""Логика скачивания VOD через yt-dlp.

Поддерживаются Twitch, YouTube и Kick. Видео кэшируется в постоянную папку
по id ролика: если файл уже скачан, повторная загрузка не выполняется —
это спасает от перекачивания после ошибки на следующих этапах пайплайна.
"""

import glob
import os
import re

import yt_dlp


class DownloadError(Exception):
    """Ошибка на этапе скачивания VOD."""


def _sanitize_filename(name: str) -> str:
    """Убирает из названия символы, недопустимые в именах папок/файлов."""
    # Заменяем всё, что не буква/цифра/пробел/дефис/подчёркивание, на "_"
    cleaned = re.sub(r"[^\w\s-]", "_", name, flags=re.UNICODE)
    # Схлопываем повторяющиеся пробелы и подчёркивания
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:120] or "vod"


def _find_cached_file(cache_dir: str, video_id: str) -> str | None:
    """Ищет уже скачанный файл VOD по его id (исключая недокачанные .part)."""
    matches = [
        path
        for path in glob.glob(os.path.join(cache_dir, f"{video_id}.*"))
        if not path.endswith(".part") and os.path.isfile(path)
    ]
    return matches[0] if matches else None


def download_vod(url: str, cache_dir: str = "./vod_cache", progress_callback=None) -> dict:
    """Скачивает VOD по ссылке (с кэшированием).

    Параметры:
        url: ссылка на стрим/VOD (Twitch, YouTube, Kick).
        cache_dir: папка для постоянного кэша скачанных VOD.
        progress_callback: необязательная функция вида callback(percent, message)
            для обновления прогресс-бара в UI.

    Возвращает словарь:
        {
            "video_path": путь к скачанному файлу,
            "title": очищенное название VOD,
            "duration": длительность в секундах (или None),
            "from_cache": True, если файл взят из кэша без скачивания,
        }

    Бросает DownloadError при любой проблеме.
    """
    if not url or not url.strip():
        raise DownloadError("Не указана ссылка на VOD.")

    url = url.strip()
    os.makedirs(cache_dir, exist_ok=True)

    # Сначала забираем метаданные без скачивания, чтобы узнать id ролика
    # и проверить кэш до начала тяжёлой загрузки.
    meta_opts = {"noplaylist": True, "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"yt-dlp не смог получить данные о VOD: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — ловим всё, чтобы не уронить UI
        raise DownloadError(f"Неожиданная ошибка при чтении метаданных: {exc}") from exc

    video_id = info.get("id") or "vod"
    title = _sanitize_filename(info.get("title") or video_id)
    duration = info.get("duration")

    # Если этот VOD уже скачан — переиспользуем файл, не качаем заново.
    cached = _find_cached_file(cache_dir, video_id)
    if cached:
        if progress_callback:
            progress_callback(1.0, "VOD найден в кэше — скачивание пропущено.")
        return {
            "video_path": cached,
            "title": title,
            "duration": duration,
            "from_cache": True,
        }

    # Хук прогресса yt-dlp -> прокидываем процент в UI.
    def _progress_hook(status: dict) -> None:
        if progress_callback is None:
            return
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes", 0)
            if total:
                percent = downloaded / total
                progress_callback(percent, f"Скачивание VOD… {percent * 100:.0f}%")
        elif status.get("status") == "finished":
            progress_callback(1.0, "Скачивание завершено, обработка файла…")

    ydl_opts = {
        # Лучшее качество, но в одном файле (нужен для последующей нарезки).
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(cache_dir, "%(id)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"yt-dlp не смог скачать VOD: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — ловим всё, чтобы не уронить UI
        raise DownloadError(f"Неожиданная ошибка при скачивании: {exc}") from exc

    # yt-dlp мог сохранить файл с другим расширением после постобработки.
    if not os.path.exists(video_path):
        found = _find_cached_file(cache_dir, video_id)
        if not found:
            raise DownloadError("Файл скачан, но не найден на диске.")
        video_path = found

    return {
        "video_path": video_path,
        "title": title,
        "duration": duration,
        "from_cache": False,
    }
