"""Логика скачивания VOD через yt-dlp.

Поддерживаются Twitch, YouTube и Kick. Видео сохраняется во временную
папку, откуда его дальше забирают транскрайбер и нарезчик.
"""

import os
import re
import tempfile

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


def download_vod(url: str, progress_callback=None) -> dict:
    """Скачивает VOD по ссылке.

    Параметры:
        url: ссылка на стрим/VOD (Twitch, YouTube, Kick).
        progress_callback: необязательная функция вида callback(percent, message)
            для обновления прогресс-бара в UI.

    Возвращает словарь:
        {
            "video_path": путь к скачанному файлу,
            "title": очищенное название VOD,
            "duration": длительность в секундах (или None),
        }

    Бросает DownloadError при любой проблеме.
    """
    if not url or not url.strip():
        raise DownloadError("Не указана ссылка на VOD.")

    url = url.strip()

    # Временная папка для скачанного видео — чистится ОС/пользователем позже.
    temp_dir = tempfile.mkdtemp(prefix="highlight_clipper_")

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
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Скачиваем и сразу получаем метаданные.
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"yt-dlp не смог скачать VOD: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — ловим всё, чтобы не уронить UI
        raise DownloadError(f"Неожиданная ошибка при скачивании: {exc}") from exc

    # yt-dlp мог сохранить файл с другим расширением после постобработки.
    if not os.path.exists(video_path):
        base = os.path.splitext(video_path)[0]
        candidates = [
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.startswith(os.path.basename(base))
        ]
        if not candidates:
            raise DownloadError("Файл скачан, но не найден на диске.")
        video_path = candidates[0]

    title = _sanitize_filename(info.get("title") or info.get("id") or "vod")

    return {
        "video_path": video_path,
        "title": title,
        "duration": info.get("duration"),
    }
