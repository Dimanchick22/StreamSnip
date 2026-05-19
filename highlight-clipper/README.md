# Highlight Clipper

Автоматическая нарезка highlights из стримов (Twitch, YouTube, Kick).

Пайплайн полностью локальный:

1. **yt-dlp** скачивает VOD по ссылке.
2. **faster-whisper** (CUDA) транскрибирует аудио с таймстампами.
3. **ollama** (локальная LLM) выбирает топ-N лучших моментов.
4. **ffmpeg** нарезает клипы без перекодирования (потоковое копирование).
5. **Gradio** показывает результат и даёт скачать клипы.

## Железо

Рассчитано на RTX 5090 + 96 ГБ RAM, работает на Windows и Linux.
Модели `qwen2.5:72b` / `llama3.3:70b` и `whisper large-v3` помещаются
в видеопамять 5090 (32 ГБ) с запасом.

## Установка

### 1. Python-зависимости

Нужен Python 3.10+.

```bash
cd highlight-clipper
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Для CUDA-ускорения `faster-whisper` нужны библиотеки cuDNN/cuBLAS.
Самый простой способ — поставить их через pip:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Если CUDA недоступна, поставь в `CONFIG` (в `main.py`)
`whisper_device = "cpu"` и `whisper_compute_type = "int8"`.

### 2. ffmpeg

`ffmpeg` должен быть доступен в `PATH`.

- **Windows:** скачай сборку с https://www.gyan.dev/ffmpeg/builds/
  и добавь папку `bin` в переменную окружения `PATH`.
- **Linux (Debian/Ubuntu):** `sudo apt install ffmpeg`
- **Linux (Arch):** `sudo pacman -S ffmpeg`

Проверка: `ffmpeg -version`.

### 3. ollama

1. Установи ollama с https://ollama.com/download
   (есть инсталляторы для Windows и Linux).
2. Запусти сервер (на Windows запускается автоматически после установки):

   ```bash
   ollama serve
   ```

3. Скачай модель для анализа:

   ```bash
   ollama pull qwen2.5:72b
   # или
   ollama pull llama3.3:70b
   ```

По умолчанию ollama слушает `http://localhost:11434`.

## Запуск

```bash
python main.py
```

Gradio откроет веб-интерфейс (по умолчанию http://127.0.0.1:7860).
Вставь ссылку на VOD, нажми **Обработать** и следи за прогресс-баром.

Готовые клипы сохраняются в `./clips/[название VOD]/` и доступны
для скачивания прямо из интерфейса.

## Конфигурация

Все параметры собраны в словаре `CONFIG` в начале `main.py`:

| Параметр | Назначение |
|---|---|
| `ollama_url` | Адрес ollama API |
| `ollama_model` | Модель LLM (`qwen2.5:72b` / `llama3.3:70b`) |
| `top_highlights` | Сколько моментов выбирать |
| `whisper_model` | Размер модели Whisper (`large-v3`, `medium`, …) |
| `whisper_device` | `cuda` или `cpu` |
| `whisper_compute_type` | `float16` (CUDA) или `int8` (CPU) |
| `whisper_language` | Код языка (`ru`/`en`) или `None` для автоопределения |
| `clips_dir` | Папка для готовых клипов |
| `buffer_seconds` | Буфер до/после момента (±15 c) |

## Структура проекта

```
highlight-clipper/
  main.py          # Gradio UI + оркестрация пайплайна
  downloader.py    # Скачивание VOD через yt-dlp
  transcriber.py   # Транскрипция через faster-whisper
  analyzer.py      # Запросы к ollama + парсинг JSON
  clipper.py       # Нарезка клипов через ffmpeg
  requirements.txt
  README.md
```

## Замечания

- Нарезка идёт с `-c copy` (без перекодирования) — это быстро, но границы
  клипа выравниваются по ближайшему ключевому кадру. Для кадрово-точной
  нарезки потребовалось бы перекодирование.
- Длинные транскрипты обрезаются перед отправкой в LLM, чтобы уложиться
  в контекст модели (см. `_build_transcript_text` в `analyzer.py`).
- Временные файлы VOD сохраняются в системную временную папку и не
  удаляются автоматически — при необходимости чисти их вручную.
