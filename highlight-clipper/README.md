# Highlight Clipper

Автоматическая нарезка highlights из стримов (Twitch, YouTube, Kick).

Пайплайн полностью локальный:

1. **yt-dlp** скачивает VOD по ссылке.
2. **faster-whisper** (CUDA) транскрибирует аудио с таймстампами.
3. **ollama** (локальная LLM) выбирает топ-N лучших моментов.
4. **ffmpeg** нарезает клипы без перекодирования (потоковое копирование).
5. **Gradio** показывает результат и даёт скачать клипы.

## Железо

Рассчитано на RTX 5090 (32 ГБ VRAM) + 96 ГБ RAM, работает на Windows и Linux.

**Выбор моделей под это железо:**

- **Транскрипция:** по умолчанию `large-v3-turbo` — дистилляция `large-v3`
  (≈4-8x быстрее при почти том же качестве). Если важна максимальная
  точность — выбери `large-v3` прямо в интерфейсе. Для черновой быстрой
  расшифровки есть `tiny`/`base`/`small`/`medium`.
- **LLM:** по умолчанию `qwen2.5:32b` — при Q4 (~20 ГБ) полностью влезает
  в 32 ГБ VRAM и работает быстро. Модели `qwen2.5:72b` / `llama3.3:70b`
  точнее, но при Q4 (~40-47 ГБ) частично уходят в offload на RAM и работают
  медленнее (96 ГБ RAM это позволяют — меняй `ollama_model` в `CONFIG`).

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
   ollama pull qwen2.5:32b      # по умолчанию, влезает в 32 ГБ VRAM
   # или, если нужна максимальная точность (медленнее из-за offload):
   ollama pull qwen2.5:72b
   ollama pull llama3.3:70b
   ```

По умолчанию ollama слушает `http://localhost:11434`.

## Запуск

```bash
python main.py
```

Gradio откроет веб-интерфейс (по умолчанию http://127.0.0.1:7860).
Вставь ссылку на VOD, выбери модель транскрипции, нажми **Обработать**
и следи за прогресс-баром.

Готовые клипы сохраняются в `./clips/[название VOD]/` и доступны
для скачивания прямо из интерфейса.

**Кэш VOD:** скачанные видео складываются в `./vod_cache/` по id ролика.
Если на этапе транскрипции/анализа/нарезки произойдёт ошибка, повторный
запуск той же ссылки не будет качать VOD заново — файл берётся из кэша.

## Конфигурация

Все параметры собраны в словаре `CONFIG` в начале `main.py`:

| Параметр | Назначение |
|---|---|
| `ollama_url` | Адрес ollama API |
| `ollama_model` | Модель LLM (`qwen2.5:32b` / `qwen2.5:72b` / `llama3.3:70b`) |
| `top_highlights` | Сколько моментов выбирать |
| `whisper_model` | Модель Whisper по умолчанию (можно менять в UI) |
| `whisper_device` | `cuda` или `cpu` |
| `whisper_compute_type` | `float16` (CUDA) или `int8` (CPU) |
| `whisper_language` | Код языка (`ru`/`en`) или `None` для автоопределения |
| `cache_dir` | Папка кэша скачанных VOD |
| `clips_dir` | Папка для готовых клипов |
| `buffer_seconds` | Буфер до/после момента (±15 c) |

Модель транскрипции выбирается прямо в интерфейсе из выпадающего списка
(`tiny` → `base` → `small` → `medium` → `large-v3-turbo` → `large-v3`).

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
- Скачанные VOD кэшируются в `./vod_cache/` и не удаляются автоматически —
  при необходимости чисти папку вручную.
