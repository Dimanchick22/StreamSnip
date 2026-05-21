"""Детекция боёв в видео через обученную модель combat_detector.pt (ResNet18).

Модуль семплирует кадры видео с фиксированным шагом, прогоняет их через
бинарный классификатор "combat"/"no_combat" и строит из временной шкалы
вероятностей список боевых сегментов с гистерезисом (раздельные пороги
длительности на вход и выход из боя).

Модель загружается один раз (load_model) и переиспользуется между запросами.
Если файл модели не найден — load_model вернёт None, и приложение продолжит
работу в режиме только LLM.
"""

import os

import cv2
import torch
import torchvision
from torch import nn
from torchvision import transforms

# Индекс класса "combat" в выходе модели (для бинарной классификации на 2
# логита принимаем соглашение: 0 = no_combat, 1 = combat).
_COMBAT_CLASS_INDEX = 1

# Стандартная ImageNet-нормализация + ресайз под вход ResNet.
_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class CombatModel:
    """Обёртка над загруженной моделью: хранит сеть, устройство и режим."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: str,
        half: bool,
        combat_index: int = _COMBAT_CLASS_INDEX,
    ):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.half = half
        self.combat_index = combat_index

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> list[float]:
        """Возвращает вероятности боя для батча кадров (тензор N×3×224×224)."""
        batch = batch.to(self.device)
        if self.half:
            batch = batch.half()
        logits = self.model(batch)
        if self.num_classes == 1:
            # Один логит -> сигмоида -> вероятность класса "combat".
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            # Несколько логитов -> softmax -> берём вероятность класса combat.
            probs = torch.softmax(logits, dim=1)[:, self.combat_index]
        return probs.float().cpu().tolist()


def _combat_index_from_classes(classes, num_classes: int) -> int:
    """Определяет индекс класса "combat" по списку имён классов из чекпойнта."""
    if isinstance(classes, (list, tuple)):
        for i, name in enumerate(classes):
            low = name.lower() if isinstance(name, str) else ""
            if "combat" in low and "no" not in low:
                return i
    # Запасной вариант: для 2 классов считаем combat = индекс 1.
    return _COMBAT_CLASS_INDEX if num_classes > 1 else 0


def load_model(
    model_path: str,
    device: str = "cuda",
    compute_type: str = "float16",
) -> CombatModel | None:
    """Загружает combat_detector.pt один раз при старте приложения.

    Поддерживает как сохранённый целиком nn.Module, так и state_dict
    (в т. ч. вложенный в {"state_dict": …} и с префиксом "module.").
    Число выходных классов определяется по форме fc.weight.

    Возвращает CombatModel или None, если файл не найден / не загрузился —
    в этом случае приложение работает в режиме только LLM.
    """
    if not model_path or not os.path.exists(model_path):
        print(f"[combat_segmenter] Модель не найдена: {model_path}. "
              f"Работаем в режиме только LLM.")
        return None

    # На системах без CUDA откатываемся на CPU/float32, чтобы не падать.
    if device == "cuda" and not torch.cuda.is_available():
        print("[combat_segmenter] CUDA недоступна — переключаюсь на CPU.")
        device = "cpu"
    half = compute_type == "float16" and device == "cuda"

    try:
        try:
            # torch>=2.6 по умолчанию weights_only=True (грузит только тензоры).
            checkpoint = torch.load(model_path, map_location=device)
        except Exception:  # noqa: BLE001
            # Откат для полностью сохранённой модели (nn.Module) — файл локальный
            # и доверенный, поэтому разрешаем десериализацию объекта целиком.
            checkpoint = torch.load(
                model_path, map_location=device, weights_only=False
            )

        classes = None
        if isinstance(checkpoint, nn.Module):
            # Сохранена вся модель целиком.
            model = checkpoint
            fc = getattr(model, "fc", None)
            num_classes = fc.out_features if isinstance(fc, nn.Linear) else 2
            combat_index = _combat_index_from_classes(None, num_classes)
        else:
            # Чекпойнт-словарь: достаём сам state_dict из известных ключей.
            state = checkpoint
            if isinstance(checkpoint, dict):
                classes = checkpoint.get("classes")
                for key in ("model_state", "state_dict", "model"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        state = checkpoint[key]
                        break
            # Убираем префикс "module." от DataParallel, если он есть.
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

            fc_weight = state.get("fc.weight")
            num_classes = int(fc_weight.shape[0]) if fc_weight is not None else 2
            combat_index = _combat_index_from_classes(classes, num_classes)

            model = torchvision.models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(state)
    except Exception as exc:  # noqa: BLE001 — не валим UI из-за модели
        print(f"[combat_segmenter] Не удалось загрузить модель: {exc}. "
              f"Работаем в режиме только LLM.")
        return None

    model.eval().to(device)
    if half:
        model.half()

    print(f"[combat_segmenter] Модель загружена: classes={num_classes} "
          f"({classes if classes else 'имена не заданы'}), "
          f"combat_index={combat_index}, device={device}, fp16={half}")
    return CombatModel(model, num_classes, device, half, combat_index)


def _build_probability_timeline(
    video_path: str,
    combat_model: CombatModel,
    sample_step: float,
    batch_size: int,
    progress_callback,
) -> tuple[list[float], list[float], float]:
    """Семплирует кадры с шагом sample_step и возвращает (времена, вероятности, шаг)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise CombatDetectionError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps <= 0:
        fps = 30.0  # запасное значение, если контейнер не сообщил FPS
    duration = frame_count / fps if frame_count > 0 else None

    # Сколько кадров между сэмплами. Читаем поток ПОСЛЕДОВАТЕЛЬНО и пропускаем
    # лишние кадры дешёвым grab() (без декодирования в картинку), а декодируем
    # через retrieve() только нужные. Это на порядок быстрее, чем перематывать
    # видео cap.set(POS_FRAMES) на каждый сэмпл (перемотка декодит весь GOP).
    step_frames = max(1, round(fps * sample_step))

    times: list[float] = []
    probs: list[float] = []
    batch_tensors: list[torch.Tensor] = []
    batch_times: list[float] = []

    def _flush() -> None:
        if not batch_tensors:
            return
        stacked = torch.stack(batch_tensors)
        batch_probs = combat_model.predict_batch(stacked)
        probs.extend(batch_probs)
        times.extend(batch_times)
        batch_tensors.clear()
        batch_times.clear()

    try:
        frame_idx = 0
        while True:
            # grab() продвигает поток на кадр вперёд, не конвертируя его —
            # это дёшево по сравнению с полным read().
            if not cap.grab():
                break

            if frame_idx % step_frames == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                t = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_tensors.append(_TRANSFORM(rgb))
                batch_times.append(t)

                if len(batch_tensors) >= batch_size:
                    _flush()
                    if progress_callback and duration:
                        progress_callback(
                            min(t / duration, 1.0),
                            f"Детекция боёв… {min(t / duration * 100, 100):.0f}%",
                        )
            frame_idx += 1
        _flush()
    finally:
        cap.release()

    # Фактический шаг между сэмплами (может чуть отличаться от запрошенного
    # из-за округления до целого числа кадров) — нужен для точной стейт-машины.
    actual_step = step_frames / fps
    return times, probs, actual_step


def _detect_segments(
    times: list[float],
    probs: list[float],
    step: float,
    threshold: float,
    start_hold: float,
    end_hold: float,
) -> list[dict]:
    """Стейт-машина с гистерезисом: строит сырые боевые сегменты.

    - Бой начинается, когда вероятность >= threshold держится start_hold секунд.
    - Бой заканчивается, когда вероятность < threshold держится end_hold секунд
      (короткие провалы — перекаты/уклонения — игнорируются).
    """
    n = len(probs)
    if n == 0:
        return []

    # Переводим длительности удержания в количество соседних сэмплов.
    start_samples = max(1, round(start_hold / step))
    end_samples = max(1, round(end_hold / step))

    hot = [p >= threshold for p in probs]
    segments: list[dict] = []

    in_combat = False
    seg_start = 0.0
    i = 0
    while i < n:
        if not in_combat:
            if hot[i]:
                # Длина непрерывной "горячей" серии начиная с i.
                j = i
                while j < n and hot[j]:
                    j += 1
                if (j - i) >= start_samples:
                    in_combat = True
                    seg_start = times[i]
                i = j  # короткую серию просто пропускаем
            else:
                i += 1
        else:
            if not hot[i]:
                # Длина непрерывной "холодной" серии начиная с i.
                j = i
                while j < n and not hot[j]:
                    j += 1
                if (j - i) >= end_samples:
                    # Бой закончился в начале холодной серии.
                    segments.append({"start": seg_start, "end": times[i]})
                    in_combat = False
                i = j  # короткий провал игнорируем, бой продолжается
            else:
                i += 1

    # Бой не закрылся до конца видео — закрываем последним сэмплом.
    if in_combat:
        segments.append({"start": seg_start, "end": times[n - 1] + step})

    return segments


def _merge_close(segments: list[dict], max_gap: float) -> list[dict]:
    """Объединяет соседние бои, если пауза между ними меньше max_gap секунд."""
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        if seg["start"] - merged[-1]["end"] < max_gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


class CombatDetectionError(Exception):
    """Ошибка на этапе детекции боёв."""


def segment_combat(
    video_path: str,
    combat_model: CombatModel,
    sample_step: float = 0.5,
    threshold: float = 0.7,
    start_hold: float = 2.0,
    end_hold: float = 4.0,
    min_length: float = 8.0,
    merge_gap: float = 6.0,
    batch_size: int = 64,
    progress_callback=None,
) -> list[dict]:
    """Находит боевые сегменты в видео.

    Параметры:
        video_path: путь к видеофайлу.
        combat_model: загруженная модель (из load_model).
        sample_step: шаг семплирования кадров, секунды (по умолчанию 0.5).
        threshold: порог вероятности боя (0.7).
        start_hold: сколько секунд подряд держать порог, чтобы начать бой (2с).
        end_hold: сколько секунд подряд быть ниже порога, чтобы завершить бой (4с).
        min_length: минимальная длина боя, секунды (8с).
        merge_gap: объединять бои с паузой меньше этого значения (6с).
        batch_size: размер батча для инференса.
        progress_callback: функция callback(percent, message) для UI.

    Возвращает список сегментов: [{"start": float, "end": float}, …].
    Бросает CombatDetectionError при проблемах с видео.
    """
    if combat_model is None:
        raise CombatDetectionError("Модель детекции боя не загружена.")

    if progress_callback:
        progress_callback(0.0, "Сканирование кадров для детекции боёв…")

    times, probs, step = _build_probability_timeline(
        video_path, combat_model, sample_step, batch_size, progress_callback
    )

    # Сырые сегменты по гистерезису.
    segments = _detect_segments(
        times, probs, step, threshold, start_hold, end_hold
    )
    # Сначала объединяем близкие бои, затем отсекаем слишком короткие:
    # так два близких коротких всплеска корректно сольются в один длинный бой.
    segments = _merge_close(segments, merge_gap)
    segments = [s for s in segments if (s["end"] - s["start"]) >= min_length]

    if progress_callback:
        progress_callback(1.0, f"Детекция завершена: найдено боёв — {len(segments)}.")

    return segments
