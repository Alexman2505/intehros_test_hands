import os
import io
import uuid
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
from flask import Flask, request, render_template, url_for

# import cv2

"""
 ============================================
 НАСТРОЙКА ЛОГИРОВАНИЯ
 ============================================
"""

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/alexman2505/flask_hand_app/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

"""
 ============================================
 ЗАГРУЗКА ONNX МОДЕЛИ
 ============================================
 ONNX — это формат модели, который работает быстрее и легче, чем PyTorch
 Модель загружается ОДИН РАЗ при старте приложения и живёт в памяти,
 чтобы каждый раз не тратить время на загрузку при новом запросе
"""

# MODEL_PATH = 'best.onnx'
MODEL_PATH = '/home/alexman2505/flask_hand_app/best.onnx'
logger.info(f"Загружаю ONNX модель из {MODEL_PATH}")

try:
    session = ort.InferenceSession(
        MODEL_PATH, providers=['CPUExecutionProvider']
    )
    logger.info("Модель успешно загружена")

    # Получаем информацию о входах и выходах модели — это нужно для отладки
    # Если форма не совпадает с ожидаемой, модель не сможет обработать картинку
    input_name = session.get_inputs()[0].name  # "images"
    input_shape = session.get_inputs()[0].shape  # [1, 3, 640, 640]
    logger.info(f"Вход модели: {input_name}, форма: {input_shape}")

    output_name = session.get_outputs()[0].name  # "output0"
    output_shape = session.get_outputs()[0].shape  # [1, 37, 8400]
    logger.info(f"Выход модели: {output_name}, форма: {output_shape}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

CLASS_NAME = 'hands'

"""
 ============================================
 МЕТРИКИ МОДЕЛИ
 ============================================
 Эти метрики получены при обучении и показывают качество модели:
 - Precision: точность (сколько из найденных объектов действительно хвосты)
 - Recall: полнота (сколько реальных хвостов нашла модель)
 - mAP50: средняя точность при IoU=0.5
 - mAP50-95: главная метрика, средняя точность при разных порогах
 - F1-score: гармоническое среднее Precision и Recall


}

 ============================================
 ФУНКЦИИ ДЛЯ ОБРАБОТКИ СЕГМЕНТАЦИИ
 ============================================
"""


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Изменяет размер с сохранением пропорций и добавляет паддинг.

    Зачем: YOLO ожидает на входе квадратное изображение 640x640.
    Но пользователь может загрузить фото любого размера (например, 1920x1080).
    Мы не можем просто сжать его до квадрата, потому что пропорции исказятся.

    Решение: масштабируем так, чтобы бОльшая сторона стала 640,
    а недостающее пространство заполняем серым цветом (паддинг).

    Параметры:
        im: входное изображение (numpy array)
        new_shape: целевой размер (640, 640)
        color: цвет паддинга (серый, чтобы не отвлекать)

    Возвращает:
        im_resized: изображение с паддингом (640x640)
        ratio: коэффициент масштабирования (нужен позже, чтобы вернуть координаты)
        (left, top): отступы слева и сверху (нужны, чтобы убрать паддинг)
    """
    shape = im.shape[:2]  # (height, width) исходного изображения
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Вычисляем коэффициент масштабирования так, чтобы изображение вписалось в квадрат
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Новый размер после масштабирования (без паддинга)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # Вычисляем отступы слева/справа и сверху/снизу
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (
        new_shape[0] - new_unpad[1]
    ) / 2

    # Создаём пустое изображение с паддингом (залитое серым)
    im_resized = np.zeros(
        (new_shape[0], new_shape[1], 3), dtype=im.dtype
    ) + np.array(color, dtype=im.dtype)

    # Масштабируем исходное изображение
    im_pil = Image.fromarray(im)
    im_small_pil = im_pil.resize(new_unpad, Image.Resampling.BILINEAR)
    im_small = np.array(im_small_pil)

    # Вставляем масштабированное изображение в центр серого фона
    top, left = int(round(dh)), int(round(dw))
    im_resized[top : top + new_unpad[1], left : left + new_unpad[0]] = im_small

    return im_resized, r, (left, top)


def xywh2xyxy(x):
    """
    Преобразует координаты из формата YOLO в формат для рисования.

    YOLO возвращает координаты в виде [x_center, y_center, width, height] (центр + размер).
    Для рисования прямоугольника нужны координаты левого верхнего и правого нижнего углов.

    Формулы:
        x1 = x_center - width/2
        y1 = y_center - height/2
        x2 = x_center + width/2
        y2 = y_center + height/2

    Пояснение для тех, кто дочитал досюда.
    Это не классический срез питона. Это двумерная матрица numpy. Пример

         Было (x):
         [
           [cx1, cy1, w1, h1],  # строка 0
           [cx2, cy2, w2, h2],  # строка 1
           [cx3, cy3, w3, h3]   # строка 2
         ]

        x[:, 0]  # → [cx1, cx2, cx3]  (все x_center)
        x[:, 2]  # → [w1,  w2,  w3]   (все ширины)

         После вычислений в первой колонке y будут:
        y[:, 0] = [cx1 - w1/2, cx2 - w2/2, cx3 - w3/2]  # все x1
    """
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def compute_iou(box1, box2):
    """
    Вычисляет Intersection over Union (IoU) между двумя bounding box'ами.

    IoU = площадь_пересечения / площадь_объединения

    Зачем: если два бокса сильно перекрываются (IoU большой), значит,
    они скорее всего нашли один и тот же объект. Нужно оставить только один.

    Значения:
        1.0 — идеальное совпадение (один и тот же объект)
        0.0 — полное отсутствие пересечения (разные объекты)
    """
    # Координаты области пересечения
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # Площади боксов
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def nms(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression (NMS) — подавление немаксимумов.

    ЧТО ДЕЛАЕТ:
    Удаляет дублирующиеся детекции одного и того же объекта.

    ЗАЧЕМ:
    YOLO генерирует множество предсказаний для одного объекта:
    - Немного разные координаты
    - Немного разная уверенность
    - Все они указывают на одно и то же место

    БЕЗ NMS: на картинке будет 50 боксов вокруг одного хвоста
    С NMS: остаётся только лучший бокс для каждого хвоста

    АЛГОРИТМ (пошагово):
    1. Берём все боксы, которые модель нашла
    2. Сортируем их по уверенности (самые уверенные — первые)
    3. Выбираем самый уверенный бокс — ОН ТОЧНО ХОРОШИЙ
    4. Удаляем все боксы, которые сильно перекрываются с ним
       (потому что они скорее всего про тот же объект)
    5. С оставшимися повторяем с шага 3

    Параметры:
        boxes: массив координат боксов в формате [[x1,y1,x2,y2], ...]
        scores: массив уверенностей [0.95, 0.87, 0.45, ...]
        iou_threshold: порог перекрытия (0.45 = 45%)
                      Если два бокса перекрываются больше чем на 45%,
                      считаем, что это один и тот же объект

    Возвращает:
        list: индексы боксов, которые нужно оставить
    """

    # Если боксов нет — возвращаем пустой список
    if len(boxes) == 0:
        return []

    """
     ===== ШАГ 1: СОРТИРОВКА ПО УВЕРЕННОСТИ =====
     np.argsort(scores) возвращает индексы в порядке ВОЗРАСТАНИЯ уверенности
     [::-1] переворачивает список, получаем УБЫВАНИЕ (самые уверенные первые)
     Пример: scores = [0.5, 0.9, 0.7]
     argsort -> [0, 1, 2] (индексы по возрастанию)
     [::-1]  -> [2, 1, 0] (индексы по убыванию)
    """
    indices = np.argsort(scores)[::-1]

    # Здесь будем хранить индексы боксов, которые ОСТАВЛЯЕМ
    keep = []

    """
    ===== ШАГ 2: ЦИКЛИЧЕСКИ ВЫБИРАЕМ ЛУЧШИЕ =====
    """
    # Пока есть необработанные боксы
    while len(indices) > 0:
        # Берём первый (самый уверенный) бокс
        current = indices[0]
        keep.append(current)  # Этот бокс — хороший, оставляем

        # Если это был последний бокс — выходим
        if len(indices) == 1:
            break

        # ===== ШАГ 3: УДАЛЯЕМ ПЕРЕКРЫВАЮЩИЕСЯ БОКСЫ =====
        # Остальные боксы (без первого)
        rest = indices[1:]

        """
        Считаем IoU текущего бокса со всеми остальными
         compute_iou(box1, box2) возвращает число от 0 до 1
         0 — нет пересечения, 1 — идеальное совпадение
        """
        ious = np.array([compute_iou(boxes[current], boxes[i]) for i in rest])

        """
         ===== ШАГ 4: ФИЛЬТРАЦИЯ =====
         Оставляем только те боксы, у которых IoU МЕНЬШЕ порога
         ious < iou_threshold создаёт булевый массив [True, False, True, ...]
         rest[ious < iou_threshold] оставляет только индексы с True
         Пример:
           ious = [0.8, 0.2, 0.9, 0.1]
           ious < 0.45 -> [False, True, False, True]
           rest = [1, 3, 5, 7] (индексы)
           Результат: [3, 7] (оставили только те, что слабо перекрываются)
        """
        indices = rest[ious < iou_threshold]

        # Повторяем цикл с оставшимися боксами

    # Возвращаем индексы боксов, которые нужно оставить
    return keep


def process_yolo_output(
    output,
    prototypes=None,
    img_shape=640,
    conf_thres=0.25,
    iou_threshold=0.45,
    pad=None,
    ratio=None,
    orig_size=None,
):
    """
    Обрабатывает сырой выход модели YOLO и преобразует в список детекций с точками контура.

    Модель YOLO для сегментации возвращает тензор размером [1, 37, 8400], где:
        - 1: размер батча (всегда 1, т.к. обрабатываем одно изображение)
        - 37: количество параметров для каждого якоря:
            * 0-3: координаты [x_center, y_center, width, height]
            * 4: уверенность (confidence score)
            * 5-36: коэффициенты для масок (32 шт.)
        - 8400: количество якорей (точек, где модель искала объекты).

    Алгоритм работы:
        1. Транспонирует выход в удобный формат [8400, 37]
        2. Отфильтровывает якоря с низкой уверенностью (conf_thres)
        3. Конвертирует координаты из формата [x_center, y_center, width, height]
           в формат [x1, y1, x2, y2]
        4. Если координаты нормализованы (0..1), масштабирует к img_shape
        5. Применяет NMS для удаления дублирующихся детекций
        6. Если переданы prototypes, генерирует точки контура для каждого объекта
        7. Формирует список словарей с результатами

    Аргументы:
        output (numpy.ndarray): Сырой выход модели формата (1, 37, 8400)
        prototypes (numpy.ndarray, optional): Прототипы масок формата (1, 32, 160, 160)
        img_shape (int): Размер изображения, к которому нужно масштабировать координаты
        conf_thres (float): Порог уверенности для фильтрации детекций
        iou_threshold (float): Порог IoU для NMS
        pad (tuple, optional): Паддинг (pad_w, pad_h) для обратного масштабирования
        ratio (float, optional): Коэффициент масштабирования
        orig_size (tuple, optional): Оригинальный размер (orig_w, orig_h)

    Возвращает:
        list: Список словарей, каждый словарь содержит:
            - 'bbox' (list): Координаты бокса [x1, y1, x2, y2] в масштабе 640
            - 'score' (float): Уверенность модели (от 0 до 1)
            - 'class_id' (int): Идентификатор класса (всегда 0)
            - 'contour_points' (list, optional): Список точек контура [(x1,y1), (x2,y2), ...]
              в координатах оригинального изображения (если переданы prototypes)
    """
    out = output[0].T  # (8400, 37)

    if out.size == 0:
        return []

    boxes_xywh = out[:, :4]  # координаты
    confidences = out[:, 4]  # уверенность
    mask_coeffs = (
        out[:, 5:] if prototypes is not None else None
    )  # коэффициенты масок

    # Фильтр по уверенности
    mask = confidences >= conf_thres
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    if mask_coeffs is not None:
        mask_coeffs = mask_coeffs[mask]

    # Конвертируем координаты в формат xyxy
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # Если координаты нормализованы (0..1), масштабируем к img_shape
    if boxes_xyxy.max() <= 1.0:
        boxes_xyxy = boxes_xyxy * img_shape

    # Удаляем дубликаты через NMS
    keep_indices = nms(boxes_xyxy, confidences, iou_threshold)

    # Формируем итоговый список детекций
    final = []

    # Если есть прототипы, генерируем контуры
    if (
        prototypes is not None
        and mask_coeffs is not None
        and pad
        and ratio
        and orig_size
    ):
        proto = prototypes[0]  # (32, 160, 160)
        orig_w, orig_h = orig_size
        pad_w, pad_h = pad

        for idx in keep_indices:
            # Коэффициенты для этого объекта
            coeffs = mask_coeffs[idx]

            # Генерируем маску
            mask = np.matmul(coeffs, proto.reshape(32, -1)).reshape(160, 160)
            mask = 1.0 / (1.0 + np.exp(-mask))
            mask = (mask > 0.5).astype(np.uint8) * 255

            # Масштабируем маску
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize((640, 640), Image.NEAREST)
            mask_np = np.array(mask_img)

            # Получаем координаты bounding box
            x1, y1, x2, y2 = boxes_xyxy[idx]

            # Убираем паддинг
            x1_orig = (x1 - pad_w) / ratio
            x2_orig = (x2 - pad_w) / ratio
            y1_orig = (y1 - pad_h) / ratio
            y2_orig = (y2 - pad_h) / ratio

            x1_orig = max(0, min(orig_w, int(round(x1_orig))))
            x2_orig = max(0, min(orig_w, int(round(x2_orig))))
            y1_orig = max(0, min(orig_h, int(round(y1_orig))))
            y2_orig = max(0, min(orig_h, int(round(y2_orig))))

            # Вырезаем маску по bounding box
            mask_cropped = mask_np[
                int(y1_orig * 640 / orig_h) : int(y2_orig * 640 / orig_h),
                int(x1_orig * 640 / orig_w) : int(x2_orig * 640 / orig_w),
            ]

            # Извлекаем точки контура
            contour_points = []
            if mask_cropped.size > 0:
                mask_cropped_img = Image.fromarray(mask_cropped)
                mask_cropped_img = mask_cropped_img.resize(
                    (x2_orig - x1_orig, y2_orig - y1_orig), Image.NEAREST
                )
                mask_cropped = np.array(mask_cropped_img)

                # Находим граничные точки
                h, w = mask_cropped.shape
                boundary = []
                for y in range(h):
                    for x in range(w):
                        if mask_cropped[y, x] > 0:
                            is_boundary = False
                            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ny, nx = y + dy, x + dx
                                if (
                                    ny < 0
                                    or ny >= h
                                    or nx < 0
                                    or nx >= w
                                    or mask_cropped[ny, nx] == 0
                                ):
                                    is_boundary = True
                                    break
                            if is_boundary:
                                boundary.append((x1_orig + x, y1_orig + y))

                # Упрощаем: берём каждый 10-й пиксель для читаемости
                if len(boundary) > 30:
                    contour_points = [
                        boundary[i] for i in range(0, len(boundary), 10)
                    ]
                else:
                    contour_points = boundary

            final.append(
                {
                    "bbox": boxes_xyxy[idx].tolist(),
                    "score": float(confidences[idx]),
                    "class_id": 0,
                    "contour_points": contour_points[
                        :100
                    ],  # ограничиваем до 100 точек
                }
            )
    else:
        # Если нет прототипов, возвращаем только боксы
        for idx in keep_indices:
            final.append(
                {
                    "bbox": boxes_xyxy[idx].tolist(),
                    "score": float(confidences[idx]),
                    "class_id": 0,
                    "contour_points": [],
                }
            )

    return final


def run_inference(image_bytes, conf_thres=0.5):
    """
    ГЛАВНАЯ ФУНКЦИЯ ПАЙПЛАЙНА — выполняет полный цикл обработки изображения через YOLO ONNX модель.

    Функция принимает сырые байты изображения, прогоняет его через модель сегментации,
    обрабатывает результаты и возвращает аннотированное изображение с нарисованными
    bounding box'ами и координатами углов, а также количество найденных объектов.

    АРХИТЕКТУРА ПАЙПЛАЙНА:
    ========================
    1. ЗАГРУЗКА ИЗОБРАЖЕНИЯ
       └─ Преобразование байтов → PIL Image → numpy array

    2. ПРЕДОБРАБОТКА (letterbox)
       ├─ Масштабирование с сохранением пропорций до 640×640
       ├─ Добавление серого паддинга (чтобы получить квадрат)
       ├─ Преобразование RGB → BGR (требование модели)
       ├─ Изменение порядка размерностей HWC → CHW (Height, Width, Channels → Channels, Height, Width)
       └─ Нормализация пикселей [0, 255] → [0, 1]

    3. ИНФЕРЕНС (ONNX Runtime)
       ├─ Передача подготовленного тензора в модель
       └─ Получение двух выходов:
           └─ output0: детекции + коэффициенты масок [1, 37, 8400]
           └─ output1: прототипы масок [1, 32, 160, 160] (в текущей версии не используются)

    4. ОБРАБОТКА ДЕТЕКЦИЙ (process_yolo_output)
       ├─ Транспонирование в формат [8400, 37]
       ├─ Фильтрация по порогу уверенности (conf_thres)
       ├─ Конвертация координат xywh → xyxy
       ├─ Масштабирование координат к размеру 640
       ├─ Применение NMS (Non-Maximum Suppression) для удаления дубликатов
       └─ Формирование списка словарей с детекциями

    5. ОТРИСОВКА РЕЗУЛЬТАТОВ
       ├─ Для каждого найденного объекта:
           ├─ Убираем паддинг и масштабируем координаты к оригинальному размеру
           ├─ Рисуем красный прямоугольник
           ├─ Подписываем координаты всех четырёх углов (синим)
           ├─ Добавляем подпись с вероятностью P(tail) = X.XX (красным на белом фоне)
       └─ Возвращаем итоговое изображение

    Аргументы:
        image_bytes (bytes): Сырые байты загруженного изображения.
                            Может быть получен из request.files['image'].read().
        conf_thres (float, optional): Порог уверенности для фильтрации детекций.
                                      Объекты с уверенностью ниже этого значения не будут показаны.
                                      По умолчанию: 0.5 (50%).
                                      Влияет на:
                                      - Количество найденных объектов
                                      - Точность/полноту (precision/recall)
                                      Рекомендации:
                                          - 0.3 — много объектов, но возможны ложные срабатывания
                                          - 0.5 — баланс (по умолчанию)
                                          - 0.7 — только очень уверенные объекты

    Возвращает:
        tuple: (image, detections_data) где:
            - image (PIL.Image): аннотированное изображение (только прямоугольник)
            - detections_data (list): список словарей с данными о каждом найденном объекте
              [{
                  'coords': (x1,y1,x2,y2),
                  'score': 0.95,
                  'label': 'hand',
                  'contour_points': [(x1,y1), (x2,y2), ...]  # список точек контура
              }, ...]

    Исключения:
        Не выбрасывает исключений напрямую, но логирует ошибки через logger.
        В случае критических ошибок (например, модель вернула только один выход)
        возвращает исходное изображение и 0.

    Пример использования:
        >>> with open('hand.jpg', 'rb') as f:
        ...     image_bytes = f.read()
        >>> result_image, count = run_inference(image_bytes, conf_thres=0.5)
        >>> print(f"Найдено хвостов: {count}")
        >>> result_image.save('annotated_hand.jpg')

    Примечания:
        - Модель обучена только на одном классе ('hand').
        - Функция ожидает, что модель сегментации возвращает 2 выхода.
        - Все промежуточные шаги логируются с уровнем INFO для отладки.
        - Шрифты: попытка загрузить Arial, при неудаче — шрифт по умолчанию.
        - Координаты углов выводятся с небольшим смещением, чтобы не налезать на линии.

    Зависимости:
        - PIL (Image, ImageDraw, ImageFont)
        - numpy
        - onnxruntime
        - Вспомогательные функции: letterbox, process_yolo_output, nms

    Связанные функции:
        - letterbox: предобработка изображения
        - process_yolo_output: обработка детекций
        - nms: подавление немаксимумов
        - xywh2xyxy: конвертация координат
        - compute_iou: вычисление пересечения боксов (используется в nms)
    """

    logger.info("=" * 50)
    logger.info("НАЧАЛО ИНФЕРЕНСА (ДЕТЕКЦИЯ)")
    logger.info("=" * 50)

    # ===== ШАГ 1: Загружаем изображение =====
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size
    logger.info(f"Размер изображения: {orig_w} x {orig_h}")
    img_np = np.array(image)

    # ===== ШАГ 2: Предобработка =====
    img_pad, ratio, (pad_w, pad_h) = letterbox(img_np, new_shape=(640, 640))
    img_input = img_pad[:, :, ::-1].transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

    # ===== ШАГ 3: Инференс =====
    outputs = session.run(None, {input_name: img_input})

    if len(outputs) < 2:
        logger.error(
            "Модель вернула только один выход! Нужны 2 для сегментации"
        )
        return image, []

    detections_out = outputs[0]
    prototypes = outputs[1]

    logger.info(f"Shape детекций: {detections_out.shape}")
    logger.info(f"Shape прототипов: {prototypes.shape}")

    # ===== ШАГ 4: Обработка детекций через process_yolo_output =====
    detections_raw = process_yolo_output(
        detections_out,
        prototypes=prototypes,
        conf_thres=conf_thres,
        iou_threshold=0.45,
        pad=(pad_w, pad_h),
        ratio=ratio,
        orig_size=(orig_w, orig_h),
    )

    if not detections_raw:
        logger.info("Нет детекций после обработки")
        return image, []

    logger.info(f"Найдено {len(detections_raw)} детекций после обработки")

    # ===== ШАГ 5: Отрисовка результатов =====
    draw = ImageDraw.Draw(image)

    try:
        font_main = ImageFont.truetype("arial.ttf", 18)
        font_coords = ImageFont.truetype("arial.ttf", 12)
    except:
        font_main = ImageFont.load_default()
        font_coords = font_main

    detections_data = []

    for idx, det in enumerate(detections_raw):
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        contour_points = det.get("contour_points", [])

        # Убираем паддинг
        x1 = (x1 - pad_w) / ratio
        x2 = (x2 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        y2 = (y2 - pad_h) / ratio

        # Ограничиваем координаты
        x1 = max(0, min(orig_w, int(round(x1))))
        x2 = max(0, min(orig_w, int(round(x2))))
        y1 = max(0, min(orig_h, int(round(y1))))
        y2 = max(0, min(orig_h, int(round(y2))))

        # Сохраняем данные для шаблона
        detections_data.append(
            {
                'coords': (x1, y1, x2, y2),
                'score': float(score),
                'label': CLASS_NAME,
                'contour_points': contour_points[
                    :50
                ],  # ограничиваем для читаемости
            }
        )

        # Рисуем прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Координаты углов
        draw.text(
            (x1 - 5, y1 - 15), f"({x1}, {y1})", fill="blue", font=font_coords
        )
        draw.text(
            (x2 - 40, y1 - 15), f"({x2}, {y1})", fill="blue", font=font_coords
        )
        draw.text(
            (x1 - 5, y2 + 5), f"({x1}, {y2})", fill="blue", font=font_coords
        )
        draw.text(
            (x2 - 40, y2 + 5), f"({x2}, {y2})", fill="blue", font=font_coords
        )

        # Подпись с уверенностью
        label = f"P(tail) = {score:.2f}"
        text_x = x1
        text_y = max(0, y1 - 25)
        text_bbox = draw.textbbox((text_x, text_y), label, font=font_main)
        draw.rectangle(text_bbox, fill="white")
        draw.text((text_x, text_y), label, fill="red", font=font_main)

    logger.info(f"Инференс завершен. Найдено {len(detections_data)} хвостов")
    return image, detections_data


# ============================================
# МАРШРУТЫ FLASK (веб-интерфейс)
# ============================================
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Главная страница.

    GET: показываем форму загрузки
    POST: получаем файл, обрабатываем, показываем результат
    """
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return "Нет файла", 400

        # Читаем файл в байты
        image_bytes = file.read()

        try:
            # Запускаем инференс. Явно указанный порог уверенности conf_thres: 0.5 (50%).
            result_image, detections_data = run_inference(
                image_bytes, conf_thres=0.5
            )
        except Exception as e:
            logger.error(f"Ошибка при инференсе: {e}")
            return f"Ошибка обработки: {str(e)}", 500

        # Сохраняем результат
        filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_image.save(filepath)

        # Передаём в шаблон и картинку, и данные
        return render_template(
            'index.html',
            result_image=url_for('static', filename=f'uploads/{filename}'),
            detections=detections_data,
            found=len(detections_data),
        )

    return render_template('index.html', result_image=None, detections=[])


if __name__ == '__main__':
    logger.info("Запуск Flask приложения")
    app.run(debug=True, host='0.0.0.0', port=5000)
