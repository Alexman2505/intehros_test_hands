"""
Обучение YOLOv8-seg для сегментации рук
"""

import torch
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    # ===== 1. НАСТРОЙКИ =====
    DATA_YAML = "data.yaml"
    EPOCHS = 25
    BATCH_SIZE = 8
    IMG_SIZE = 640
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Папка для результатов (в текущей директории)
    PROJECT_DIR = Path(__file__).parent
    RUNS_DIR = PROJECT_DIR / 'runs'

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ===== 2. ОБУЧЕНИЕ =====
    model = YOLO('yolov8n-seg.pt')

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=str(RUNS_DIR),  # ← всё в папку runs
        name='hand_experiment',  # ← имя эксперимента
        augment=True,
        copy_paste=0.2,  # Копирует объект с одного изображения и вставляет на другое 20% вероятности
        # mosaic=1.0,
        fliplr=0.5,  # Отражает изображение горизонтально (зеркально)
        plots=True,
        val=True,
    )

    # ===== 3. КОНВЕРТАЦИЯ В ONNX =====
    best_model_path = RUNS_DIR / 'hand_experiment' / 'weights' / 'best.pt'

    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))

        # Валидация (покажет метрики)
        metrics = best_model.val(data=DATA_YAML)
        print(f"\n📊 mAP50-95: {metrics.box.map:.4f}")

        # Конвертация в ONNX
        onnx_path = best_model.export(
            format='onnx', imgsz=IMG_SIZE, device='cpu'
        )

        # Копируем в текущую папку
        import shutil

        shutil.copy(onnx_path, 'best.onnx')
        print(f"✅ ONNX модель: best.onnx")
    else:
        print("❌ Модель не найдена!")
