import os
import torch
from roboflow import Roboflow
from ultralytics import YOLO

# =========================
# Пути
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# =========================
# Загрузка датасета с Roboflow
# =========================
rf = Roboflow(api_key="PPvWh3zYDUwdXSmGOJai")
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)

dataset = version.download(
    model_format="yolov11",
    location=DATASET_DIR
)

# =========================
# Загрузка модели
# =========================
model = YOLO("yolov10m.pt")

device = 0 if torch.cuda.is_available() else "cpu"
print("Используемое устройство:", device)

# =========================
# Обучение
# =========================
model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=15,
    imgsz=640,
    batch=16,
    device=device,
    project=RUNS_DIR,
    name="rps_train",
)

# =========================
# Проверка на одном изображении
# =========================
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test.jpg")  # положи картинку сюда

if os.path.exists(TEST_IMAGE_PATH):
    model.predict(
        source=TEST_IMAGE_PATH,
        conf=0.25,
        save=True,
        project=RUNS_DIR,
        name="predictions"
    )
    print("Предсказание сохранено в папке runs/predictions")
else:
    print("test.jpg не найден — этап предсказания пропущен")
