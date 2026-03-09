from ultralytics import YOLO
import shutil
import os

# This downloads the pretrained weights automatically
model = YOLO("yolov8n.pt")  # 'n' = nano (smallest/fastest)

# Move the downloaded weights to your models folder
if os.path.exists("yolov8n.pt"):
    shutil.move("yolov8n.pt", "models/pretrained/yolov8n.pt")
    print("Model moved to models/pretrained/")
