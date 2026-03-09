from ultralytics import YOLO
import os

# ── Config ──────────────────────────────────────────
MODEL_PATH  = "models/pretrained/yolov8s.pt"
CONFIG_PATH = "configs/dataset.yaml"
PROJECT_DIR = "models/trained"
RUN_NAME    = "cats_dogs_v1"
# ────────────────────────────────────────────────────

def train():
    print("=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    # Load pretrained model
    model = YOLO(MODEL_PATH)

    # Train
    results = model.train(
        data    = CONFIG_PATH,
        epochs  = 50,          # Number of training rounds
        imgsz   = 640,         # Image size
        batch   = 16,          # Reduce to 8 if memory error
        workers = 4,
        device  = "cpu",       # Change to 0 if you have GPU
        project = PROJECT_DIR,
        name    = RUN_NAME,
        save    = True,
        plots   = True,        # Saves training graphs
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {PROJECT_DIR}/{RUN_NAME}")
    print("=" * 50)

if __name__ == "__main__":
    train()
