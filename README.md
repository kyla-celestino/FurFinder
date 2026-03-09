# FurFinder 🐱🐶

An object detection system that identifies cats and dogs in images using YOLOv8.

---

## Results

| Metric        | Score  |
|---------------|--------|
| mAP50         | 96.50% |
| mAP50-95      | 79.52% |
| Precision     | 92.07% |
| Recall        | 94.70% |
| Accuracy      | 97.70% |

### Per Class
| Class | AP50   | Accuracy        |
|-------|--------|-----------------|
| Cat   | 95.42% | 100.0% (82/82)  |
| Dog   | 97.57% | 96.4% (132/137) |

---

## Dataset

| Property       | Details                        |
|----------------|--------------------------------|
| Total Images   | 4,737                          |
| Classes        | Cat, Dog                       |
| Image Format   | JPG                            |
| Label Format   | YOLO (.txt)                    |
| Source         | Kaggle                         |

### Dataset Split
| Split | Images |
|-------|--------|
| Train | 3,800+ |
| Val   | 500+   |
| Test  | 220    |

---

## Model

| Property      | Details          |
|---------------|------------------|
| Architecture  | YOLOv8s          |
| Trained on    | Google Colab T4 GPU |
| Epochs        | 50               |
| Image Size    | 640x640          |
| Batch Size    | 16               |

---

## Project Structure

FurFinder/
│
├── data/
│   ├── raw_images/
│   ├── annotated/
│   └── splits/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
├── models/
│   ├── pretrained/
│   ├── trained/
│   └── exported/
│
├── notebooks/
├── scripts/
│   ├── download_model.py
│   ├── check_dataset.py
│   ├── check_filenames.py
│   ├── train.py
│   ├── test_model.py
│   └── evaluate.py
│
├── configs/
│   └── dataset.yaml
│
├── results/
│   ├── metrics/
│   └── predictions/
│
├── logs/
├── .gitignore
├── requirements.txt
└── README.md


---

## Setup

### 1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/FurFinder.git
cd FurFinder


### 2. Create Conda Environment

conda create -n FurFinder python=3.10 -y
conda activate FurFinder


### 3. Install Dependencies

pip install -r requirements.txt


### 4. Download Pretrained Weights

python scripts/download_model.py

---

## Usage

### Run Detection on Test Images

python scripts/test_model.py


### Run Full Evaluation

python scripts/evaluate.py


### Check Dataset

python scripts/check_dataset.py

---

## Sample Output

DETECTION RESULTS


Image 1: cat_image.jpg
  Actual    : cat
  Detected  : cat (92.3% confidence)
  Status    : CORRECT


Image 2: dog_image.jpg
  Actual    : dog
  Detected  : dog (90.2% confidence)
  Status    : CORRECT


SUMMARY
  Correct       : 6/6
  Wrong         : 0/6
  Not detected  : 0/6
  Accuracy      : 100.0%

---

## Requirements

ultralytics
opencv-python
torch
torchvision
numpy
matplotlib
pandas
scikit-learn
Pillow
tqdm
pyyaml
jupyter
albumentations

---

## How It Works

Input Image
     ↓
YOLOv8s Model
     ↓
Bounding Box Detection
     ↓
Class Label (Cat or Dog)
     ↓
Confidence Score
     ↓
Output Image with Annotations

---

## Future Improvements

- [ ] Add more animal classes
- [ ] Build a live webcam detection demo
- [ ] Deploy as a web application
- [ ] Export model to ONNX or TFLite for mobile
- [ ] Increase dataset size for even better accuracy

---

## License

This project is for educational purposes.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Kaggle](https://kaggle.com) — Dataset source
- [Google Colab](https://colab.research.google.com) — Training platform
- [Roboflow](https://roboflow.com) — Dataset tools