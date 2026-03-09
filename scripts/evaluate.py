from ultralytics import YOLO
import glob
import os

MODEL_PATH  = "models/trained/furfinder_v1/weights/best.pt"
TEST_IMAGES = "data/splits/test/images"
TEST_LABELS = "data/splits/test/labels"
CONFIG_PATH = "configs/dataset.yaml"

def get_class_from_label(img_path):
    base = os.path.basename(img_path)
    name = base.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    label_path = f"{TEST_LABELS}/{name}.txt"

    if not os.path.exists(label_path):
        return None

    with open(label_path, 'r') as f:
        lines = f.readlines()

    classes = set()
    for line in lines:
        parts = line.strip().split()
        if parts:
            classes.add(int(parts[0]))

    if classes == {0}:
        return "cat"
    elif classes == {1}:
        return "dog"
    else:
        return "both"

def evaluate():
    print("=" * 50)
    print("FURFINDER FULL EVALUATION")
    print("=" * 50)

    print("\nLoading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")

    all_images = glob.glob(f"{TEST_IMAGES}/*")
    print(f"\nTotal test images: {len(all_images)}")

    print("\nRunning official evaluation metrics...")
    metrics = model.val(
        data      = CONFIG_PATH,
        split     = "test",
        plots     = True,
        save_json = True,
    )

    print("\n" + "=" * 50)
    print("OFFICIAL METRICS")
    print("=" * 50)
    print(f"mAP50       : {metrics.box.map50:.4f}")
    print(f"mAP50-95    : {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    print("\n" + "=" * 50)
    print("PER CLASS METRICS")
    print("=" * 50)
    class_names = ["cat", "dog"]
    for i, name in enumerate(class_names):
        try:
            ap50 = metrics.box.ap50[i]
            print(f"{name.upper()}")
            print(f"  AP50 : {ap50:.4f}")
        except:
            print(f"{name.upper()} : metrics not available")

    print("\n" + "=" * 50)
    print("RUNNING ON ALL 220 TEST IMAGES")
    print("=" * 50)

    correct     = 0
    incorrect   = 0
    no_detect   = 0
    cat_correct = 0
    cat_total   = 0
    dog_correct = 0
    dog_total   = 0

    for img_path in all_images:
        actual = get_class_from_label(img_path)
        if actual is None or actual == "both":
            continue

        result = model.predict(
            source  = img_path,
            conf    = 0.5,
            verbose = False
        )

        res   = result[0]
        boxes = res.boxes

        if actual == "cat":
            cat_total += 1
        elif actual == "dog":
            dog_total += 1

        if len(boxes) == 0:
            no_detect += 1
        else:
            detected_class = res.names[int(boxes[0].cls)]
            if detected_class == actual:
                correct += 1
                if actual == "cat":
                    cat_correct += 1
                elif actual == "dog":
                    dog_correct += 1
            else:
                incorrect += 1

    total = correct + incorrect + no_detect

    print(f"\nTotal images tested : {total}")
    print(f"Correct             : {correct}")
    print(f"Wrong               : {incorrect}")
    print(f"Not detected        : {no_detect}")
    print(f"Overall Accuracy    : {(correct/total*100):.1f}%")
    print(f"\nCat Accuracy        : {(cat_correct/cat_total*100):.1f}% ({cat_correct}/{cat_total})")
    print(f"Dog Accuracy        : {(dog_correct/dog_total*100):.1f}% ({dog_correct}/{dog_total})")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()
