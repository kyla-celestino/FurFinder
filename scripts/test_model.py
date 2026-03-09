from ultralytics import YOLO
import glob
import random
import os

# Config
MODEL_PATH  = "models/trained/furfinder_v1/weights/best.pt"
TEST_IMAGES = "data/splits/test/images"
TEST_LABELS = "data/splits/test/labels"
RESULTS_DIR = "results"
CONF        = 0.5

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

def test_model():
    print("=" * 50)
    print("FURFINDER MODEL TEST")
    print("=" * 50)

    # Load model
    print("\nLoading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")

    # Get all images
    all_images = glob.glob(f"{TEST_IMAGES}/*")

    # Filter by label
    cat_images  = [img for img in all_images if get_class_from_label(img) == "cat"]
    dog_images  = [img for img in all_images if get_class_from_label(img) == "dog"]

    print(f"\nCat images found  : {len(cat_images)}")
    print(f"Dog images found  : {len(dog_images)}")

    # Pick 3 from each
    sample_cats = random.sample(cat_images, min(3, len(cat_images)))
    sample_dogs = random.sample(dog_images, min(3, len(dog_images)))
    sample_images = sample_cats + sample_dogs

    print(f"\nTesting on {len(sample_images)} images (3 cats + 3 dogs)...")

    # Run detection
    results = model.predict(
        source     = sample_images,
        conf       = CONF,
        save       = True,
        project    = RESULTS_DIR,
        name       = "predictions",
        line_width = 2,
    )

    # Print results
    print("\n" + "=" * 50)
    print("DETECTION RESULTS")
    print("=" * 50)

    correct   = 0
    incorrect = 0
    no_detect = 0

    for i, result in enumerate(results):
        image_name   = os.path.basename(result.path)
        boxes        = result.boxes
        actual_class = get_class_from_label(sample_images[i])

        print(f"\nImage {i+1}: {image_name}")
        print(f"  Actual    : {actual_class}")

        if len(boxes) == 0:
            print(f"  Detected  : nothing")
            print(f"  Status    : MISSED")
            no_detect += 1
        else:
            for box in boxes:
                class_id   = int(box.cls)
                class_name = result.names[class_id]
                confidence = float(box.conf) * 100
                status     = "CORRECT" if class_name == actual_class else "WRONG"

                print(f"  Detected  : {class_name} ({confidence:.1f}% confidence)")
                print(f"  Status    : {status}")

                if class_name == actual_class:
                    correct += 1
                else:
                    incorrect += 1

    # Summary
    total = correct + incorrect + no_detect
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Correct       : {correct}/{total}")
    print(f"  Wrong         : {incorrect}/{total}")
    print(f"  Not detected  : {no_detect}/{total}")
    print(f"  Accuracy      : {(correct/total*100):.1f}%" if total > 0 else "N/A")
    print("=" * 50)
    print(f"\nOutput images saved to: {RESULTS_DIR}/predictions/")

if __name__ == "__main__":
    test_model()
