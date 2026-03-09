import os
import glob

TEST_IMAGES = "data/splits/test/images"
TEST_LABELS = "data/splits/test/labels"

all_images = glob.glob(f"{TEST_IMAGES}/*")

cat_images = []
dog_images = []
both_images = []
unknown = []

for img_path in all_images:
    base = os.path.basename(img_path)
    name = base.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    label_path = f"{TEST_LABELS}/{name}.txt"

    if not os.path.exists(label_path):
        unknown.append(img_path)
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    classes_in_image = set()
    for line in lines:
        parts = line.strip().split()
        if parts:
            class_id = int(parts[0])
            classes_in_image.add(class_id)

    if classes_in_image == {0}:
        cat_images.append(img_path)
    elif classes_in_image == {1}:
        dog_images.append(img_path)
    elif 0 in classes_in_image and 1 in classes_in_image:
        both_images.append(img_path)
    else:
        unknown.append(img_path)

print("=" * 50)
print("DATASET LABEL ANALYSIS")
print("=" * 50)
print(f"Total images    : {len(all_images)}")
print(f"Cat only        : {len(cat_images)}")
print(f"Dog only        : {len(dog_images)}")
print(f"Both            : {len(both_images)}")
print(f"Unknown         : {len(unknown)}")

print("\n" + "=" * 50)
print("SAMPLE CAT IMAGE FILENAMES (first 5)")
print("=" * 50)
for img in cat_images[:5]:
    print(f"  {os.path.basename(img)}")

print("\n" + "=" * 50)
print("SAMPLE DOG IMAGE FILENAMES (first 5)")
print("=" * 50)
for img in dog_images[:5]:
    print(f"  {os.path.basename(img)}")
