import os

splits = ['train', 'val', 'test']

print("=" * 50)
print("DATASET CHECK")
print("=" * 50)

total_images = 0
total_labels = 0

for split in splits:
    img_path = f"data/splits/{split}/images"
    lbl_path = f"data/splits/{split}/labels"
    
    imgs = len(os.listdir(img_path)) if os.path.exists(img_path) else 0
    lbls = len(os.listdir(lbl_path)) if os.path.exists(lbl_path) else 0
    
    total_images += imgs
    total_labels += lbls
    
    match = "✅" if imgs == lbls else "⚠️ MISMATCH"
    print(f"\n{split.upper()}")
    print(f"  Images : {imgs}")
    print(f"  Labels : {lbls}")
    print(f"  Status : {match}")

print("\n" + "=" * 50)
print(f"Total Images : {total_images}")
print(f"Total Labels : {total_labels}")
print("=" * 50)
