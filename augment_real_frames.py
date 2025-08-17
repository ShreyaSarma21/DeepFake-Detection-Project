import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from albumentations import (
    HorizontalFlip, Rotate, RandomBrightnessContrast, HueSaturationValue,
    ShiftScaleRotate, OneOf, MotionBlur, GaussianBlur, Compose
)

# === Folder Setup ===
input_root = Path("frames_real_clahe")
output_root = Path("frames_real_augmented")

# === Augmentation Pipeline ===
def get_aug_pipeline():
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=10, p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        OneOf([
            MotionBlur(blur_limit=3, p=0.5),
            GaussianBlur(blur_limit=(3, 5), p=0.5)
        ], p=0.3),
        RandomBrightnessContrast(p=0.3),
        HueSaturationValue(p=0.3),
    ])

augmentor = get_aug_pipeline()

# === Identify real frame paths ===
def is_real_video_path(path: Path):
    return "frames_real_clahe" in str(path)


# === Augment image once and save ===
def augment_image_once(img_path: Path, save_base_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f" Failed to read: {img_path}")
        return
    augmented = augmentor(image=img)["image"]
    filename = img_path.stem + f"_aug.jpg"
    save_path = save_base_path / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), augmented)

# === Main Function ===
def augment_real_images_once():
    all_real_images = []

    # Step 1: Find all real frames
    print("🔍 Scanning for real images...")
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".jpg"):
                img_path = Path(root) / file
                if is_real_video_path(img_path):
                    all_real_images.append(img_path)

    original_count = len(all_real_images)
    print(f"📸 Found {original_count} real images.")

    if original_count == 0:
        print("⚠️ No real frames found.")
        return

    # Step 2: Copy original real images
    print("📥 Copying original real frames...")
    for img_path in tqdm(all_real_images, desc="Copying Originals", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        relative = img_path.relative_to(input_root)
        save_path = output_root / relative
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

    # Step 3: Augment each image once
    print("🧪 Creating 1 augmentation per image...")
    for img_path in tqdm(all_real_images, desc="Augmenting", unit="img"):
        relative = img_path.relative_to(input_root)
        save_base_path = output_root / relative.parent
        augment_image_once(img_path, save_base_path)

    print("✅ Augmentation completed.")
    print(f"   ➤ Originals : {original_count}")
    print(f"   ➤ Augmented: {original_count}")
    print(f"   ➤ Total     : {original_count * 2}")

# === Run the script ===
if __name__ == "__main__":
    augment_real_images_once()

