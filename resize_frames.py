import cv2
import os
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# Input and output root folders
real_input_root = Path("frames_real")
synth_input_root = Path("frames_synthesis")

real_output_root = Path("frames_real_resized")
synth_output_root = Path("frames_synthesis_resized")

# Target image size
target_size = (256, 256)

def resize_and_save(image_path: Path, save_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to read: {image_path}")
        return
    resized = cv2.resize(image, target_size)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), resized)

def resize_all(input_root: Path, output_root: Path):
    image_paths = []

    # First gather all image paths
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_paths.append(Path(root) / file)

    print(f"📦 Found {len(image_paths)} images in: {input_root}")

    for image_path in tqdm(image_paths, desc=f"Resizing {input_root.name}"):
        relative_path = image_path.relative_to(input_root)
        save_path = output_root / relative_path
        resize_and_save(image_path, save_path)

if __name__ == "__main__":
    print("🔁 Resizing frames_real...")
    resize_all(real_input_root, real_output_root)

    print("🔁 Resizing frames_synthesis...")
    resize_all(synth_input_root, synth_output_root)

    print("🎉 Done resizing all frames!")

