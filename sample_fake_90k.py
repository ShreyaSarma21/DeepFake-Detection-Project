import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
input_root = Path("frames_synthesis_clahe")
output_root = Path("fake_frames_90614")
target_sample_size = 90614

def get_all_images(input_dir):
    valid_exts = ['.jpg', '.jpeg', '.png']
    img_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_exts):
                img_paths.append(Path(root) / file)
    return img_paths

def sample_and_copy_images():
    all_images = get_all_images(input_root)
    print(f"📦 Found total fake frames: {len(all_images)}")

    if len(all_images) < target_sample_size:
        raise ValueError(f"Not enough fake frames to sample {target_sample_size}. Only found {len(all_images)}.")

    # Randomly select sample
    sampled_images = random.sample(all_images, target_sample_size)

    print(f"📤 Copying {target_sample_size} sampled fake frames...")
    for img_path in tqdm(sampled_images, desc="Copying sampled images", unit="img"):
        relative_path = img_path.relative_to(input_root)
        save_path = output_root / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, save_path)

    print("✅ Sampling and copying complete.")
    print(f"Sampled fake frames saved in: {output_root}")

if __name__ == "__main__":
    sample_and_copy_images()
