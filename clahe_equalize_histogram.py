import cv2
import os
from pathlib import Path
from tqdm import tqdm

# === Input & Output Directories ===
real_input = Path("frames_real_resized")
synth_input = Path("frames_synthesis_resized")

real_output = Path("frames_real_clahe")
synth_output = Path("frames_synthesis_clahe")

# === CLAHE Equalization Function ===
def equalize_image_color_clahe(image):
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE on L-channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Merge and convert back to BGR
    lab_eq = cv2.merge((l_eq, a, b))
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return bgr_eq

# === CLAHE Equalization for a Folder ===
def equalize_folder_clahe(input_dir: Path, output_dir: Path):
    image_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(Path(root) / file)

    print(f"📸 Found {len(image_paths)} images in: {input_dir}")

    for img_path in tqdm(image_paths, desc=f"CLAHE: {input_dir.name}"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ Could not read: {img_path}")
            continue

        eq_img = equalize_image_color_clahe(image)

        # Maintain same folder structure
        rel_path = img_path.relative_to(input_dir)
        save_path = output_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), eq_img)

# === MAIN ===
if __name__ == "__main__":
    equalize_folder_clahe(real_input, real_output)
    equalize_folder_clahe(synth_input, synth_output)
    print("✅ CLAHE Equalization complete!")
