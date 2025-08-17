import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import matlab.engine
from scipy.ndimage import laplace

# Start MATLAB engine
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\SHREYA SARMA\Documents\MATLAB\CurveLab-2.1.3\fdct_wrapping_matlab', nargout=0)

def safe_entropy(x):
    hist, _ = np.histogram(x, bins=256, range=(np.min(x), np.max(x)), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def extract_advanced_curvelet_features(coeffs):
    energies = []
    entropies = []
    sharpness_scores = []
    features = []
    prev_scale_energies = []

    for scale_idx, scale in enumerate(coeffs):
        scale_energies = []
        scale_entropies = []
        for wedge in scale:
            rows, cols = wedge.size
            real = np.array(eng.real(wedge)).reshape((rows, cols))
            imag = np.array(eng.imag(wedge)).reshape((rows, cols))
            mag = np.sqrt(real**2 + imag**2)

            energy = np.sum(mag**2)
            entropy = safe_entropy(mag.flatten())
            sharpness = np.mean(np.abs(laplace(mag)))

            energies.append(energy)
            entropies.append(entropy)
            sharpness_scores.append(sharpness)

            scale_energies.append(energy)
            scale_entropies.append(entropy)

        if scale_idx > 0:
            prev = np.array(prev_scale_energies)
            curr = np.array(scale_energies)
            min_len = min(len(prev), len(curr))
            corr = np.corrcoef(prev[:min_len], curr[:min_len])[0, 1] if min_len >= 2 else 0
            features.append(corr)

        ded = np.std(scale_entropies)
        features.append(ded)

        decay = np.mean(scale_energies) / (np.max(scale_energies) + 1e-5)
        features.append(decay)

        prev_scale_energies = scale_energies

    features.append(np.mean(sharpness_scores))

    if len(energies) % 2 == 0:
        n = len(energies)
        symmetry_diffs = [abs(energies[i] - energies[n - i - 1]) for i in range(n // 2)]
        features.append(np.mean(symmetry_diffs))

    top_energies = sorted(energies, reverse=True)[:10]
    features.append(np.mean(top_energies))

    features.append(np.std(energies))

    return features

def apply_curvelet(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, (256, 256))
    matlab_img = matlab.double(img.tolist())
    coeffs = eng.fdct_wrapping(matlab_img, False, 2)
    return coeffs

def process_balanced_curvelet(dataset_dir, output_csv):
    all_data = []
    for label_dir, label in [("real_balanced", 0), ("fake_balanced", 1)]:
        full_dir = os.path.join(dataset_dir, label_dir)
        for filename in tqdm(os.listdir(full_dir), desc=f"Processing {label_dir}"):
            try:
                path = os.path.join(full_dir, filename)
                coeffs = apply_curvelet(path)
                feats = extract_advanced_curvelet_features(coeffs)
                frame_name = os.path.splitext(filename)[0]
                all_data.append([frame_name, label] + feats)
            except Exception as e:
                print(f"Error with {filename}: {e}")

    feat_cols = [f"curvelet_f{i}" for i in range(len(all_data[0]) - 2)]
    df = pd.DataFrame(all_data, columns=["frame_name", "label"] + feat_cols)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv} with {len(df)} rows")

# Example usage
if __name__ == "__main__":
    base_dir = r"C:\Users\SHREYA SARMA\Desktop\Dataset_2_Capstone"
    process_balanced_curvelet(base_dir, "curvelet_features_dataset2.csv")
