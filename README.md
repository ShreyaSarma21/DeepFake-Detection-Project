# 🔍 DeepFake Detection with Hybrid Wavelet Features

## 📌 Overview
This repository contains our **DeepFake Detection pipeline** that leverages **Curvelet–Shearlet hybrid features** and **Cross-Wavelet Fusion Mechanism**.  
We preprocess videos, extract discriminative features, and train models for robust classification of **real vs. fake videos**, with emphasis on adversarial robustness.  

---

## 🛠️ Pipeline

1. **Frame Extraction**  
   - Extracts frames from real & fake videos at 5 FPS  
   - `extract_frames.py`

2. **Resizing**  
   - Resizes all frames to `256x256`  
   - `resize_frames.py`

3. **Histogram Equalization (CLAHE)**  
   - Enhances contrast using LAB color-space equalization  
   - `clahe_equalize_histogram.py`

4. **Data Augmentation (Real Frames)**  
   - Augments real frames using **Albumentations** (flip, rotate, blur, brightness, HSV)  
   - `augment_real_frames.py`

5. **Dataset Balancing**  
   - Samples ~90k fake frames to match real distribution  
   - `sample_fake_90k.py`

6. **Feature Extraction**  
   - **Curvelet Transform**: via MATLAB CurveLab (`curvelet_extraction.py`)  
   - **Shearlet Transform**: via PyShearlab (notebook)  
   - Extracts statistical features (energy, entropy, sharpness, symmetry, decay, correlation)

7. **Hybrid Fusion**  
   - Combines Curvelet & Shearlet features using **Cross-Wavelet Fusion**  
   - Implemented in `cross_wavelet_fusion.ipynb`

8. **Classification & Evaluation**  
   - Train ML/DL classifiers on extracted features  
   - Evaluate with **standard & biometric spoofing metrics**:  
     - Accuracy, Precision, Recall, F1-score  
     - ROC-AUC, EER  
     - APCER, BPCER, ACER, HTER  


