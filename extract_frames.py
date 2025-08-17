import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Frame saving interval
FRAME_RATE = 5

# Input and output folder paths
real_input_root = Path("celebdf/Celeb-real")
synth_input_root = Path("celebdf/Celeb-synthesis")
real_output_root = Path("frames_real")
synth_output_root = Path("frames_synthesis")


def extract_frames(video_path: Path, save_dir: Path, frame_rate: int = FRAME_RATE):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = cap.read()
    if not success:
        print(f"⚠️ No readable frames in: {video_path}")
        cap.release()
        return

    count = 0
    saved = 0

    with tqdm(total=total_frames, desc=f"Extracting {video_path.name}", leave=False) as pbar:
        while success:
            if count % frame_rate == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                frame_name = f"{video_path.stem}_frame{saved}.jpg"
                cv2.imwrite(str(save_dir / frame_name), frame)
                saved += 1
            success, frame = cap.read()
            count += 1
            pbar.update(1)

    cap.release()
    print(f"✅ Saved {saved} frames from: {video_path.relative_to(video_path.parents[2])}")


def extract_from_folder(input_root: Path, output_root: Path):
    video_files = []

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(Path(root) / file)

    print(f"🎬 Found {len(video_files)} videos in: {input_root.name}")

    for video_path in tqdm(video_files, desc=f"Processing {input_root.name}"):
        relative_path = video_path.relative_to(input_root).parent
        target_dir = output_root / relative_path
        extract_frames(video_path, target_dir)


if __name__ == "__main__":
    extract_from_folder(real_input_root, real_output_root)
    extract_from_folder(synth_input_root, synth_output_root)


