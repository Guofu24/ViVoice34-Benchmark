import os
from pathlib import Path
import soundfile as sf
import numpy as np

def calculate_speaker_stats(dataset_path):
    dataset_root = Path(dataset_path)
    
    # Cấu trúc: 
    # Train: train/<speaker_id>/*.wav
    # Val: val/wav/<speaker_id>/*.wav
    paths = {
        "Train": dataset_root / "train",
        "Val": dataset_root / "val" / "wav"
    }

    print(f"{'Split':<10} | {'Speakers':<10} | {'Min(s)':<10} | {'Max(s)':<10} | {'Mean(s)':<10} | {'Total(h)':<10}")
    print("-" * 75)

    for split_name, split_path in paths.items():
        if not split_path.exists():
            print(f"Cảnh báo: Không tìm thấy đường dẫn {split_path}")
            continue

        speaker_totals = []
        
        # Lấy danh sách các thư mục con (mỗi thư mục là 1 speaker)
        speaker_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        for speaker_dir in speaker_dirs:
            current_speaker_duration = 0
            # Quét tất cả file wav của speaker này (kể cả trong sub-folder nếu có)
            for wav_file in speaker_dir.rglob("*.wav"):
                try:
                    info = sf.info(wav_file)
                    current_speaker_duration += info.duration
                except Exception:
                    continue
            
            if current_speaker_duration > 0:
                speaker_totals.append(current_speaker_duration)
        
        if speaker_totals:
            arr = np.array(speaker_totals)
            print(f"{split_name:<10} | "
                  f"{len(arr):<10} | "
                  f"{arr.min():<10.2f} | "
                  f"{arr.max():<10.2f} | "
                  f"{arr.mean():<10.2f} | "
                  f"{arr.sum()/3600:<10.2f}")
        else:
            print(f"{split_name:<10} | Không có dữ liệu")

if __name__ == "__main__":
    # Thay đổi đường dẫn đến thư mục của bạn
    DATASET_DIR = "D:\Viettel\dataset\ViVoice34\VoxVietnamese_dataset\ViVoice34" 
    calculate_speaker_stats(DATASET_DIR)