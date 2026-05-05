import os
import random
from itertools import combinations

# --- CẤU HÌNH ---
VAL_WAV_DIR = r"D:\Viettel\dataset\ViVoice34\VoxVietnamese_dataset\ViVoice34\val\wav"  # Đường dẫn đến thư mục chứa các folder speaker
OUTPUT_TRIALS = r"D:\Viettel\dataset\ViVoice34\VoxVietnamese_dataset\ViVoice34\val\trials.txt"
NUM_POSITIVE_PER_SPEAKER = 5  # Số cặp cùng người nói tối đa cho mỗi speaker
NEG_POS_RATIO = 1             # Tỉ lệ Negative/Positive (1:1 là cân bằng)

def generate_trials():
    speaker_dict = {}
    
    # 1. Quét toàn bộ speaker và file wav
    print(f"📂 Đang quét thư mục: {VAL_WAV_DIR}")
    speakers = [d for d in os.listdir(VAL_WAV_DIR) if os.path.isdir(os.path.join(VAL_WAV_DIR, d))]
    
    for spk in speakers:
        spk_path = os.path.join(VAL_WAV_DIR, spk)
        # Lấy tất cả file wav (bao gồm cả trong sub-folder nếu có)
        wavs = []
        for root, _, files in os.walk(spk_path):
            for f in files:
                if f.endswith('.wav'):
                    # Lưu đường dẫn tương đối từ thư mục wav/ để file trials gọn sạch
                    rel_path = os.path.relpath(os.path.join(root, f), VAL_WAV_DIR)
                    wavs.append(rel_path.replace("\\", "/")) # Format chuẩn Linux path
        
        if len(wavs) >= 2:
            speaker_dict[spk] = wavs

    if not speaker_dict:
        print("❌ Không tìm thấy speaker nào có ít nhất 2 file audio để tạo cặp!")
        return

    pos_pairs = []
    # 2. Tạo Positive Pairs (Label 1)
    for spk, wavs in speaker_dict.items():
        # Lấy các tổ hợp cặp file của cùng 1 người
        all_pos = list(combinations(wavs, 2))
        random.shuffle(all_pos)
        pos_pairs.extend(all_pos[:NUM_POSITIVE_PER_SPEAKER])

    num_pos = len(pos_pairs)
    print(f"✅ Đã tạo {num_pos} cặp Positive.")

    # 3. Tạo Negative Pairs (Label 0)
    neg_pairs = []
    all_speakers = list(speaker_dict.keys())
    target_neg_count = int(num_pos * NEG_POS_RATIO)

    while len(neg_pairs) < target_neg_count:
        # Chọn ngẫu nhiên 2 speaker khác nhau
        spk1, spk2 = random.sample(all_speakers, 2)
        
        # Chọn ngẫu nhiên 1 file từ mỗi người
        wav1 = random.choice(speaker_dict[spk1])
        wav2 = random.choice(speaker_dict[spk2])
        
        neg_pairs.append((wav1, wav2))

    print(f"✅ Đã tạo {len(neg_pairs)} cặp Negative.")

    # 4. Ghi ra file trials.txt
    all_trials = []
    for pair in pos_pairs:
        all_trials.append(f"1 {pair[0]} {pair[1]}")
    for pair in neg_pairs:
        all_trials.append(f"0 {pair[0]} {pair[1]}")

    # Trộn ngẫu nhiên các dòng để model không bị học theo thứ tự
    random.shuffle(all_trials)

    with open(OUTPUT_TRIALS, "w", encoding="utf-8") as f:
        for line in all_trials:
            f.write(line + "\n")

    print(f"🚀 Hoàn tất! File trials đã được lưu tại: {OUTPUT_TRIALS}")

if __name__ == "__main__":
    generate_trials()