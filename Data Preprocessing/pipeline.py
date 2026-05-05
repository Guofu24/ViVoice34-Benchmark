import os
import shutil
import librosa
import soundfile as sf
import multiprocessing
from functools import partial

# Cấu hình đường dẫn
TARGET_DIR = r"D:\Viettel\dataset\VoxVietnamese AND datasplited (voxceleb_and_MozillaViWav )\VoxVietnamese_dataset\VoxVietnamese_dataset"
TRAIN_DIR = os.path.join(TARGET_DIR, "train")
VAL_DIR = os.path.join(TARGET_DIR, "val")

def flatten_audio_directory(speaker_path):
    """
    Di chuyển các file .wav từ các sub folder (gallery, query) ra directory gốc của speaker (speaker_path).
    Sau đó xóa sub-folders này đi trước khi bắt đầu VAD logic
    """
    for sub in ["gallery", "query"]:
        sub_path = os.path.join(speaker_path, sub)
        if os.path.isdir(sub_path):
            for file_name in os.listdir(sub_path):
                if file_name.lower().endswith(".wav"):
                    old_path = os.path.join(sub_path, file_name)
                    new_path = os.path.join(speaker_path, file_name)
                    
                    # Đổi tên tránh trùng lặp
                    if os.path.exists(new_path):
                        new_path = os.path.join(speaker_path, f"moved_{sub}_{file_name}")
                        
                    try:
                        shutil.move(old_path, new_path)
                    except:
                        pass
            
            # Xóa sub-folder query/gallery      
            try:
                if not os.listdir(sub_path):
                    os.rmdir(sub_path)
            except:
                pass

def process_single_audio(file_path):
    """
    Apply VAD (trim silence) and return the new duration.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        sf.write(file_path, y_trimmed, sr)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        return duration
    except Exception as e:
        # Nếu audio lỗi -> Xóa file
        try:
            os.remove(file_path)
        except:
            pass
        return 0.0

def process_speaker(speaker, folders_to_scan):
    """
    Hàm xử lý cho một speaker: 
    1. Chuẩn hóa Folder (Gom từ query/gallery)
    2. VAD: bỏ khoảng lặng, lấy audio hợp lệ 
    3. Sum thời gian -> Quyết định DROP/ TRAIN / VAL.
    """
    all_files = []
    
    # Chuẩn hóa cấu trúc thư mục trước tiên và thu thập file .wav của nhánh train/val
    for root_dir in folders_to_scan:
        speaker_path = os.path.join(root_dir, speaker)
        if os.path.isdir(speaker_path):
            # 1. Gom file
            flatten_audio_directory(speaker_path)
            
            # 2. Lấy tên các file sau khi đã kéo ra ngoài gốc
            for file_name in os.listdir(speaker_path):
                f_path = os.path.join(speaker_path, file_name)
                if os.path.isfile(f_path) and file_name.lower().endswith('.wav'):
                    all_files.append(f_path)
                    
    if not all_files:
        return speaker, "BỎ QUA DO KHÔNG CÓ FILE WAV HỢP LỆ"

    # 3. Tính toán VAD
    total_dur = 0.0
    valid_files = []
    
    for f in all_files:
        dur = process_single_audio(f)
        if dur > 0:
            total_dur += dur
            valid_files.append(f)
            
    # 4. Phân loại theo total_dur
    if total_dur < 5.0:
        action = "DROP"
        # Xóa tất cả các file
        for f in valid_files:
            try:
                os.remove(f)
            except:
                pass
                
        # Xóa folder trống
        for root_dir in folders_to_scan:
            s_path = os.path.join(root_dir, speaker)
            if os.path.exists(s_path) and not os.listdir(s_path):
                try:
                    os.rmdir(s_path)
                except:
                    pass
                    
    elif 5.0 <= total_dur < 8.0:
        action = "Di Chuyển vào VAL"
        val_speaker_folder = os.path.join(VAL_DIR, speaker)
        os.makedirs(val_speaker_folder, exist_ok=True)
        
        for f in valid_files:
            if not f.startswith(VAL_DIR):
                dest_path = os.path.join(val_speaker_folder, os.path.basename(f))
                try:
                    shutil.move(f, dest_path)
                except:
                    pass
                    
        # Dọn dẹp folder bên train nếu rỗng
        train_s_path = os.path.join(TRAIN_DIR, speaker)
        if os.path.exists(train_s_path) and not os.listdir(train_s_path):
            try:
                os.rmdir(train_s_path)
            except:
                pass
                
    else:  # >= 8.0
        action = "Di chuyển vào TRAIN"
        train_speaker_folder = os.path.join(TRAIN_DIR, speaker)
        os.makedirs(train_speaker_folder, exist_ok=True)
        
        for f in valid_files:
            if not f.startswith(TRAIN_DIR):
                dest_path = os.path.join(train_speaker_folder, os.path.basename(f))
                try:
                    shutil.move(f, dest_path)
                except:
                    pass
                    
        # Dọn dẹp folder bên val nếu rỗng
        val_s_path = os.path.join(VAL_DIR, speaker)
        if os.path.exists(val_s_path) and not os.listdir(val_s_path):
            try:
                os.rmdir(val_s_path)
            except:
                pass

    return speaker, f"{action} (Tổng cộng: {total_dur:.2f} s)"

def main():
    folders_to_scan = [TRAIN_DIR, VAL_DIR]
    speakers = set()
    
    # Quét tất cả thư mục Speaker từ dataset
    for root_dir in folders_to_scan:
        if os.path.exists(root_dir):
            for spk in os.listdir(root_dir):
                s_path = os.path.join(root_dir, spk)
                if os.path.isdir(s_path):
                    speakers.add(spk)
                    
    speakers = list(speakers)
    total_speakers = len(speakers)
    print(f"Tổng số speaker: {total_speakers}")
    print("Vui lòng chờ đợi xử lý Audio Pipeline bằng Đa luồng xử lý CPU...")
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        process_func = partial(process_speaker, folders_to_scan=folders_to_scan)
        for i, (speaker, result) in enumerate(pool.imap_unordered(process_func, speakers), 1):
            if i % 100 == 0 or i == total_speakers:
                print(f"Tiến độ tổng: [{i}/{total_speakers}] | Speaker: {speaker} => {result}")
                
if __name__ == "__main__":
    main()
