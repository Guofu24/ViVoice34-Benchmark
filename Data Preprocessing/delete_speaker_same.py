import pandas as pd
import os
import shutil

# --- CẤU HÌNH ---
DATASET_ROOT = r"D:\Viettel\dataset\ViVoice34\VoxVietnamese_dataset\VoxVietnamese_dataset"
EXCEL_FILE = "speaker_duplicates_check.xlsx"

# Chế độ chạy: 
# True = Chỉ in ra danh sách sẽ xóa (An toàn)
# False = Thực hiện xóa thật trên ổ đĩa (Cẩn thận!)
DRY_RUN = False 

def get_actual_path(unique_name):
    """
    Chuyển đổi 'train_spk1' -> 'D:/.../train/spk1'
    """
    parts = unique_name.split('_', 1)
    if len(parts) < 2:
        return None
    split, spk_name = parts[0], parts[1]
    return os.path.join(DATASET_ROOT, split, spk_name)

def main():
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Không tìm thấy file {EXCEL_FILE}")
        return

    # 1. Đọc dữ liệu từ Excel
    df = pd.read_excel(EXCEL_FILE)
    if df.empty:
        print("✅ File Excel trống, không có gì để xử lý.")
        return

    # 2. Thuật toán gom nhóm để tìm danh sách cần xóa
    # Chúng ta dùng một set để theo dõi những người đã được "giữ lại" 
    # và một set để lưu những người "cần xóa"
    all_speakers_in_pairs = set(df['Speaker A']).union(set(df['Speaker B']))
    to_delete = set()
    kept = set()

    for _, row in df.iterrows():
        spk_a = row['Speaker A']
        spk_b = row['Speaker B']

        # Nếu A đã được giữ lại và B chưa bị đánh dấu xóa, thì xóa B
        if spk_a in kept:
            if spk_b not in kept:
                to_delete.add(spk_b)
        # Nếu B đã được giữ lại, thì xóa A
        elif spk_b in kept:
            if spk_a not in kept:
                to_delete.add(spk_a)
        # Nếu cả 2 đều chưa xuất hiện trong danh sách "giữ lại"
        else:
            # Ưu tiên giữ lại A (giả sử A là train, B là val)
            kept.add(spk_a)
            to_delete.add(spk_b)

    print(f"📊 Tổng số cặp trùng: {len(df)}")
    print(f"🗑️ Số lượng Speaker dự kiến xóa: {len(to_delete)}")
    print("-" * 50)

    # 3. Thực hiện xóa
    success_count = 0
    error_count = 0

    for unique_name in sorted(list(to_delete)):
        path = get_actual_path(unique_name)
        
        if path and os.path.exists(path):
            if DRY_RUN:
                print(f"[DRY-RUN] Sẽ xóa: {unique_name} tại {path}")
                success_count += 1
            else:
                try:
                    shutil.rmtree(path)
                    print(f"✅ Đã xóa: {unique_name}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Lỗi khi xóa {unique_name}: {e}")
                    error_count += 1
        else:
            print(f"⚠️ Không tìm thấy thư mục của: {unique_name}")

    print("-" * 50)
    if DRY_RUN:
        print(f"👉 Đây là chế độ chạy thử. Đổi DRY_RUN = False để thực hiện xóa thật.")
    else:
        print(f"🚀 Hoàn tất! Đã xóa thành công {success_count} thư mục. Lỗi: {error_count}")

if __name__ == "__main__":
    main()