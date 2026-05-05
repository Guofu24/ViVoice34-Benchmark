import os
import shutil

TARGET_DIR = r"D:\Viettel\dataset\VoxVietnamese AND datasplited (voxceleb_and_MozillaViWav )\VoxVietnamese_dataset\VoxVietnamese_dataset"
TRAIN_DIR = os.path.join(TARGET_DIR, "train")
VAL_DIR = os.path.join(TARGET_DIR, "val")

def merge_gallery_query():
    folders_to_scan = [TRAIN_DIR, VAL_DIR]
    
    print("Bắt đầu dọn dẹp và gom file từ các folders 'gallery' và 'query' ra ngoài...")
    
    move_count = 0
    remove_count = 0

    for root_folder in folders_to_scan:
        if not os.path.exists(root_folder):
            continue
            
        # Lặp qua từng Thư mục speaker
        for speaker_name in os.listdir(root_folder):
            speaker_path = os.path.join(root_folder, speaker_name)
            
            if not os.path.isdir(speaker_path):
                continue
                
            # Duyệt qua các thư mục con trong speaker_path (vì có thể có hoặc không có gallery/query)
            # Dùng listdir thay vì đặt tay tên để có thể linh hoạt (phòng khi bạn muốn apply thêm folder khác)
            for sub_name in ["gallery", "query"]:
                sub_path = os.path.join(speaker_path, sub_name)
                
                # Nếu tồn tại thư mục dạng speaker/gallery hoặc speaker/query
                if os.path.exists(sub_path) and os.path.isdir(sub_path):
                    
                    # Duyệt và di chuyển TẤT CẢ CÁC FILE .wav bên trong ra folder của speaker
                    for f_name in os.listdir(sub_path):
                        f_path = os.path.join(sub_path, f_name)
                        
                        if os.path.isfile(f_path) and f_name.lower().endswith('.wav'):
                            dest_path = os.path.join(speaker_path, f_name)
                            
                            # Xử lý đụng độ tên file: NẾU file đã tồn tại ở cấp ngoài, thêm prefix tránh overwite
                            if os.path.exists(dest_path):
                                dest_path = os.path.join(speaker_path, f"moved_{sub_name}_{f_name}")
                            
                            try:
                                shutil.move(f_path, dest_path)
                                move_count += 1
                            except Exception as e:
                                print(f"Lỗi di chuyển file {f_path}: {e}")
                                
                    # Cuối cùng, xóa folder con nếu nó đã rỗng
                    try:
                        if not os.listdir(sub_path):
                            os.rmdir(sub_path)
                            remove_count += 1
                    except Exception as e:
                        print(f"Không thể xóa thư mục rỗng {sub_path} - Chi tiết: {e}")
                        
    print("\n------------------------------")
    print("HOÀN TẤT!")
    print(f"- Tổng số file .wav đã được chuyển ra ngoài: {move_count}")
    print(f"- Tổng số thư mục 'gallery'/'query' đã được xóa bỏ: {remove_count}")

if __name__ == "__main__":
    merge_gallery_query()
