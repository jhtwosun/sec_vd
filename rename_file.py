import os
import glob

def rename_files(base_path):
    # 각 하위 폴더에 대한 루프
    for i in range(1, 1001):
        folder_path = os.path.join(base_path, f"{i:04d}", "images")
        if os.path.exists(folder_path):
            # images 폴더 내의 모든 png, raw 파일 찾기
            file_list = glob.glob(os.path.join(folder_path, "frame_*.png")) + glob.glob(os.path.join(folder_path, "frame_*.raw"))
            for file_path in file_list:
                # 새 파일 이름 생성 (예: frame_0000.png -> 0000.png)
                base_name = os.path.basename(file_path)
                new_name = base_name.replace("frame_", "")
                new_path = os.path.join(folder_path, new_name)
                # 파일 이름 변경
                os.rename(file_path, new_path)
                print(f"Renamed {file_path} to {new_path}")
        else:
            print(f"No images folder found in {folder_path}")

# 스크립트 실행 (기본 경로는 수정해주세요)
base_directory = "/home/gyojin.han/ftp_shared_data/dataset/Inter4K/Inter4K/60fps/UHD/"
rename_files(base_directory)