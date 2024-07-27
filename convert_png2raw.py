import os
import cv2

# GTAV_1080 폴더 경로
# base_dir = '/mnt/nfs_shared_data/dataset/MVS-Synth/GTAV_1080'
base_dir = '/home/gyojin.han/ftp_shared_data/dataset/Inter4K/Inter4K/60fps/UHD'
folders = [f'{i:04d}' for i in range(901, 1001)]

for folder in folders:
    images_path = os.path.join(base_dir, folder, 'images')
    
    for file_name in os.listdir(images_path):
        if file_name.endswith('.png'):
            img_path = os.path.join(images_path, file_name)
            raw_path = img_path.replace('.png', '.raw')
            
            # PNG 파일 읽기
            img = cv2.imread(img_path, -1)
            # RGBA 이미지를 RGB로 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # RAW 파일로 저장
            with open(raw_path, 'wb') as out:
                img.tofile(out)

print('PNG to RAW conversion completed for all files.')