import os
import OpenEXR
import Imath
import numpy as np
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def max_disparity_adjust(d, max_disparity_mu = 1.2, max_disparity_sigma = 0.2):
	return d * np.random.normal(max_disparity_mu, max_disparity_sigma)

def normalize_array(arr):
    # Inf 값이 아닌 요소들의 인덱스 마스크 생성
    mask = ~np.isinf(arr)
    
    # Inf 값이 아닌 요소들의 최소값과 최대값 계산
    min_val = np.min(arr[mask])
    max_val = np.max(arr[mask])
    
    # Inf 값이 아닌 요소들에 대해 정규화 수행
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    # Inf 값은 그대로 유지
    normalized_arr[~mask] = arr[~mask]
    
    return normalized_arr

def exr_to_numpy(exr_path):
    # EXR 파일 열기
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    # 이미지 크기 가져오기
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 채널 정보 가져오기
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_data = exr_file.channels('Y', pt)

    # NumPy 배열로 변환
    depth_array = np.frombuffer(depth_data[0], dtype=np.float32).reshape(height, width)
    return depth_array

# GTAV_1080 폴더 경로 설정
base_dir = '/mnt/nfs_shared_data/dataset/MVS-Synth/GTAV_1080'
folders = [f'{i:04d}' for i in range(120)]

for folder in folders:
    depths_path = os.path.join(base_dir, folder, 'depths')
    
    for file_name in os.listdir(depths_path):
        if file_name.endswith('.exr'):
            exr_path = os.path.join(depths_path, file_name)
            npy_path = exr_path.replace('.exr', '.npy')
            
            # EXR 파일을 NumPy 배열로 변환
            img_array = exr_to_numpy(exr_path)
            
            # NumPy 배열을 .npy 파일로 저장
            np.save(npy_path, img_array)

print('EXR to NumPy conversion completed for all files.')