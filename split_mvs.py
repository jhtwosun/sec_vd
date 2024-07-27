import os
import random

# 경로 설정
base_dir = '/mnt/nfs_shared_data/dataset/MVS-Synth/GTAV_1080'

# 폴더 리스트 생성
folders = [f'{i:04d}' for i in range(120)]

# 파일 경로 수집
file_paths = []
for folder in folders:
    images_path = os.path.join(base_dir, folder, 'images')
    depths_path = os.path.join(base_dir, folder, 'depths')
    poses_path = os.path.join(base_dir, folder, 'poses')
    
    for file_name in os.listdir(images_path):
        if file_name.endswith('.png'):
            file_base = os.path.splitext(file_name)[0]
            image_file = os.path.join(folder, 'images', file_name)
            depth_file = os.path.join(folder, 'depths', f'{file_base}.exr')
            pose_file = os.path.join(folder, 'poses', f'{file_base}.json')
            file_paths.append(f'{image_file} {depth_file} {pose_file}')

# 데이터 셔플
random.shuffle(file_paths)

# 데이터 분할 (70% train, 15% val, 15% test)
total_files = len(file_paths)
train_split = int(total_files * 0.7)
val_split = int(total_files * 0.85)

train_files = file_paths[:train_split]
val_files = file_paths[train_split:val_split]
test_files = file_paths[val_split:]

# 파일 저장
def write_to_file(file_list, file_name):
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)

write_to_file(train_files, 'train.txt')
write_to_file(val_files, 'val.txt')
write_to_file(test_files, 'test.txt')

print('Data split completed. Files saved as train.txt, val.txt, test.txt')