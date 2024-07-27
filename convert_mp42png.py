import os
import cv2

# Define the base directory
base_dir = '/home/gyojin.han/ftp_shared_data/dataset/Inter4K/Inter4K/60fps/UHD'

# Function to convert video to images
def video_to_images(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as PNG file
        frame_name = f"{output_dir}/frame_{frame_count:04d}.png"
        cv2.imwrite(frame_name, frame)
        frame_count += 1
    
    # Release the capture
    cap.release()

# Loop over all the video files
for i in range(1, 1001):
    video_name = f"{i}.mp4"
    video_path = os.path.join(base_dir, video_name)
    output_dir = os.path.join(base_dir, f"{i:04d}", "images")
    
    if os.path.exists(video_path):
        print(f"Processing {video_name}...")
        video_to_images(video_path, output_dir)
    else:
        print(f"{video_name} not found.")

print("Conversion completed.")