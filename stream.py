import time
import cv2
import os
import shutil
import pyautogui
from threading import Thread, Lock
from UAV import split_image, main

# 定義調整參數
big_map_img = cv2.imread("Big_map_collect/test_big_map0920.jpg")


# 指定要分割的行和列數
num_rows = 6
num_cols = 6
all_blocks, block_height, block_width = split_image(big_map_img, num_rows, num_cols)
blocks = all_blocks
print(f"Blocks length before calling main: {len(blocks)}")


# 定義RTSP流的URL
url = 'rtsp://admin:admin@192.168.42.108:554/cam/realmonitor?channel=1&subtype=2'
cap = cv2.VideoCapture(url)

# 檢查是否成功打開
if not cap.isOpened():
    print("無法打開RTSP流")
    exit()

print("RTSP流已打開")

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"刪除 {file_path} 時發生錯誤: {e}")

# 清空 UAV_path-drone 資料夾，確保地圖存入
image_folder_drone = 'UAV_path-drone'
clear_folder(image_folder_drone)
os.makedirs(image_folder_drone, exist_ok=True)

center_points = []
# 創建 UAV_path 資料夾，作為存檔資料夾
image_folder_record = 'UAV_path'
os.makedirs(image_folder_record, exist_ok=True)

# 將大地圖存入 UAV_path-drone 和 UAV_path 資料夾中，命名為 'path_0.jpg'
big_map_path_drone = os.path.join(image_folder_drone, 'path_0.jpg')
big_map_path_record = os.path.join(image_folder_record, 'path_0.jpg')
cv2.imwrite(big_map_path_drone, big_map_img)
cv2.imwrite(big_map_path_record, big_map_img)
print(f"已保存大地圖至 {big_map_path_drone} 和 {big_map_path_record}")

# 創建保存影像的資料夾
output_folder = 'captured_frames'
clear_folder(output_folder)  # 清空資料夾
os.makedirs(output_folder, exist_ok=True)
current_directory = os.getcwd()
print("當前工作目錄:", current_directory)

# 設置緩衝區大小為1，確保處理的是最新幀
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
frame_count = 0
lock = Lock()
capture_frame = None
process_frame = None

def get_latest_image(folder):
    """從資料夾中獲取最新的圖片"""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if files:
        latest_file = max(files, key=os.path.getctime)  # 根據創建時間取得最新文件
        return latest_file
    return None

blocks = all_blocks  # 初始化 blocks

def process_image():
    global process_frame, blocks
    while True:
        if process_frame is not None:
            start_time = time.time()
            try:
                if blocks is None:
                    blocks = all_blocks

                blocks = main(all_blocks, blocks, process_frame, block_height, block_width, center_points)

                output_image_drone = os.path.join(image_folder_drone, f'path_{frame_count}.jpg')
                output_image_record = os.path.join(image_folder_record, f'path_{frame_count}.jpg')
                cv2.imwrite(output_image_drone, process_frame)
                cv2.imwrite(output_image_record, process_frame)
                print(f"已保存處理後圖片到 {output_image_drone} 和 {output_image_record}")

            except Exception as e:
                print(f"處理圖片時發生錯誤: {e}")

            end_time = time.time()
            print("辨識流程所花時間 = %.3f 秒" % (end_time - start_time))

            with lock:
                process_frame = None

try:
    processing_thread = Thread(target=process_image, daemon=True)
    processing_thread.start()

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            print("無法讀取視頻幀")
            break

        with lock:
            if process_frame is None:
                frame_count += 1
                image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"已儲存 {image_path}")

                process_frame = frame.copy()
                blocks = all_blocks  # 確保每次都重設 blocks

        screen_width, screen_height = pyautogui.size()
        latest_image = get_latest_image(image_folder_drone)
        if latest_image:
            path_image = cv2.imread("UAV_path-drone/path_0.jpg")
            height, width, _ = path_image.shape
            aspect_ratio = width / height
            new_width = screen_width // 2
            new_height = int(new_width / aspect_ratio)

            resized_image = cv2.resize(path_image, (new_width, new_height))
            image_name = os.path.basename(latest_image)
            cv2.namedWindow('Path_Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Path_Image', resized_image)

        end_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
