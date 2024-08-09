import time
import cv2
import os
from UAV import split_image, main
import shutil


# 定義調整參數
big_map_img = cv2.imread("Big_map_collect/23-1.jpg")

# 指定要分割的行和列數
num_rows = 6
num_cols = 6
new_num_rows = 3
new_num_cols = 3
all_blocks, block_height, block_width = split_image(big_map_img, num_rows, num_cols)
blocks = all_blocks

# 定義RTSP流的URL
url = 'rtsp://admin:53373957@192.168.144.108:554/cam/realmonitor?channel=1&subtype=1'

# 打開RTSP流
cap = cv2.VideoCapture(url)

# 檢查是否成功打開
if not cap.isOpened():
    print("無法打開RTSP流")
    exit()

print("RTSP流已打開")


def clear_folder(folder_path):
    # 檢查資料夾是否存在
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在")
        return

    # 刪除資料夾中的所有文件和子資料夾
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪除文件或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 刪除子資料夾
        except Exception as e:
            print(f"刪除 {file_path} 時發生錯誤: {e}")

# 指定要清空的資料夾路徑
folder_path = 'captured_frames'
# 清空資料夾
clear_folder(folder_path)
print(f"資料夾 {folder_path} 已清空")

# 創建保存圖片的資料夾
output_folder = 'captured_frames'
os.makedirs(output_folder, exist_ok=True)
current_directory = os.getcwd()
print("當前工作目錄:", current_directory)


# 設置緩衝區大小為1，確保處理的是最新帧
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0.5)

center_points = []
frame_count = 0

# 循環處理每一帧
while cap.isOpened():
    # 讀取一帧視頻
    ret, frame = cap.read()

    # 檢查是否成功讀取
    if not ret:
        print("無法讀取視頻帧")
        break

        # 儲存圖片
    frame_count += 1
    image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"已儲存 {image_path}")

    # 讀取保存的圖片，並將其傳遞給main函數
    image = cv2.imread(image_path)
    try:
        blocks = main(all_blocks,blocks, image,block_height,block_width)
    except Exception as e:
        print(f"處理圖片時發生錯誤: {e}")
        continue

    image_folder = 'UAV_path-drone'
    # 將影像寬度調整為螢幕寬度的一半，高度按比例縮放
    image = cv2.imread(f'{image_folder}/path_0.jpg', 1)
    
    image_name = os.path.basename(f'{image_folder}/path_0.jpg')
    # 在視窗中顯示影像名稱顯示圖片
    cv2.setWindowTitle('Image', image_name)
    cv2.imshow('Image', image)

    time.sleep(2)
    
    # 顯示原始影像帧
    cv2.imshow('RTSP Stream', frame)

    
    # 按下 'q' 鍵退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放視頻捕獲對象
cap.release()

# 關閉所有OpenCV窗口
cv2.destroyAllWindows()
