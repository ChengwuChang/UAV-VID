
import time
import cv2
import os
import pyautogui
import shutil
from time import sleep
from UAV_vedio import split_image, main


# 定義調整參數
big_map_img = cv2.imread("Big_map_collect/test_big_map0917.jpg")

# 指定要分割的行和列數
num_rows = 6
num_cols = 6
new_num_rows = 3
new_num_cols = 3
all_blocks,block_height,block_width = split_image(big_map_img, num_rows, num_cols)
blocks = all_blocks

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

# 獲取螢幕的寬度和高度
screen_width, screen_height = pyautogui.size()

# 設定截圖區域為影片螢幕
# left_half_region = (0, 0, screen_width // 2, screen_height)
x1, y1, x2, y2 = (0,90,1100,550)
# 等待開始截圖的條
input("按 'p' 鍵開始截圖，並按 Enter 確認。")

output_folder = 'captured_frames_video'
os.makedirs(output_folder, exist_ok=True)
current_directory = os.getcwd()
print("當前工作目錄:", current_directory)
center_points = []
frame_count = 0
while True:
    # 截取指定區域
    start_time = time.time()#0831修改處(計算單張時間)
    myScreenshot = pyautogui.screenshot(region=(x1, y1, x2, y2))
    save_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
    myScreenshot.save(save_path)
    frame = cv2.imread(save_path)
    print(f"frame_{frame_count}")
    frame_count += 1
    try:
        blocks = main(blocks, frame, center_points)
    except Exception as e:
        print(f"處理圖片時發生錯誤: {e}")
        continue

    image_folder = 'UAV_path-drone'
    # 將影像寬度調整為螢幕寬度的一半，高度按比例縮放
    image = cv2.imread(f'{image_folder}/path_0.jpg', 1)
    # 取得影像的寬度和高度，計算影像的長寬比
    height, width, _ = image.shape
    aspect_ratio = width / height
    new_width = screen_width // 2
    new_height = int(new_width / aspect_ratio)
    # 縮放圖像大小
    resized_image = cv2.resize(image, (new_width, new_height))
    image_name = os.path.basename(f'{image_folder}/path_0.jpg')
    end_time = time.time()#0831修改處(計算單張時間)
    # 在視窗中顯示影像名稱顯示縮放後的圖片
    cv2.setWindowTitle('Image', image_name)
    cv2.imshow('Image', resized_image)

    time.sleep(2)
    print("單張辨識所花時間 = %.3f 秒"%(end_time - start_time))#0831修改處(計算單張時間)


    # 檢查是否按下 'q' 鍵來終止
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
