import cv2
import os
from pathlib import Path

class path_video():
    def __init__(self):
        super().__init__()
        self.image_folder = "UAV_path"
        self.image_folder2 = "output_frames"
        self.image_files = [os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder) if
                            os.path.isfile(os.path.join(self.image_folder, file))]
        self.image_files2 = [os.path.join(self.image_folder2, file) for file in os.listdir(self.image_folder2) if
                             os.path.isfile(os.path.join(self.image_folder2, file))]
        self.screen_width = 1920  # 螢幕寬度為1920

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')          # 設定影片的格式為 mp4v
        self.out = cv2.VideoWriter('output.mp4', self.fourcc, 1.0, (960,  540))  # 產生空的影片
        self.out2 = cv2.VideoWriter('output2.mp4', self.fourcc, 1.0, (960,  540))  # 產生空的影片

        self.Get_video()

    def Get_video(self):
        for i in range(0, 41):
            image = cv2.imread(f'UAV_path/path_{i}.jpg', 1)
            # 取得影像的寬度和高度，計算影像的長寬比
            height, width, _ = image.shape
            aspect_ratio = width / height

            # 將影像寬度調整為螢幕寬度的一半，高度按比例縮放
            new_width = self.screen_width // 2
            new_height = int(new_width / aspect_ratio)
            # 縮放圖像大小
            resized_image = cv2.resize(image, (new_width, new_height))

            self.out.write(resized_image)  # 將取得的每一幀圖像寫入空的影片

            image2 = cv2.imread(f'output_frames/frame_{i}.jpg', 1)
            if image2 is None:
                print(f"Unable to read image: output_frames/{i}.jpg")
                continue
            # 获取图像的宽度和高度，計算影像的長寬比
            height2, width2, _ = image2.shape
            aspect_ratio2 = width2 / height2

            # 將影像寬度調整為螢幕寬度的一半，高度按比例縮放
            new_height2 = int(new_width / aspect_ratio2)
            resized_image2 = cv2.resize(image2, (new_width, new_height2))
            self.out2.write(resized_image2)  # 將取得的每一幀圖像寫入空的影片
        self.out.release()
        self.out2.release()

    def get_file_path(self):
        # 獲取當前文件路徑
        script_path = Path(__file__).resolve()
        # 獲取當前文件目錄
        script_folder = script_path.parent
        return script_folder

    def get_video_path(self, filename):
        video_folder = self.get_file_path()
        video_path = video_folder / filename
        video_path_str = str(video_path).replace('\\', '/')

        if not video_path.exists():
            raise FileNotFoundError(f"The file {filename} does not exist in the 'video' folder.")

        return video_path_str

