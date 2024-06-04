import cv2
import numpy as np
import os
import cv2 as cv
import open3d as o3d
import math
from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5 import QtCore, QtMultimedia

from path_video_output import path_video


class VideoProcessingThread(QThread):
    """
        QThread寫法，在run()裡面寫上需要運行的動作: 會使用線程是為了防止在進行特徵點比對時，GUI介面會有無法回應的問題。
        video_screen_shot(影片路徑, 儲存的資料夾名稱): 讀取影片截成圖片儲存在output_frames資料夾中，會回傳儲存的圖片張數。
        (這部分需要改成直接讀取攝影機畫面，並將圖片儲存。)
        build_path(影片圖片張數): 實現SIFT特徵點比對，繪製出路徑，將比對結果存成圖片，放入UAV_path資料夾。
        path_video(): 從path_video_output.py匯入的function。將圖片轉成影片儲存成"output.mp4"和"output2.mp4"
        回傳結束線程訊號
    """
    processing_done = pyqtSignal(str)
    frame_count = 0

    def __init__(self, video_path, output_folder_name):
        super().__init__()
        self.video_path = video_path
        self.output_folder_name =output_folder_name

        self.all_boxes = []  # 儲存所有方框的座標
        self.center_points = []  # 儲存所有中心點座標
        self.target_points = []
        self.matched_points = []

        self.average_keypoints_in_boxes = []  # 儲存每個方框的平均特徵點
        self.farthest_keypoints_in_boxes = []  # 儲存每個方框內距離平均特徵點最遠的特徵點
        self.distances = []  # 儲存每個方框內最遠特徵點與平均特徵點之間的距離
        self.img_keypoints = []  # 儲存影像 img 的所有特徵點座標

    def run(self):
        self.frame_count = self.video_screen_shot(self.video_path, self.output_folder_name)
        self.build_path(self.frame_count)
        path_video()
        self.processing_done.emit(self.video_path)

    def video_screen_shot(self, video_file, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_file)  # 打開影片文件
        # check
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)  # 取得影片幀率
        frame_interval = int(fps * 3)

        frame_count = 0
        total_frames = 0

        while cap.isOpened() and total_frames < 150:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                # 存圖像
                frame_file = os.path.join(output_folder, f'frame_{total_frames}.jpg')
                cv2.imwrite(frame_file, frame)
                print(f'Frame {total_frames} saved.')
                total_frames += 1
            frame_count += 1
        cap.release()
        print('Video frames extraction completed.')
        return total_frames

    def build_path(self, frame_count):
        total_path = 0
        MIN_MATCH_COUNT = 10
        img3 = cv2.imread(r"C:\Users\ASUS\image-stitching-opencv-master\Map_GUI\23-2.jpg", 1)  # trainImage
        sift = cv2.SIFT_create()  # 初始化 SIFT 檢測器

        kp3, des3 = sift.detectAndCompute(img3, None)  # 使用 SIFT 找到img3的關鍵點和描述子

        # 定義 FLANN 參數
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)  # 創建 FLANN 匹配器

        for i in range(0, frame_count):
            img = cv2.imread(f'output_frames/frame_{i}.jpg', 1)  # queryImage
            # 使用 SIFT 找到關鍵點和描述子
            kp1, des1 = sift.detectAndCompute(img, None)
            # 進行特徵匹配
            matches = flann.knnMatch(des1, des3, k=2)

            # 根據 Lowe's ratio 測試存儲所有良好的匹配
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # 執行 Homography 變換
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp3[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                # 計算旋轉角度（以度為單位）
                rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

                h, w, d = img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)

                x1, y1 = np.int32(dst[0][0])
                x2, y2 = np.int32(dst[1][0])
                x3, y3 = np.int32(dst[2][0])
                x4, y4 = np.int32(dst[3][0])
                center_x = (x1 + x2 + x3 + x4) / 4
                center_y = (y1 + y2 + y3 + y4) / 4

                self.center_points.append((center_x, center_y))
                self.matched_points.append(src_pts.squeeze())
                self.target_points.append(dst_pts.squeeze())
                self.all_boxes.append(np.int32(dst))

                point_cloud = o3d.geometry.PointCloud()
                target_point_cloud = o3d.geometry.PointCloud()
                for points in self.matched_points:
                    # 將每個特徵點轉換為 3D 點
                    points_3d = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
                    # 將 3D 點添加到點雲中
                    point_cloud.points.extend(o3d.utility.Vector3dVector(points_3d))
                for points in self.target_points:
                    # 將每個特徵點轉換為 3D 點
                    target_points_3d = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
                    # 將 3D 點添加到點雲中
                    target_point_cloud.points.extend(o3d.utility.Vector3dVector(target_points_3d))

                print(f"Angle {i}")
                print(rotation_angle)

                color2 = (0, 0, 255)  # BGR
                # 將所有中心點標示出來
                for point in self.center_points:
                    center_x, center_y = map(int, point)
                    cv.circle(img3, (center_x, center_y), radius=5, color=(255, 255, 255), thickness=25)

                # 畫三角形
                center_x, center_y = map(int, self.center_points[i])
                base_length = 50
                half_base = base_length // 2
                # Height of an equilateral triangle is sqrt(3)/2 times the base length
                height = int(0.866 * base_length)
                offset_factor = 50  # Adjust this value as needed

                # Calculate coordinates of base vertices
                vertex1 = (int(center_x - half_base - offset_factor), int(center_y + height))
                vertex2 = (int(center_x - offset_factor), int(center_y - height))
                vertex3 = (int(center_x + half_base - offset_factor), int(center_y + height))

                # Rotate the base vertices around the center point by a given angle (e.g., 45 degrees)
                vertex1_rotated = self.rotate_point(vertex1[0], vertex1[1], center_x, center_y, rotation_angle)
                vertex2_rotated = self.rotate_point(vertex2[0], vertex2[1], center_x, center_y, rotation_angle)
                vertex3_rotated = self.rotate_point(vertex3[0], vertex3[1], center_x, center_y, rotation_angle)

                # Draw the rotated triangle
                vertices_rotated = np.array([vertex1_rotated, vertex2_rotated, vertex3_rotated], dtype=np.int32)
                cv.fillPoly(img3, [vertices_rotated], color=(0 + 20 * i, 255 - 20 * i, 0))
                # 將所有中心點連接成直線
                for i in range(len(self.center_points) - 1):
                    cv.line(img3, tuple(map(int, self.center_points[i])), tuple(map(int, self.center_points[i + 1])), color2, 10)
                # 存進檔案中
                frame_file = os.path.join('UAV_path', f'path_{total_path}.jpg')
                cv2.imwrite(frame_file, img3)
                total_path += 1
                print("------------")
            else:
                print(f"Not enough matches are found in frame {i} - {len(good)}/{MIN_MATCH_COUNT}")


    def rotate_point(self, x, y, cx, cy, angle):
        # Convert angle to radians
        theta = math.radians(angle)
        # Perform rotation
        x_rotated = cx + (x - cx) * math.cos(theta) - (y - cy) * math.sin(theta)
        y_rotated = cy + (x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
        return x_rotated, y_rotated


class video_controller():
    """
        video_controller(ui, 影片路徑): 用來播放影片
    """
    def __init__(self, ui, video_path):
        super().__init__()
        self.ui = ui
        self.video_path = video_path
        self.media_player = QMediaPlayer()
        self.init_video_info()
        # self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)

    def init_video_info(self):
        self.media_player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(
            self.video_path)))
        self.media_player.setVideoOutput(self.ui.path_widget)

    def play(self):
        self.media_player.play()


class navigation(object):
    """
        navigation(ui): 會在Map_Controller.py中調用。
        open_file(): 按下path按鈕會出發這項功能，點選需要切片的影片(這邊要改成攝影機鏡頭的圖片，變成點選按鈕是開啟攝影機)，
                     之後產生UAV_path資料夾，然後進行VideoProcessingThread()的功能。最後將影片顯示在GUI介面上。
        on_processing_done(): 原先只有一個畫面的做法。
        two_video(): 播放路徑影片。
    """
    def __init__(self, ui):
        self.ui = ui
        self.output_folder_name = 'output_frames'
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_6.clicked.connect(self.open_file)

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(None, "Open file Window", "./", "Video Files(*.mp4 *.avi)")
        self.video_path = filename
        self.Create_UAV_file()
        # 使用線程以避免視窗無回應
        self.thread = VideoProcessingThread(self.video_path, self.output_folder_name)
        self.thread.processing_done.connect(self.two_video)
        self.thread.start()

    # 播放路徑影片
    def on_processing_done(self):
        output_video_path = path_video().get_video_path(filename="output.mp4")
        self.video_controller = video_controller(ui=self.ui, video_path=output_video_path)
        self.video_controller.play()

    def two_video(self):
        # Layout to hold the main video and overlay video
        layout = QVBoxLayout(self.ui.path_widget)

        # Main video widget
        self.main_video_widget = QVideoWidget()
        layout.addWidget(self.main_video_widget)
        self.main_media_player = QMediaPlayer()
        self.overlay_media_player = QMediaPlayer()
        self.main_media_player.setVideoOutput(self.main_video_widget)

        # Overlay video widget
        self.overlay_video_widget = QVideoWidget()
        self.overlay_video_widget.setFixedSize(320, 240)  # Set fixed size for the overlay video
        self.overlay_video_widget.setParent(self.main_video_widget)
        self.overlay_media_player.setVideoOutput(self.overlay_video_widget)

        # Load videos
        main_video_url = QUrl.fromLocalFile("output.mp4")
        overlay_video_url = QUrl.fromLocalFile("output2.mp4")

        self.main_media_player.setMedia(QMediaContent(main_video_url))
        self.overlay_media_player.setMedia(QMediaContent(overlay_video_url))

        # Play videos
        self.main_media_player.play()
        self.overlay_media_player.play()

        # Make sure overlay video is on top
        self.overlay_video_widget.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Adjust the position of the overlay video widget
        self.overlay_video_widget.move(self.width() - self.overlay_video_widget.width() - 70, 70)

    def Create_UAV_file(self):
        folder_name = "./UAV_path"  # 定義文件夾名稱
        current_dir = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(current_dir, folder_name)  # 使用os.path.join連接路径和文件夾名稱
        if not os.path.exists(folder_path):  # 檢查是否存在
            os.mkdir(folder_path)
            print(f"文件夾 '{folder_path}' 已創建。")
        else:
            print(f"文件夾 '{folder_path}' 已存在。")
