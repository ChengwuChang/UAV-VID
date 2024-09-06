import os
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

class ImageDisplayer:
    def __init__(self, label, image_folder):
        self.label = label
        self.image_folder = image_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.current_image_index = 0
        self.timer = QTimer()  # 用於自動更新的定時器
        self.auto_play_timer = QTimer()  # 用於自動播放的定時器
        self.auto_playing = False  # 標記是否正在自動播放

        self.timer.timeout.connect(self.check_for_new_images)  # 每次時間到時檢查是否有新圖片
        self.timer.start(5000)  # 每5秒檢查一次
        self.label.setScaledContents(True)

    def display_current_image(self):
        if self.image_files:
            image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(self.label.size(), aspectRatioMode=Qt.KeepAspectRatio,
                                   transformMode=Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def next_image(self):
        self.update_image_list()
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_current_image()
        else:
            print("已經是最後一張圖片。")

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
        else:
            print("已經是第一張圖片。")

    def exit_display(self):
        self.label.clear()
        self.stop_auto_play()  # 停止自動播放

    def check_for_new_images(self):
        # 檢查是否有新增的圖片
        updated_image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if len(updated_image_files) > len(self.image_files):
            self.image_files = updated_image_files
            print("檢測到新圖片")
            if self.auto_playing:
                self.next_image()

    def toggle_auto_play(self):
        if self.auto_playing:
            self.stop_auto_play()
        else:
            self.start_auto_play()

    def start_auto_play(self):
        if not self.auto_playing:
            print("開始自動播放")
            self.auto_playing = True
            self.auto_play_timer.timeout.connect(self.next_image)
            self.auto_play_timer.start(3000)  # 每3秒顯示下一張圖片

    def stop_auto_play(self):
        if self.auto_playing:
            print("停止自動播放")
            self.auto_playing = False
            self.auto_play_timer.stop()

    def update_image_list(self):
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def display_images(self):
        self.display_current_image()
        if not self.auto_playing:
            self.start_auto_play()  # 開始自動播放
