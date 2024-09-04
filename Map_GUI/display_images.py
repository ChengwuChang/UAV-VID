import os
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageDisplayer:
    def __init__(self, label, image_folder):
        self.label = label
        self.image_folder = image_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.current_image_index = 0
        # 設定 QLabel 自動縮放圖片
        self.label.setScaledContents(True)

    def display_current_image(self):
        if self.image_files:
            image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
            pixmap = QPixmap(image_path)
            # 如果需要，根據 QLabel 的大小縮放圖片
            pixmap = pixmap.scaled(self.label.size(), aspectRatioMode=Qt.KeepAspectRatio,
                                   transformMode=Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self.display_current_image()

    def previous_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        self.display_current_image()
