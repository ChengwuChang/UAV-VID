#-*- coding: utf-8 -*-
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5 import QtCore, QtMultimedia

from opencv_engine import opencv_engine

class video_controller(object):
    def __init__(self, video_path, ui):
        self.video_path = video_path
        self.ui = ui
        self.media_player = QMediaPlayer()
        self.init_video_info()

        # 添加標誌，用於檢查是否正在手動拖動滑塊
        self.is_slider_pressed = False

    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)
        self.video_fps = videoinfo["fps"]
        self.video_total_frame_count = videoinfo["frame_count"]
        self.ui.horizontalSlider.setRange(0, self.video_total_frame_count - 1)

        self.ui.horizontalSlider.sliderMoved.connect(self.set_frame_position)
        self.media_player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.video_path)))
        self.media_player.setVideoOutput(self.ui.widget_2)
        self.media_player.mediaStatusChanged.connect(self.handle_error)

    def set_frame_position(self):
        # 從滑塊獲取值並設置為影片的播放位置
        position = self.ui.horizontalSlider.value()
        self.media_player.setPosition(position)

    def play(self):
        self.media_player.play()

    def stop(self):
        self.media_player.stop()

    def pause(self):
        self.media_player.pause()

    def handle_error(self, error):
        if error != QMediaPlayer.NoError:
            print("Error:", self.media_player.errorString())
