# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtWebEngineWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal, QThread
import pyqtgraph as pg
import pandas as pd
import numpy as np
from folium.plugins import Draw, AntPath
import folium, io, sys, json
from itertools import chain

from Map_GUI_sidebar import Ui_MainWindow
from video_controller_gpu import video_controller
from Visual_navigation import navigation
from PyTeapot_csv import PyTeapot
from display_images import ImageDisplayer


class WebEnginePage(QtWebEngineWidgets.QWebEnginePage):
    lonlat = pyqtSignal(list)

    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        coords_dict = json.loads(msg)
        self.coords = coords_dict['geometry']['coordinates'][0]
        self.lonlat.emit(self.coords)

class PyTeapotThread(QThread):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        pyteapot = PyTeapot()
        pyteapot.main()
        self.finished.emit()


class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.navigation = navigation(ui=self.ui)

        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.home_btn_2.setChecked(True)
        # 初始化 ImageDisplayer
        self.image_displayer = ImageDisplayer(self.ui.display_image, "output_frames")

        # 連接按鈕點擊信號與對應的槽函數
        self.ui.home_btn_1.clicked.connect(self.on_home_btn_1_togged)
        self.ui.home_btn_2.clicked.connect(self.on_home_btn_2_togged)
        self.ui.map_btn_1.clicked.connect(self.on_map_btn_1_togged)
        self.ui.map_btn_2.clicked.connect(self.on_map_btn_2_togged)
        self.ui.chart_btn_1.clicked.connect(self.on_chart_btn_1_togged)
        self.ui.chart_btn_2.clicked.connect(self.on_chart_btn_2_togged)
        self.ui.video_btn_1.clicked.connect(self.on_video_btn_1_togged)
        self.ui.video_btn_2.clicked.connect(self.on_video_btn_2_togged)
        # 連接 Image_display 按鈕的點擊事件到 display_images 方法
        self.ui.Image_display_btn.clicked.connect(self.Image_display_btn_toggled)
        self.ui.Image_display.clicked.connect(self.display_images)
        # initial parameters
        self.t = np.zeros(10)
        self.y = np.zeros(10)
        self.plots = []
        self.scenes = []
        self.curves = []
        self.plot_tittle = ["X (m)", "Y (m)", "Z (m)", "Row (deg)", "Yaw(deg)", "Pitch(deg)"]
        self.history_t = []
        self.history_y = []

        self.route_coordinates = []  # 存放路徑座標

        self.charts = []
        self.chart_curves = []
        self.counter = 0
        self.csv_x = np.zeros(30)
        self.csv_y_1 = np.zeros(30)
        self.csv_y_2 = np.zeros(30)

        self.pyteapot_thread = None


        # functions
        self.navigation
        self.plot_data()
        self.folium_map()
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_4.clicked.connect(self.video_play)
        self.ui.pushButton_5.clicked.connect(self.start_visualization)
        self.ui.chart_open_file_btn.clicked.connect(self.load_csv)


    def display_images(self):
        self.image_displayer.display_current_image()

        # 如果你有設置其他按鈕來瀏覽圖片（下一張、上一張），可以在這裡連接
        # self.ui.next_button.clicked.connect(self.image_displayer.next_image)
        # self.ui.prev_button.clicked.connect(self.image_displayer.previous_image)
    def start_visualization(self):
        if self.pyteapot_thread is None or not self.pyteapot_thread.isRunning():
            self.pyteapot_thread = PyTeapotThread()
            self.pyteapot_thread.finished.connect(self.on_pyteapot_finished)
            self.pyteapot_thread.start()

    def on_pyteapot_finished(self):
        self.pyteapot_thread = None

    def video_play(self):
        # start path
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)")
        self.video_path = filename
        self.video_controller = video_controller(video_path=self.video_path,
                                                 ui=self.ui)
        self.video_controller.play()
        self.ui.pushButton_3.clicked.connect(self.video_controller.play)  # connect to function()
        self.ui.pushButton.clicked.connect(self.video_controller.stop)
        self.ui.pushButton_2.clicked.connect(self.video_controller.pause)

    # Function for searching
    def on_search_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        search_text = self.ui.search_input.text().strip()
        if search_text:
            self.ui.label_8.setText(search_text)

    # Change QPushbutton Checkable status when stackedWidget index changed
    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                   + self.ui.full_menu_widget.findChildren(QPushButton)

        for btn in btn_list:
            if index in [5]:
                btn.setAutoExclusive(False)
                btn.setChecked(False)
            else:
                btn.setAutoExclusive(True)

    # functions for changing menu page
    def on_home_btn_1_togged(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def on_home_btn_2_togged(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def on_map_btn_1_togged(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def on_map_btn_2_togged(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def on_chart_btn_1_togged(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_chart_btn_2_togged(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_video_btn_1_togged(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_video_btn_2_togged(self):
        self.ui.stackedWidget.setCurrentIndex(3)
    def Image_display_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(5)
    def plot_data(self):
        # set canvas
        pg.setConfigOptions(leftButtonPan=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        for i in range(6):
            plot_widget = pg.PlotWidget()
            plot_widget.addLegend()
            plot_widget.showGrid(x=True, y=True)
            curve = plot_widget.plot(pen=pg.mkPen(color=(i, 6), width=3))
            styles = {"color": "black", "font-size": "15px"}
            plot_widget.setTitle(self.plot_tittle[i], size="15px")
            plot_widget.setLabel("bottom", "Time", "sec", **styles)
            self.plots.append(plot_widget)
            self.curves.append(curve)

        # Add PlotWidgets to QVBoxLayout
        layout = QVBoxLayout()
        for plot in self.plots:
            layout.addWidget(plot)
        self.ui.graphicsView_6.setLayout(layout)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    # data update
    def update_plot(self):
        self.history_t.append(self.t)
        self.history_y.append(self.y)

        self.t = self.t[1:]
        self.t = np.append(self.t, self.t[-1] + 1)
        self.y = self.y[1:]
        self.y = np.append(self.y, np.random.randint(1, 50, size=1))
        # 在每個圖表中設置相同的資料
        for curve in self.curves:
            curve.setData(self.t, self.y)

        # 限制歷史資料最大保存量為 5
        if len(self.history_t) > 5:
            self.history_t.pop(0)
            self.history_y.pop(0)

        self.text(self.history_t, self.history_y)
        self.setFontSize(self.ui.show_data_4, 12)
        self.setFontSize(self.ui.show_coordinate_3, 12)

    # show datas
    def text(self, t, y):
        text = ""
        for time_data, value in zip(t[-5:], y[-5:]):
            text += "Time：{}，Data：{}\n".format(time_data[-1], value[-1])
        self.ui.show_data_4.setText(text)

    def setFontSize(self, widget, size):
        font = QFont("Times New Roman")
        font.setPointSize(size)
        widget.setFont(font)

    def folium_map(self):
        self.m = folium.Map(location=[23.562559063008074, 120.47881278473741],
                            zoom_start=13,
                            tiles='https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                            attr='default')
        Draw(export=True).add_to(self.m)
        self.data = io.BytesIO()
        self.m.save(self.data, close_file=False)
        self.view = QtWebEngineWidgets.QWebEngineView()
        self.page = WebEnginePage(self.view)
        self.page.lonlat.connect(self.getLonLat)
        self.view.setPage(self.page)
        self.view.setHtml(self.data.getvalue().decode())

        # 添加 QLabel 以顯示座標
        self.coordinates_label = self.ui.show_coordinate_3
        self.ui.gridLayout_12.addWidget(self.view, 0, 0, 1, 1)

        # 添加 QPushButton 以導出座標
        self.export_button = self.ui.pushButton_14
        self.export_button.clicked.connect(self.export_coordinates)

    def getLonLat(self, lonlat):
        # print("Received coordinates:", lonlat)
        self.route_coordinates.append(lonlat)  # 將座標添加到路徑座標列表中
        self.reload_map()
        # 顯示座標
        coordinates_text = "\n".join([f"{lonlat[i]}" for i in range(len(lonlat))])
        self.coordinates_label.setText(f"座標點:\n{coordinates_text}")

    def reload_map(self):
        # 重新加載地圖
        self.view.reload()
        self.m = None
        self.m = folium.Map(location=[23.562559063008074, 120.47881278473741],
                            zoom_start=13,
                            tiles='https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                            attr='default')
        locations = list(chain.from_iterable(self.route_coordinates))  # 降低list維度
        corrected_coordinates = [[coord[1], coord[0]] for coord in locations]  # 經緯度交換順序
        AntPath(corrected_coordinates, delay=400, weight=6, color="red", dash_array=[30, 15]).add_to(self.m)
        for i, coord in enumerate(corrected_coordinates, start=0):
            folium.Marker(location=coord, radius=5, color='blue', fill=True, fill_color='blue',
                          tooltip=f"Point {i}").add_to(self.m)

        # 保存地圖到 BytesIO
        self.data = None
        self.data = io.BytesIO()
        self.m.save(self.data, close_file=False)
        # 重新設置地圖的 HTML，以顯示更新後的地圖
        self.view.setHtml(self.data.getvalue().decode())

    def export_coordinates(self):
        # 將座標寫入到.txt中
        try:
            with open("coordinates.txt", "w") as file:
                for coord in self.route_coordinates:
                    file.write(f"{coord}\n")
            print("座標存取成功.")
        except Exception as e:
            print("座標存取失敗", e)

    def plot_csv_data(self):
        pg.setConfigOptions(leftButtonPan=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        tittle = ["Alt(m)", "VSpd(m/s)"]

        for i in range(2):
            plot_widget = pg.PlotWidget()
            plot_widget.addLegend()
            plot_widget.showGrid(x=True, y=True)
            curve = plot_widget.plot(pen=pg.mkPen(color=(i, 1), width=5))
            styles = {"color": "black", "font-size": "15px"}
            plot_widget.setTitle(tittle[i], size="15px")
            plot_widget.setLabel("bottom", "Time", "sec", **styles)
            self.charts.append(plot_widget)
            self.chart_curves.append(curve)

        layout = QVBoxLayout()
        for plot in self.charts:
            layout.addWidget(plot)
        self.ui.csv_chart.setLayout(layout)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.csv_data)
        self.timer.start()

    def load_csv(self):
        # 選擇要讀取的檔案
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'CSV Files (*.csv)')
        if file_path:
            # 從 CSV 文件中讀取資料
            self.datas_csv = pd.read_csv(file_path)

        self.plot_csv_data()

    def csv_data(self):
        # 建立 x 軸數據
        self.csv_x = self.csv_x[1:]
        self.csv_x = np.append(self.csv_x, self.csv_x[-1] + 0.5)
        # 取得第一個圖的資料（CSV檔案中的第12欄）
        self.csv_y_1 = self.csv_y_1[1:]
        self.csv_y_1 = np.append(self.csv_y_1, float(self.datas_csv.iloc[self.counter, 12]))
        # 取得第二個圖的資料（CSV檔案中的第13欄）
        self.csv_y_2 = self.csv_y_2[1:]
        self.csv_y_2 = np.append(self.csv_y_2, float(self.datas_csv.iloc[self.counter, 13]))
        # 更新第一個圖的曲線
        self.chart_curves[0].setData(self.csv_x, self.csv_y_1)
        # 更新第二個圖的曲線
        self.chart_curves[1].setData(self.csv_x, self.csv_y_2)
        self.counter += 1
