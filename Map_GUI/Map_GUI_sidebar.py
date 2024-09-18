# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Map_GUI_sidebar.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1017, 657)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.icon_only_widget = QtWidgets.QWidget(self.centralwidget)
        self.icon_only_widget.setObjectName("icon_only_widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.icon_only_widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.icon_only_widget)
        self.label.setMinimumSize(QtCore.QSize(50, 50))
        self.label.setMaximumSize(QtCore.QSize(50, 50))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/icon/uav.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.home_btn_1 = QtWidgets.QPushButton(self.icon_only_widget)
        self.home_btn_1.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.home_btn_1.setIcon(icon)
        self.home_btn_1.setCheckable(True)
        self.home_btn_1.setAutoExclusive(True)
        self.home_btn_1.setObjectName("home_btn_1")
        self.verticalLayout.addWidget(self.home_btn_1)
        self.map_btn_1 = QtWidgets.QPushButton(self.icon_only_widget)
        self.map_btn_1.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/map.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.map_btn_1.setIcon(icon1)
        self.map_btn_1.setCheckable(True)
        self.map_btn_1.setAutoExclusive(True)
        self.map_btn_1.setObjectName("map_btn_1")
        self.verticalLayout.addWidget(self.map_btn_1)
        self.chart_btn_1 = QtWidgets.QPushButton(self.icon_only_widget)
        self.chart_btn_1.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/chart.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.chart_btn_1.setIcon(icon2)
        self.chart_btn_1.setCheckable(True)
        self.chart_btn_1.setAutoExclusive(True)
        self.chart_btn_1.setObjectName("chart_btn_1")
        self.verticalLayout.addWidget(self.chart_btn_1)
        self.video_btn_1 = QtWidgets.QPushButton(self.icon_only_widget)
        self.video_btn_1.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/youtube.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.video_btn_1.setIcon(icon3)
        self.video_btn_1.setCheckable(True)
        self.video_btn_1.setAutoExclusive(True)
        self.video_btn_1.setObjectName("video_btn_1")
        self.verticalLayout.addWidget(self.video_btn_1)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 496, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.exit_btn_1 = QtWidgets.QPushButton(self.icon_only_widget)
        self.exit_btn_1.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/cancel.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exit_btn_1.setIcon(icon4)
        self.exit_btn_1.setCheckable(True)
        self.exit_btn_1.setAutoExclusive(True)
        self.exit_btn_1.setObjectName("exit_btn_1")
        self.verticalLayout_3.addWidget(self.exit_btn_1)
        self.horizontalLayout_20.addWidget(self.icon_only_widget)
        self.full_menu_widget = QtWidgets.QWidget(self.centralwidget)
        self.full_menu_widget.setObjectName("full_menu_widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.full_menu_widget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.full_menu_widget)
        self.label_2.setMaximumSize(QtCore.QSize(40, 40))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/icon/uav.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.home_btn_2 = QtWidgets.QPushButton(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.home_btn_2.setFont(font)
        self.home_btn_2.setIcon(icon)
        self.home_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.home_btn_2.setCheckable(True)
        self.home_btn_2.setAutoExclusive(True)
        self.home_btn_2.setObjectName("home_btn_2")
        self.verticalLayout_2.addWidget(self.home_btn_2)
        self.map_btn_2 = QtWidgets.QPushButton(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.map_btn_2.setFont(font)
        self.map_btn_2.setIcon(icon1)
        self.map_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.map_btn_2.setCheckable(True)
        self.map_btn_2.setAutoExclusive(True)
        self.map_btn_2.setObjectName("map_btn_2")
        self.verticalLayout_2.addWidget(self.map_btn_2)
        self.chart_btn_2 = QtWidgets.QPushButton(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.chart_btn_2.setFont(font)
        self.chart_btn_2.setIcon(icon2)
        self.chart_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.chart_btn_2.setCheckable(True)
        self.chart_btn_2.setAutoExclusive(True)
        self.chart_btn_2.setObjectName("chart_btn_2")
        self.verticalLayout_2.addWidget(self.chart_btn_2)
        self.video_btn_2 = QtWidgets.QPushButton(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.video_btn_2.setFont(font)
        self.video_btn_2.setIcon(icon3)
        self.video_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.video_btn_2.setCheckable(True)
        self.video_btn_2.setAutoExclusive(True)
        self.video_btn_2.setObjectName("video_btn_2")
        self.verticalLayout_2.addWidget(self.video_btn_2)
        self.Image_display_btn = QtWidgets.QPushButton(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.Image_display_btn.setFont(font)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("picture.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Image_display_btn.setIcon(icon5)
        self.Image_display_btn.setObjectName("Image_display_btn")
        self.verticalLayout_2.addWidget(self.Image_display_btn)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 481, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.exit_btn_2 = QtWidgets.QPushButton(self.full_menu_widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.exit_btn_2.setFont(font)
        self.exit_btn_2.setIcon(icon4)
        self.exit_btn_2.setIconSize(QtCore.QSize(14, 14))
        self.exit_btn_2.setCheckable(True)
        self.exit_btn_2.setAutoExclusive(True)
        self.exit_btn_2.setObjectName("exit_btn_2")
        self.verticalLayout_4.addWidget(self.exit_btn_2)
        self.horizontalLayout_20.addWidget(self.full_menu_widget)
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.widget = QtWidgets.QWidget(self.widget_3)
        self.widget.setEnabled(True)
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.menu_btn = QtWidgets.QPushButton(self.widget)
        self.menu_btn.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icon/list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menu_btn.setIcon(icon6)
        self.menu_btn.setIconSize(QtCore.QSize(14, 14))
        self.menu_btn.setCheckable(True)
        self.menu_btn.setObjectName("menu_btn")
        self.horizontalLayout_4.addWidget(self.menu_btn)
        spacerItem2 = QtWidgets.QSpacerItem(287, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.search_input = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.search_input.setFont(font)
        self.search_input.setObjectName("search_input")
        self.horizontalLayout.addWidget(self.search_input)
        self.search_btn = QtWidgets.QPushButton(self.widget)
        self.search_btn.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icon/search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.search_btn.setIcon(icon7)
        self.search_btn.setIconSize(QtCore.QSize(14, 14))
        self.search_btn.setObjectName("search_btn")
        self.horizontalLayout.addWidget(self.search_btn)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        spacerItem3 = QtWidgets.QSpacerItem(287, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.verticalLayout_5.addWidget(self.widget)
        self.stackedWidget = QtWidgets.QStackedWidget(self.widget_3)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.page)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.frame_28 = QtWidgets.QFrame(self.frame_2)
        self.frame_28.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_28.setObjectName("frame_28")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.frame_28)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_13 = QtWidgets.QLabel(self.frame_28)
        font = QtGui.QFont()
        font.setFamily("ROG Fonts")
        font.setPointSize(16)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_21.addWidget(self.label_13)
        self.verticalLayout_16.addWidget(self.frame_28)
        self.frame_29 = QtWidgets.QFrame(self.frame_2)
        self.frame_29.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_29.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_29.setObjectName("frame_29")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_29)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView_6 = QtWidgets.QGraphicsView(self.frame_29)
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.gridLayout.addWidget(self.graphicsView_6, 0, 0, 1, 1)
        self.verticalLayout_16.addWidget(self.frame_29)
        self.frame_30 = QtWidgets.QFrame(self.frame_2)
        self.frame_30.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_30.setObjectName("frame_30")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.frame_30)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.frame_31 = QtWidgets.QFrame(self.frame_30)
        self.frame_31.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_31.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_31.setObjectName("frame_31")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout(self.frame_31)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.show_data_4 = QtWidgets.QLabel(self.frame_31)
        self.show_data_4.setObjectName("show_data_4")
        self.horizontalLayout_23.addWidget(self.show_data_4)
        self.horizontalLayout_22.addWidget(self.frame_31)
        self.frame_32 = QtWidgets.QFrame(self.frame_30)
        self.frame_32.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_32.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_32.setObjectName("frame_32")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.frame_32)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.pushButton_11 = QtWidgets.QPushButton(self.frame_32)
        self.pushButton_11.setObjectName("pushButton_11")
        self.verticalLayout_17.addWidget(self.pushButton_11)
        self.pushButton_12 = QtWidgets.QPushButton(self.frame_32)
        self.pushButton_12.setObjectName("pushButton_12")
        self.verticalLayout_17.addWidget(self.pushButton_12)
        self.horizontalLayout_22.addWidget(self.frame_32)
        self.verticalLayout_16.addWidget(self.frame_30)
        self.horizontalLayout_27.addWidget(self.frame_2)
        self.frame_33 = QtWidgets.QFrame(self.frame)
        self.frame_33.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_33.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_33.setObjectName("frame_33")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.frame_33)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.frame_34 = QtWidgets.QFrame(self.frame_33)
        self.frame_34.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_34.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_34.setObjectName("frame_34")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.frame_34)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.graphicsView_7 = QtWidgets.QGraphicsView(self.frame_34)
        self.graphicsView_7.setObjectName("graphicsView_7")
        self.gridLayout_12.addWidget(self.graphicsView_7, 0, 0, 1, 1)
        self.verticalLayout_18.addWidget(self.frame_34)
        self.frame_35 = QtWidgets.QFrame(self.frame_33)
        self.frame_35.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_35.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_35.setObjectName("frame_35")
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout(self.frame_35)
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.label_14 = QtWidgets.QLabel(self.frame_35)
        font = QtGui.QFont()
        font.setFamily("ROG Fonts")
        font.setPointSize(16)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_24.addWidget(self.label_14)
        self.verticalLayout_18.addWidget(self.frame_35)
        self.frame_36 = QtWidgets.QFrame(self.frame_33)
        self.frame_36.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_36.setObjectName("frame_36")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout(self.frame_36)
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.frame_37 = QtWidgets.QFrame(self.frame_36)
        self.frame_37.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_37.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_37.setObjectName("frame_37")
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout(self.frame_37)
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.show_coordinate_3 = QtWidgets.QLabel(self.frame_37)
        self.show_coordinate_3.setObjectName("show_coordinate_3")
        self.horizontalLayout_26.addWidget(self.show_coordinate_3)
        self.horizontalLayout_25.addWidget(self.frame_37)
        self.frame_38 = QtWidgets.QFrame(self.frame_36)
        self.frame_38.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_38.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_38.setObjectName("frame_38")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(self.frame_38)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.pushButton_13 = QtWidgets.QPushButton(self.frame_38)
        self.pushButton_13.setObjectName("pushButton_13")
        self.verticalLayout_19.addWidget(self.pushButton_13)
        self.pushButton_14 = QtWidgets.QPushButton(self.frame_38)
        self.pushButton_14.setObjectName("pushButton_14")
        self.verticalLayout_19.addWidget(self.pushButton_14)
        self.horizontalLayout_25.addWidget(self.frame_38)
        self.verticalLayout_18.addWidget(self.frame_36)
        self.horizontalLayout_27.addWidget(self.frame_33)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_7 = QtWidgets.QFrame(self.page_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame_9)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.path_widget = QVideoWidget(self.frame_9)
        self.path_widget.setMinimumSize(QtCore.QSize(800, 450))
        self.path_widget.setObjectName("path_widget")
        self.gridLayout_8.addWidget(self.path_widget, 0, 0, 1, 1)
        self.verticalLayout_8.addWidget(self.frame_9)
        self.frame_8 = QtWidgets.QFrame(self.frame_7)
        self.frame_8.setMinimumSize(QtCore.QSize(0, 100))
        self.frame_8.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_6 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_6.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_6.addWidget(self.pushButton_6)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_8)
        self.pushButton_5.setMinimumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("ROG Fonts")
        self.pushButton_5.setFont(font)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icon/uav.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_5.setIcon(icon8)
        self.pushButton_5.setIconSize(QtCore.QSize(14, 14))
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_6.addWidget(self.pushButton_5)
        self.verticalLayout_8.addWidget(self.frame_8)
        self.gridLayout_3.addWidget(self.frame_7, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.frame_6 = QtWidgets.QFrame(self.page_3)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.csv_chart = QtWidgets.QWidget(self.frame_6)
        self.csv_chart.setObjectName("csv_chart")
        self.verticalLayout_7.addWidget(self.csv_chart)
        self.chart_open_file_btn = QtWidgets.QPushButton(self.frame_6)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.chart_open_file_btn.setFont(font)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icon/open-folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.chart_open_file_btn.setIcon(icon9)
        self.chart_open_file_btn.setIconSize(QtCore.QSize(14, 14))
        self.chart_open_file_btn.setObjectName("chart_open_file_btn")
        self.verticalLayout_7.addWidget(self.chart_open_file_btn)
        self.gridLayout_4.addWidget(self.frame_6, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.page_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_3 = QtWidgets.QFrame(self.page_4)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_4 = QtWidgets.QFrame(self.frame_3)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.widget_2 = QVideoWidget(self.frame_4)
        self.widget_2.setMinimumSize(QtCore.QSize(800, 450))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_7.addWidget(self.widget_2, 0, 0, 1, 1)
        self.verticalLayout_6.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setMinimumSize(QtCore.QSize(0, 100))
        self.frame_5.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_4.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton_4.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.pushButton_4.setFont(font)
        self.pushButton_4.setIcon(icon9)
        self.pushButton_4.setIconSize(QtCore.QSize(14, 14))
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_5.addWidget(self.pushButton_4)
        self.horizontalSlider = QtWidgets.QSlider(self.frame_5)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(150, 30))
        self.horizontalSlider.setMaximumSize(QtCore.QSize(150, 30))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_5.addWidget(self.horizontalSlider)
        self.pushButton = QtWidgets.QPushButton(self.frame_5)
        self.pushButton.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.pushButton.setFont(font)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icon/stop-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon10)
        self.pushButton.setIconSize(QtCore.QSize(14, 14))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_2.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton_2.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.pushButton_2.setFont(font)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/icon/video-pause-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon11)
        self.pushButton_2.setIconSize(QtCore.QSize(14, 14))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_5.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_3.setMinimumSize(QtCore.QSize(100, 30))
        self.pushButton_3.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.pushButton_3.setFont(font)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/icon/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon12)
        self.pushButton_3.setIconSize(QtCore.QSize(14, 14))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_5.addWidget(self.pushButton_3)
        self.verticalLayout_6.addWidget(self.frame_5)
        self.gridLayout_5.addWidget(self.frame_3, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_4)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.page_5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_8 = QtWidgets.QLabel(self.page_5)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(15)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_6.addWidget(self.label_8, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setEnabled(True)
        self.page_6.setObjectName("page_6")
        self.Image_display = QtWidgets.QPushButton(self.page_6)
        self.Image_display.setGeometry(QtCore.QRect(50, 520, 131, 31))
        self.Image_display.setObjectName("Image_display")
        self.display_image = QtWidgets.QLabel(self.page_6)
        self.display_image.setEnabled(True)
        self.display_image.setGeometry(QtCore.QRect(40, 20, 801, 471))
        self.display_image.setMinimumSize(QtCore.QSize(801, 471))
        self.display_image.setAutoFillBackground(False)
        self.display_image.setText("")
        self.display_image.setObjectName("display_image")
        self.last_page = QtWidgets.QPushButton(self.page_6)
        self.last_page.setGeometry(QtCore.QRect(410, 520, 75, 23))
        self.last_page.setObjectName("last_page")
        self.next_page = QtWidgets.QPushButton(self.page_6)
        self.next_page.setGeometry(QtCore.QRect(520, 520, 75, 23))
        self.next_page.setObjectName("next_page")
        self.exit_display = QtWidgets.QPushButton(self.page_6)
        self.exit_display.setGeometry(QtCore.QRect(630, 520, 75, 23))
        self.exit_display.setObjectName("exit_display")
        self.auto_display = QtWidgets.QPushButton(self.page_6)
        self.auto_display.setGeometry(QtCore.QRect(210, 520, 75, 23))
        self.auto_display.setObjectName("auto_display")
        self.display_image.raise_()
        self.Image_display.raise_()
        self.last_page.raise_()
        self.next_page.raise_()
        self.exit_display.raise_()
        self.auto_display.raise_()
        self.stackedWidget.addWidget(self.page_6)
        self.verticalLayout_5.addWidget(self.stackedWidget)
        self.horizontalLayout_20.addWidget(self.widget_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.display_image.setBuddy(self.display_image)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(5)
        self.menu_btn.toggled['bool'].connect(self.icon_only_widget.setVisible) # type: ignore
        self.menu_btn.toggled['bool'].connect(self.full_menu_widget.setHidden) # type: ignore
        self.home_btn_1.toggled['bool'].connect(self.home_btn_2.setChecked) # type: ignore
        self.home_btn_2.toggled['bool'].connect(self.home_btn_1.setChecked) # type: ignore
        self.map_btn_1.toggled['bool'].connect(self.map_btn_2.setChecked) # type: ignore
        self.map_btn_2.toggled['bool'].connect(self.map_btn_1.setChecked) # type: ignore
        self.exit_btn_2.clicked.connect(MainWindow.close) # type: ignore
        self.exit_btn_1.clicked.connect(MainWindow.close) # type: ignore
        self.chart_btn_1.toggled['bool'].connect(self.chart_btn_2.setChecked) # type: ignore
        self.chart_btn_2.toggled['bool'].connect(self.chart_btn_1.setChecked) # type: ignore
        self.video_btn_1.toggled['bool'].connect(self.video_btn_2.setChecked) # type: ignore
        self.video_btn_2.toggled['bool'].connect(self.video_btn_1.setChecked) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "UAV"))
        self.home_btn_2.setText(_translate("MainWindow", "Home"))
        self.map_btn_2.setText(_translate("MainWindow", "Map"))
        self.chart_btn_2.setText(_translate("MainWindow", "Chart"))
        self.video_btn_2.setText(_translate("MainWindow", "Video"))
        self.Image_display_btn.setText(_translate("MainWindow", "Image"))
        self.exit_btn_2.setText(_translate("MainWindow", "Exit"))
        self.search_input.setPlaceholderText(_translate("MainWindow", "Search..."))
        self.label_13.setText(_translate("MainWindow", "DATA CHART"))
        self.show_data_4.setText(_translate("MainWindow", "Data"))
        self.pushButton_11.setText(_translate("MainWindow", "Import"))
        self.pushButton_12.setText(_translate("MainWindow", "Outport"))
        self.label_14.setText(_translate("MainWindow", "COORDINATE LOG"))
        self.show_coordinate_3.setText(_translate("MainWindow", "Coordinate"))
        self.pushButton_13.setText(_translate("MainWindow", "Import"))
        self.pushButton_14.setText(_translate("MainWindow", "Outport"))
        self.pushButton_6.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_5.setText(_translate("MainWindow", "無人機姿態"))
        self.chart_open_file_btn.setText(_translate("MainWindow", "Openfile"))
        self.pushButton_4.setText(_translate("MainWindow", "Openfile"))
        self.pushButton.setText(_translate("MainWindow", "Stop"))
        self.pushButton_2.setText(_translate("MainWindow", "Pause"))
        self.pushButton_3.setText(_translate("MainWindow", "Play"))
        self.label_8.setText(_translate("MainWindow", "Search Page"))
        self.Image_display.setText(_translate("MainWindow", "顯示影像"))
        self.last_page.setText(_translate("MainWindow", "上一張"))
        self.next_page.setText(_translate("MainWindow", "下一張"))
        self.exit_display.setText(_translate("MainWindow", "退出"))
        self.auto_display.setText(_translate("MainWindow", "即時影像化"))
from PyQt5.QtMultimediaWidgets import QVideoWidget
import resource_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
