from PyQt5 import QtWidgets, QtWebEngineWidgets
from PyQt5.QtCore import pyqtSignal
import folium, io, sys, json
from folium.plugins import Draw


class WebEnginePage(QtWebEngineWidgets.QWebEnginePage):
    lonlat = pyqtSignal(list)

    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        coords_dict = json.loads(msg)
        self.coords = coords_dict['geometry']['coordinates'][0]
        self.lonlat.emit(self.coords)
        print("emit")


class MapWidget(QtWidgets.QWidget):
    def __init__(self, coordinates_label, export_button, gridlayout):
        super().__init__()
        self.route_coordinates = []  # 存放路徑座標
        # 添加 QLabel 以顯示座標
        self.coordinates_label = coordinates_label

        # 添加 QPushButton 以導出座標
        self.export_button = export_button
        self.export_button.clicked.connect(self.export_coordinates)

        self.folium_map(gridlayout)

    def folium_map(self, gridLayout):
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
        gridLayout.addWidget(self.view, 0, 0, 1, 1)

    def getLonLat(self, lonlat):
        print("Received coordinates:", lonlat)
        self.route_coordinates.append(lonlat)  # 將座標添加到路徑座標列表中
        self.reload_map()
        # 顯示座標
        self.coordinates_label.setText(f"Received coordinates: {lonlat}")

    def reload_map(self):
        # 重新加載地圖
        self.m = folium.Map(location=[23.562559063008074, 120.47881278473741],
                            zoom_start=13,
                            tiles='https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                            attr='default')
        # 畫 PolyLine
        folium.PolyLine(locations=self.route_coordinates, color='blue').add_to(self.m)
        # 保存地圖到 BytesIO
        self.m.save(self.data, close_file=False)
        # 重新設置地圖的 HTML，以顯示更新後的地圖
        self.view.setHtml(self.data.getvalue().decode())

    def export_coordinates(self):
        # 將座標寫入到.txt中
        try:
            with open("coordinates.txt", "w") as file:
                for coord in self.route_coordinates:
                    file.write(f"{coord}\n")
            print("座標存取成功")
        except Exception as e:
            print("座標存取失敗", e)


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     widget = MapWidget()
#     widget.show()
#     sys.exit(app.exec_())
