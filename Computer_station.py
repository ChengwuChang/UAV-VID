import socket  # 導入 socket 庫，用於網路通訊
import numpy as np  # 導入 NumPy 庫，用於數組操作
import cv2  # 導入 OpenCV 庫，用於影像處理

# 地面站的 IP 和埠號
HOST = '地面站的IP'  # 替換成地面站的 IP
PORT = 9000        # 與地面站使用的埠號一致，注意防火牆通行

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 創建一個 TCP 客戶端套接字
client.connect((HOST, PORT))  # 連接到指定的地面站 IP 和埠號

while True:  # 持續循環接收影像數據
    # 接收數據長度
    length = client.recv(16)  # 接收影像數據的長度
    length = int(length)  # 將長度轉換為整數

    # 接收影像數據
    data = b""  # 初始化一個空的字節串
    while len(data) < length:  # 直到接收到完整的影像數據
        packet = client.recv(4096)  # 每次最多接收 4096 字節數據
        if not packet:  # 如果接收的數據包為空，則表示連接已關閉
            break  # 跳出循環
        data += packet  # 將接收到的數據添加到 data 中

    # 解碼影像
    frame_data = np.frombuffer(data, np.uint8)  # 將接收到的字節數據轉換為 NumPy 陣列
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # 解碼影像數據為 OpenCV 格式

    # 在這裡可以進行影像處理
    # 例如：cv2.imshow("接收到的影像", frame)  # 顯示接收到的影像（如果需要顯示）

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 檢查是否按下 'q' 鍵
        break  # 跳出循環以結束程序

client.close()  # 關閉與地面站的連接
cv2.destroyAllWindows()  # 銷毀所有 OpenCV 創建的窗口
