import cv2  # 導入 OpenCV 庫，用於影像處理
import socket  # 導入 socket 庫，用於網路通訊
import numpy as np  # 導入 NumPy 庫，用於數組操作

#要先連接圖傳，接著跑這段程式，再跑電腦的程式
# RTSP 串流地址
url = 'rtsp://admin:53373957@192.168.144.108:554/cam/realmonitor?channel=1&subtype=1'

# 設定伺服器端的 TCP socket
HOST = '0.0.0.0'  # 接受所有的 IP
PORT = 9000        # 使用的埠號，還要注意防火牆能允許這樣的流量
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 創建一個 TCP 套接字
server.bind((HOST, PORT))  # 綁定 IP 和埠號
server.listen(1)  # 開始監聽來自客戶端的連接請求

print('等待連接...')  # 輸出等待連接的提示信息
conn, addr = server.accept()  # 接受客戶端的連接請求
print(f'已連接: {addr}')  # 輸出已連接的客戶端地址

# 開始接收 RTSP 串流
cap = cv2.VideoCapture(url)  # 使用 OpenCV 打開 RTSP 串流

while True:  # 持續循環接收影像
    ret, frame = cap.read()  # 從攝影機獲取一幀影像
    if not ret:  # 檢查是否成功獲取影像
        print("無法接收影像")  # 如果無法獲取影像，輸出錯誤信息
        break  # 跳出循環

    # 將影像轉換為字節數據
    _, buffer = cv2.imencode('.jpg', frame)  # 將影像編碼為 JPEG 格式的字節數據
    data = np.array(buffer)  # 將編碼後的數據轉換為 NumPy 陣列
    stringData = data.tobytes()  # 將 NumPy 陣列轉換為字節串

    # 發送數據長度，然後發送影像數據
    conn.send(str(len(stringData)).ljust(16).encode('utf-8'))  # 發送影像數據的長度，填充至 16 個字元
    conn.send(stringData)  # 發送實際的影像數據

cap.release()  # 釋放 VideoCapture 對象
conn.close()  # 關閉與客戶端的連接
server.close()  # 關閉伺服器
