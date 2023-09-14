import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个空窗口
cv2.namedWindow('Color Selector')

# 创建滑动条来调整HSV范围
cv2.createTrackbar('Hue Min', 'Color Selector', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Color Selector', 179, 179, nothing)
cv2.createTrackbar('Saturation Min', 'Color Selector', 0, 255, nothing)
cv2.createTrackbar('Saturation Max', 'Color Selector', 255, 255, nothing)
cv2.createTrackbar('Value Min', 'Color Selector', 0, 255, nothing)
cv2.createTrackbar('Value Max', 'Color Selector', 255, 255, nothing)

# 打开摄像头
cap = cv2.VideoCapture(1)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    # 将帧转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 读取滑动条的当前位置
    h_min = cv2.getTrackbarPos('Hue Min', 'Color Selector')
    h_max = cv2.getTrackbarPos('Hue Max', 'Color Selector')
    s_min = cv2.getTrackbarPos('Saturation Min', 'Color Selector')
    s_max = cv2.getTrackbarPos('Saturation Max', 'Color Selector')
    v_min = cv2.getTrackbarPos('Value Min', 'Color Selector')
    v_max = cv2.getTrackbarPos('Value Max', 'Color Selector')

    # 定义HSV颜色范围
    lower_range = np.array([h_min, s_min, v_min])
    upper_range = np.array([h_max, s_max, v_max])

    # 根据HSV颜色范围创建掩码
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # 将掩码应用于原始帧
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 显示原始帧、掩码和结果
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()