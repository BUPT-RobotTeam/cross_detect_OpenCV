import cv2
import numpy as np
import undistort

video = cv2.VideoCapture(1)
img_size = (640, 480)
video.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
video.set(cv2.CAP_PROP_FPS, 30)

while True:
    _, frame = video.read()
    frame = undistort.undistort(undistort.s908_params, frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('./test.jpg', frame)
        print('saved')
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
