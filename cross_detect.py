import cv2
import numpy as np
from sklearn.cluster import KMeans

video = cv2.VideoCapture(1)
img_size = (640, 480)
video.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
video.set(cv2.CAP_PROP_FPS, 30)
kernel = np.ones((5, 5), np.uint8)

def get_params(camera_matrix, dist_coefs):
    global img_size
    return {
        'camera_matrix' : camera_matrix,
        'dist_coefs' : dist_coefs,
        'new_cammtx' : cv2.getOptimalNewCameraMatrix(
             camera_matrix, dist_coefs, img_size, 0
        )[0]
    }

def undistort(params, frame):
    return cv2.undistort(frame, params['camera_matrix'],
        params['dist_coefs'], None, params['new_cammtx']
    )

g200_params = get_params(camera_matrix=np.array([
        [391.459091, 0.000000, 329.719318],
        [0.000000, 391.714735, 229.722416],
        [0.000000, 0.000000, 1.000000]
    ]),
    dist_coefs=np.array([
        -0.34456177150589806, 0.08938559391911026,
        0.0026686183140887153, -0.0035206005954522388, 0.0
    ])
)

s908_params = get_params(camera_matrix=np.array([
        [452.2764826544423 , 0.0, 359.89953113602235],
        [0.0, 451.6034994124971, 254.37194889475958],
        [0.000000, 0.000000, 1.000000]
    ]),
    dist_coefs=np.array([
        -0.35302786455375235, 0.08773253097338021,
        0.00402433714716149, -0.00787628412411706, 0.0
    ])
)



while True:
    _, frame = video.read()
    frame = undistort(s908_params, frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([108, 36, 141])
    hsv_high = np.array([121, 163, 255])
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    min_area = 300

    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    filtered_mask = np.zeros_like(mask)
    cv2.drawContours(filtered_mask, filtered_contours, -1, 255, cv2.FILLED)

    filtered_mask = cv2.bitwise_not(filtered_mask)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    result = cv2.drawContours(filtered_mask, filtered_contours, -1, (0, 255, 0), 3)

    result = cv2.bitwise_not(result)

    edges = cv2.Canny(result, 0, 0, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)

    # 绘制检测到的直线
    intercepts_slopes = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            slope = -a / b if b != 0 else np.inf
            intercept = y0 - slope * x0
            if slope != np.inf :
                intercepts_slopes.append((intercept, slope))

    # 将截距和斜率转换为NumPy数组
    intercepts_slopes = np.array(intercepts_slopes)

    # 使用K-means聚类算法进行分组
    kmeans = KMeans(n_clusters=4, random_state=0).fit(intercepts_slopes)

    labels = kmeans.labels_

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # 不同类别的颜色
    if lines is not None:
        for i, line in enumerate(lines):
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            slope = -a / b if b != 0 else np.inf
            intercept = y0 - slope * x0
            # color = colors[labels[i]]
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow('edges', edges)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break