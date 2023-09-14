import cv2
import numpy as np
from sklearn.cluster import KMeans
import undistort
import math

video = cv2.VideoCapture(1)
img_size = (640, 480)
video.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
video.set(cv2.CAP_PROP_FPS, 30)
kernel = np.ones((5, 5), np.uint8)

def pre_processing(frame):
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
    ret = cv2.drawContours(filtered_mask, filtered_contours, -1, (0, 255, 0), 3)
    ret = cv2.bitwise_not(ret)
    return ret


def get_mid_point(slope, intercept):
    x0 = 0
    y0 = intercept + slope * x0
    x1 = 640
    y1 = intercept + slope * x1
    if abs(y0) > 480:
        y0 = 480 if y0 > 0 else 0
        x0 = (y0 - intercept) / slope
    if abs(y1) > 480:
        y1 = 480 if y1 > 0 else 0
        x1 = (y1 - intercept) / slope
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def rad2deg(rad):
    return rad / np.pi * 180


while True:
    _, frame = video.read()
    frame = undistort.undistort(undistort.s908_params, frame)
    result = pre_processing(frame)
    edges = cv2.Canny(result, 0, 0, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)
    vertical_lines = [[],[]]
    vertical_lines_cnt = [0, 0]
    vertical_lines_rho = [0, 0]
    horizontal_lines = [[],[]]
    horizontal_lines_cnt = [0, 0]
    horizontal_lines_rho = [0, 0]
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            theta_deg = rad2deg(theta)
            print(theta_deg)
            if 120 >= theta_deg >= 70:
                if vertical_lines_cnt[0] == 0:
                    vertical_lines_rho[0] = rho
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                slope = -a / b if b != 0 else np.inf
                intercept = y0 - slope * x0
                if slope != np.inf:
                    cur_index = 0
                    if abs(rho - vertical_lines_rho[0]) > 50:
                        cur_index = 1
                        if vertical_lines_cnt[1] == 0:
                            vertical_lines_rho[1] = rho
                    vertical_lines[cur_index].append([slope, intercept])
                    vertical_lines_cnt[cur_index] += 1
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0) if cur_index==0 else (255,0,0), 2)
                else:
                    cur_index = 0
                    if abs(rho - vertical_lines_rho[0]) > 50:
                        cur_index = 1
                        if vertical_lines_cnt[1] == 0:
                            vertical_lines_rho[1] = rho
                    vertical_lines[cur_index].append([slope, x0])
                    vertical_lines_cnt[cur_index] += 1
                    cv2.line(frame, (int(x0), 0), (int(x0), 480), (0, 0, 0) if cur_index==0 else (255,0,0) , 2)
            elif 0 <= theta_deg <= 30 or theta_deg >= 150:
                if horizontal_lines_cnt[0] == 0:
                    horizontal_lines_rho[0] = rho
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                slope = -a / b if b != 0 else np.inf
                intercept = y0 - slope * x0
                if slope != np.inf:
                    cur_index = 0
                    if abs(rho - horizontal_lines_rho[0]) > 50:
                        cur_index = 1
                        if horizontal_lines_cnt[1] == 0:
                            horizontal_lines_rho[1] = rho
                    horizontal_lines[cur_index].append([slope, intercept])
                    horizontal_lines_cnt[cur_index] += 1
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255) if cur_index==0 else (0,255,0), 2)
                else:
                    cur_index = 0
                    if abs(rho - horizontal_lines_rho[0]) > 50:
                        cur_index = 1
                        if horizontal_lines_cnt[1] == 0:
                            horizontal_lines_rho[1] = rho
                    horizontal_lines.append([slope, x0])
                    horizontal_lines_cnt[cur_index] += 1
                    cv2.line(frame, (int(x0), 0), (int(x0), 480), (0, 0, 255) if cur_index==0 else (0,255,0) , 2)

    ax = ay = 0

    if (len(vertical_lines[0]) == 0) or (len(horizontal_lines[0]) == 0) or (len(vertical_lines[1]) == 0) or (len(horizontal_lines[1]) == 0):
        cv2.imshow('edges', edges)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for line in vertical_lines[0]:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            x, y = get_mid_point(slope, intercept)
        else:
            x = intercept
            y = 240

        ax += x
        ay += y

    ax /= len(vertical_lines[0])
    ay /= len(vertical_lines[0])

    cv2.circle(frame,(int(ax), int(ay)), 5, (0, 255, 255), -1)

    bx = 0
    by = 0

    for line in vertical_lines[1]:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            x, y = get_mid_point(slope, intercept)
        else:
            x = intercept
            y = 240

        bx += x
        by += y

    bx /= len(vertical_lines[1])
    by /= len(vertical_lines[1])

    cv2.circle(frame,(int(bx), int(by)), 5, (0, 255, 255), -1)

    vertical_mid_point = [(ax + bx) / 2, (ay + by) / 2]

    ax = ay = 0

    for line in horizontal_lines[0]:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            x, y = get_mid_point(slope, intercept)
        else:
            x = intercept
            y = 320

        ax += x
        ay += y

    ax /= len(horizontal_lines[0])
    ay /= len(horizontal_lines[0])

    cv2.circle(frame,(int(ax), int(ay)), 5, (255, 0, 255), -1)

    bx = by = 0

    for line in horizontal_lines[1]:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            x, y = get_mid_point(slope, intercept)
        else:
            x = intercept
            y = 320

        bx += x
        by += y

    bx /= len(horizontal_lines[1])
    by /= len(horizontal_lines[1])

    cv2.circle(frame,(int(bx), int(by)), 5, (255, 0, 255), -1)

    horizontal_mid_point = [(ax + bx) / 2, (ay + by) / 2]

    cv2.circle(frame,(int((horizontal_mid_point[0]+vertical_mid_point[0])/2), int((horizontal_mid_point[1]+vertical_mid_point[1])/2)), 5, (114, 514, 0xA), -1)

    cv2.imshow('edges', edges)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
