import cv2
import numpy as np
import undistort


def pre_processing(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([108, 36, 141])
    hsv_high = np.array([121, 163, 255])
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    kernel = np.ones((5, 5), np.uint8)

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


def divide_lines(frame, lines):
    vertical_lines = [[], []]
    vertical_lines_cnt = [0, 0]
    vertical_lines_rho = [0, 0]
    horizontal_lines = [[], []]
    horizontal_lines_cnt = [0, 0]
    horizontal_lines_rho = [0, 0]
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            theta_deg = rad2deg(theta)
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
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0) if cur_index == 0 else (255, 0, 0), 2)
                else:
                    cur_index = 0
                    if abs(rho - vertical_lines_rho[0]) > 50:
                        cur_index = 1
                        if vertical_lines_cnt[1] == 0:
                            vertical_lines_rho[1] = rho
                    vertical_lines[cur_index].append([slope, x0])
                    vertical_lines_cnt[cur_index] += 1
                    cv2.line(frame, (int(x0), 0), (int(x0), 480), (0, 0, 0) if cur_index == 0 else (255, 0, 0), 2)
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
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255) if cur_index == 0 else (0, 255, 0), 2)
                else:
                    cur_index = 0
                    if abs(rho - horizontal_lines_rho[0]) > 50:
                        cur_index = 1
                        if horizontal_lines_cnt[1] == 0:
                            horizontal_lines_rho[1] = rho
                    horizontal_lines[cur_index].append([slope, x0])
                    horizontal_lines_cnt[cur_index] += 1
                    cv2.line(frame, (int(x0), 0), (int(x0), 480), (0, 0, 255) if cur_index == 0 else (0, 255, 0), 2)
    return vertical_lines, horizontal_lines


def get_avg_mid_point(lines):
    ax = ay = 0
    for line in lines:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            x, y = get_mid_point(slope, intercept)
        else:
            x = intercept
            y = 240
        ax += x
        ay += y
    ax /= len(lines)
    ay /= len(lines)
    return ax, ay


def get_avg_mid_point_with_segment_y(lines, top, bottom):
    ax = ay = 0
    for line in lines:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            y0 = top
            x0 = (y0 - intercept) / slope
            y1 = bottom
            x1 = (y1 - intercept) / slope
            x = (x0 + x1) / 2
            y = (y0 + y1) / 2
        else:
            x = intercept
            y = 240
        ax += x
        ay += y
    ax /= len(lines)
    ay /= len(lines)
    return ax, ay


def get_avg_mid_point_with_segment_x(lines, left, right):
    ax = ay = 0
    for line in lines:
        slope, intercept = line
        x = y = 0
        if slope != np.inf:
            x0 = left
            y0 = slope * x0 + intercept
            x1 = right
            y1 = slope * x1 + intercept
            x = (x0 + x1) / 2
            y = (y0 + y1) / 2
        else:
            x = intercept
            y = 240
        ax += x
        ay += y
    ax /= len(lines)
    ay /= len(lines)
    return ax, ay


def cross_detect(frame, show_source=False, show_edge=False, need_undistort=False):
    if need_undistort:
        frame = undistort.undistort(undistort.s908_params, frame)
    result = pre_processing(frame)
    edges = cv2.Canny(result, 0, 0, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)
    vertical_lines, horizontal_lines = divide_lines(frame, lines)

    if (len(vertical_lines[0]) == 0) or (len(horizontal_lines[0]) == 0) or (len(vertical_lines[1]) == 0) or (
            len(horizontal_lines[1]) == 0):
        cv2.imshow('edges', edges)
        cv2.imshow('frame', frame)
        return

    vertical_lines_mid_point = [get_avg_mid_point(vertical_lines[0]), get_avg_mid_point(vertical_lines[1])]
    top_y = min(vertical_lines_mid_point[0][1], vertical_lines_mid_point[1][1])
    bottom_y = max(vertical_lines_mid_point[0][1], vertical_lines_mid_point[1][1])

    horizontal_lines_mid_point = [get_avg_mid_point_with_segment_y(horizontal_lines[0], top_y, bottom_y),
                                  get_avg_mid_point_with_segment_y(horizontal_lines[1], top_y, bottom_y)]
    cv2.circle(frame, (int(horizontal_lines_mid_point[0][0]), int(horizontal_lines_mid_point[0][1])),
               5, (0, 255, 255), -1)
    cv2.circle(frame, (int(horizontal_lines_mid_point[1][0]), int(horizontal_lines_mid_point[1][1])),
               5, (0, 255, 255), -1)

    left_x = min(horizontal_lines_mid_point[0][0], horizontal_lines_mid_point[1][0])
    right_x = max(horizontal_lines_mid_point[0][0], horizontal_lines_mid_point[1][0])
    horizontal_mid_point = [(horizontal_lines_mid_point[0][0] + horizontal_lines_mid_point[1][0]) / 2,
                            (horizontal_lines_mid_point[0][1] + horizontal_lines_mid_point[1][1]) / 2]

    vertical_lines_mid_point = [get_avg_mid_point_with_segment_x(vertical_lines[0], left_x, right_x),
                                get_avg_mid_point_with_segment_x(vertical_lines[1], left_x, right_x)]
    cv2.circle(frame, (int(vertical_lines_mid_point[0][0]), int(vertical_lines_mid_point[0][1])),
               5, (255, 0, 255), -1)
    cv2.circle(frame, (int(vertical_lines_mid_point[1][0]), int(vertical_lines_mid_point[1][1])),
               5, (255, 0, 255), -1)

    vertical_mid_point = [(vertical_lines_mid_point[0][0] + vertical_lines_mid_point[1][0]) / 2,
                          (vertical_lines_mid_point[0][1] + vertical_lines_mid_point[1][1]) / 2]

    cross_mid_point = [(horizontal_mid_point[0] + vertical_mid_point[0]) / 2,
                       (horizontal_mid_point[1] + vertical_mid_point[1]) / 2]
    cv2.circle(frame, (int(cross_mid_point[0]), int(cross_mid_point[1])), 5, (255, 255, 0), -1)

    if (280 < cross_mid_point[0] < 360) and (200 < cross_mid_point[1] < 280):
        print('cross detected')

    if show_edge:
        cv2.imshow('edges', edges)
    if show_source:
        cv2.imshow('frame', frame)


if __name__ == "__main__":
    choice = input("Mode: \n 1. Photo\n 2. Camera\n")
    if choice == '1':
        frame = cv2.imread('./test.jpg')
        cross_detect(frame, show_source=True, show_edge=True, need_undistort=False)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)
    elif choice == '2':
        video_number = input("Give the number of the video device:")
        video = cv2.VideoCapture(int(video_number))
        img_size = (640, 480)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
        video.set(cv2.CAP_PROP_FPS, 30)

        while True:
            _, frame = video.read()
            cross_detect(frame, show_source=True, show_edge=True, need_undistort=False)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
        exit(0)
    else:
        print('Invalid input')
        exit(1)
