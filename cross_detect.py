import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import undistort

# video = cv2.VideoCapture(1)
# img_size = (640, 480)
# video.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
# video.set(cv2.CAP_PROP_FPS, 30)
kernel = np.ones((5, 5), np.uint8)

def pre_processing(frame):
    # frame = undistort.undistort(undistort.s908_params, frame)
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


frame = cv2.imread('./test.jpg')
result = pre_processing(frame)
edges = cv2.Canny(result, 0, 0, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)
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
        if slope != np.inf:
            Kx = a
            Ky = b
            C = -intercept*b
            intercepts_slopes.append((Kx, Ky, C))
        else:
            Kx = 1
            Ky = 0
            C = -x0
            intercepts_slopes.append((Kx, Ky, C))

intercepts_slopes = np.array(intercepts_slopes)

if len(intercepts_slopes) > 4:
    kmeans = KMeans(n_clusters=4).fit(intercepts_slopes)
    labels = kmeans.labels_

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # 不同类别的颜色
    if lines is not None:
        line_groups = [[] for _ in range(4)]
        tags = [False for _ in range(4)]
        for i, line in enumerate(lines):
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            slope = -a / b if b != 0 else np.inf
            intercept = y0 - slope * x0
            # if i < len(labels):
            cur_label = labels[i]
            color = colors[cur_label]
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            if slope != np.inf:
                line_groups[cur_label].append([slope, intercept])
            else:
                line_groups[cur_label].append([x0, slope])
                tags[cur_label] = True

        points = []

        if not tags[0]:
            mean_slope_group1 = np.mean([line[0] for line in line_groups[0]])
            mean_intercept_group1 = np.mean([line[1] for line in line_groups[0]])
            print("Group 1: y =", mean_slope_group1, "x +", mean_intercept_group1)
            x, y = get_mid_point(mean_slope_group1, mean_intercept_group1)
            points.append([x, y])
        else:
            mean_x0_group1 = np.mean([line[0] for line in line_groups[0]])
            print("Group 1: x =", mean_x0_group1)
            points.append([mean_x0_group1, 240])
        if not tags[1]:
            mean_slope_group2 = np.mean([line[0] for line in line_groups[1]])
            mean_intercept_group2 = np.mean([line[1] for line in line_groups[1]])
            print("Group 2: y =", mean_slope_group2, "x +", mean_intercept_group2)
            x, y = get_mid_point(mean_slope_group2, mean_intercept_group2)
            points.append([x, y])
        else:
            mean_x0_group2 = np.mean([line[0] for line in line_groups[1]])
            print("Group 2: x =", mean_x0_group2)
            points.append([mean_x0_group2, 240])
        if not tags[2]:
            mean_slope_group3 = np.mean([line[0] for line in line_groups[2]])
            mean_intercept_group3 = np.mean([line[1] for line in line_groups[2]])
            print("Group 3: y =", mean_slope_group3, "x +", mean_intercept_group3)
            x, y = get_mid_point(mean_slope_group3, mean_intercept_group3)
            points.append([x, y])
        else:
            mean_x0_group3 = np.mean([line[0] for line in line_groups[2]])
            print("Group 3: x =", mean_x0_group3)
            points.append([mean_x0_group3, 240])
        if not tags[3]:
            mean_slope_group4 = np.mean([line[0] for line in line_groups[3]])
            mean_intercept_group4 = np.mean([line[1] for line in line_groups[3]])
            print("Group 4: y =", mean_slope_group4, "x +", mean_intercept_group4)
            x, y = get_mid_point(mean_slope_group4, mean_intercept_group4)
            points.append([x, y])
        else:
            mean_x0_group4 = np.mean([line[0] for line in line_groups[3]])
            print("Group 4: x =", mean_x0_group4)
            points.append([mean_x0_group4, 240])

        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)

        mean_x = np.mean([point[0] for point in points])
        mean_y = np.mean([point[1] for point in points])

        cv2.circle(frame, (int(mean_x), int(mean_y)), 5, (255, 0, 0), -1)

cv2.imshow('edges', edges)
cv2.imshow('frame', frame)

cv2.waitKey(0)


# while True:
#     _, frame = video.read()
#     result = pre_processing(frame)
#     frame = undistort.undistort(undistort.s908_params, frame)
#     edges = cv2.Canny(result, 0, 0, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)
#     intercepts_slopes = []
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             slope = -a / b if b != 0 else np.inf
#             intercept = y0 - slope * x0
#             if slope != np.inf:
#                 intercepts_slopes.append((intercept, slope))
#
#     intercepts_slopes = np.array(intercepts_slopes)
#
#     if len(intercepts_slopes) > 4:
#         kmeans = KMeans(n_clusters=4, random_state=0).fit(intercepts_slopes)
#         labels = kmeans.labels_
#
#         colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # 不同类别的颜色
#         if lines is not None:
#             line_groups = [[] for _ in range(4)]
#             tags = [False for _ in range(4)]
#             for i, line in enumerate(lines):
#                 rho, theta = line[0]
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a * rho
#                 y0 = b * rho
#                 slope = -a / b if b != 0 else np.inf
#                 intercept = y0 - slope * x0
#                 if i < len(labels):
#                     cur_label = labels[i]
#                     color = colors[cur_label]
#                     x1 = int(x0 + 1000 * (-b))
#                     y1 = int(y0 + 1000 * a)
#                     x2 = int(x0 - 1000 * (-b))
#                     y2 = int(y0 - 1000 * a)
#                     cv2.line(frame, (x1, y1), (x2, y2), color, 2)
#                     if slope != np.inf:
#                         line_groups[cur_label].append([slope, intercept])
#                     else:
#                         line_groups[cur_label].append([x0, slope])
#                         tags[cur_label] = True
#
#             points = []
#
#             if not tags[0]:
#                 mean_slope_group1 = np.mean([line[0] for line in line_groups[0]])
#                 mean_intercept_group1 = np.mean([line[1] for line in line_groups[0]])
#                 print("Group 1: y =", mean_slope_group1, "x +", mean_intercept_group1)
#                 x, y = get_mid_point(mean_slope_group1, mean_intercept_group1)
#                 points.append([x, y])
#             else:
#                 mean_x0_group1 = np.mean([line[0] for line in line_groups[0]])
#                 print("Group 1: x =", mean_x0_group1)
#                 points.append([mean_x0_group1, 240])
#             if not tags[1]:
#                 mean_slope_group2 = np.mean([line[0] for line in line_groups[1]])
#                 mean_intercept_group2 = np.mean([line[1] for line in line_groups[1]])
#                 print("Group 2: y =", mean_slope_group2, "x +", mean_intercept_group2)
#                 x, y = get_mid_point(mean_slope_group2, mean_intercept_group2)
#                 points.append([x, y])
#             else:
#                 mean_x0_group2 = np.mean([line[0] for line in line_groups[1]])
#                 print("Group 2: x =", mean_x0_group2)
#                 points.append([mean_x0_group2, 240])
#             if not tags[2]:
#                 mean_slope_group3 = np.mean([line[0] for line in line_groups[2]])
#                 mean_intercept_group3 = np.mean([line[1] for line in line_groups[2]])
#                 print("Group 3: y =", mean_slope_group3, "x +", mean_intercept_group3)
#                 x, y = get_mid_point(mean_slope_group3, mean_intercept_group3)
#                 points.append([x, y])
#             else:
#                 mean_x0_group3 = np.mean([line[0] for line in line_groups[2]])
#                 print("Group 3: x =", mean_x0_group3)
#                 points.append([mean_x0_group3, 240])
#             if not tags[3]:
#                 mean_slope_group4 = np.mean([line[0] for line in line_groups[3]])
#                 mean_intercept_group4 = np.mean([line[1] for line in line_groups[3]])
#                 print("Group 4: y =", mean_slope_group4, "x +", mean_intercept_group4)
#                 x, y = get_mid_point(mean_slope_group4, mean_intercept_group4)
#                 points.append([x, y])
#             else:
#                 mean_x0_group4 = np.mean([line[0] for line in line_groups[3]])
#                 print("Group 4: x =", mean_x0_group4)
#                 points.append([mean_x0_group4, 240])
#
#             for point in points:
#                 cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)
#
#             mean_x = np.mean([point[0] for point in points])
#             mean_y = np.mean([point[1] for point in points])
#
#             cv2.circle(frame, (int(mean_x), int(mean_y)), 5, (255, 0, 0), -1)
#
#     cv2.imshow('edges', edges)
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
