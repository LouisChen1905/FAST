import sys
import cv2
import math
import numpy as np

radius = 3
pixel_t = 5
compare_num = 16
threshold_num = 12
fast_index = ((-3, 0),
              (-3, 1),
              (-2, 2),
              (-1, 3),
              (0, 3),
              (1, 3),
              (2, 2),
              (3, 1),
              (3, 0),
              (3, -1),
              (2, -2),
              (1, -3),
              (0, -3),
              (-1, -3),
              (-2, -2),
              (-3, -1))


def FAST(argv=sys.argv[1:]):
    img = cv2.imread(argv[1])

    height, width, channels = img.shape
    height = int(height / 2)
    width = int(width / 2)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.GaussianBlur(gray, (3, 3), 0, gray)
    result = img.copy()

    keypoints = {}
    filtered_keypoints = []
    keypoints_score = np.zeros(shape=[0, width], dtype=np.int)
    for i in range(radius, height - radius):
        score_line = np.zeros(shape=[1, width], dtype=np.int)
        keypoints_line = np.empty(shape=[0, 2], dtype=np.int)
        for j in range(radius, width - radius):
            center = gray[i, j]
            a = 0
            pixel_diff = np.int16(gray[i + fast_index[0][0], j + fast_index[0][1]]) - center
            a += 1 if math.fabs(pixel_diff) > pixel_t else 0
            pixel_diff = np.int16(gray[i + fast_index[8][0], j + fast_index[8][1]]) - center
            a += 1 if math.fabs(pixel_diff) > pixel_t else 0
            if a < 2:
                continue
            pixel_diff = np.int16(gray[i + fast_index[4][0], j + fast_index[4][1]]) - center
            a += 1 if math.fabs(pixel_diff) > pixel_t else 0
            pixel_diff = np.int16(gray[i + fast_index[11][0], j + fast_index[11][1]]) - center
            a += 1 if math.fabs(pixel_diff) > pixel_t else 0

            if a < 3:
                continue

            same_pixel_index = []
            for k in range(len(fast_index)):
                pixel_diff = np.int16(gray[i + fast_index[k][0], j + fast_index[k][1]]) - center
                if math.fabs(pixel_diff) <= pixel_t:
                    same_pixel_index.append(k)

            if len(same_pixel_index) > 1:
                for l in range(len(same_pixel_index)):
                    dist = 0
                    if l >= 1:
                        dist = same_pixel_index[l] - same_pixel_index[l - 1]
                    else:
                        dist = 16 - (same_pixel_index[len(same_pixel_index) - 1] - same_pixel_index[l])
                    if math.fabs(dist) >= threshold_num:
                        # print("Found key point")
                        # cv2.drawMarker(result, (j,i), (255, 0, 0),
                        #                markerType=cv2.MARKER_SQUARE,
                        #                markerSize=6, thickness=1)
                        score = corner_score((i, j), gray)
                        score_line[0,j] = score
                        keypoints_line = np.append(keypoints_line, [[i, j]], axis=0)
                        # keypoints = np.sort(keypoints, axis=0, kind='quicksort')
                        break;
            else:
                score = corner_score((i, j), gray)
                score_line[0, j] = score
                keypoints_line = np.append(keypoints_line, [[i, j]], axis=0)

        keypoints_score = np.append(keypoints_score, score_line, axis=0)
        keypoints[i] = keypoints_line

        # 不够non-maximal suppression计算
        if np.size(keypoints_score, axis=0) < 3:
            continue

        prev_idx = i - 2
        curr_idx = i - 1
        next_idx = i
        try:
            for k in range(len(keypoints[curr_idx])):
                index = keypoints[curr_idx][k]
                score = keypoints_score[curr_idx-3, index[1]]
                if score > keypoints_score[prev_idx-3, index[1] - 1] \
                        and score > keypoints_score[prev_idx-3, index[1]] \
                        and score > keypoints_score[prev_idx-3, index[1] + 1] \
                        and score > keypoints_score[curr_idx-3, index[1] - 1] \
                        and score > keypoints_score[curr_idx-3, index[1] + 1] \
                        and score > keypoints_score[next_idx-3, index[1] - 1] \
                        and score > keypoints_score[next_idx-3, index[1]] \
                        and score > keypoints_score[next_idx-3, index[1] + 1]:
                    filtered_keypoints.append((index[1], index[0]))
        except KeyError:
            continue

    for i in range(len(filtered_keypoints)):
        cv2.drawMarker(result, filtered_keypoints[i], (255, 0, 0),
                       markerType=cv2.MARKER_SQUARE, markerSize=6, thickness=1)
    cv2.imshow("FAST result", result)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv2.imshow("FAST opencv result", img2)

    while (1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def corner_score(keypoints, gray):
    center = int(gray[keypoints[0], keypoints[1]])
    sum_absolute_diff = 0
    for j in range(len(fast_index)):
        pixel = gray[keypoints[0] + fast_index[j][0], keypoints[1] + fast_index[j][1]]
        sum_absolute_diff += math.fabs(
            center - pixel
        )
    return sum_absolute_diff


if __name__ == '__main__':
    FAST(sys.argv)
