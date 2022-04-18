import cv2
import numpy as np
import math


# 获得图片sift特征
def get_sift(img_origin):
    img = np.array(img_origin)
    sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(to_gray(img), None)
    cv2.drawKeypoints(img, kp, img, color=(255, 0, 255))
    return kp, des


# 转化成灰度图片
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Canny边缘检测
def edge(img):
    # 高斯模糊,降低噪声
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # 灰度图像
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # 图像梯度
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    # 计算边缘
    # 50和150参数必须符合1：3或者1：2
    edge_output2 = cv2.Canny(xgrad, ygrad, 50, 100)
    return edge_output2


# 返回好的匹配
def get_good_matches(matches, rate=0.8):
    good = []

    for m, n in matches:
        if m.distance < rate * n.distance:
            good.append([m])

    return good


def get_angle(kp1, kp2, m):
    [m] = m
    [x1, y1] = kp1[m.queryIdx].pt
    [x2, y2] = kp2[m.trainIdx].pt
    x2 = x2 + 400
    y2 = y2 + 400
    return math.atan2(y2 - y1, x2 - x1)


def work(img1, img2):
    kp1, des1 = get_sift(img1)
    kp2, des2 = get_sift(img2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = get_good_matches(matches, 0.7)
    print(len(good))

    # good = [[m] for m in matches[22]]
    # for m in matches[133]: print (m.distance)
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imwrite('result3.png', img5)

    # RANSAC
    if len(good) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for [m] in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for [m] in good]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return imgOut, H, status

        nice = np.array(good)[status]
        print(len(nice))

        angles = np.array([get_angle(kp1, kp2, m) for [m] in nice])
        mean_angle = angles.mean()
        print(angles)
        print(mean_angle)

        good = get_good_matches(matches, 0.9)
        print(len(good))
        better = []
        for [m] in good:
            t = get_angle(kp1, kp2, [m])
            if abs((t - mean_angle) / mean_angle) < 0.3:
                better.append([m])

        print(len(better))

        ptsA = np.float32([kp1[m.queryIdx].pt for [m] in better]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for [m] in better]).reshape(-1, 1, 2)
        ransacReprojThreshold = 10
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return imgOut, H, status


img = cv2.imread('01.jpeg.png')
tamplate = cv2.imread('02.jpeg')


'''
cv2.imshow('dilation', dilation)
cv2.imwrite('dilation.png', dilation)
cv2.imshow('opening', opening2)
cv2.imwrite('opening.png', opening2)
cv2.imshow('closing', closing2)
cv2.imwrite('closing.png', closing2)

cv2.waitKey()
cv2.destroyAllWindows()
'''
#binary, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imgOut, H, status = work(img, tamplate)


cv2.imshow('roi', imgOut)

cv2.waitKey()
cv2.destroyAllWindows()

