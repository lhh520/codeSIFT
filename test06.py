# -*- coding: utf-8 -*
from PIL import Image
import numpy as np
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rebuild_img(u, sigma, v, p):  # p表示奇异值的百分比
    # print p
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))

    count = (int)(sum(sigma))
    curSum = 0
    k = 0
    print
    sigma[0:2], count * p
    while curSum <= count * p:
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        # print curSum,count,'--------',k
        a += sigma[k] * np.dot(uk, vk)
        curSum += sigma[k]
        k += 1
        # print k

    print
    'k:', k
    a[a < 0] = 0
    a[a > 255] = 255
    # 按照最近距离取整数，并设置参数类型为uint8
    return np.rint(a).astype("uint8")


if __name__ == '__main__':
    #SVD操作
    img = Image.open(u'3.png', 'r')
    a = np.array(img)
    # print a[:, :, 0]
    # u, sigma, v = np.linalg.svd(a[:, :, 0])
    # R = rebuild_img(u, sigma, v, 0.9)
    p=0.9
    u, sigma, v = np.linalg.svd(a[:, :, 0])
    R = rebuild_img(u, sigma, v, p)
    u, sigma, v = np.linalg.svd(a[:, :, 1])
    G = rebuild_img(u, sigma, v, p)
    u, sigma, v = np.linalg.svd(a[:, :, 2])
    B = rebuild_img(u, sigma, v, p)
    I = np.stack((R, G, B), 2)
    # 保存图片在img文件夹下
    Image.fromarray(I).save("svd1_"+ ".jpg")
    ##
    img = Image.open(u'4.png', 'r')
    a = np.array(img)
    # print a[:, :, 0]
    # u, sigma, v = np.linalg.svd(a[:, :, 0])
    # R = rebuild_img(u, sigma, v, 0.9)
    p = 0.9
    u, sigma, v = np.linalg.svd(a[:, :, 0])
    R = rebuild_img(u, sigma, v, p)
    u, sigma, v = np.linalg.svd(a[:, :, 1])
    G = rebuild_img(u, sigma, v, p)
    u, sigma, v = np.linalg.svd(a[:, :, 2])
    B = rebuild_img(u, sigma, v, p)
    I = np.stack((R, G, B), 2)
    # 保存图片在img文件夹下
    Image.fromarray(I).save("svd2_" + ".jpg")
MIN_MATCH_COUNT = 5

img1 = cv2.imread('svd1_.jpg', 0)  # 目标图像
img2 = cv2.imread('svd2_.jpg', 0) # 原图像

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)#这个kp1好类似类的实例化返回的是一个<class ....>一样
kp2, des2 = sift.detectAndCompute(img2,None)
#print(dir(000002DB2058DBD0))
#print(type(kp1),type(des1))
print(kp1)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)#这个是陷进去init还是先进入call
#print(type(flann))
#先去找到des2中的和des1中特征描述相近的
matches = flann.knnMatch(des1, des2, k=2)
print("matches:\n",len(matches))
print(matches)

#matches counts
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.85*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    print(src_pts)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #print(dst_pts)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) #calib3d模块
    '''
    RANSAC算法：RANSAC
    cv2.findHomography(kpA, kpB, cv2.RANSAC, reproThresh) # 计算出单应性矩阵
    参数说明：kpA表示图像A关键点的坐标, kpB图像B关键点的坐标, 
    使用随机抽样一致性算法来进行迭代，reproThresh表示每次抽取样本的个数
    '''
    matchesMask = mask.ravel().tolist()

    h,w, = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
img1 = cv2.imread('3.png')  # 目标图像
img2 = cv2.imread('4.png') # 原图像
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
img1 = cv2.imread('svd1_.jpg')  # 目标图像
img2 = cv2.imread('svd2_.jpg') # 原图像
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

cv2.imshow('matched', img3)
cv2.waitKey()
