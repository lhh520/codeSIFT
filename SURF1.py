import cv2
import numpy as np
pic_path='17.jpg'
img = cv2.imread(pic_path, 0)
img.resize(750,500)
surf = cv2.xfeatures2d.SURF_create(400)
# 需要调整阈值到合适的大小，示例说的是len(kp)在50以下最好
#surf.setHessianThreshold(18000)
kp, des = surf.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),40)
cv2.imshow("SURF",img2)
cv2.waitKey()
#
img = cv2.imread(pic_path)  # 读取文件
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
sift = cv2.xfeatures2d_SURF.create()
keyPoint, descriptor = sift.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
img2 = cv2.drawKeypoints(img,keyPoint,None,(255,0,0),40)
cv2.imshow("SURF",img2)
cv2.waitKey()
#
img = cv2.imread(pic_path) # 读取文件
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转化为灰度图
sift = cv2.ORB_create()
keyPoint, descriptor = sift.detectAndCompute(img, None) # 特征提取得到关键点以及对应的描述符（特征向量）
img2 = cv2.drawKeypoints(img,keyPoint,None,(255,0,0),40)
cv2.imshow("ORB",img2)
cv2.waitKey()