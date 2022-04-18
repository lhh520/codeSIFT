#导入需要的库
import numpy as np
import cv2
from matplotlib import pyplot as plt

#图片地址，改为自己的
img_path = r"12.jpg"
#引入sift创建函数
sift = cv2.xfeatures2d.SIFT_create()
#读入图片
img = cv2.imread(img_path)
#进行sift特征点检测
kp,des = sift.detectAndCompute(img,None)

#画出关键点
img1=cv2.drawKeypoints(img,kp,img,color=(255,0,0))#color为标记特征点的颜色，按（B，G，R）排的

#显示图片
cv2.imshow('point',img1)

#写入图片，保存的位置改为自己的
cv2.waitKey(0)#按下任意键退出
cv2.destroyAllWindows()
