import math
# 最近邻近似匹配 2NN
from matplotlib import pyplot as plt
import matplotlib
import cv2

book_l = cv2.imread('C:\\Users\\123\\Desktop\\zhang.jpg')
res=cv2.resize(book_l,(295,413),interpolation=cv2.INTER_CUBIC)
cv2.imshow('iker',res)
cv2.imwrite('zhangyang.jpg',res)
cv2.imwrite('C:\\Users\\123\\Desktop\\zhangy.jpg',res)
cv2.waitKey(0)
cv2.destoryAllWindows()

