import numpy as np
import cv2

# original image
image = cv2.imread('17.jpg')
cv2.resize(image,1000,2000,3)
h, w, c = image.shape
print('image shape --> h:%d  w:%d  c:%d' % (h, w, c))
cv2.imshow('image', image)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# harris dst
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=3, ksize=5, k=0.05)
image_dst = image[:, :, :]
image_dst[dst > 0.01 * dst.max()] = [255, 0,0]
cv2.imshow('dst', image_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
