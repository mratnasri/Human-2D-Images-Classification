import cv2
import matplotlib as plt
import numpy as np

img = cv2.imread('../../stretched2.jpg')
img2 = cv2.imread('../../Complete 2D Human Images Dataset/1-14.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray_img = np.array(gray_img)
gray_img = gray_img.reshape(480, 640, 1)
minmax = (gray_img-np.min(gray_img))/(np.max(gray_img)-np.min(gray_img))*255
minmax = minmax.astype(np.uint8)
minmax = minmax.astype('float32')


"""stretched = 255*gray_img/np.max(gray_img)
stretched = stretched.astype(np.uint8)"""
#stretched = stretched.astype('float32')

minmax2 = (gray_img2-np.min(gray_img2)) / \
    (np.max(gray_img2)-np.min(gray_img2))*255
minmax2 = minmax2.astype(np.uint8)
minmax2 = minmax2.astype('float32')

gray_img2 = np.array(gray_img2)
gray_img2 = gray_img2.reshape(480, 640, 1)
stretched2 = 255*gray_img2/np.max(gray_img2)
stretched2 = stretched2.astype(np.uint8)
stretched2 = stretched2.astype('float32')

cv2.imwrite("../../minmax1_updated.jpg", minmax)
cv2.imwrite("../../minmax14.jpg", minmax2)
#cv2.imwrite("../../max1.jpg", stretched)
cv2.imwrite("../../max14_new.jpg", stretched2)
# plt.show()
