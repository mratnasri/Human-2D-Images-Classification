print("start")
import cv2
import matplotlib as plt

img = cv2.imread('../../Complete 2D Human Images Dataset/1-01.jpg')
img2 = cv2.imread('../../Complete 2D Human Images Dataset/1-14.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE()
gray_img = clahe.apply(gray_img)
gray_img2 = clahe.apply(gray_img2)

cv2.imwrite("../../trial1.jpg", gray_img)
cv2.imwrite("../../trial2.jpg", gray_img2)
# plt.show()
print("end")
