import cv2

img = cv2.imread(r"py\Screenshot 2023-12-29 225317.png")
cv2.imshow("output",img)
cv2.waitKey(0)

#Thresholding
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)


cv2.imshow("Original", img)
cv2.imshow("Binary", binary)
cv2.imshow("Binary Inv", binary_inv)
cv2.imshow("Trunc", trunc)
cv2.imshow("ToZero", tozero)
cv2.imshow("ToZero Inv", tozero_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()
#rotate
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#filter
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
