import cv2
import numpy as np
import pytesseract


image = cv2.imread(r'C:\Users\chinmaya bindhani\OneDrive\Desktop\python\download.jpg', cv2.IMREAD_GRAYSCALE)


image = cv2.resize(image, None, fx=1.5, fy=1.5)


sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])


sharpened = cv2.filter2D(image, -1, sharpen_kernel)


_, binary = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)


text = pytesseract.image_to_string(binary)


cv2.imshow('Original', image)
cv2.imshow('Sharpened', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


print("Extracted Text:\n", text)
