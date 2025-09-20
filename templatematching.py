import cv2
import numpy as np
image = cv2.imread(r'ten.jpg', 0)
template = cv2.imread(r'7.jpg', 0)
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where(result >= threshold)
w, h = template.shape[::-1]
for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
cv2.imshow('Match Result with Red Borders', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
