import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
count = 0
limit = 1000 # limit for Number of Saved Images
while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)
    cv2.imwrite("images/%d.png" % count , frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count == limit:
        break

cap.release()
cv2.destroyAllWindows()