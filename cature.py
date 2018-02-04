import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
count = 0
limit = 1000 # limit for Number of Saved Images
time.sleep(10)
print("Start Capturing")
while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)
    cv2.imwrite("images/class1T/%d.png" % count , frame)
    count += 1
    print(count)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count == limit:
        break

cap.release()
cv2.destroyAllWindows()