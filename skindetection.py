import cv2
import numpy as np

cap = cv2.VideoCapture(0)

count = 0
limit = 10000

minValue = 70
low_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])

# low_range = np.array([15, 50, 50])
# upper_range = np.array([30, 255, 255])


# low_range = np.array([94, 100, 100])
# upper_range = np.array([114, 255, 255])


def skinMode(frame):
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)

    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    #cv2.imshow("Blur", mask)

    #bitwise and mask original frame
    res = cv2.bitwise_and(frame, frame, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

def binaryMode(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

# test = cv2.imread("E:\Python\Deep Learning\Tensorflow-Bootcamp-master\Gesture Reco. Project\images2\class1\\192.png")

# testSkinMode = skinMode(test)
# cv2.imshow('Test' , testSkinMode)
# cv2.imwrite('images\\Test.png' , testSkinMode)


# kernel = np.ones((15,15),np.uint8)
# kernel2 = np.ones((1,1),np.uint8)
# skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

while(cap.isOpened()):
    ret, frame = cap.read()

    res = skinMode(frame)
    # res2 = binaryMode(frame)
    cv2.imshow("res", res)
    # cv2.imshow("res2", res2)
    if count == 0:
        cv2.imwrite("resp.png", res)
    count += 1
    if count == limit:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()