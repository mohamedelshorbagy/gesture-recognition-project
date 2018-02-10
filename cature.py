import numpy as np
import cv2
import time
import os
# import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import PIL
import os
from keras.models import load_model
from keras.optimizers import Adam
# from imutils.perspective import four_point_transform
# from imutils import contours
# import imutils

# Load Model
model = load_model('model3.h5')
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

img_width = 128
img_height = 128
cap = cv2.VideoCapture(0)


# def removeBG(bgModel,frame):
#     fgmask = bgModel.apply(frame,learningRate=0)
#     kernel = np.ones((3, 3), np.uint8)
#     fgmask = cv2.erode(fgmask, kernel, iterations=1)
#     res = cv2.bitwise_and(frame, frame, mask=fgmask)
#     return res  # return eroded binary image


im2 = Image.open('/media/amrgalal7/Files/GesturesDataset/300.png')
# resized_image = cv2.resize(im2 , (img_width , img_height))
img2 = im2.resize((img_width , img_height))
# gray2 = img2.convert('L')

images_matrix2 = np.array(img2).flatten()


images_matrix2 = images_matrix2.reshape(-1 , img_width , img_height , 1)

# images_matrix2.shape
classes = model.predict_classes(images_matrix2)
print(classes)


print("****************")
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 40  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
# bgSubThreshold = 50
# bgModel = None
# image_store = None
# limit = 500 # limit for Number of Saved Images
time.sleep(10)
isCount = 0
count = 0
print("Start Capturing")
# os.mkdir('E:\Python\Deep Learning\Tensorflow-Bootcamp-master\Gesture Reco. Project\images2\class1A')
while(cap.isOpened()):
    ret, frame = cap.read()
    # image_store = frame
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('frame' , frame)
    # if isCount == 1:
    #     # cv2.imwrite('images2/class9/%d.png' % count , frame)
    #     count += 1
    #     print(count)
    # frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    # # frame = cv2.flip(frame, 1)  # flip the frame horizontally
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    # # frame = cv2.GaussianBlur(frame , (5 , 5) , 0)
    # # image_adapt = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # # image_adapt = cv2.bilateralFilter(image_adapt, 5, 50, 100)
    # # image_thresh = cv2.threshold(frame , 0 , 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow('frame',frame)
    # # if count == 0:
    # #     image_store = frame
    # # if count % 50 == 0:
    # if isBgCaptured == 1:
    #     # frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    #     # img = removeBG(bgModel,frame)
    #     # img = frame[0:int(cap_region_y_end * frame.shape[0]),
    #     #             int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the blue rectangle
    #     # img = frame
    #     # cv2.imshow('image' , img)
    #     # image_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #     # frame = cv2.absdiff(frame,image_store)
    #     # cv2.imshow('newFrame' , frame)
    #     # cv2.imwrite('images/thresh%d.png' % count , image_thresh)
    #     # gray2 = cv2.cvtColor(image_thresh, cv2.COLOR_BGR2GRAY)
    #     # image_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #     # cv2.imshow('image2' , image_adapt)
    #     img = removeBG(bgModel,frame)
    #     # img = img[0:int(cap_region_y_end * frame.shape[0]),
    #     #             int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
    #     # data =  frame[0:int(cap_region_y_end * frame.shape[0]),
    #     #             int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]# clip the blue rectangle
    #     #cv2.imshow('mask', img)
    #     data = frame

    #     # convert the image into binary image
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gaussian_filtered = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    #     #cv2.imshow('blur', blur)
    #     ret, thresh = cv2.threshold(gaussian_filtered, threshold, 255, cv2.THRESH_BINARY)
    #     cv2.imshow('extracted', thresh)
    #     col = cv2.bitwise_and(data , data , mask=thresh)
    #     cv2.imshow('col', col)
    if count % 50 == 0:
        frame =  frame[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        # frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        # kernel = np.ones((5 , 5) , np.uint8)
        frame = cv2.GaussianBlur(frame,(15,15), 0)
        thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # cv2.imwrite('/media/amrgalal7/Files/GesturesDataset/%d.png' %count, thresh)
        # dilated = cv2.dilate(thresh , kernel , iterations = 1)
        resized_image = cv2.resize(thresh , (img_width , img_height))
        cv2.imshow('thresh' , resized_image)
        #     # cv2.imshow('resized' , resized_image)
        images_matrix = np.array(resized_image).flatten()
        images_matrix = images_matrix.reshape(-1 , img_width , img_height , 1)
        classes = model.predict_classes(images_matrix, batch_size=10)
        #     # cv2.imwrite('images/%d.png' % count , frame)
        print(classes)
    # if cv2.waitKey(10) & 0xFF == ord('b'):
    #     bgModel = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=bgSubThreshold)
    #     for i in range(16):
    #         fgmask = bgModel.apply(frame, learningRate=0.5)
    #     print("bgModel instantiated", type(bgModel))
    #     isBgCaptured = 1
    #     # image_store = frame
    #     print ('!!!Background Captured!!!')
    count += 1
    # if count == limit:
    #     isCount = 0
    # if cv2.waitKey(1) & 0xFF == ord('s'):
    #     isCount = 1
    #     print('S Clicked!')
    # elif cv2.waitKey(1) & 0xFF == ord('o'):
    #     isCount = 0
    #     print('O Clicked!')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
