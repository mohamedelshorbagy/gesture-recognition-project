import cv2
import numpy as np
import image_analysis
import pyautogui

def draw_hand_rect(frame):  
    rows,cols,_ = frame.shape
    hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,10*rows/20,10*rows/20,10*rows/20,14*rows/20,14*rows/20,14*rows/20])

    hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20])

    hand_row_se = hand_row_nw + 10
    hand_col_se = hand_col_nw + 10
    size = hand_row_nw.size
    for i in range(size):
        cv2.rectangle(frame,(int(hand_col_nw[i]),int(hand_row_nw[i])),(int(hand_col_se[i]),int(hand_row_se[i])),(0,255,0),1)
        black = np.zeros(frame.shape, dtype=frame.dtype)
        frame_final = np.vstack([black, frame])
    return frame_final

def set_hand_hist(frame):  

    rows,cols,_ = frame.shape
    hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,10*rows/20,10*rows/20,10*rows/20,14*rows/20,14*rows/20,14*rows/20])
    hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90,10,3], dtype=hsv.dtype)

    size = hand_row_nw.size
    for i in range(size):
        roi[i*10:i*10+10,0:10] = hsv[int(hand_row_nw[i]):int(hand_row_nw[i])+10, int(hand_col_nw[i]):int(hand_col_nw[i])+10]

    hand_hist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

def draw_final(frame, hand_hist):  
    hand_masked = image_analysis.apply_hist_mask(frame,hand_hist)

    contours = image_analysis.contours(hand_masked)
    # image_analysis.plot_contours(frame , contours)
    if contours is not None and len(contours) > 0:
        max_contour = image_analysis.max_contour(contours)
        hull = image_analysis.hull(max_contour)
        centroid = image_analysis.centroid(max_contour)
        defects = image_analysis.defects(max_contour)
        cnt = max_contour
        pyautogui.moveTo(centroid)
        image_analysis.plot_centroid(frame , centroid)
        image_analysis.plot_hull(frame , hull)
        # image_analysis.plot_defects(frame , defects , max_contour)
        # if centroid is not None and defects is not None and len(defects) > 0:   
        #     farthest_point = image_analysis.farthest_point(defects, max_contour, centroid)
        #     if farthest_point is not None:
        #         image_analysis.plot_farthest_point(frame, farthest_point)
def resize(frame):
    rows,cols,_ = frame.shape
    
    ratio = float(cols)/float(rows)
    new_rows = 400
    new_cols = int(ratio*new_rows)
    
    row_ratio = float(rows)/float(new_rows)
    col_ratio = float(cols)/float(new_cols)
    
    resized = cv2.resize(frame, (new_cols, new_rows))	
    return resized


cap = cv2.VideoCapture(0)

count = 0
handHist = None
trained_hand = False
while(cap.isOpened()):
    ret, frame = cap.read()
    orig = frame.copy()
    frame = resize(frame)
    if cv2.waitKey(1) & 0xFF == ord('h'):
        if trained_hand == False:
            handHist = set_hand_hist(frame)
            trained_hand = True
    if trained_hand == False:
        frame_final = draw_hand_rect(frame)
    elif trained_hand == True:
        draw_final(frame , handHist)

    cv2.imshow("frame" , frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()