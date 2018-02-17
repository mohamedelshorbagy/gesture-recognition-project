import numpy as np
import argparse
import cv2
import pyautogui

frame = None
roiPts = []
inputMode = False


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help = "path to the (optional) video file")
	args = vars(ap.parse_args())

	global frame, roiPts, inputMode


	if not args.get("video", False):
		camera = cv2.VideoCapture(0)

	else:
		camera = cv2.VideoCapture(args["video"])

	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
	roiBox = None

	while True:
		(grabbed, frame) = camera.read()
		# cv2.rectangle(frame,(384,0),(640,256),(0,0,255),3)

		cv2.imshow("frame" , frame)
		if not grabbed:
			break

		if roiBox is not None:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

			backProj[backProj <= 254] = 0

			# (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
			# pts = np.int0(cv2.boxPoints(r))
			# cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
			# cv2.drawContours(frame,[pts],0,(0,0,255),2)
			# xPosition = (pts[0][0] + pts[3][0]) / 2
			# yPosition = (pts[0][1] + pts[1][1]) / 2
			# cv2.circle(frame, (int(xPosition), int(yPosition)), 4, (255, 0, 0), 2)

			im, contours, hierarchy = cv2.findContours(backProj,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			# # showing all contours
			# cv2.drawContours(frame, contours, -1, 255, -1)

			# getting the largest contour
			cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
			cv2.drawContours(frame, cnt, -1, (0,0,255), 2)
			# x, y, w, h = cv2.boundingRect(cnt)
			# hand = frame[y:y+h, x:x+w]
			# imshow()
            #
			M = cv2.moments(cnt)
			cx = int(M['m10']/ (0.00001+M['m00']))
			cy = int(M['m01']/ (0.00001+M['m00']))

			cv2.circle(frame, (cx, cy), 4, (255, 0, 0), 2)
			# pyautogui.moveTo(xPosition, yPosition)
			pyautogui.moveTo(cx, cy)
			print("mouse pos: " + str(pyautogui.position()) )
			# print(pyautogui.size())

			cv2.imshow("backProj", backProj)

		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("i") and len(roiPts) < 4:
			roiPts.append((420,150))
			roiPts.append((520,150))
			roiPts.append((420,220))
			roiPts.append((520,220))
			cv2.circle(frame, (420,150) , 4, (0, 255, 0), 2)
			cv2.circle(frame, (520,150) , 4, (0, 255, 0), 2)
			cv2.circle(frame, (420,220) , 4, (0, 255, 0), 2)
			cv2.circle(frame, (520,220) , 4, (0, 255, 0), 2)
			cv2.imshow("frame", frame)
			inputMode = True
			orig = frame.copy()

			while len(roiPts) < 4:
				cv2.imshow("frame", frame)
				cv2.waitKey(0)

			roiPts = np.array(roiPts)
			s = roiPts.sum(axis = 1)
			tl = roiPts[np.argmin(s)]
			br = roiPts[np.argmax(s)]

			roi = orig[tl[1]:br[1], tl[0]:br[0]]
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

			roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
			roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
			roiBox = (tl[0], tl[1], br[0], br[1])

		elif key == ord("q"):
			break

	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
