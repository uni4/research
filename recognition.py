#https://note.nkmk.me/python-opencv-face-detection-haar-cascade/
import cv2
import numpy as np
import sys

def main():
	hand_cascade_path = '/Users/dennomaaya/Desktop/py/cascade_directory/cascade/test_data/hand.xml'
	#hand_cascade_path = '/Users/dennomaaya/Desktop/py/cascade_directory/cascade/test_data/Hand.Cascade.1.xml'
	#hand_cascade_path = '/usr/local/opt/opencv@3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'

	hand_cascade = cv2.CascadeClassifier(hand_cascade_path)

	img = cv2.imread("%s"%param[1])
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
	print(hands)

	for (x, y, w, h) in hands:
	    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
	    #roi_gray = gray[y: y + h, x: x + w]
	    #roi_color = img[y: y + h, x: x + w]

	cv2.imwrite('hand_detect.jpg', img)


if __name__ == '__main__':
	param = sys.argv
	main()