#http://www.cellstat.net/centroid/
#輪郭が最大の重心座標を求める
import cv2
import numpy as np
import sys

def main():
	im = cv2.imread("%s"%param[1])
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	image, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#image, contours, hierarchy = cv2.findContours(gray,1,2)
	im_con = cv2.drawContours(im, contours, 0, (255,255,255), -1,1)
	cnt = contours[0]
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	cv2.circle(im_con,(cx,cy), 20,(255,0,0),-1)

	#凸包の取得
	hull = cv2.convexHull(cnt,returnPoints = False)
	defects = cv2.convexityDefects(cnt,hull)

	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
		#cv2.line(im_con,start,end,[0,255,0],2)
		cv2.circle(im_con,far,20,[0,0,255],-1)

	"""
	for i in range(defects.shape[0]):
		far = int(defects[:,0][i,0])
		cv2.circle(im_con,far,20[0,0,255],-1)

	"""

	cv2.imwrite("fingertip_test.jpg", im_con)


	print("凸包", defects[:,0])
	#print("輪郭画像", image)
	#print("輪郭", contours)
	#print("輪郭の階層", hierarchy)
	




if __name__ == '__main__':
	param = sys.argv
	main()