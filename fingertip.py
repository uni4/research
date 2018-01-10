#http://www.cellstat.net/centroid/
#輪郭が最大の重心座標を求める 指の位置を計算する
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
	cv2.circle(im_con,(cx,cy), 20,(0,0,255),-1)

	aa,bb,cc = (0,100,200)
	list = [[]]
	ren = len(contours[0][:,:])


	while cc <= ren - 1:
		p1 = contours[0][aa,0]
		p2 = contours[0][bb,0]
		p3 = contours[0][cc,0]
		c1x = p1[0] - 0#本来はp2だが原点をp2にする際に必要な計算であり、p2を(0,0)としている
		c1y = p1[1] - 0#
		c2x = p3[0] - 0#
		c2y = p3[1] - 0#
		nai_c1 = np.sqrt(pow(c1x,2) + pow(c1y,2))
		nai_c2 = np.sqrt(pow(c2x,2) + pow(c2y,2))
		c1_c2 = (c1x * c2x) + (c1y * c2y)#c1・c2のこと
		cos = c1_c2 / (nai_c1 * nai_c2)
		deg = np.rad2deg(cos)

		if(cos >0 and cos < 60):
			cv2.circle(im_con,(p2[0],p2[1]), 10,(0,0,255),-1)

		aa += 1
		bb += 1
		cc += 1
		#print(cos)

	cv2.imwrite("fingertip_test.jpg", im_con)

	#print("輪郭", contours[0][0,0])
	#print("輪郭の階層", hierarchy)
	print("長さ", ren)
	print("リスト", list)


if __name__ == '__main__':
	param = sys.argv
	main()