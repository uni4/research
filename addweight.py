import cv2
import numpy as np
import sys

def addweight(img,alpha,beta,ganma):	
	gauss = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3)

	kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
      
	himg2 = cv2.filter2D(img, -1, kernel)
	im_add = cv2.addWeighted(gauss, alpha, himg2, beta, ganma)
	return im_add


def main():
	img = cv2.imread("%s"%param[1])

	#画像の合成
	img_addweight = addweight(img,1,0.5,0)
	cv2.imwrite("imadd.png",img_addweight)
	    

 
if __name__ == "__main__":
	param = sys.argv
	main()