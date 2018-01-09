import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image

def crop(img,x,y):
	height, width = img.shape[:2]
	img_cut = img[(height - y)//2:(height + y)//2,
								(width - x)//2:(width + x) // 2]
	return img_cut

def main():
	image = cv2.imread("%s"%param[1])
	height, width = image.shape[:2]
	cut = crop(image,width//2,height//2)
	cv2.imwrite("image_cut.jpg",cut)


if __name__ == '__main__':
	param = sys.argv
	main()