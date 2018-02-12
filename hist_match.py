#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""hist matching."""

import cv2
import os
import sys
def hist_match():
	TARGET_FILE = param[1]
	IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/hikaku/'
	IMG_SIZE = (200, 200)

	target_img_path = IMG_DIR + TARGET_FILE
	target_img = cv2.imread(target_img_path)
	target_img = cv2.resize(target_img, IMG_SIZE)
	target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

	print('TARGET_FILE: %s' % (TARGET_FILE))

	files = os.listdir(IMG_DIR)
	for file in files:
	    if file == '.DS_Store':
	        continue

	    comparing_img_path = IMG_DIR + file
	    comparing_img = cv2.imread(comparing_img_path)
	    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
	    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

	    ret = cv2.compareHist(target_hist, comparing_hist, 0)
	    print(file, ret)

def chara_match():
	TARGET_FILE = param[1]
	IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/hikaku/'
	IMG_SIZE = (200, 200)

	target_img_path = IMG_DIR + TARGET_FILE
	target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
	target_img = cv2.resize(target_img, IMG_SIZE)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	detector = cv2.ORB_create()
	#detector = cv2.AKAZE_create()
	(target_kp, target_des) = detector.detectAndCompute(target_img, None)

	print('TARGET_FILE: %s' % (TARGET_FILE))

	files = os.listdir(IMG_DIR)
	for file in files:
	    if file == '.DS_Store':
	        continue

	    comparing_img_path = IMG_DIR + file
	    try:
	        comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
	        comparing_img = cv2.resize(comparing_img, IMG_SIZE)
	        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
	        matches = bf.match(target_des, comparing_des)
	        dist = [m.distance for m in matches]
	        ret = sum(dist) / len(dist)
	    except cv2.error:
	        ret = 100000

	    print(file, ret)

def main():
	hist_match()
	chara_match()

if __name__ == "__main__":
	param = sys.argv
	main()