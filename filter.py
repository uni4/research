import cv2
import numpy as np
import sys
    
def highpass_filter(src, a):
	# 高速フーリエ変換(2次元)
	src = np.fft.fft2(src)

	# 画像サイズ
	h, w = src.shape[:2]

	# 画像の中心座標
	cy, cx =  int(h/2), int(w/2)

	# フィルタのサイズ(矩形の高さと幅)
	rh, rw = int(a*cy), int(a*cx)

	# 第1象限と第3象限、第1象限と第4象限を入れ替え
	fsrc =  np.fft.fftshift(src)  

	# 入力画像と同じサイズで値0の配列を生成
	fdst = fsrc.copy()

	# 中心部分だけ0を代入（中心部分以外は元のまま）
	fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0

	# 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
	fdst =  np.fft.fftshift(fdst)

	# 高速逆フーリエ変換 
	dst = np.fft.ifft2(fdst)

	# 実部の値のみを取り出し、符号なし整数型に変換して返す
	return  np.uint8(dst.real)


def addweight(imgA,imgB,alpha,beta,ganma):
	im_add = cv2.addWeighted(imgA, alpha, imgB, beta, ganma)
	return im_add


def main():
	# 入力画像を読み込み
	img = cv2.imread("%s"%param[1])

	# グレースケール変換
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	    
	# ガウシアンフィルタ	
	dst3 = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3)

	# ハイパスフィルタ処理
	himg = highpass_filter(img, 0.8)

	#ハイパスフィルタの係数　ようはラプラシアンフィルタ
	kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
	k = 1.0
	shape_kernel = np.array([
				[-k, -k, -k],
				[-k, 1 + 8 * k, -k],
				[-k, -k, -k]
            ])
    
    #ハイパスフィルタ方法2  
	#himg2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
	himg2 = cv2.filter2D(img, -1, kernel)

	#先鋭化フィルター
	img_shape = cv2.filter2D(img, -1, shape_kernel)
	

	#画像の合成
	img_addweight = addweight(dst3,himg2,1,0.5,0)
	    
	# 結果を出力
	cv2.imwrite("gaussian.jpg", dst3)
	cv2.imwrite("highpass.jpg", himg)
	cv2.imwrite("highpass2.jpg", himg2)
	cv2.imwrite("addweight.jpg", img_addweight)
	cv2.imwrite("shapeeing.jpg", img_shape)

 
if __name__ == "__main__":
	param = sys.argv
	main()