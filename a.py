import numpy as np
from scipy.spatial import distance

def normalize(v, axis=-1, order=2):
	l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
	l2[l2==0] = 1
	return v/l2

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def main():
	a = np.array([1,2,3,2,1])
	#a[1] =  distance.euclidean(a[1],20)

	z1 = a - np.average(a)
	z2 = np.average(z1)
	print(z1)
	print(z2)

	z_score = zscore(a)
	z_ave = np.average(z_score)
	n_lize = normalize(a)
	mm = min_max(a)

	print("ベクトルのnormalize",n_lize)
	print("zscore",z_score)
	print("zスコアの平均",z_ave)
	print("min_max",mm)

if __name__ == '__main__':
	main()