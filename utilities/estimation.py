import numpy as np
import matplotlib.pyplot as plt

def ESTIMATION_1(X,Y):
	try:
		assert (len(X) == len(Y))
	except:
		print ('Size of arrays are not the same. Quitting .... ')
		exit(0)
		
	n = len(X)
	sum_Y = sum(Y)
	sum_X = sum(X)

	A = np.matrix([[n]])
	B = np.matrix([[sum_Y]])
	A_inverse = np.linalg.inv(A)
	# Calculating the coefficients of least square
	min_square = A_inverse*B
	return (min_square)

	
def ESTIMATION_2(X,Y):
	try:
		assert (len(X) == len(Y))
	except:
		print ('Size of arrays are not the same. Quitting .... ')
		exit(0)
		
	n = len(X)
	sum_Y = sum(Y)
	sum_X = sum(X)
	sum_X2 = sum([x*x for x in X])
	sum_XY = sum([x*y for x,y in zip(X,Y)])

	A2 = np.matrix([[n, sum_X],[sum_X,sum_X2]])
	B2 = np.matrix([[sum_Y],[sum_XY]])
	A2_inverse = np.linalg.inv(A2)
	# Calculating the coefficients of least square
	min_square2 = A2_inverse*B2
	return (min_square2)

def ESTIMATION_3(X,Y):
	try:
		assert (len(X) == len(Y))
	except:
		print ('Size of arrays are not the same. Quitting .... ')
		exit(0)
	
	n = len(X)
	sum_Y = sum(Y)
	sum_X = sum(X)
	sum_X2 = sum([x*x for x in X])
	sum_X3 = sum([x*x*x for x in X])
	sum_X4 = sum([x*x*x*x for x in X])
	sum_XY = sum([x*y for x,y in zip(X,Y)])
	sum_X2Y = sum([x*x*y for x,y in zip(X,Y)])

	A3 = np.matrix([[n, sum_X, sum_X2],[sum_X,sum_X2, sum_X3],[sum_X2,sum_X3, sum_X4]])
	B3 = np.matrix([[sum_Y],[sum_XY],[sum_X2Y]])
	A3_inverse = np.linalg.inv(A3)

	# Calculating the coefficients of least square
	min_square3 = A3_inverse*B3
	return (min_square3)
