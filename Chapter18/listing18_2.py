import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


ArrayK = []
XS_1 = 1.2
XS_2 = 0.2
XS_3 = 2.9
XS_4 = 2.1
ArrayD = [XS_1,XS_2,XS_3,XS_4]

AK = 0.0
BK = 0.0
K = 1
ArrayA = [AK]
ArrayB = [BK]

ArrayK = [0]

while (K <= 3):
	XH_1 = BK
	XH_2 = AK + BK
	XH_3 = 2*AK + BK
	XH_4 = 3*AK +BK
	deltaX1 = XS_1 - XH_1
	deltaX2 = XS_2 - XH_2
	deltaX3 = XS_3 - XH_3
	deltaX4 = XS_4 - XH_4
	AK = AK - 0.3*deltaX1 - 0.1*deltaX2 + 0.1*deltaX3 + 0.3*deltaX4
	BK = BK + 0.7*deltaX1 + 0.4*deltaX2 + 0.1*deltaX3 - 0.2*deltaX4
	ArrayA.append(AK)
	ArrayB.append(BK)
	ArrayK.append(K)
	K = K+1
	
ArrayT = [1,2,3,4]	
plt.figure(1)
plt.grid(True)
plt.plot(ArrayK,ArrayA,label='a', linewidth=0.6)
plt.plot(ArrayK,ArrayB,label='b', linewidth=0.6)
plt.plot(ArrayK,[a*x+b for a,b,x in zip(ArrayA,ArrayB,ArrayT)],label='b', linewidth=0.6)
plt.plot(ArrayK,ArrayD, 'ro',label='b')

plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.legend()
plt.show()
