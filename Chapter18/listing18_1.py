import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


ArrayK = []
XS_1 = 1.2
XS_2 = 0.2
XS_3 = 2.9
XS_4 = 2.1

ave = 0.25*(XS_1 + XS_2 + XS_3 + XS_4)
AK = 0.0
K = 1
ArrayA = [AK]
ArrayK = [0]
while (K <= 5):
	XH_1 = AK
	XH_2 = AK
	XH_3 = AK
	XH_4 = AK
	deltaX1 = XS_1 - XH_1
	deltaX2 = XS_2 - XH_2
	deltaX3 = XS_3 - XH_3
	deltaX4 = XS_4 - XH_4
	AK = AK + 0.25*(deltaX1 + deltaX2 + deltaX3 + deltaX4)
	print(AK)
	ArrayA.append(AK)
	ArrayK.append(K)
	K = K+1
		
plt.figure(1)
plt.grid(True)
plt.plot(ArrayK,ArrayA,label='a', linewidth=0.6)
plt.plot(ArrayK,[ave for a in range(0,6)],label='ave', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.legend()
plt.show()



