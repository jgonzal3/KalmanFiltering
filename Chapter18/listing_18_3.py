import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


ArrayK = []
XS_1 = 1.2
XS_2 = 0.2
XS_3 = 2.9
XS_4 = 2.1
XS = np.matrix([[XS_1],[XS_2],[XS_3],[XS_4]])

AK = 0.0
BK = 0.0
CK = 0.0

K = 1
ArrayA = [AK]
ArrayB = [BK]
ArrayC = [CK]

AK = np.matrix([[AK],[BK],[CK]])

A = np.matrix([[0,0,1],[1,1,1],[4,2,1],[9,3,1]])
AT = A.transpose()
INVATAAT = inv(AT*A)*AT
ArrayK = [0]

while (K <= 3):
	#XH = A*AK
	#DELTA_X = XS - XH
	#DELTA_A = INVATAAT*DELTA_X
	#AK = AK + DELTA_A
  # Combining all the statements above, we can use only one line to estimate AK
	AK = AK + INVATAAT*(XS - A*AK)
	ArrayA.append(AK[0,0])
	ArrayB.append(AK[1,0])
	ArrayC.append(AK[2,0])
	ArrayK.append(K)
	K = K+1
	
print(AK)	
ArrayT = [1,2,3,4]	
plt.figure(1)
plt.grid(True)
plt.plot(ArrayK,ArrayA,label='a', linewidth=0.6)
plt.plot(ArrayK,ArrayB,label='b', linewidth=0.6)
plt.plot(ArrayK,ArrayC,label='c', linewidth=0.6)
plt.plot(ArrayK,XS, 'ro',label='measurements')
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.legend()
plt.show()
