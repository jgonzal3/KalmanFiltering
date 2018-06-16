import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

ORDER=2
T=0.
S=0.
H=.001
TS=.1
TF=10.
PHIS=0.
XJ=1.

F = np.matrix([[0,1],[0,0]])
P = np.matrix([[100,0],[0,100]])
Q = np.matrix([[0,0],[0,PHIS]])
POLD = P

HMAT= np.matrix([[1,0]])
HT =HMAT.transpose()

SIGN2=1.**2
PHIN=SIGN2*TS
t=[]
K1=[]
K2=[]

while (T<=TF):
	POLD = P
	FP = F*P
	PFT = FP.transpose()
	PHT = P*HT
	HP =  HMAT*P
	PHTHP = PHT*HP
	PHTHPR = (1/PHIN)*PHTHP
	PFTFP = PFT + FP
	PFTFPQ = PFTFP + Q
	PD = PFTFPQ - PHTHPR
	P=P+H*PD
	T = T + H
	FP = F*P
	PFT = FP.transpose()
	PHT = P*HT
	HP =  HMAT*P
	PHTHP = PHT*HP
	PHTHPR = (1/PHIN)*PHTHP
	PFTFP = PFT + FP
	PFTFPQ = PFTFP + Q
	PD = PFTFPQ - PHTHPR
	P =.5*(POLD+P+H*PD)
	K = PHT/PHIN
	K1.append(TS*K[0,0])
	K2.append(TS*K[1,0])	
	t.append(T)

	
plt.figure(1)
plt.grid(True)
plt.plot(t,K1)
plt.xlabel('Time (Sec)')
plt.ylabel('Kalman Gain 1')
plt.xlim(0,10)
plt.ylim(0,1.2)

plt.figure(2)
plt.grid(True)
plt.plot(t,K2)
plt.xlabel('Time (Sec)')
plt.ylabel('Kalman Gain 2')
plt.xlim(0,10)
plt.ylim(0,10)

plt.show()

