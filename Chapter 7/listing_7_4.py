import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
import random

def GAUSS(SIG):
	SUM=0
	for j in range(1,7):
		# THE NEXT STATEMENT PRODUCES A UNIF. DISTRIBUTED NUMBER FROM -0.5 and 0.5
		IRAN=random.uniform(-0.5, 0.5)
		SUM=SUM+IRAN
		# In this case we have to multiply the resultant random variable by the square root of 2,
	X=math.sqrt(2)*SUM*SIG
	return (X)
	
def GAUSS_PY(SIG):
	X=(random.uniform(-3.0,3.0))*SIG
	return (X)	

def PROJECT(TS,XP,XDP,HP):
	T=0.
	X=XP
	XD=XDP
	H=HP
	while (T<=(TS-.0001)):
		XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2
		XD=XD+H*XDD
		X=X+H*XD
		T=T+H
	XH=X
	XDH=XD
	XDDH=XDD
	return [XH, XDH, XDDH]
	
TS=.1
TF=30.
T=0.
S=0.
H=.001
HP=.1
BETA=500
errx = []
errxd = []
t =[]

for HP in [0.1, 0.01, 0.001]:
	T = 0
	X=200000.
	XD=-6000.
	XH=X
	XDH=XD
	while (T<=TF):
		XOLD=X
		XDOLD=XD
		XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2
		X=X+H*XD
		XD=XD+H*XDD
		T=T+H
		XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2
		X=.5*(XOLD+X+H*XD)
		XD=.5*(XDOLD+XD+H*XDD)
		S=S+H
		if (S>=(TS-.00001)):
			S=0.
			SOL = PROJECT(TS,XH,XDH,HP)
			XH=SOL[0]
			XDH=SOL[1]
			XDDH= SOL[2]
			ERRX=X-XH
			ERRXD=XD-XDH
			errx.append(ERRX)
			errxd.append(ERRXD)
			t.append(T)
	plt.figure(1)
	plt.plot(t,errx,label='H='+str(HP), linewidth=0.6)
	plt.figure(2)
	plt.plot(t,errxd,label='H='+str(HP), linewidth=0.6)
	t=[]
	errx=[]
	errxd=[]


plt.figure(1)
plt.grid(True)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in altitude')
plt.xlim(0,30)
plt.legend()
plt.ylim(-30,30)

plt.figure(2)
plt.grid(True)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in velocity')
plt.xlim(0,30)
plt.legend()
plt.ylim(-5,15)
plt.show()
