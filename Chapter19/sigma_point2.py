import numpy as np
#import numpy.linalg.cholesky as chsky
import math
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def PROJECTC19L1(TS,XP,HP):
	T=0;
	XT =0
	XTD=1
	W  =1
	H=HP;
	X= []
	XD =[]
	TD = []	
	while T<=30:
		XTDD=-W*W*XT;
		XTD=XTD+H*XTDD;
		XT=XT+H*XTD;
		T=T+H;
		TD.append(T)
		X.append(XT)
		XD.append(XTD)
	return (X, XD, TD)


def f1(x1,x2,w,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,w,t):
	dx2dt = -w*w*x1
	return (dx2dt)
	
def rk4(TS,XP, HP):
	X1  =0
	X1D =1
	W   =1
	T = 0.0
	dt = HP
	X= []
	XD =[]
	TD = []
	while (T<=30.0):
		k11 = dt*f1(X1,X1D,W,T)
		k21 = dt*f2(X1,X1D,W,T)
		k12 = dt*f1(X1+0.5*k11,X1D+0.5*k21,W,T+0.5*dt)
		k22 = dt*f2(X1+0.5*k11,X1D+0.5*k21,W,T+0.5*dt)
		k13 = dt*f1(X1+0.5*k12,X1D+0.5*k22,W,T+0.5*dt)
		k23 = dt*f2(X1+0.5*k12,X1D+0.5*k22,W,T+0.5*dt)
		k14 = dt*f1(X1+k13,X1D+k23,W,T+dt)
		k24 = dt*f2(X1+k13,X1D+k23,W,T+dt)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X1D = X1D + (k21+2*k22+2*k23+k24)/6
		T = T+dt
		TD.append(T)
		X.append(X1)
		XD.append(X1D)
	return (X, XD, TD)
	
SIGX = 0.1
PHIS = 1.0
TS = 0.1
XH=0.0
XDH= 0.0
WH = 3.0
HP = 0.0001
RK = 0 # Set to 1 uses the Runge-Kutta; set to 0 uses Euler
P=np.matrix([[SIGX**2,0,0],[0,10.**2,0],[0,0,3.**2]])

Q=np.matrix([[0,0,0],[0,PHIS*TS,0],[0,0,PHIS*TS]])
XHAT = np.matrix([[XH], [XDH], [WH]])

XS0 = XHAT
(x1,x2,t) = rk4(TS, XS0, HP)
(x1a,x2a,ta) = PROJECTC19L1(TS, XS0, HP*10);

plt.figure(1)
plt.plot(t,x1,linewidth=0.8, label='x1')
plt.plot(t,x2,linewidth=0.8, label='x2')
plt.legend()
plt.grid(True)
plt.ylim(-3, 3)

plt.figure(2)
plt.plot(ta,x1a,linewidth=0.8, label='x1a')
plt.plot(ta,x2a,linewidth=0.8, label='x2a')
plt.legend()
plt.grid(True)
plt.ylim(-3, 3)
plt.show()
