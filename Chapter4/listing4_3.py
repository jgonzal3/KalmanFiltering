import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

t=[]
x=[]
x_hat=[]
xdot=[]
xdot_hat=[]
xddot=[]
xddot_hat=[]
x_hat_ERR=[]
sp11=[]
sp11P=[]
xdot_hat_ERR=[]
sp22=[]
sp22P=[]
xddot_hat_ERR=[]
sp33=[]
sp33P=[]


PHIS=0.
TS=.1
A0=400000
A1=-6000.
A2=-16.1
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=1000.

PHI = np.matrix([[1, TS, 0.5*TS*TS],[0, 1, TS] ,[0, 0, 1] ])
P = np.matrix([[99999999., 0, 0],[0,99999999. , 0], [0, 0, 99999999.]])
I = np.matrix([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
Q = np.matrix([[TS**5/20, TS**4/8, TS**3/6],[TS**4/8, TS**3/3, TS*TS/2] ,[TS**3/6, TS*TS/2, TS] ])
H = np.matrix([[1, 0 , 0]])
R = np.matrix([[SIGMA_NOISE**2]])


for T in [x*TS for x in range(0,301)]:
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))
	for v in range(0,4):
		print (K[0,0], K[1,0], K[2,0])
	P=(I-K*H)*M
	for v in range(0,4):
		print (P[0,0], P[1,1], P[2,2])	
	XNOISE = GAUSS_PY(SIGMA_NOISE)
	X=A0+A1*T+A2*T*T
	XD=A1+2*A2*T
	XDD=2*A2
	XS=X+XNOISE
	RES=XS-XH-TS*XDH-.5*TS*TS*XDDH
	XH=XH+XDH*TS+.5*TS*TS*XDDH+K[0,0]*RES
	XDH=XDH+XDDH*TS+K[1,0]*RES
	XDDH=XDDH+K[2,0]*RES
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	SP33=math.sqrt(P[2,2])
	XHERR=X-XH
	XDHERR=XD-XDH
	XDDHERR=XDD-XDDH
	SP11P=-SP11
	SP22P=-SP22
	SP33P=-SP33
	#print(T,X,XH,XD,XDH,XDD,XDDH)
	#print(T,XHERR,SP11,-SP11,XDHERR,SP22,-SP22,XDDHERR,SP33,-SP33)
	t.append(T)
	x.append(X)
	x_hat.append(XH)
	xdot.append(XD)
	xdot_hat.append(XDH)
	xddot.append(XDD)
	xddot_hat.append(XDDH)
	x_hat_ERR.append(XHERR)
	sp11.append(SP11)
	sp11P.append(SP11P)
	xdot_hat_ERR.append(XDHERR)
	sp22.append(SP22)
	sp22P.append(SP22P)
	xddot_hat_ERR.append(XDDHERR)
	sp33.append(SP33)
	sp33P.append(SP33P)

plt.figure(1)
plt.grid(True)
plt.plot(t,x)
plt.plot(t,x_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('Altitude (Ft)')
plt.xlim(0,30)
plt.ylim(0,400000)

plt.figure(2)
plt.grid(True)
plt.plot(t,xdot)
plt.plot(t,xdot_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity (Ft/Sec)')
plt.xlim(0,30)
plt.ylim(-10000,0)

plt.figure(3)
plt.grid(True)
plt.plot(t,xddot)
plt.plot(t,xddot_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('Acceleration (Ft/Sec^2)')
plt.xlim(0,30)
plt.ylim(-100,100)

plt.figure(4)
plt.grid(True)
plt.plot(t,x_hat_ERR)
plt.plot(t,sp11)
plt.plot(t,sp11P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.xlim(0,30)
plt.ylim(-1500,1500)

plt.figure(5)
plt.grid(True)
plt.plot(t,xdot_hat_ERR)
plt.plot(t,sp22)
plt.plot(t,sp22P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Velocity (Ft/Sec)')
plt.xlim(0,30)
plt.ylim(-500,500)

plt.show()
