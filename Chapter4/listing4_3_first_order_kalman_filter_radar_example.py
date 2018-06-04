import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY



'''
Errors in estimates of first state and second state of a first-order polynomial Kalman filter:
Example of radar. The acceleration state is disregarded.
fairly insensitive to the initial covariance matrix.
First order polynomial is a 2x2 matrix
Zero order is a single number
Second Order is a 3x3 matrix'''

t=[]
x=[]
x_hat=[]
xdot=[]
xdot_hat=[]
x_hat_ERR=[]
sp11=[]
sp11P=[]
xdot_hat_ERR=[]
sp22=[]
sp22P=[]
xddot_hat_ERR=[]


PHIS=0.
TS=.1
A0=400000
A1=-6000.
A2=-16.1
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=1000.
P0 = 99999999.0
PHI = np.matrix([[1, TS],[0,1]])
M = np.matrix([[0, 0],[0,0]])
H = np.matrix([[1, 0]])
I = np.matrix([[1, 0],[0,1]])
R = np.matrix([[SIGMA_NOISE**2]])
Q = np.matrix([[TS**3/3, TS*TS/2] ,[TS*TS/2, TS] ])
P = np.matrix([[P0, 0],[0,P0]])

for T in [x*TS for x in range(0,301)]:
	M=PHI*P*PHI.transpose() +PHIS*Q
	# Kalman gaing given by the solution of Riccati equation
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))  
	P=(I-K*H)*M;
	XNOISE = GAUSS_PY(SIGMA_NOISE)
	X=A0+A1*T+A2*T*T
	XD=A1+2*A2*T
	XDD=2*A2
	XS=X+XNOISE
	RES=XS-XH-TS*XDH-.5*TS*TS*XDDH
	XH=XH+XDH*TS+.5*TS*TS*XDDH+K[0,0]*RES
	XDH=XDH+XDDH*TS+K[1,0]*RES
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	SP11P=-SP11
	SP22P=-SP22
	XHERR=X-XH
	XDHERR=XD-XDH
	sp11.append(SP11)
	sp22.append(SP22)
	t.append(T)
	x.append(X)
	x_hat.append(XH)
	xdot.append(XD)
	xdot_hat.append(XDH)
	x_hat_ERR.append(XHERR)
	xdot_hat_ERR.append(XDHERR)
	sp11P.append(SP11P)
	sp22P.append(SP22P)

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
plt.plot(t,x_hat_ERR)
plt.plot(t,sp11)
plt.plot(t,sp11P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.xlim(0,30)
plt.ylim(-1500,1500)

plt.figure(4)
plt.grid(True)
plt.plot(t,xdot_hat_ERR)
plt.plot(t,sp22)
plt.plot(t,sp22P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Velocity (Ft/Sec)')
plt.xlim(0,30)
plt.ylim(-500,500)

plt.show()
