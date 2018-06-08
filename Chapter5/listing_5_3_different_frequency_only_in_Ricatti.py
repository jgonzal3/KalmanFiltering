import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

'''
Let us now assume  that  the  fundamental  matrix  was known precisely.
However, our knowledge of the signal frequency W, was in error by 100%. 
The real  signal frequency was 1 rad/s, but in  the fundamental matrix, 
PHI and  in the  filter we  thought  the weave  frequency  was actually 
2 r/s, WH. The estimate of the signal  deteriorates substantially. This 
shows the sensitivity of this type of Kalman filter to modelling errors
The polynomial Kalman  filter would have been  insensitive to this type 
of error because it made no  use of knowledge of the frequency o of the 
sinusoidal signal.
'''
t=[]
x=[]
x_hat=[]
xdot=[]
xdot_hat=[]
xs=[]
xddot_hat=[]
x_hat_ERR=[]
sp11=[]
sp11P=[]
xdot_hat_ERR=[]
sp22=[]
sp22P=[]
xddot_hat_ERR=[]

PHIS=0.
TS=.1
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=1.
W = 1
A = 1
WH = 2

PHI = np.matrix([[math.cos(WH*TS), math.sin(WH*TS)/WH],[-WH*math.sin(WH*TS), math.cos(WH*TS)]])
P = np.matrix([[99999999., 0],[0,99999999.]])
I = np.matrix([[1, 0],[0, 1]])
Q = np.matrix([[TS**3/3, TS*TS/2] ,[TS*TS/2, TS]])
H = np.matrix([[1, 0]])
R = np.matrix([[SIGMA_NOISE**2]])


for T in [x*TS for x in range(0,201)]:
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))
	P=(I-K*H)*M	
	XNOISE = GAUSS_PY(SIGMA_NOISE)
	X=A*math.sin(W*T)
	XD=A*W*math.cos(W*T)
	XS=X+XNOISE
	XHOLD=XH
	RES=XS-XH*math.cos(W*TS)-math.sin(W*TS)*XDH/W
	XH=math.cos(W*TS)*XH+XDH*math.sin(W*TS)/W+K[0,0]*RES
	XDH=-W*math.sin(W*TS)*XHOLD+XDH*math.cos(W*TS)+K[1,0]*RES
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	XHERR=X-XH
	XDHERR=XD-XDH
	SP11P=-SP11
	SP22P=-SP22
	t.append(T)
	x.append(X)
	xs.append(XS)
	x_hat.append(XH)
	xdot.append(XD)
	xdot_hat.append(XDH)
	x_hat_ERR.append(XHERR)
	sp11.append(SP11)
	sp11P.append(SP11P)
	xdot_hat_ERR.append(XDHERR)
	sp22.append(SP22)
	sp22P.append(SP22P)

plt.figure(1)
plt.grid(True)
plt.plot(t,x)
plt.plot(t,xs)
plt.xlabel('Time (Sec)')
plt.ylabel('Measurement and True Signal')
plt.xlim(0,20)
plt.ylim(-4,4)

plt.figure(2)
plt.grid(True)
plt.plot(t,x)
plt.plot(t,x_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,20)
plt.ylim(-4,4)

plt.figure(3)
plt.grid(True)
plt.plot(t,xdot)
plt.plot(t,xdot_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('XD and XDH')
plt.xlim(0,20)
plt.ylim(-4,4)

plt.show()
