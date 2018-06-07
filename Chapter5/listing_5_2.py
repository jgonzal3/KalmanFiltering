import math
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

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
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=1.

PHI = np.matrix([[1, TS, 0.5*TS*TS],[0, 1, TS] ,[0, 0, 1] ])
P = np.matrix([[99999999., 0, 0],[0,99999999. , 0], [0, 0, 99999999.]])
I = np.matrix([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
Q = np.matrix([[TS**5/20, TS**4/8, TS**3/6],[TS**4/8, TS**3/3, TS*TS/2] ,[TS**3/6, TS*TS/2, TS] ])
H = np.matrix([[1, 0 , 0]])
R = np.matrix([[SIGMA_NOISE**2]])


for T in [x*TS for x in range(0,201)]:
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))
	P=(I-K*H)*M
	XNOISE = GAUSS_PY(SIGMA_NOISE)
	X=math.sin(T)
	XD=math.cos(T)
	XDD=-math.sin(T)
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

textstr = 'Second-Order Filter\n$Q=0, \Phi_s=%.2f$' % (PHIS)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.figure(1)
plt.grid(True)
plt.plot(t,x)
plt.plot(t,x_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('x')
plt.xlim(0,20)
plt.text(14, 2.5, textstr,bbox=props)

plt.figure(2)
plt.grid(True)
plt.plot(t,xdot)
plt.plot(t,xdot_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('x dot (Ft/Sec)')
plt.xlim(0,20)
plt.ylim(-4,4)
plt.text(14, 2.5, textstr,bbox=props)

plt.figure(3)
plt.grid(True)
plt.plot(t,xddot)
plt.plot(t,xddot_hat)
plt.xlabel('Time (Sec)')
plt.ylabel('Acceleration (Ft/Sec^2)')
plt.xlim(0,20)
plt.text(14, 2.5, textstr,bbox=props)

plt.figure(4)
plt.grid(True)
plt.plot(t,x_hat_ERR)
plt.plot(t,sp11)
plt.plot(t,sp11P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.xlim(0,20)
plt.text(14, 2.5, textstr,bbox=props)

plt.show()
