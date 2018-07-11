import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

t=[]
x=[]
a=[]
xd=[]
x_hat=[]
xd=[]
xdh=[]
xdd=[]
xdddd=[]
xd_hat=[]
xs=[]
xh=[]
xdd_hat=[]
xddd_hat=[]
xdddd_hat=[]
x_hat_ERR=[]
sp11=[]
sp11P=[]
xd_hat_ERR=[]
sp22=[]
sp22P=[]
xdd_hat_ERR=[]
sp33=[]
sp33P=[]
xddd_hat_ERR=[]
sp44=[]
sp44P=[]
ArrayRES = []

PHIS=1.
XLOSE = 99
SIGMA_NOISE=50.0
P0 = 999999999999999.0
add_noise = 1

# ***************************************************************************************
#
# The higher-order polynomial Kalman filter is less accurate than  the lower-order filter 
# in estimating the lower-order  derivatives of  the signal when only a few  measurements 
# are available.
# Of course, the advantage of a higher-order polynomial Kalman filter is that we are able
# to estimate higher-order derivatives of the signal.
#
# ***************************************************************************************
W = .1
TS = 1.0

I = np.identity(3)
P = np.matrix([[P0, 0.0, 0.0],[0.0, P0, 0.0],[0.0, 0.0, P0]])
F = np.matrix([[0.0,  1.0, 0.0], [0.0, 0.0, 1.0],[0.0, 0.0, 0.0]])
XH = np.matrix([[0.0],[0.0],[0.0]])
Q = np.matrix([[TS**5/20.0, TS**4/8.0, TS**3/6.0],
			   [TS**4/8.0,  TS**3/3.0, TS**2/2.0],
			   [TS**3/3.0,  TS**2/2.0, TS]])
HMAT = np.matrix([[1.0, 0.0, 0.0]])
PHI = I + TS*F + TS**2*F*F/2
RMAT = np.matrix([[SIGMA_NOISE**2]])

for T in [x*TS for x in range(0,201)]:
	if(T>XLOSE):
		RMAT[0,0]=999999999999999.
	# Estimate the Riccatti matrices and solve the Riccati differential equation
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
	P=(I-K*HMAT)*M	
	if add_noise == 1:
		XNOISE = GAUSS_PY(SIGMA_NOISE)
	else:
		XNOISE = 0.0
	# Instead of using a single value for each variable, a matrix is defined
	X=np.matrix([[100*T - (20*math.cos(W*T))/W + 20/W],[100 + 20*math.sin(W*T)],[20*W*math.cos(W*T)]])
	XS=X[0,0]+XNOISE
	# The is the residual.
	RES = XS - HMAT*PHI*XH
	XH = PHI*XH + K*(XS - HMAT*PHI*XH)
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	# Theoretical value of the residuals estimated in the book
	SP44=math.sqrt(HMAT*M*HMAT.transpose()+RMAT)
	Xhat = XH[0,0]
	XDhat= XH[1,0]
	XDDhat = XH[2,0]
	XHERR=X-XH
	SP11P=-SP11
	SP22P=-SP22
	SP44P=-SP44
	t.append(T)
	ArrayRES.append(RES[0,0])
	xs.append(XS)
	x.append(X[0,0])
	x_hat.append(XH[0,0])
	xd.append(X[1,0])
	xd_hat.append(XH[1,0])
	xdd.append(X[2,0])
	xdd_hat.append(XH[2,0])
	
	x_hat_ERR.append(XHERR[0,0])
	xd_hat_ERR.append(XHERR[1,0])
	sp11.append(SP11)
	sp11P.append(SP11P)
	sp22.append(SP22)
	sp22P.append(SP22P)

plt.figure(1)
plt.grid(True)
plt.plot(t,x,label='X actual', linewidth=0.6)
plt.plot(t,x_hat,label='X estimate', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.ylim(0,20000)
plt.xlim(0,200)
plt.legend()

plt.figure(2)
plt.grid(True)
plt.plot(t,xd,label='XD actual', linewidth=0.6)
plt.plot(t,xd_hat,label='XD estimate', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('XD Estimate and True Signal')
plt.ylim(0,200)
plt.xlim(0,200)

plt.figure(3)
plt.grid(True)
plt.plot(t,x_hat_ERR,label='ERROR', linewidth=0.6)
plt.plot(t,sp11,label='sp11', linewidth=0.6)
plt.plot(t,sp11P,label='sp11p', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity Estimate and True Signal')
plt.xlim(0,200)
plt.ylim(-200,200)

plt.figure(4)
plt.grid(True)
plt.plot(t,x_hat_ERR,label='ERROR', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity Estimate and True Signal')
plt.xlim(0,200)
plt.ylim(0,8000)
plt.show()
