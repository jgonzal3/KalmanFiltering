import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

t=[]
x=[]
x_hat=[]
xd=[]
xdd=[]
xddd=[]
xdddd=[]
xd_hat=[]
xs=[]
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

PHIS=0000.
TS=0.1
SIGMA_NOISE=5.0
P0 = 9999999999999.0
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
A0=3.
A1=1.
A2=0.
XH=0
XDH=0
XDDH=0


ITERM = 2
I = np.identity(ITERM)
if ITERM == 2:
	F = np.matrix([[0.0,  1.0],  [0.0, 0.0]])
	P = np.matrix([[P0, 0.0],[0.0 ,P0]])
	HMAT = np.matrix([[1.0, 0.0]])
	PHI = I+TS*F
	Q = np.matrix([[TS*TS*TS/3.0,0.5*TS*TS],[0.5*TS*TS, TS]])
if ITERM == 3:
	P = np.matrix([[P0, 0.0, 0.0],[0.0, P0, 0.0],[0.0, 0.0, P0]])
	F = np.matrix([[0.0,  1.0, 0.0], [0.0, 0.0, 1.0],[0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0]])
	PHI = I + TS*F + TS**2*F*F/2
if ITERM == 4:
	P = np.matrix([[P0, 0.0, 0.0,0.0],[0.0, P0, 0.0,0.0],[0.0, 0.0, P0,0.0],[0.0, 0.0, 0.0,P0]])
	F = np.matrix([[0.0,  1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0,1.0], [0.0, 0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0], [0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0 , 0.0]])
	PHI = I + TS*F +TS**2*F*F/2 + TS**3*F*F*F/6
if ITERM == 5:
	P = np.matrix([[P0, 0.0, 0.0, 0.0, 0.0],[0.0, P0, 0.0, 0.0, 0.0],[0.0, 0.0, P0, 0.0, 0.0],[0.0, 0.0, 0.0, P0, 0.0],[0.0, 0.0, 0.0, 0.0, P0]])
	F = np.matrix([[0.0,  1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0], [0.0], [0.0]])
	PHI = I + TS*F +TS**2*F*F/2 + TS**3*F*F*F/6 + TS**4*F*F*F*F/24
	
RMAT = np.matrix([[SIGMA_NOISE**2]])

for T in [x*TS for x in range(0,101)]:
	# Estimate the Riccatti matrices and solve the Riccati differential equation
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
	P=(I-K*HMAT)*M	
	if add_noise == 1:
		XNOISE = GAUSS_PY(SIGMA_NOISE)
	else:
		XNOISE = 0.0
	# Instead of using a single value for each variable, a matrix is defined
	X=np.matrix([[A0+A1*T+A2*T*T],[A1+2.0*A2*T]])
	XS=X[0,0]+XNOISE
	# The is the residual.
	RES = XS - HMAT*PHI*XH
	XH = PHI*XH + K*(XS - HMAT*PHI*XH)
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	#SP33=math.sqrt(P[2,2])
	# Theoretical value of the residuals estimated in the book
	SP44=math.sqrt(HMAT*M*HMAT.transpose()+RMAT)
	XHERR=X-XH
	SP11P=-SP11
	SP22P=-SP22
	#SP33P=-SP33
	SP44P=-SP44
	t.append(T)
	ArrayRES.append(RES[0,0])
	xs.append(XS)
	x.append(X[0,0])
	x_hat.append(XH[0,0])
	xd.append(X[1,0])
	xd_hat.append(XH[1,0])
	#xdd.append(X[2,0])
	#xdd_hat.append(XH[2,0])
	
	x_hat_ERR.append(XHERR[0,0])
	xd_hat_ERR.append(XHERR[1,0])
	#xdd_hat_ERR.append(XHERR[2,0])
	sp11.append(SP11)
	sp11P.append(SP11P)
	sp22.append(SP22)
	sp22P.append(SP22P)
	#sp33.append(SP33)
	#sp33P.append(SP33P)
	sp44.append(SP44)
	sp44P.append(SP44P)
	
plt.figure(1)
plt.grid(True)
plt.plot(t,x,label='X', linewidth=0.6)
plt.plot(t,x_hat,label='XH', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.ylim(0,12)
plt.xlim(0,10)
plt.show()
fafd
plt.figure(1)
plt.grid(True)
plt.plot(t,x_hat_ERR,label='XH error', linewidth=0.6)
plt.plot(t,sp11,label='sp11', linewidth=0.6)
plt.plot(t,sp11P,label='sp11p', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.ylim(-100,100)
plt.xlim(0,10)

plt.figure(2)
plt.grid(True)
plt.plot(t,xd_hat_ERR,label='Xdot error', linewidth=0.6)
plt.plot(t,sp22,label='sp22', linewidth=0.6)
plt.plot(t,sp22P,label='sp22p', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity Estimate and True Signal')
plt.xlim(0,10)
plt.ylim(-150,150)

plt.figure(3)
plt.grid(True)
plt.plot(t,ArrayRES,label='Residual', linewidth=0.6)
plt.plot(t,sp44,label='sp44', linewidth=0.6)
plt.plot(t,sp44P,label='sp44p', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X dot Estimate and True Signal')
plt.xlim(0,10)
plt.ylim(-50,50)
plt.show()
