import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

t=[]
x=[]
a=[]
ad=[]
a_hat=[]
xd=[]
xdh=[]
xddd=[]
xdddd=[]
ad_hat=[]
xs=[]
xh=[]
xdd_hat=[]
xddd_hat=[]
xdddd_hat=[]
a_hat_ERR=[]
sp11=[]
sp11P=[]
ad_hat_ERR=[]
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
	F = np.matrix([[0.0,  0.0],  [0.0, 0.0]])
	P = np.matrix([[P0, 0.0],[0.0 ,P0]])
	HMAT = np.matrix([[1.0, 0.0]])
	AH = np.matrix([[0.0], [100.0]])
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
	HMAT = np.matrix([[1.0, T]])
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
	P=(I-K*HMAT)*M	
	if add_noise == 1:
		XNOISE = GAUSS_PY(SIGMA_NOISE)
	else:
		XNOISE = 0.0
	# Instead of using a single value for each variable, a matrix is defined
	X=np.matrix([[A0+A1*T],[A1]])
	XS=X[0,0]+XNOISE
	# The is the residual.
	RES = XS - HMAT*PHI*AH
	AH = PHI*AH + K*(XS - HMAT*PHI*AH)
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	#SP33=math.sqrt(P[2,2])
	# Theoretical value of the residuals estimated in the book
	SP44=math.sqrt(HMAT*M*HMAT.transpose()+RMAT)
	A0H = AH[0,0]
	A1H = AH[1,0]
	xh.append(A0H + A1H*T)
	xdh.append(A1H)
	XHERR=X-AH
	SP11P=-SP11
	SP22P=-SP22
	#SP33P=-SP33
	SP44P=-SP44
	t.append(T)
	ArrayRES.append(RES[0,0])
	xs.append(XS)
	x.append(X[0,0])
	a_hat.append(AH[0,0])
	xd.append(X[1,0])
	ad_hat.append(AH[1,0])
	#xdd.append(X[2,0])
	#xdd_hat.append(XH[2,0])
	
	a_hat_ERR.append(XHERR[0,0])
	ad_hat_ERR.append(XHERR[1,0])
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
plt.plot(t,a_hat,label='X', linewidth=0.6)
plt.plot(t,[A0 for x in t],label='XH', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.ylim(-20,100)
plt.xlim(0,10)

plt.figure(2)
plt.grid(True)
plt.plot(t,ad_hat,label='Actaul', linewidth=0.6)
plt.plot(t,[A1 for x in t],label='Estimate', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('XD Estimate and True Signal')
plt.ylim(-20,100)
plt.xlim(0,10)

plt.figure(3)
plt.grid(True)
plt.plot(t,x,label='X actual', linewidth=0.6)
plt.plot(t,xh,label='X hat', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity Estimate and True Signal')
plt.xlim(0,10)
plt.ylim(-15,15)

plt.figure(4)
plt.grid(True)
plt.plot(t,xd,label='X actual', linewidth=0.6)
plt.plot(t,xdh,label='X hat', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X dot Estimate and True Signal')
plt.xlim(0,10)
plt.ylim(-15,15)
plt.show()
