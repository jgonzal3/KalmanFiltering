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
a0_hat=[]
a1_hat=[]
a2_hat=[]
xd=[]
xdh=[]

xs=[]
xh=[]
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
A1=0.1
A2=0.9
XH=0
XDH=0
XDDH=0

ITERM = 3
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
	F = np.matrix([[0.0,  0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0]])
	AH = np.matrix([[0.0], [0.0], [0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0]])
	Q = np.identity(3)
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

# Compare results using the methodology given in the book.
ArrayX=[]
ArrayXD=[]
ArrayA0=[]
ArrayA1=[]
ArrayA2=[]
A0Hl = 0.
A1Hl = 0.
A2Hl = 0.

for T in [x*TS for x in range(0,101)]:
	# Estimate the Riccatti matrices and solve the Riccati differential equation
	HMAT = np.matrix([[1.0, T, T]])
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
	P=(I-K*HMAT)*M	
	if add_noise == 1:
		XNOISE = GAUSS_PY(SIGMA_NOISE)
	else:
		XNOISE = 0.0
	# Instead of using a single value for each variable, a matrix is defined
	X=np.matrix([[A0+A1*T+A2*T],[A1+A2]])
	XS=X[0,0]+XNOISE
	# The is the residual.
	RES = XS - HMAT*PHI*AH
	AH = PHI*AH + K*(XS - HMAT*PHI*AH)
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	A0H = AH[0,0]
	A1H = AH[1,0]
	A2H = AH[2,0]
	xh.append(A0H + A1H*T + A2H*T)
	xdh.append(A1H+A2H)
	SP11P=-SP11
	SP22P=-SP22
	t.append(T)
	ArrayRES.append(RES[0,0])
	xs.append(XS)
	x.append(X[0,0])
	xd.append(X[1,0])
	a0_hat.append(A0H)
	a1_hat.append(A1H)
	a2_hat.append(A2H)

	sp11.append(SP11)
	sp11P.append(SP11P)
	sp22.append(SP22)
	sp22P.append(SP22P)
	
	Xl=A0+A1*T+A2*T
	XDl=A1+A2
	XSl=Xl+XNOISE
	RESl=XSl-A0Hl-A1Hl*T-A2Hl*T
	A0Hl=A0Hl+K[0,0]*RESl
	A1Hl=A1Hl+K[1,0]*RESl
	A2Hl=A2Hl+K[2,0]*RESl
	ArrayA0.append(A0Hl)
	ArrayA1.append(A1Hl)
	ArrayA2.append(A2Hl)
	
	SP11l=math.sqrt(P[0,0])
	SP22l=math.sqrt(P[1,1])
	SP33l=math.sqrt(P[2,2])
	A0HERRl=A0-A0H
	A1HERRl=A1-A1H
	A2HERR=A2-A2H
	
plt.figure(1)
plt.grid(True)
plt.plot(t,a0_hat,label='A0', linewidth=0.6)
plt.plot(t,ArrayA0,label='A0 book', linewidth=0.6)
plt.plot(t,[A0 for x in t],label='XH', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('A0 Estimate and True Signal')
plt.legend()
plt.ylim(-10,10)
plt.xlim(0,10)

plt.figure(2)
plt.grid(True)
plt.plot(t,a1_hat,label='A1', linewidth=0.6)
plt.plot(t,ArrayA1,label='A1 book', linewidth=0.6)
plt.plot(t,[A1 for x in t],label='Estimate', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('A1 Estimate and True Signal')
plt.ylim(-10,10)
plt.xlim(0,10)
plt.legend()

plt.figure(3)
plt.grid(True)
plt.plot(t,ArrayA2,label='A2 book', linewidth=0.6)
plt.plot(t,a2_hat,label='A2', linewidth=0.6)
plt.plot(t,[A2 for x in t],label='Estimate', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('A2 Estimate and True Signal')
plt.ylim(-10,10)
plt.xlim(0,10)
plt.legend()
plt.show()
af
plt.figure(4)
plt.grid(True)
plt.plot(t,x,label='X actual', linewidth=0.6)
plt.plot(t,xh,label='X hat', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity Estimate and True Signal')
plt.xlim(0,10)
plt.ylim(-15,15)

plt.figure(5)
plt.grid(True)
plt.plot(t,xd,label='X actual', linewidth=0.6)
plt.plot(t,xdh,label='X hat', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X dot Estimate and True Signal')
plt.xlim(0,10)
plt.ylim(-15,15)
