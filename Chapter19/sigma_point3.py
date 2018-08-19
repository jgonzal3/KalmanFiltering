import numpy as np
#import numpy.linalg.cholesky as chsky
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def PROJECTC19L1(TS,XP,HP):
	T=0;
	XT =XP[0,0]
	XTD=XP[1,0]
	W  =XP[2,0]
	H=HP;
	while T<=(TS-.0001):
		XTDD=-W*W*XT;
		XTD=XTD+H*XTDD;
		XT=XT+H*XTD;
		T=T+H;
	XB=XT;
	XDB=XTD;
	WB=W;
	XB = np.matrix([[XB],[XDB],[WB]])
	return (XB)

def f1(x1,x2,w,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,w,t):
	dx2dt = -w*w*x1
	return (dx2dt)
	
def rk4(TS,XP, HP):
	X1  =XP[0,0]
	X1D =XP[1,0]
	W   =XP[2,0]
	T = 0.0
	dt = HP
	while (T<=(TS-.0001)):
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
	X1a = X1
	X2a = X1D
	XB = np.matrix([[X1a], [X2a], [W]])
	return (XB)
	
ArrayT = []
ArrayW = []
ArrayWH = []
ArrayERRW = []
ArraySP33 = []
ArraySP33P = []
	
SIGX = 0.1
PHIS = 1.0
TS = 0.1
XH=0.0
XDH= 1.0
WH = 3.0
HP = 0.001
XT =0.0
XTD=3.0
SP=0.0

P=np.matrix([[SIGX**2,0,0],[0,10.**2,0],[0,0,3.**2]])
QMAT=np.matrix([[0,0,0],[0,0,0],[0,0,PHIS*TS]])
RMAT=np.matrix([[SIGX*SIGX]])
XHAT = np.matrix([[XH], [XDH], [WH]])
HMAT=np.matrix([[1,0,0]])

N = 3
KAPPA = 0.0
H=0.001
T=0.0
QCONSTANT = 1
NOISE = 1

RK = 1# Set to 1 uses the Runge-Kutta; set to 0 uses Euler

while (T < 30.0):
	XTOLD=XT;
	XTDOLD=XTD;
	if QCONSTANT==1:
		W=5
	else:
		W=8.3*T/30.+.2
	XTDD=-W*W*XT;
	XT=XT+H*XTD;
	XTD=XTD+H*XTDD;
	T=T+H;
	if QCONSTANT==1:
		W=5
	else:
		W=8.3*T/30.+.2;
	XTDD=-W*W*XT
	XT=.5*(XTOLD+XT+H*XTD)
	XTD=.5*(XTDOLD+XTD+H*XTDD)
	SP=SP+H;
	if SP>=(TS-.00001):
		SP=0.;
		C = np.linalg.cholesky((N+KAPPA)*P)
		#STEP1: State estimate vector
		XS0 = XHAT
		XS1 = XHAT + C[:,0]
		XS2 = XHAT + C[:,1]
		XS3 = XHAT + C[:,2]
		XS4 = XHAT - C[:,0]
		XS5 = XHAT - C[:,1]
		XS6 = XHAT - C[:,2]
		#Step 2: Build the weighting factors
		W0 = KAPPA/(N+KAPPA)
		W1 = 1/(2*(N + KAPPA))
		W2 = 1/(2*(N + KAPPA))
		W3 = 1/(2*(N + KAPPA))
		W4 = 1/(2*(N + KAPPA))
		W5 = 1/(2*(N + KAPPA))
		W6 = 1/(2*(N + KAPPA))
		#Step 3: Propagate the sigma points one time interval
		if (RK == 0):
			XSB0 = PROJECTC19L1(TS, XS0, HP);
			XSB1 = PROJECTC19L1(TS, XS1, HP);
			XSB2 = PROJECTC19L1(TS, XS2, HP);
			XSB3 = PROJECTC19L1(TS, XS3, HP);
			XSB4 = PROJECTC19L1(TS, XS4, HP);
			XSB5 = PROJECTC19L1(TS, XS5, HP);
			XSB6 = PROJECTC19L1(TS, XS6, HP);
		else:
			XSB0 = rk4(TS, XS0, HP);
			XSB1 = rk4(TS, XS1, HP);
			XSB2 = rk4(TS, XS2, HP);
			XSB3 = rk4(TS, XS3, HP);
			XSB4 = rk4(TS, XS4, HP);
			XSB5 = rk4(TS, XS5, HP);
			XSB6 = rk4(TS, XS6, HP);
		#Step 4: Calculate the weighing average
		XBAR = W0*XSB0 + W1*XSB1 + W2*XSB2 + W3*XSB3 + W4*XSB4 + W5*XSB5 + W6*XSB6
		#STEP 5: Estimate the covariance matrix for the sigma points
		M0 = (XSB0 - XBAR)*(XSB0 - XBAR).transpose()
		M1 = (XSB1 - XBAR)*(XSB1 - XBAR).transpose()
		M2 = (XSB2 - XBAR)*(XSB2 - XBAR).transpose()
		M3 = (XSB3 - XBAR)*(XSB3 - XBAR).transpose()
		M4 = (XSB4 - XBAR)*(XSB4 - XBAR).transpose()
		M5 = (XSB5 - XBAR)*(XSB5 - XBAR).transpose()
		M6 = (XSB6 - XBAR)*(XSB6 - XBAR).transpose()
		#STEP6: build the new weighted matrix M and add process noise matrix QMAT
		M = (W0*M0 + W1*M1 + W2*M2 + W3*M3 + W4*M4 + W5*M5 + W6*M6) + QMAT
		#STEP 7: Estimate a new sigma points using the recent calculated M matrix
		CM = np.linalg.cholesky((N+KAPPA)*M)
		XS0 = XBAR
		XS1 = XBAR + CM[:,0]
		XS2 = XBAR + CM[:,1]
		XS3 = XBAR + CM[:,2]
		XS4 = XBAR - CM[:,0]
		XS5 = XBAR - CM[:,1]
		XS6 = XBAR - CM[:,2]
		#STEP 8: Estimate the measurements using the H matrix
		YS0 = HMAT*XS0
		YS1 = HMAT*XS1
		YS2 = HMAT*XS2
		YS3 = HMAT*XS3
		YS4 = HMAT*XS4
		YS5 = HMAT*XS5 
		YS6 = HMAT*XS6
		#STEP 9: Estimate the weighted measurements		
		YH = (W0*YS0 + W1*YS1 + W2*YS2 + W3*YS3 + W4*YS4 + W5*YS5 + W6*YS6)
		#STEP 10: Estimate new variance
		S0 = (YS0 - YH)*(YS0 - YH).transpose()
		S1 = (YS1 - YH)*(YS1 - YH).transpose()
		S2 = (YS2 - YH)*(YS2 - YH).transpose()
		S3 = (YS3 - YH)*(YS3 - YH).transpose()
		S4 = (YS4 - YH)*(YS4 - YH).transpose()
		S5 = (YS5 - YH)*(YS5 - YH).transpose()
		S6 = (YS6 - YH)*(YS6 - YH).transpose()
		# Form weighted average and add measurement noise matrix (scalar) (STEP 11)
		SMAT = (W0*S0 + W1*S1 + W2*S2 + W3*S3 + W4*S4 + W5*S5 + W6*S6 ) + RMAT
		#STEP 12 and STEP 13: Estimate the cross covariance between x and y
		Pxy0 = (XS0 - XBAR)*(YS0 - YH)
		Pxy1 = (XS1 - XBAR)*(YS1 - YH)
		Pxy2 = (XS2 - XBAR)*(YS2 - YH)
		Pxy3 = (XS3 - XBAR)*(YS3 - YH)
		Pxy4 = (XS4 - XBAR)*(YS4 - YH)
		Pxy5 = (XS5 - XBAR)*(YS5 - YH)
		Pxy6 = (XS6 - XBAR)*(YS6 - YH)
		PXY = W0*Pxy0 + W1*Pxy1 + W2*Pxy2 + W3*Pxy3 + W4*Pxy4 + W5*Pxy5 + W6*Pxy6
		#STEP 14: Estimate Kalman gain
		KPZ = PXY*inv(SMAT)
		if NOISE == 1:
			XNOISE = GAUSS_PY(SIGX)
		else:
			XNOISE = 0.0
		XMEASU = XT + XNOISE
		RESK = XMEASU - YH[0,0]
		XHAT = XBAR + KPZ*RESK
		P = M - KPZ*SMAT*KPZ.transpose()
		WH = XHAT[2,0]
		if (WH<0):
			WH=abs(WH);
		ERRW=W-WH
		SP33=math.sqrt(P[2,2]);
		ArrayT.append(T);
		ArrayW.append(W);
		ArrayWH.append(WH)
		ArrayERRW.append(ERRW)
		ArraySP33.append(math.sqrt(P[2,2]))
		ArraySP33P.append(-math.sqrt(P[2,2]))

plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayW,label='x-hat', linewidth=0.6)
plt.plot(ArrayT,ArrayWH,label='sp11', linewidth=0.6)
#plt.plot(t,sp11n,label='sp11n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.legend()

plt.figure(2)
plt.grid(True)
plt.plot(ArrayT,ArrayERRW,label='Error W', linewidth=0.6)
plt.plot(ArrayT,ArraySP33,label='sp11', linewidth=0.6)
plt.plot(ArrayT,ArraySP33P,label='sp11', linewidth=0.6)
#plt.plot(t,sp11n,label='sp11n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.legend()


plt.show()
