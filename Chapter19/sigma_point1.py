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
N = 3
KAPPA = 0.0
H=0.001
T=0.0
QCONSTANT = 1
NOISE = 1

RK = 0# Set to 1 uses the Runge-Kutta; set to 0 uses Euler

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
	#Step 3: Propagate the sigma points one time interval
	if RK == 1:
		XS_LIST = [XS0,XS1,XS2,XS3,XS4,XS5,XS6]
		W_LIST = [W0] + [1/(2*(N + KAPPA)) for i in range(0,2*N-1)]
		XSB_LIST = [rk4(TS, xs, HP) for xs in XS_LIST]
		#Step 4: Calculate the weighing average
		XBAR = sum([w*xb for w,xb in zip(W_LIST, XSB_LIST)])
		M = [(XP - XBAR)*(XP-XBAR).transpose() for XP in XSB_LIST]
		MK = sum([w*m for w, m in zip(W_LIST, M)]) + QMAT
	else:
		XSB0 = PROJECTC19L1(TS, XS0, HP);
		XSB1 = PROJECTC19L1(TS, XS1, HP);
		XSB2 = PROJECTC19L1(TS, XS2, HP);
		XSB3 = PROJECTC19L1(TS, XS3, HP);
		XSB4 = PROJECTC19L1(TS, XS4, HP);
		XSB5 = PROJECTC19L1(TS, XS5, HP);
		XSB6 = PROJECTC19L1(TS, XS6, HP);
		W1 = 1/(2*(N + KAPPA))
		W2 = 1/(2*(N + KAPPA))
		W3 = 1/(2*(N + KAPPA))
		W4 = 1/(2*(N + KAPPA))
		W5 = 1/(2*(N + KAPPA))
		W6 = 1/(2*(N + KAPPA))
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
		MK = (W0*M0 + W1*M1 + W2*M2 + W3*M3 + W4*M4 + W5*M5 + W6*M6) + QMAT
	#STEP 7: Estimate a new sigma points using the recent calculated M matrix
	CM = np.linalg.cholesky((N+KAPPA)*MK)
	XS0 = XBAR
	XS1 = XBAR + CM[:,0]
	XS2 = XBAR + CM[:,1]
	XS3 = XBAR + CM[:,2]
	XS4 = XBAR - CM[:,0]
	XS5 = XBAR - CM[:,1]
	XS6 = XBAR - CM[:,2]
	if RK == 1:
		XS_LIST_NEW = [XS0,XS1,XS2,XS3,XS4,XS5,XS6]
		#STEP 8: Estimate the measurements using the H matrix
		Y_LIST = [H*XS for XS in XS_LIST_NEW]
		#STEP 9: Estimate the weighted measurements		
		YH = sum([W*Y for W,Y in zip(W_LIST, Y_LIST)])
		#STEP10: Estimate new variance
		SMAT = sum([W*((Y - YH)*(Y - YH).transpose()) for W,Y in zip(W_LIST,Y_LIST)]) + RMAT
	else:
		#STEP 8: Estimate the measurements using the H matrix
		Y0 = H*XS0
		Y1 = H*XS1
		Y2 = H*XS2
		Y3 = H*XS3
		Y4 = H*XS4
		Y5 = H*XS5 
		Y6 = H*XS6
		#STEP 9: Estimate the weighted measurements		
		YH = (W0*Y0 + W1*Y1 + W2*Y2 + W3*Y3 + W4*Y4 + W5*Y5 + W6*Y6)
		#STEP 10: Estimate new variance
		S0 = (Y0 - YH)*(Y0 - YH).transpose()
		S1 = (Y1 - YH)*(Y1 - YH).transpose()
		S2 = (Y2 - YH)*(Y2 - YH).transpose()
		S3 = (Y3 - YH)*(Y3 - YH).transpose()
		S4 = (Y4 - YH)*(Y4 - YH).transpose()
		S5 = (Y5 - YH)*(Y5 - YH).transpose()
		S6 = (Y6 - YH)*(Y6 - YH).transpose()
		SMAT = (W0*S0 + W1*S1 + W2*S2 + W3*S3 + W4*S4 + W5*S5 + W6*S6 ) + RMAT
	if (RK == 1):
		#STEP 12 and STEP 13: Estimate the cross covariance between x and y
		PXY_LIST = [(X - XBAR)*(Y - YH).transpose() for X,Y in zip(XSB_LIST, Y_LIST)]
		PXY = sum([W*PXY for W,PXY in zip(W_LIST, PXY_LIST)])
	else:
		#STEP 12 and STEP 13: Estimate the cross covariance between x and y
		Pxy0 = (XSB0 - XBAR)*(Y0 - YH).transpose()
		Pxy1 = (XSB1 - XBAR)*(Y1 - YH).transpose()
		Pxy2 = (XSB2 - XBAR)*(Y2 - YH).transpose()
		Pxy3 = (XSB3 - XBAR)*(Y3 - YH).transpose()
		Pxy4 = (XSB4 - XBAR)*(Y4 - YH).transpose()
		Pxy5 = (XSB5 - XBAR)*(Y5 - YH).transpose()
		Pxy6 = (XSB6 - XBAR)*(Y6 - YH).transpose()
		PXY = W0*Pxy0 + W1*Pxy1 + W2*Pxy2 + W3*Pxy3 + W4*Pxy4 + W5*Pxy5 + W6*Pxy6
	#STEP 14: Estimate Kalman gain
	KZ = PXY*inv(SMAT)
	if NOISE == 1:
		XNOISE = GAUSS_PY(SIGX)
	else:
		XNOISE = 0.0
	XMEASU = XT + XNOISE
	RESK = XMEASU - YH[0,0]
	XHAT = XBAR + KZ*RESK
	P = MK - KZ*SMAT*KZ.transpose()
	T=T+H
	ERRW=W-abs(WH);
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
