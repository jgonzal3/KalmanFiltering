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

def g1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)

def g2(x1,x2,t):
	dx2dt = -w*w*x1
	return (dx2dt)
	
def f1(x1,x2,BETA,t):
	dx1dt = x2
	return (dx1dt)
	
def f2(x1,x2,BETA,t):
	G = 32.2
	dx2dt = (0.0034*G*math.exp(-x1/20000)*x2*x2/(2*BETA)) - G
	return (dx2dt)
	
def rk4(TS,XP, HP):
	X1  =XP[0,0]
	X1D =XP[1,0]
	W   =XP[2,0]
	T = 0.0
	dt = HP
	X= []
	XD =[]
	TD = []
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
		XS_LIST = [XS0,XS1,XS2,XS3,XS4,XS5,XS6]
		#Step 2: Build the weighting factors
		W0 = KAPPA/(N+KAPPA)
		W_LIST = [W0] + [1/(2*(N + KAPPA)) for i in range(0,2*N)]
		#Step 3: Propagate the sigma points one time interval
		XSB_LIST = [rk4(TS, xs, HP) for xs in XS_LIST]
		#Step 4: Calculate the weighing average
		XBAR = sum([w*xb for w,xb in zip(W_LIST, XSB_LIST)])
		#STEP 5: Estimate the covariance matrix for the sigma points
		MK = [(XP - XBAR)*(XP-XBAR).transpose() for XP in XSB_LIST]
		#STEP6: build the new weighted matrix M and add process noise matrix QMAT
		M = sum([w*m for w, m in zip(W_LIST, MK)]) + QMAT
		#STEP 7: Estimate a new sigma points using the recent calculated M matrix
		CM = np.linalg.cholesky((N+KAPPA)*M)
		XS0 = XBAR
		XS1 = XBAR + CM[:,0]
		XS2 = XBAR + CM[:,1]
		XS3 = XBAR + CM[:,2]
		XS4 = XBAR - CM[:,0]
		XS5 = XBAR - CM[:,1]
		XS6 = XBAR - CM[:,2]
		XS_LIST_NEW = [XS0,XS1,XS2,XS3,XS4,XS5,XS6]
		#STEP 8: Estimate the measurements using the H matrix
		YS_LIST = [HMAT*XS for XS in XS_LIST_NEW]
		#STEP 9: Estimate the weighted measurements		
		YH = sum([W*Y for W,Y in zip(W_LIST, YS_LIST)])
		#STEP 10: Estimate new variance
		S_LIST = [(YS - YH)*(YS - YH).transpose() for YS in YS_LIST]
		#STEP 11: Estimate the weighted S Matrix		
		SMAT = sum([W*S for W, S in zip(W_LIST, S_LIST)]) + RMAT
		#STEP 12 and STEP 13: Estimate the cross covariance between x and y
		PXY_LIST = [(XS - XBAR)*(YS - YH) for XS,YS in zip(XS_LIST_NEW, YS_LIST)]
		PXY = sum([W*PXY for W,PXY in zip(W_LIST, PXY_LIST)])	
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
plt.ylabel('Error in Frequency')
plt.legend()


plt.show()
