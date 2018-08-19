import numpy as np
#import numpy.linalg.cholesky as chsky
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def PROJECTC19L1(TS,XP,HP):
	T=0.0;
	XT =XP[0,0]
	XTD=XP[1,0]
	BETA =XP[2,0]
	H=HP;
	while T<=(TS-.0001):
		XTDD=.5*.0034*32.2*math.exp(-XT/22000.)*XTD*XTD/BETA-32.2;
		XTD=XTD+H*XTDD;
		XT=XT+H*XTD;
		T=T+H;
	XTH=XT;
	XTDH=XTD;
	BETAH = BETA;
	XB = np.matrix([[XTH],[XTDH],[BETAH]])
	return (XB)

def PROJECTC19L2(TS,XP,HP):
	T=0.0;
	XT =XP[0,0]
	XTD=XP[1,0]
	BETA =XP[2,0]
	H=HP;
	while T<=(TS-.0001):
		XTDD=.5*.0034*32.2*math.exp(-XT/22000.)*XTD*XTD*BETA-32.2;
		XTD=XTD+H*XTDD;
		XT=XT+H*XTD;
		T=T+H;
	XTH=XT;
	XTDH=XTD;
	BETAH = BETA;
	XB = np.matrix([[XTH],[XTDH],[BETAH]])
	return (XB)
	
def f1(x1,x2,w,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,BETA,t):
	dx2dt = .5*.0034*32.2*math.exp(-XT/22000.)*XTD*XTD/BETA-32.2
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

def g1(x1,x2,w,t):
	dx1dt = x2
	return (dx1dt)

def g2(x1,x2,BETA,t):
	dx2dt = .5*.0034*32.2*math.exp(-XT/22000.)*XTD*XTD*BETA-32.2
	return (dx2dt)
	
def rk4_INV(TS,XP, HP):
	X1  =XP[0,0]
	X1D =XP[1,0]
	W   =XP[2,0]
	T = 0.0
	dt = HP
	X= []
	XD =[]
	TD = []
	while (T<=(TS-.0001)):
		k11 = dt*g1(X1,X1D,W,T)
		k21 = dt*g2(X1,X1D,W,T)
		k12 = dt*g1(X1+0.5*k11,X1D+0.5*k21,W,T+0.5*dt)
		k22 = dt*g2(X1+0.5*k11,X1D+0.5*k21,W,T+0.5*dt)
		k13 = dt*g1(X1+0.5*k12,X1D+0.5*k22,W,T+0.5*dt)
		k23 = dt*g2(X1+0.5*k12,X1D+0.5*k22,W,T+0.5*dt)
		k14 = dt*g1(X1+k13,X1D+k23,W,T+dt)
		k24 = dt*g2(X1+k13,X1D+k23,W,T+dt)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X1D = X1D + (k21+2*k22+2*k23+k24)/6
		T = T+dt
	X1a = X1
	X2a = X1D
	XB = np.matrix([[X1a], [X2a], [W]])
	return (XB)
		
ArrayT = []
ArrayT = []
ArrayX = []
ArrayXH = []
ArrayX = []
ArrayXDH = []
ArrayXD = []
ArrayBETA = []
ArrayBETAH = []
ArrayERRX = []
ArrayERRXD = []
ArrayERRBETA = []
ArraySP33 = []
ArraySP33P = []
	
count=0;
QINVERSE=0;
ORDER=3;
XORDER=3.;
K=-1.;
K=0.;
XORDERP=XORDER+K;
G=32.2;
HP=.001;
# Initial state values
BETA=500.;
XT=150000.;
XTD=-6000.;
# Initial state estimates
XTH=150025;
XTDH=-6150.;
BETAH=1000.;
BETAINVH=1./BETAH;
BETAINV=1./BETA;
TS=.1
SIGX=25;
TF = 30;
T=0.;
SP=0.;
H=.001;
BETA=1800*T/30+500;
BETAINV=1./BETA;
RMAT=np.matrix([[SIGX*SIGX]])

P=np.matrix([[25.**2.0,0,0],[0,150.**2,0],[0,0,(BETA - BETAH)**2]])
RMAT=np.matrix([[SIGX*SIGX]])
if QINVERSE==1:
	P=np.matrix([[SIGX**2.0,0,0],[0,150.**2,0],[0,0,(1./BETA-1./BETAH)**2]])
	XHAT = np.matrix([[XTH], [XTDH], [BETAINVH]])
	PHIS=1.e-7;
else:
	P=np.matrix([[SIGX**2.0,0,0],[0,150.**2,0],[0,0,(BETA - BETAH)**2]])
	XHAT = np.matrix([[XTH], [XTDH], [BETAH]])
	PHIS=1000;
	
QMAT=np.matrix([[0,0,0],[0,0,0],[0,0,PHIS*TS]])
HMAT=np.matrix([[1,0,0]])

NOISE = 1
KAPPA = 0
RK = 1# Set to 1 uses the Runge-Kutta; set to 0 uses Euler

while (T < 30.0):
	XOLD=XT;
	XDOLD=XTD;
	XDD=.0034*32.2*XTD*XTD*math.exp(-XT/22000.)/(2.*BETA)-32.2;
	XT=XT+H*XTD;
	XTD=XTD+H*XDD;
	T=T+H;
	XDD=.0034*32.2*XTD*XTD*math.exp(-XT/22000.)/(2.*BETA)-32.2;
	XT=.5*(XOLD+XT+H*XTD);
	XTD=.5*(XDOLD+XTD+H*XDD);
	SP = SP+H
	if SP>=(TS-.00001):
		SP=0.;
		try:
			assert(np.linalg.det(P) > 0)
		except:
			print ('P is not positive definite')
			exit(0)
		C = np.linalg.cholesky(((ORDER + K)*P))
		#STEP1: State estimate vector
		XS0 = XHAT
		XS1 = XHAT + C[:,0]
		XS2 = XHAT + C[:,1]
		XS3 = XHAT + C[:,2]
		XS4 = XHAT - C[:,0]
		XS5 = XHAT - C[:,1]
		XS6 = XHAT - C[:,2]
		#Step 2: Build the weighting factors
		W0 = KAPPA/(ORDER+KAPPA)
		W1 = 1/(2*(ORDER + KAPPA))
		W2 = 1/(2*(ORDER + KAPPA))
		W3 = 1/(2*(ORDER + KAPPA))
		W4 = 1/(2*(ORDER + KAPPA))
		W5 = 1/(2*(ORDER + KAPPA))
		W6 = 1/(2*(ORDER + KAPPA))
		W_LIST = [W0] + [1/(2*(ORDER + KAPPA)) for i in range(0,2*ORDER)]
		#Step 3: Propagate the sigma points one time interval
		if (RK == 0):
			if QINVERSE == 1:
				XSB0 = PROJECTC19L2(TS, XS0, HP);
				XSB1 = PROJECTC19L2(TS, XS1, HP);
				XSB2 = PROJECTC19L2(TS, XS2, HP);
				XSB3 = PROJECTC19L2(TS, XS3, HP);
				XSB4 = PROJECTC19L2(TS, XS4, HP);
				XSB5 = PROJECTC19L2(TS, XS5, HP);
				XSB6 = PROJECTC19L2(TS, XS6, HP);
			else:
				XSB0 = PROJECTC19L1(TS, XS0, HP);
				XSB1 = PROJECTC19L1(TS, XS1, HP);
				XSB2 = PROJECTC19L1(TS, XS2, HP);
				XSB3 = PROJECTC19L1(TS, XS3, HP);
				XSB4 = PROJECTC19L1(TS, XS4, HP);
				XSB5 = PROJECTC19L1(TS, XS5, HP);
				XSB6 = PROJECTC19L1(TS, XS6, HP);
			
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
			M = (W0*M0 + W1*M1 + W2*M2 + W3*M3 + W4*M4 + W5*M5 + W6*M6) + QMAT
		else:
			XS_LIST = [XS0,XS1,XS2,XS3,XS4,XS5,XS6]
			if QINVERSE == 1:
				XSB_LIST = [rk4_INV(TS, xs, HP) for xs in XS_LIST]
			else:
				XSB_LIST = [rk4(TS, xs, HP) for xs in XS_LIST]			
			#Step 4: Calculate the weighing average
			XBAR = sum([w*xb for w,xb in zip(W_LIST, XSB_LIST)])
			#STEP 5: Estimate the covariance matrix for the sigma points
			MK = [(XP - XBAR)*(XP-XBAR).transpose() for XP in XSB_LIST]
			#STEP6: build the new weighted matrix M and add process noise matrix QMAT
			M = sum([w*m for w, m in zip(W_LIST, MK)]) + QMAT
		#STEP 7: Estimate a new sigma points using the recent calculated M matrix
		try:
			assert(np.linalg.det(P) > 0)
		except:
			print ('M is not positive definite')
			exit(0)
		C = np.linalg.cholesky(((ORDER + K)*M))
		XS0 = XBAR
		XS1 = XBAR + C[:,0]
		XS2 = XBAR + C[:,1]
		XS3 = XBAR + C[:,2]
		XS4 = XBAR - C[:,0]
		XS5 = XBAR - C[:,1]
		XS6 = XBAR - C[:,2]
		if (RK == 0):
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
		else:
			#STEP 8: Estimate the measurements using the H matrix
			XS_LIST_NEW = [XS0,XS1,XS2,XS3,XS4,XS5,XS6]
			#STEP 8: Estimate the measurements using the H matrix
			YS_LIST = [HMAT*XS for XS in XS_LIST_NEW]
			#STEP 9: Estimate the weighted measurements		
			YH = sum([W*Y for W,Y in zip(W_LIST, YS_LIST)])
			#STEP10: Estimate new variance
			S_LIST = [(YS - YH)*(YS - YH).transpose() for YS in YS_LIST]
			SMAT = sum([W*S for W, S in zip(W_LIST, S_LIST)]) + RMAT
			#STEP 12 and STEP 13: Estimate the cross covariance between x and y
			PXY_LIST = [(XS - XBAR)*(YS - YH) for XS,YS in zip(XS_LIST_NEW, YS_LIST)]
			PXY = sum([W*PXY for W,PXY in zip(W_LIST, PXY_LIST)])	
		#STEP 14: Estimate Kalman gain
		KPZ = PXY*inv(SMAT)
		P = M - KPZ*SMAT*KPZ.transpose()
		if NOISE == 1:
			XNOISE = GAUSS_PY(SIGX)
		else:
			XNOISE = 0.0
		XMEASU = XT + XNOISE
		RESK = XMEASU - YH[0,0]
		XHAT = XBAR + KPZ*RESK
		XH = XHAT[0,0]
		XDH = XHAT[1,0]
		if QINVERSE==1:
			BETAINVH=XHAT[2,0]
			if BETAINVH>.01:
				BETAINVH=.01;
			if BETAINVH<.0001:
				BETAINVH=.0001;
			BETAH=1./BETAINVH;
		else:
			BETAH=XHAT[2,0]
			if BETAH<100:
				BETAH=100
			if BETAH>10000:
				BETAH=10000
		ERRX = XT - XH
		ERRXD = XTD - XDH
		ERRBETA=BETA-BETAH
		SP33=math.sqrt(P[2,2])
		SP22=math.sqrt(P[1,1])
		SP11=math.sqrt(P[0,0])
		ArrayT.append(T);
		ArrayX.append(XT);
		ArrayXH.append(XH);
		ArrayERRX.append(ERRX)
		ArrayXD.append(XTD);
		ArrayXDH.append(XDH);
		ArrayERRXD.append(ERRXD)
		#ArrayT.append(T);
		ArrayBETA.append(BETA);
		ArrayBETAH.append(BETAH)
		ArrayERRBETA.append(ERRBETA)
		ArraySP33.append(SP33)
		ArraySP33P.append(-SP33)


plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayBETA,label='BETA('+str(QINVERSE)+')', linewidth=0.6)
plt.plot(ArrayT,ArrayBETAH,label='sp11', linewidth=0.6)
#plt.plot(t,sp11n,label='sp11n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.legend()

plt.show()
