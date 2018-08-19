import numpy as np
#import numpy.linalg.cholesky as chsky
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def PROJECTC19L4A(TS,XTP,XTDP,BETAINVP,HP):
	T=0.;
	XT=XTP;
	XTD=XTDP;
	BETAINV=BETAINVP;
	H=HP;
	while(T<=(TS-.0001)):
		XTDD=.5*.0034*32.2*math.exp(-XT/22000.)*XTD*XTD*BETAINV-32.2;
		XTD=XTD+H*XTDD;
		XT=XT+H*XTD;
		T=T+H;
	XTH=XT;
	XTDH=XTD;
	BETAINVH = BETAINV;
	return [XTH, XTDH, BETAINVH]

def PROJECTC19L4B(TS,XTP,XTDP,BETAP,HP):
	T=0.;
	XT=XTP;
	XTD=XTDP;
	BETA=BETAP;
	H=HP;
	while(T<=(TS-.0001)):
		XTDD=.5*.0034*32.2*math.exp(-XT/22000.)*XTD*XTD/BETA-32.2;
		XTD=XTD+H*XTDD;
		XT=XT+H*XTD;
		T=T+H;
	XTH=XT;
	XTDH=XTD;
	BETAH = BETA;
	return [XTH, XTDH, BETAH]


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
	
RK = 0# Set to 1 uses the Runge-Kutta; set to 0 uses Euler

count=0;
QINVERSE=1;
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
# Initial covariance matrix
if QINVERSE==1:
	P=np.matrix([[SIGX**2.0,0,0],[0,150.**2,0],[0,0,(1./BETA-1./BETAH)**2]])
	PHIS=1.e-7;
else:
	P=np.matrix([[SIGX**2.0,0,0],[0,150.**2,0],[0,0,(BETA - BETAH)**2]])
	PHIS=1000;
	
# Measurement matrix
HMAT=np.matrix([[1,0,0]])

while(T<=TF):
	XTOLD=XT;
	XTDOLD=XTD;
	XTDD=.0034*32.2*XTD*XTD*math.exp(-XT/22000.)/(2.*BETA)-32.2;
	XT=XT+H*XTD;
	XTD=XTD+H*XTDD;
	T=T+H;
	XTDD=.0034*32.2*XTD*XTD*math.exp(-XT/22000.)/(2.*BETA)-32.2;
	XT=.5*(XTOLD+XT+H*XTD);
	XTD=.5*(XTDOLD+XTD+H*XTDD);
	SP=SP+H;
	if(SP>=(TS-.00001)):
		SP=0.;
# Process noise matrix
		Q=np.matrix([[0,0,0],[0,0,0],[0,0,PHIS*TS]])
# State estimate vector (STEP 1)
		if QINVERSE==1:
			XHAT=np.matrix([[XTH],[XTDH],[BETAINVH]]);
		else:
			XHAT=np.matrix([[XTH],[XTDH],[BETAH]]);
		C = np.linalg.cholesky(((ORDER + K)*P))
		C1 = np.matrix([[C[0,0]],[C[1,0]],[C[2,0]]]);
		C2 = np.matrix([[C[0,1]],[C[1,1]],[C[2,1]]]);
		C3 = np.matrix([[C[0,2]],[C[1,2]],[C[2,2]]]);
# Calculate sigma points
		XS0 = XHAT;
		XS1 = XHAT + C1;
		XS2 = XHAT + C2; 
		XS3 = XHAT + C3;
		XS4 = XHAT - C1;
		XS5 = XHAT - C2;
		XS6 = XHAT - C3;
# Calculate weights (STEP 2)
		W0 = K/(XORDER+K);
		W1 = 1/(2*(XORDER+K));
		W2 = 1/(2*(XORDER+K));
		W3 = 1/(2*(XORDER+K));
		W4 = 1/(2*(XORDER+K));
		W5 = 1/(2*(XORDER+K));
		W6 = 1/(2*(XORDER+K));
# Use numerical integration to propagate each sigma point ahead TS seconds (STEP 3)
		if QINVERSE==1:
			[XTI0, XTDI0, BETAINVI0] = PROJECTC19L4A (TS, XS0[0,0], XS0[1,0], XS0[2,0], HP);
			XI0 = np.matrix([[XTI0],[XTDI0],[BETAINVI0]])

			[XTI1, XTDI1, BETAINVI1] = PROJECTC19L4A (TS, XS1[0,0], XS1[1,0], XS1[2,0], HP);
			XI1 = np.matrix([[XTI1],[XTDI1],[BETAINVI1]])

			[XTI2, XTDI2, BETAINVI2] = PROJECTC19L4A (TS, XS2[0,0], XS2[1,0], XS2[2,0], HP);
			XI2 = np.matrix([[XTI2],[XTDI2],[BETAINVI2]])

			[XTI3, XTDI3, BETAINVI3] = PROJECTC19L4A (TS, XS3[0,0], XS3[1,0], XS3[2,0], HP);
			XI3 = np.matrix([[XTI3],[XTDI3],[BETAINVI3]])

			[XTI4, XTDI4, BETAINVI4] = PROJECTC19L4A (TS, XS4[0,0], XS4[1,0], XS4[2,0], HP);
			XI4 = np.matrix([[XTI4],[XTDI4],[BETAINVI4]])

			[XTI5, XTDI5, BETAINVI5] = PROJECTC19L4A (TS, XS5[0,0], XS5[1,0], XS5[2,0], HP);
			XI5 = np.matrix([[XTI5],[XTDI5],[BETAINVI5]])
			
			[XTI6, XTDI6, BETAINVI6] = PROJECTC19L4A (TS,XS6[0,0], XS6[1,0], XS6[2,0], HP);
			XI6 = np.matrix([[XTI6],[XTDI6],[BETAINVI6]])
		else:
			[XTI0, XTDI0, BETAI0] = PROJECTC19L4B (TS, XS0[0,0], XS0[1,0], XS0[2,0], HP);
			XI0 = np.matrix([[XTI0],[XTDI0],[BETAI0]])

			[XTI1, XTDI1, BETAI1] = PROJECTC19L4B (TS, XS1[0,0], XS1[1,0], XS1[2,0], HP);
			XI1 = np.matrix([[XTI1],[XTDI1],[BETAI1]])

			[XTI2, XTDI2, BETAI2] = PROJECTC19L4B (TS, XS2[0,0], XS2[1,0], XS2[2,0], HP);
			XI2 = np.matrix([[XTI2],[XTDI2],[BETAI2]])

			[XTI3, XTDI3, BETAI3] = PROJECTC19L4B (TS, XS3[0,0], XS3[1,0], XS3[2,0], HP);
			XI3 = np.matrix([[XTI3],[XTDI3],[BETAI3]])

			[XTI4, XTDI4, BETAI4] = PROJECTC19L4B (TS, XS4[0,0], XS4[1,0], XS4[2,0], HP);
			XI4 = np.matrix([[XTI4],[XTDI4],[BETAI4]])

			[XTI5, XTDI5, BETAI5] = PROJECTC19L4B (TS, XS5[0,0], XS5[1,0], XS5[2,0], HP);
			XI5 = np.matrix([[XTI5],[XTDI5],[BETAI5]])
			
			[XTI6, XTDI6, BETAI6] = PROJECTC19L4B (TS,XS6[0,0], XS6[1,0], XS6[2,0], HP);
			XI6 = np.matrix([[XTI6],[XTDI6],[BETAI6]])
# Find weighted average of all propagated states (STEP 4)
		XBAR = W0*XI0 + W1*XI1 + W2*XI2 + W3*XI3 + W4*XI4 + W5*XI5 + W6*XI6;
# Find components of propagated state vector
		XTB=XBAR[0,0];
		XTDB=XBAR[1,0];
		if QINVERSE==1:
			BETAINVB=XBAR[2,0];
		else:
			BETAB=XBAR[2,0];
# Find covariance matrix for each sigma point (STEP 5)
		M0 = (XI0 - XBAR)*(XI0 - XBAR).transpose();
		M1 = (XI1 - XBAR)*(XI1 - XBAR).transpose();
		M2 = (XI2 - XBAR)*(XI2 - XBAR).transpose();
		M3 = (XI3 - XBAR)*(XI3 - XBAR).transpose();
		M4 = (XI4 - XBAR)*(XI4 - XBAR).transpose();
		M5 = (XI5 - XBAR)*(XI5 - XBAR).transpose();
		M6 = (XI6 - XBAR)*(XI6 - XBAR).transpose();
# Find weighted average of covariance matrices and add process noise (STEP 6)
		M = W0*M0 + W1*M1 + W2*M2 + W3*M3 + W4*M4 + W5*M5 + W6*M6   + Q  ;
# Obtain new sigma points (STEP 7)
		C = np.linalg.cholesky(((ORDER + K)*M)) 
		C1 = np.matrix([[C[0,0]],[C[1,0]],[C[2,0]]]);
		C2 = np.matrix([[C[0,1]],[C[1,1]],[C[2,1]]]);
		C3 = np.matrix([[C[0,2]],[C[1,2]],[C[2,2]]]);
		XS0 = XBAR;
		XS1 = XBAR + C1;
		XS2 = XBAR + C2; 
		XS3 = XBAR + C3;
		XS4 = XBAR - C1;
		XS5 = XBAR - C2;
		XS6 = XBAR - C3;
# Map new sigma points via linear measurement matrix (STEP 8a)
		YS0 = HMAT*XS0;
		YS1 = HMAT*XS1;
		YS2 = HMAT*XS2;
		YS3 = HMAT*XS3;
		YS4 = HMAT*XS4;
		YS5 = HMAT*XS5;
		YS6 = HMAT*XS6;
# Form weighted average (STEP 9)
		YH = W0*YS0 + W1*YS1 + W2*YS2 + W3*YS3 + W4*YS4 + W5*YS5 + W6*YS6;
# Calculate residual covariance (STEP 10)
		S0 = (YS0 - YH)*(YS0 - YH).transpose();
		S1 = (YS1 - YH)*(YS1 - YH).transpose();
		S2 = (YS2 - YH)*(YS2 - YH).transpose();
		S3 = (YS3 - YH)*(YS3 - YH).transpose();
		S4 = (YS4 - YH)*(YS4 - YH).transpose();
		S5 = (YS5 - YH)*(YS5 - YH).transpose();
		S6 = (YS6 - YH)*(YS6 - YH).transpose();
# Form weighted average and add measurement noise matrix(STEP 11)
		SMAT=W0*S0+W1*S1+W2*S2+W3*S3+W4*S4+W5*S5+W6*S6+RMAT
# Calculate cross covariance between propagated state estimates and measurement (STEP 12)
		Pxy0 = (XS0 - XBAR)*(YS0 - YH);
		Pxy1 = (XS1 - XBAR)*(YS1 - YH);
		Pxy2 = (XS2 - XBAR)*(YS2 - YH);
		Pxy3 = (XS3 - XBAR)*(YS3 - YH);
		Pxy4 = (XS4 - XBAR)*(YS4 - YH);
		Pxy5 = (XS5 - XBAR)*(YS5 - YH);
		Pxy6 = (XS6 - XBAR)*(YS6 - YH);
# Form weighted average (STEP 13)
		Pxy = W0*Pxy0 + W1*Pxy1 + W2*Pxy2 + W3*Pxy3 + W4*Pxy4 + W5*Pxy5 + W6*Pxy6;
# Kalman filter (STEP 14)
		KPZ = Pxy*inv(SMAT) #compute Kalman gain
		P = M - KPZ*SMAT*KPZ.transpose()  #update covariance with Kalman gain
		XNOISE = GAUSS_PY(SIGX)
		XTMEAS=XT+XNOISE
		RESX=XTMEAS-YH[0,0];
		XTH=XTB+KPZ[0,0]*RESX;
		XTDH=XTDB+KPZ[1,0]*RESX;
		if QINVERSE==1:
			BETAINVH=BETAINVB+KPZ[2,0]*RESX;
			if BETAINVH>.01:
				BETAINVH=.01;
			if BETAINVH<.0001:
				BETAINVH=.0001;
			BETAH=1./BETAINVH;
		else:
			BETAH=BETAB+KPZ[2,0]*RESX;
			if BETAH<100:
				BETAH=100
			if BETAH>10000:
				BETAH=10000
# Compute things to plot
		count = count + 1;
		ERRX = XT - XTH
		ERRXD = XTD - XTDH
		ERRBETA=BETA-BETAH
		SP33=math.sqrt(P[2,2])
		SP22=math.sqrt(P[1,1])
		SP11=math.sqrt(P[0,0])
		ArrayT.append(T);
		ArrayX.append(XT);
		ArrayXH.append(XTH);
		ArrayERRX.append(ERRX)
		ArrayXD.append(XTD);
		ArrayXDH.append(XTDH);
		ArrayERRXD.append(ERRXD)
		#ArrayT.append(T);
		ArrayBETA.append(BETA);
		ArrayBETAH.append(BETAH)
		ArrayERRBETA.append(ERRBETA)
		ArraySP33.append(SP33)
		ArraySP33P.append(-SP33)


plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayBETA,label='BETA('+str(BETA)+')', linewidth=0.6)
plt.plot(ArrayT,ArrayBETAH,label='sp11', linewidth=0.6)
#plt.plot(t,sp11n,label='sp11n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('BETA')
plt.legend()
plt.show()
