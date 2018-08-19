import numpy as np
#import numpy.linalg.cholesky as chsky
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def PROJECTC19L6(TS,XTP,XTDP,YTP,YTDP,HP):
	T=0.;
	XT=XTP;
	XTD=XTDP;
	YT = YTP;
	YTD = YTDP;
	H=HP;
	while(T<=(TS-.0001)):
		YTDD = -32.2;
		XTDD=0;
		XTD=XTD+H*XTDD;
		YTD=YTD+H*YTDD;
		XT=XT+H*XTD;
		YT=YT+H*YTD;
		T=T+H;
	XTH=XT;
	XTDH=XTD;
	YTH = YT;
	YTDH = YTD;
	return ([XTH, XTDH, YTH, YTDH])

FUDGE=1;
PHIS = 1;
XT=0;
XTD=3000*math.cos(45);
YT = 0;
YTD = 3000*math.sin(45);
XR = 100000;
YR = 0;
TS=1;
ORDER=4;
SIGTH = 0.01;
SIGR=100;
TF = 130;
k  = 3 - ORDER; #parameter in unscented transformation
T=0.;
S=0.;
HP=.1;
H=.001;
XTH = XT + FUDGE*1000;
XTDH = XTD - FUDGE*100;
YTH = YT - FUDGE*1000;
YTDH = YTD + FUDGE*100;
# Initial covariance matrix
count = 0;
PHI=np.matrix([[1,TS, 0, 0],
			   [0, 1, 0, 0],
			   [0, 0, 1,TS],
			   [0, 0, 0, 1]])

P=np.matrix([[(FUDGE*1000)**2,0,0,0],
			 [0, (FUDGE*100)**2,0,0],
			 [0,0,(FUDGE*1000)**2,0],
			 [0,0,0, (FUDGE*100)**2]])
			 
IDNP=np.identity(ORDER)
	 
RMAT=np.matrix([[SIGTH**2, 0],[0, SIGR**2]])

YS0 = np.matrix([[0],[0]])
YS1 = np.matrix([[0],[0]])
YS2 = np.matrix([[0],[0]]) 
YS3 = np.matrix([[0],[0]])
YS4 = np.matrix([[0],[0]])
YS5 = np.matrix([[0],[0]])
YS6 = np.matrix([[0],[0]])
YS7 = np.matrix([[0],[0]])
YS8 = np.matrix([[0],[0]])
	
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
ArrayERRYD = []
ArraySP44 = []
ArraySP44P = []
	
		
while(T<=TF):	
	XTOLD=XT;
	XTDOLD=XTD;
	YTOLD = YT;
	YTDOLD = YTD;
	YTDD = -32.2;
	XTDD = 0;
	XT=XT+H*XTD;
	XTD=XTD+H*XTDD;
	YT=YT+H*YTD;
	YTD=YTD+H*YTDD;
	T=T+H;
	XT=.5*(XTOLD+XT+H*XTD);
	XTD=.5*(XTDOLD+XTD+H*XTDD);
	YT=.5*(YTOLD+YT+H*YTD);
	YTD=.5*(YTDOLD+YTD+H*YTDD);
	S=S+H;
	if(S>=(TS-.00001)):
		S=0;
# Process noise matrix
		TS2=TS*TS
		TS3=TS2*TS
		Q=np.matrix([[PHIS*TS3/3.,PHIS*TS2/2.,0,0],
					[PHIS*TS2/2.,PHIS*TS,0,0],
					[0,0,PHIS*TS3/3.,PHIS*TS2/2.],
					[0,0,PHIS*TS2/2.,PHIS*TS]])
# State estimate vector (STEP 1)
		XHAT = np.matrix([[XTH],[XTDH],[YTH],[YTDH]])

# Calculate sigma points
		C = np.linalg.cholesky((ORDER+k)*P)
		#STEP1: State estimate vector. Because there are 4 states, XT,XTD,YT and YTD, there are 2*4=8 sigma points
		XS0 = XHAT
		XS1 = XHAT + C[:,0]
		XS2 = XHAT + C[:,1]
		XS3 = XHAT + C[:,2]
		XS4 = XHAT + C[:,3]
		XS5 = XHAT - C[:,0]
		XS6 = XHAT - C[:,1]
		XS7 = XHAT - C[:,2]
		XS8 = XHAT - C[:,3]

# Calculate weights (STEP 2)
		W0 = k/(ORDER+k);
		W1 = 1/(2*(ORDER+k));
		W2 = 1/(2*(ORDER+k));
		W3 = 1/(2*(ORDER+k));
		W4 = 1/(2*(ORDER+k));
		W5 = 1/(2*(ORDER+k));
		W6 = 1/(2*(ORDER+k));
		W7 = 1/(2*(ORDER+k));
		W8 = 1/(2*(ORDER+k));
# Use numerical integration to propagate each sigma point ahead TS seconds (STEP 3)
		[XTI0, XTDI0, YTI0, YTDI0] = PROJECTC19L6(TS, XS0[0,0], XS0[1,0], XS0[2,0], XS0[3,0], HP);
		XI0 = np.matrix([[XTI0],[XTDI0],[YTI0],[YTDI0]]);

		[XTI1, XTDI1, YTI1, YTDI1] = PROJECTC19L6 (TS, XS1[0,0], XS1[1,0], XS1[2,0], XS1[3,0],HP);
		XI1 = np.matrix([[XTI1],[XTDI1],[YTI1],[YTDI1]]);

		[XTI2, XTDI2, YTI2, YTDI2] = PROJECTC19L6 (TS, XS2[0,0], XS2[1,0], XS2[2,0], XS2[3,0],HP)
		XI2 = np.matrix([[XTI2],[XTDI2],[YTI2],[YTDI2]]);

		[XTI3, XTDI3, YTI3, YTDI3] = PROJECTC19L6 (TS, XS3[0,0], XS3[1,0], XS3[2,0], XS3[3,0],HP)
		XI3 = np.matrix([[XTI3],[XTDI3],[YTI3],[YTDI3]]);

		[XTI4, XTDI4, YTI4, YTDI4] = PROJECTC19L6 (TS,XS4[0,0], XS4[1,0], XS4[2,0], XS4[3,0],HP)
		XI4 = np.matrix([[XTI4],[XTDI4],[YTI4],[YTDI4]]);

		[XTI5, XTDI5, YTI5, YTDI5] = PROJECTC19L6 (TS, XS5[0,0], XS5[1,0], XS5[2,0], XS5[3,0],HP)
		XI5 = np.matrix([[XTI5],[XTDI5],[YTI5],[YTDI5]]);

		[XTI6, XTDI6, YTI6, YTDI6] = PROJECTC19L6 (TS, XS6[0,0], XS6[1,0], XS6[2,0], XS6[3,0],HP)
		XI6 = np.matrix([[XTI6],[XTDI6],[YTI6],[YTDI6]]);

		[XTI7, XTDI7, YTI7, YTDI7] = PROJECTC19L6 (TS, XS7[0,0], XS7[1,0], XS7[2,0], XS7[3,0],HP)
		XI7 = np.matrix([[XTI7],[XTDI7],[YTI7],[YTDI7]]);

		[XTI8, XTDI8, YTI8, YTDI8] = PROJECTC19L6 (TS, XS8[0,0], XS8[1,0], XS8[2,0], XS8[3,0],HP)
		XI8 = np.matrix([[XTI8],[XTDI8],[YTI8],[YTDI8]]);
		
# Find weighted average of all propagated states (STEP 4)
		XBAR = W0*XI0 + W1*XI1 + W2*XI2 + W3*XI3 + W4*XI4 + W5*XI5 + W6*XI6 + W7*XI7 + W8*XI8;
# Find components of propagated state vector
		XTB = XBAR[0,0];
		XTDB = XBAR[1,0];
		YTB = XBAR[2,0];
		YTDB = XBAR[3,0];

# Find covariance matrix for each sigma point (STEP 5)        
		M0 = (XI0 - XBAR)*(XI0 - XBAR).transpose();
		M1 = (XI1 - XBAR)*(XI1 - XBAR).transpose()
		M2 = (XI2 - XBAR)*(XI2 - XBAR).transpose()
		M3 = (XI3 - XBAR)*(XI3 - XBAR).transpose()
		M4 = (XI4 - XBAR)*(XI4 - XBAR).transpose()
		M5 = (XI5 - XBAR)*(XI5 - XBAR).transpose()
		M6 = (XI6 - XBAR)*(XI6 - XBAR).transpose()
		M7 = (XI7 - XBAR)*(XI7 - XBAR).transpose()
		M8 = (XI8 - XBAR)*(XI8 - XBAR).transpose()
# Find weighted average of covariance matrices and add process noise (STEP 6)
		M = W0*M0 + W1*M1 + W2*M2 + W3*M3 + W4*M4 + W5*M5 + W6*M6 + W7*M7 + W8*M8  + Q  ;
# Obtain new sigma points (STEP 7)
		C = np.linalg.cholesky((ORDER+k)*M)
		XS0 = XBAR
		XS1 = XBAR + C[:,0]
		XS2 = XBAR + C[:,1]
		XS3 = XBAR + C[:,2]
		XS4 = XBAR + C[:,3]
		XS5 = XBAR - C[:,0]
		XS6 = XBAR - C[:,1]
		XS7 = XBAR - C[:,2]
		XS8 = XBAR - C[:,3]
# Transform sigma points via non-linear measurement equations (STEP 8b)
		YS0 = np.matrix([[math.atan2(XS0[2,0] - YR, XS0[0,0] - XR)],[math.sqrt( (XS0[0,0] - XR)**2 + (XS0[2,0] - YR)**2 )]])
		YS1 = np.matrix([[math.atan2(XS1[2,0] - YR, XS1[0,0] - XR)],[math.sqrt( (XS1[0,0] - XR)**2 + (XS1[2,0] - YR)**2 )]])
		YS2 = np.matrix([[math.atan2(XS2[2,0] - YR, XS2[0,0] - XR)],[math.sqrt( (XS2[0,0] - XR)**2 + (XS2[2,0] - YR)**2 )]])
		YS3 = np.matrix([[math.atan2(XS3[2,0] - YR, XS3[0,0] - XR)],[math.sqrt( (XS3[0,0] - XR)**2 + (XS3[2,0] - YR)**2 )]])
		YS4 = np.matrix([[math.atan2(XS4[2,0] - YR, XS4[0,0] - XR)],[math.sqrt( (XS4[0,0] - XR)**2 + (XS4[2,0] - YR)**2 )]])
		YS5 = np.matrix([[math.atan2(XS5[2,0] - YR, XS5[0,0] - XR)],[math.sqrt( (XS5[0,0] - XR)**2 + (XS5[2,0] - YR)**2 )]])
		YS6 = np.matrix([[math.atan2(XS6[2,0] - YR, XS6[0,0] - XR)],[math.sqrt( (XS6[0,0] - XR)**2 + (XS6[2,0] - YR)**2 )]])
		YS7 = np.matrix([[math.atan2(XS7[2,0] - YR, XS7[0,0] - XR)],[math.sqrt( (XS7[0,0] - XR)**2 + (XS7[2,0] - YR)**2 )]])
		YS8 = np.matrix([[math.atan2(XS8[2,0] - YR, XS8[0,0] - XR)],[math.sqrt( (XS8[0,0] - XR)**2 + (XS8[2,0] - YR)**2 )]])
			
# Form weighted average (STEP 9)
		YH = W0*YS0 + W1*YS1 + W2*YS2 + W3*YS3 + W4*YS4 + W5*YS5 + W6*YS6 + W7*YS7 + W8*YS8;
# Calculate residual covariance (STEP 10)
		S0 = (YS0 - YH)*(YS0 - YH).transpose()
		S1 = (YS1 - YH)*(YS1 - YH).transpose()
		S2 = (YS2 - YH)*(YS2 - YH).transpose()
		S3 = (YS3 - YH)*(YS3 - YH).transpose()
		S4 = (YS4 - YH)*(YS4 - YH).transpose()
		S5 = (YS5 - YH)*(YS5 - YH).transpose()
		S6 = (YS6 - YH)*(YS6 - YH).transpose()
		S7 = (YS7 - YH)*(YS7 - YH).transpose()
		S8 = (YS8 - YH)*(YS8 - YH).transpose()
# Form weighted average and add measurement noise matrix (STEP 11)
		SMAT = W0*S0 + W1*S1 + W2*S2 + W3*S3 + W4*S4 + W5*S5 + W6*S6 + W7*S7 + W8*S8 + RMAT;
# Calculate cross covariance between propagated state estimates and measurement (STEP 12)
		Pxy0 = (XS0 - XBAR)*(YS0 - YH).transpose()
		Pxy1 = (XS1 - XBAR)*(YS1 - YH).transpose()
		Pxy2 = (XS2 - XBAR)*(YS2 - YH).transpose()
		Pxy3 = (XS3 - XBAR)*(YS3 - YH).transpose()
		Pxy4 = (XS4 - XBAR)*(YS4 - YH).transpose()
		Pxy5 = (XS5 - XBAR)*(YS5 - YH).transpose()
		Pxy6 = (XS6 - XBAR)*(YS6 - YH).transpose()
		Pxy7 = (XS7 - XBAR)*(YS7 - YH).transpose()
		Pxy8 = (XS8 - XBAR)*(YS8 - YH).transpose()
# Form weighted average (STEP 13)
		Pxy = W0*Pxy0 + W1*Pxy1 + W2*Pxy2 + W3*Pxy3 + W4*Pxy4 + W5*Pxy5 + W6*Pxy6 + W7*Pxy7 + W8*Pxy8;
# Kalman filter (STEP 14)
		K = Pxy*inv(SMAT) #compute Kalman gain
		P = M - K*SMAT*K.transpose()  #update covariance with Kalman gain
		THNOISE=GAUSS_PY(SIGTH)
		RNOISE=GAUSS_PY(SIGR)
		TH = math.atan2(YT - YR, XT - XR);
		R = math.sqrt( (XT - XR)**2 + (YT - YR)**2 );
		THMEAS = TH + THNOISE;
		RMEAS = R + RNOISE;
		RESTH = THMEAS-YH[0,0];
		RESR = RMEAS-YH[1,0];
		XTH = XTB + K[0,0]*RESTH + K[0,1]*RESR;
		XTDH = XTDB + K[1,0]*RESTH + K[1,1]*RESR;
		YTH = YTB + K[2,0]*RESTH + K[2,1]*RESR;
		YTDH = YTDB + K[3,0]*RESTH + K[3,1]*RESR;
		ERRYD=YTD-YTDH;
		SP44=math.sqrt(P[3,3]);
		count = count + 1;
		ArrayT.append(T)
		ArrayERRYD.append(ERRYD)
		ArraySP44.append(SP44)
		ArraySP44P.append(-SP44)

plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayERRYD,label='BETA', linewidth=0.6)
plt.plot(ArrayT,ArraySP44,label='BETA', linewidth=0.6)
plt.plot(ArrayT,ArraySP44P,label='BETA', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude Velocity (Ft/Sec)')
plt.ylim(-50,50)
plt.show()
