import numpy as np
#import numpy.linalg.cholesky as chsky
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def f1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)
		
def f2(x1,x2,t):
	dx2dt = 0.0
	return (dx2dt)
		
def f3(y1,y2,t):
	dy1dt = y2
	return (dy1dt)
		
def f4(y1,y2,t):
	G = 32.2
	dy2dt = -G
	return (dy2dt)
		
		
def rk4_X(TS,XP,HP):
	X1 =XP[0]
	X1D =XP[1]
	T = 0.0
	dt = HP
	while (T<=(TS-.0001)):
		k11 = dt*f1(X1,X1D,T)
		k21 = dt*f2(X1,X1D,T)
		k12 = dt*f1(X1+0.5*k11,X1D+0.5*k21,T+0.5*dt)
		k22 = dt*f2(X1+0.5*k11,X1D+0.5*k21,T+0.5*dt)
		k13 = dt*f1(X1+0.5*k12,X1D+0.5*k22,T+0.5*dt)
		k23 = dt*f2(X1+0.5*k12,X1D+0.5*k22,T+0.5*dt)
		k14 = dt*f1(X1+k13,X1D+k23,T+dt)
		k24 = dt*f2(X1+k13,X1D+k23,T+dt)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X1D = X1D + (k21+2*k22+2*k23+k24)/6
		T = T+dt
	X1a = X1
	X2a = X1D
	return ([X1a,X2a])
		
def rk4_Y(TS,YP,HP):
	X1  =YP[0]
	X1D =YP[1]
	T = 0.0
	dt = HP
	while (T<=(TS-.0001)):
		k11 = dt*f3(X1,X1D,T)
		k21 = dt*f4(X1,X1D,T)
		k12 = dt*f3(X1+0.5*k11,X1D+0.5*k21,T+0.5*dt)
		k22 = dt*f4(X1+0.5*k11,X1D+0.5*k21,T+0.5*dt)
		k13 = dt*f3(X1+0.5*k12,X1D+0.5*k22,T+0.5*dt)
		k23 = dt*f4(X1+0.5*k12,X1D+0.5*k22,T+0.5*dt)
		k14 = dt*f3(X1+k13,X1D+k23,T+dt)
		k24 = dt*f4(X1+k13,X1D+k23,T+dt)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X1D = X1D + (k21+2*k22+2*k23+k24)/6
		T = T+dt
	Y1a = X1
	Y2a = X1D
	return ([Y1a,Y2a])

def rk4(TS,POINTS,HP):
	X = [POINTS[0,0],POINTS[1,0]]
	Y = [POINTS[2,0],POINTS[3,0]]
	X_NEXT = rk4_X(TS,X,HP)
	Y_NEXT = rk4_Y(TS,Y,HP)
	NEXT=np.matrix([[X_NEXT[0]],[X_NEXT[1]],[Y_NEXT[0]],[Y_NEXT[1]]])
	return (NEXT)
	

def EULER(TS,XP,HP):
	T=0.0;
	XT =XP[0,0]
	XTD=XP[1,0]
	YT=XP[2,0]
	YTD=XP[3,0]
	
	H=HP;
	while T<=(TS-.0001):
		YTDD = -32.2
		XTDD=0
		XTD=XTD+H*XTDD
		YTD=YTD+H*YTDD
		XT=XT+H*XTD
		YT=YT+H*YTD
		T=T+H
	XTH=XT
	XTDH=XTD
	YTH=YT
	YTDH=YTD
	XB = np.matrix([[XTH],[XTDH],[YTH],[YTDH]])
	return (XB)
	
	
FUDGE=1
PHIS=1
TS=1.
ORDER=4
SIGTH=.01
SIGR=100.
VT=3000.
GAMDEG=45.
G=32.2
XT=0.
YT=0.
XTD=VT*math.cos(GAMDEG/57.3)
YTD=VT*math.sin(GAMDEG/57.3)
XR=100000.
YR=0.
T=0.
S=0.
H=.001
HP=.001

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

PHI=np.matrix([[1,TS, 0, 0],
			   [0, 1, 0, 0],
			   [0, 0, 1,TS],
			   [0, 0, 0, 1]])

P=np.matrix([[(FUDGE*1000)**2,0,0,0],
			 [0, (FUDGE*100)**2,0,0],
			 [0,0,(FUDGE*1000)**2,0],
			 [0,0,0, (FUDGE*100)**2]])
			 
IDNP=np.identity(ORDER)

TS2=TS*TS
TS3=TS2*TS
QMAT=np.matrix([[PHIS*TS3/3.,PHIS*TS2/2.,0,0],
			 [PHIS*TS2/2.,PHIS*TS,0,0],
			 [0,0,PHIS*TS3/3.,PHIS*TS2/2.],
			 [0,0,PHIS*TS2/2.,PHIS*TS]])
			 
RMAT=np.matrix([[SIGTH**2, 0],[0, SIGR**2]])

XTH=XT+FUDGE*1000.
XTDH=XTD-FUDGE*100.
YTH=YT-FUDGE*1000.
YTDH=YTD+FUDGE*100.
RK = 0
KAPPA = 0
NOISE = 1
SIGMA_POINT = 1

XHMAT = np.matrix([[XTH],[XTDH],[YTH],[YTDH]])
count=0
while (YT>=0.):
	XTOLD=XT
	XTDOLD=XTD
	YTOLD=YT
	YTDOLD=YTD
	XTDD=0.
	YTDD=-G
	XT=XT+H*XTD
	XTD=XTD+H*XTDD
	YT=YT+H*YTD
	YTD=YTD+H*YTDD
	T=T+H
	XTDD=0.
	YTDD=-G
	XT=.5*(XTOLD+XT+H*XTD)
	XTD=.5*(XTDOLD+XTD+H*XTDD)
	YT=.5*(YTOLD+YT+H*YTD)
	YTD=.5*(YTDOLD+YTD+H*YTDD)
	S=S+H
	if (S>=(TS-.00001)):
		if NOISE == 1:
			THETNOISE=GAUSS_PY(SIGTH)
			RTNOISE=GAUSS_PY(SIGR)
		else:
			THETNOISE=0.0
			RTNOISE=0.0
		S=0.
		if RK == 0:
		# This routine propagates the previous estimates to the a-priori estimates using the fundamental matrix.
		# This can be done because higher exponentials of the fundamental matrix is zero, i.e. F*F = F*F*F = F*F*F*F = 0
			#XTB=XTH+TS*XTDH
			#XTDB=XTDH
			#YTB=YTH+TS*YTDH-.5*G*TS*TS
			#YTDB=YTDH-G*TS
			XTB=XHMAT[0,0]+TS*XHMAT[1,0]
			XTDB=XHMAT[1,0]
			YTB=XHMAT[2,0]+TS*XHMAT[3,0]-.5*G*TS*TS
			YTDB=XHMAT[3,0]-G*TS
		else:
			# Propagating the states via Runge-kutta solver
			XP_LIST = [XHMAT[0,0], XHMAT[1,0]]
			YP_LIST = [XHMAT[2,0], XHMAT[3,0]]
			[XTB,XTDB] = rk4_X(TS,XP_LIST,HP)
			[YTB,YTDB] = rk4_Y(TS,YP_LIST,HP)
		if SIGMA_POINT == 1:	
			C = np.linalg.cholesky((ORDER+KAPPA)*P)
			#STEP1: State estimate vector. Because there are 4 states, XT,XTD,YT and YTD, there are 2*4=8 sigma points
			XS0 = XHMAT
			XS1 = XHMAT + C[:,0]
			XS2 = XHMAT + C[:,1]
			XS3 = XHMAT + C[:,2]
			XS4 = XHMAT + C[:,3]
			XS5 = XHMAT - C[:,0]
			XS6 = XHMAT - C[:,1]
			XS7 = XHMAT - C[:,2]
			XS8 = XHMAT - C[:,3]
			XS_LIST = [XS0,XS1,XS2,XS3,XS4,XS5,XS6,XS7,XS8]			
			#Step 2: Build the weighting factors
			W0 = KAPPA/(ORDER+KAPPA)
			W_LIST = [W0] + [1/(2*(ORDER + KAPPA)) for i in range(0,2*ORDER)]	
			#Step 3: Propagate the sigma points one time interval
			if RK == 1:
				# Use this part of the code to numerically propagate the sigma point using Euler routine (STEP 3)
				XSB_LIST = [rk4(TS, XS, HP) for XS in XS_LIST]
			else:
				# Use numerical integration to propagate each sigma point ahead TS seconds (STEP 3)
				XSB_LIST = [EULER(TS, XS, HP) for XS in XS_LIST]
			#Step 4: Calculate the weighing average
			XBAR = sum([w*xb for w,xb in zip(W_LIST, XSB_LIST)])
			#STEP 5: Estimate the covariance matrix for the sigma points
			MK = [(XP - XBAR)*(XP-XBAR).transpose() for XP in XSB_LIST]
			#STEP6: build the new weighted matrix M and add process noise matrix QMAT
			M = sum([w*m for w, m in zip(W_LIST, MK)]) + QMAT
			#STEP 7: Estimate a new sigma points using the recent calculated M matrix
			CM = np.linalg.cholesky((ORDER+KAPPA)*M)
			XS0 = XBAR
			XS1 = XBAR + CM[:,0]
			XS2 = XBAR + CM[:,1]
			XS3 = XBAR + CM[:,2]
			XS4 = XBAR + CM[:,3]
			XS5 = XBAR - CM[:,0]
			XS6 = XBAR - CM[:,1]
			XS7 = XBAR - CM[:,2]
			XS8 = XBAR - CM[:,3]
			XS_LIST_NEW = [XS0,XS1,XS2,XS3,XS4,XS5,XS6,XS7,XS8]
			#STEP 8: Estimate the measurements using the H matrix
			#RES1=THETMEAS-THETB
			#RES2=RTMEAS-RTB
			# Transform sigma points via non-linear measurement equations (STEP 8b)
			# S_LIST = [HMAT*XS for XS in XS_LIST_NEW] cannot be used because the measurements of theta and r are
			# non-linear equations, therefore we need to transform the X,Y sigma points into a theta, r sigma points
			YS0 = np.matrix([[math.atan2(XS0[2,0] - YR, XS0[0,0] - XR)],[math.sqrt( (XS0[0,0] - XR)**2 + (XS0[2,0] - YR)**2 )]])
			YS1 = np.matrix([[math.atan2(XS1[2,0] - YR, XS1[0,0] - XR)],[math.sqrt( (XS1[0,0] - XR)**2 + (XS1[2,0] - YR)**2 )]])
			YS2 = np.matrix([[math.atan2(XS2[2,0] - YR, XS2[0,0] - XR)],[math.sqrt( (XS2[0,0] - XR)**2 + (XS2[2,0] - YR)**2 )]])
			YS3 = np.matrix([[math.atan2(XS3[2,0] - YR, XS3[0,0] - XR)],[math.sqrt( (XS3[0,0] - XR)**2 + (XS3[2,0] - YR)**2 )]])
			YS4 = np.matrix([[math.atan2(XS4[2,0] - YR, XS4[0,0] - XR)],[math.sqrt( (XS4[0,0] - XR)**2 + (XS4[2,0] - YR)**2 )]])
			YS5 = np.matrix([[math.atan2(XS5[2,0] - YR, XS5[0,0] - XR)],[math.sqrt( (XS5[0,0] - XR)**2 + (XS5[2,0] - YR)**2 )]])
			YS6 = np.matrix([[math.atan2(XS6[2,0] - YR, XS6[0,0] - XR)],[math.sqrt( (XS6[0,0] - XR)**2 + (XS6[2,0] - YR)**2 )]])
			YS7 = np.matrix([[math.atan2(XS7[2,0] - YR, XS7[0,0] - XR)],[math.sqrt( (XS7[0,0] - XR)**2 + (XS7[2,0] - YR)**2 )]])
			YS8 = np.matrix([[math.atan2(XS8[2,0] - YR, XS8[0,0] - XR)],[math.sqrt( (XS8[0,0] - XR)**2 + (XS8[2,0] - YR)**2 )]])
			YS_LIST = [YS0,YS1,YS2,YS3,YS4,YS5,YS6,YS7,YS8]
			#STEP 9: Estimate the weighted measurements		
			YH = sum([W*Y for W,Y in zip(W_LIST, YS_LIST)])
			#STEP 10: Estimate new variance
			S_LIST = [(YS - YH)*(YS - YH).transpose() for YS in YS_LIST]
			#STEP 11: Estimate the weighted S Matrix		
			SMAT = sum([W*S for W, S in zip(W_LIST, S_LIST)]) + RMAT
			#STEP 12 and STEP 13: Estimate the cross covariance between x and y
			PXY_LIST = [(XS - XBAR)*(YS - YH).transpose()  for XS,YS in zip(XS_LIST_NEW, YS_LIST)]
			PXY = sum([W*PXY for W,PXY in zip(W_LIST, PXY_LIST)])	
			#STEP 14: Estimate Kalman gain
			K = PXY*inv(SMAT)
			P = M - K*SMAT*K.transpose()
			XTB  = XBAR[0,0]
			XTDB = XBAR[1,0]
			YTB  = XBAR[2,0]
			YTDB = XBAR[3,0]
		else:
			# Required to estimate the H matrix
			RTB=math.sqrt((XTB-XR)**2+(YTB-YR)**2)
			HMAT=np.matrix([[-(YTB-YR)/RTB**2, 0, (XTB-XR)/RTB**2, 0],
							[ (XTB-XR)/RTB,    0, (YTB-YR)/RTB,    0]])
			M=PHI*P*PHI.transpose()+QMAT
			K = M*HMAT.transpose()*inv(HMAT*M*HMAT.transpose() + RMAT)
			P = (IDNP - K*HMAT)*M
		THET=math.atan2((YT-YR),(XT-XR))
		RT=math.sqrt((XT-XR)**2+(YT-YR)**2)
		THETMEAS=THET+THETNOISE
		RTMEAS=RT+RTNOISE
		if SIGMA_POINT == 1:
			XBMAT = XBAR
			THETB=YH[0,0]
			RTB = YH[1,0]
		else:
			XBMAT = np.matrix([[XTB],[XTDB],[YTB],[YTDB]])
			THETB=math.atan2((YTB-YR),(XTB-XR))
			RTB=math.sqrt((XTB-XR)**2+(YTB-YR)**2)
		RES1=THETMEAS-THETB
		RES2=RTMEAS-RTB
		XHMAT = XBMAT+K[:,0]*RES1 + K[:,1]*RES2
		# Another method to estimate the new a posteriori values without using matrix operations
		#XTH=XTB+K[0,0]*RES1+K[0,1]*RES2
		#XTDH=XTDB+K[1,0]*RES1+K[1,1]*RES2
		#YTH=YTB+K[2,0]*RES1+K[2,1]*RES2
		#YTDH=YTDB+K[3,0]*RES1+K[3,1]*RES2
		ERRYD= YTD-XHMAT[3,0]
		#ERRYD=YTD-YTDH
		SP44=math.sqrt(P[3,3])
		SP44P=-SP44
		count=count+1
		ArrayT.append(T)
		ArrayERRYD.append(ERRYD)
		ArraySP44.append(SP44)
		ArraySP44P.append(SP44P)

plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayERRYD,label='BETA', linewidth=0.6)
plt.plot(ArrayT,ArraySP44,label='BETA', linewidth=0.6)
plt.plot(ArrayT,ArraySP44P,label='BETA', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude Velocity (Ft/Sec)')
plt.ylim(-50,50)
plt.show()

