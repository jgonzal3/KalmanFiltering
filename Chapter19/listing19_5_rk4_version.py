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
# Initial vales for X and Y
XTD=VT*math.cos(GAMDEG/57.3)
YTD=VT*math.sin(GAMDEG/57.3)
XR=100000.
YR=0.
T=0.
S=0.
H=.001
HP=.001

ArrayT = []
ArrayERRYD = []
ArraySP44 = []
ArraySP44P = []

PHI=np.matrix([[1, TS, 0, 0],
			         [0, 1,  0, 0],
			         [0, 0,  1, TS],
			         [0, 0,  0, 1]])

P=np.matrix([[(FUDGE*1000)**2, 0,0,0],
			       [0,               (FUDGE*100)**2,               0,               0],
			       [0,               0,              (FUDGE*1000)**2,               0],
			       [0,               0,                            0, (FUDGE*100)**2]])
			 
IDNP=np.identity(ORDER)

TS2=TS*TS
TS3=TS2*TS
Q=np.matrix([[PHIS*TS3/3., PHIS*TS2/2.,  0,                     0],
			       [PHIS*TS2/2., PHIS*TS,      0,                     0],
			       [0,           0,            PHIS*TS3/3., PHIS*TS2/2.],
			       [0,           0,            PHIS*TS2/2.,    PHIS*TS]])
			 
RMAT=np.matrix([[SIGTH**2, 0],[0, SIGR**2]])

XTH=XT+FUDGE*1000.
XTDH=XTD-FUDGE*100.
YTH=YT-FUDGE*1000.
YTDH=YTD+FUDGE*100.
RK = 0

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
		S=0.
		if RK == 0:
		# This routine propagates the previous estimates to the a-priori estimates using the fundamental matrix.
		# This can be done because higher exponentials of the fundamental matrix are zero, i.e. F*F = F*F*F = F*F*F*F = 0
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
		XBMAT = np.matrix([[XTB],[XTDB],[YTB],[YTDB]])
		RTB=math.sqrt((XTB-XR)**2+(YTB-YR)**2)
		HMAT=np.matrix([[-(YTB-YR)/RTB**2, 0, (XTB-XR)/RTB**2, 0],
				            [ (XTB-XR)/RTB,    0, (YTB-YR)/RTB,    0]])
		M=PHI*P*PHI.transpose()+Q
		K = M*HMAT.transpose()*inv(HMAT*M*HMAT.transpose() + RMAT)
		P = (IDNP - K*HMAT)*M
		THETNOISE=GAUSS_PY(SIGTH)
		RTNOISE=GAUSS_PY(SIGR)
		THET=math.atan2((YT-YR),(XT-XR))
		RT=math.sqrt((XT-XR)**2+(YT-YR)**2)
		THETMEAS=THET+THETNOISE
		RTMEAS=RT+RTNOISE
		THETB=math.atan2((YTB-YR),(XTB-XR))
		RTB=math.sqrt((XTB-XR)**2+(YTB-YR)**2)
		RES1=THETMEAS-THETB
		RES2=RTMEAS-RTB
		XHMAT = XBMAT+K[:,0]*RES1 + K[:,1]*RES2
		ERRYD= YTD-XHMAT[3,0]
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
plt.show()

