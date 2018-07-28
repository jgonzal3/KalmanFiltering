import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

ArrayT=[]
ArrayXTD=[]
ArrayYTD=[]
ArrayXTDH=[]
ArrayYTDH=[]
ArrayERRX=[]
ArrayERRY=[]
ArraySP22=[]
ArraySP22P=[]
ArraySP44=[]
ArraySP44P=[]
ArrayERRYD = []
ArrayERRXD = []
ArraySP11=[]
ArraySP11P=[]
ArraySP33=[]
ArraySP33P=[]

CHOICE=1
EKF=1
TS=1.0
ORDER=4
PHIS=0.0
SIGTH=.01
SIGR=100.0
VT=3000.0
GAMDEG=45.0
G=32.2
XT=0.
YT=0.
XTD=VT*math.cos(GAMDEG/57.3)
YTD=VT*math.sin(GAMDEG/57.3)
XR=100000.0
YR=0.0
T=0.0
S=0.0
H=.001
IDNP=np.identity(ORDER)
Q=np.matrix([[0,0,0,0],[0,PHIS*TS,0,0],[0,0,0,0],[0,0,0,PHIS*TS]])
RMAT=np.matrix([[SIGTH**2,0],[0,SIGR**2]])
P=np.matrix([[1000.0**2,0,0,0],[0,100.**2,0,0],[0,0,1000.**2,0],[0,0,0,100.0**2]])
PHI=np.matrix([[1,TS,0,0],[0,1,0,0],[0,0,1,TS],[0,0,0,1]])
XTH=XT+1000.
XTDH=XTD-100.
YTH=YT-1000.
YTDH=YTD+100.
while (YT>=0.0):
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
		XTB=XTH+TS*XTDH
		XTDB=XTDH
		YTB=YTH+TS*YTDH-.5*G*TS*TS
		YTDB=YTDH-G*TS
		RTB=math.sqrt((XTB-XR)**2+(YTB-YR)**2)
		THETB=math.atan2((YTB-YR),(XTB-XR))
		if (EKF==1):
			HMAT=np.matrix([[-(YTB-YR)/RTB**2,0,(XTB-XR)/RTB**2,0],[(XTB-XR)/RTB,0,(YTB-YR)/RTB,0]])
		else:
			if (CHOICE==1):
				HMAT=np.matrix([[1, (THETB-XTB)/XTDB, 0, 0],[0, 0, (RTB-YTDB)/YTB, 1]])
			else:
				HMAT=np.matrix([[1,(THETB-XTB-YTDB)/XTDB,0,1],[1,0,(RTB-XTB-YTDB)/YTB,1]])
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
		XTH=XTB+K[0,0]*RES1+K[0,1]*RES2
		XTDH=XTDB+K[1,0]*RES1+K[1,1]*RES2
		YTH=YTB+K[2,0]*RES1+K[2,1]*RES2
		YTDH=YTDB+K[3,0]*RES1+K[3,1]*RES2
		ERRX=XT-XTH
		SP11=math.sqrt(P[0,0])
		ERRXD=XTD-XTDH
		SP22=math.sqrt(P[1,1])
		ERRY=YT-YTH
		SP33=math.sqrt(P[2,2])
		ERRYD=YTD-YTDH
		SP44=math.sqrt(P[3,3])
		SP11P=-SP11
		SP22P=-SP22
		SP33P=-SP33
		SP44P=-SP44
		ArrayT.append(T)
		ArrayXTD.append(XTD)
		ArrayXTDH.append(XTDH)
		ArrayYTD.append(YTD)
		ArrayYTDH.append(YTDH)
		ArrayERRX.append(ERRX)
		ArrayERRY.append(ERRY)
		ArrayERRXD.append(ERRXD)
		ArraySP22.append(SP22)
		ArraySP22P.append(SP22P)
		ArraySP11.append(SP11)
		ArraySP11P.append(SP11P)
		ArraySP33.append(SP33)
		ArraySP33P.append(SP33P)
		ArrayERRYD.append(ERRYD)
		ArraySP44.append(SP44)
		ArraySP44P.append(SP44P)

plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayERRXD,label='error XD', linewidth=0.6)
plt.plot(ArrayT,ArraySP22,label='sp22', linewidth=0.6)
plt.plot(ArrayT,ArraySP22P,label='sp22-n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Downrange Velocity (Ft/Sec)')
plt.legend()

plt.figure(2)
plt.grid(True)
plt.plot(ArrayT,ArrayERRYD,label='error YD', linewidth=0.6)
plt.plot(ArrayT,ArraySP44,label='sp44', linewidth=0.6)
plt.plot(ArrayT,ArraySP44P,label='sp44-n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude Velocity (Ft/Sec)')
plt.legend()

plt.figure(3)
plt.grid(True)
plt.plot(ArrayT,ArrayERRX,label='error X', linewidth=0.6)
plt.plot(ArrayT,ArraySP11,label='sp11', linewidth=0.6)
plt.plot(ArrayT,ArraySP11P,label='sp11-n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Downrange (Ft)')
plt.legend()

plt.figure(4)
plt.grid(True)
plt.plot(ArrayT,ArrayERRY,label='error Y', linewidth=0.6)
plt.plot(ArrayT,ArraySP33,label='sp33', linewidth=0.6)
plt.plot(ArrayT,ArraySP33P,label='sp33-n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.legend()

plt.show()
