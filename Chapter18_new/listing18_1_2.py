import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def projectc18l1(TP,TS,XTP,XTDP,HP,W):
	T=0.
	XT=XTP
	XTD=XTDP
	H=HP
	while T<=(TS-.0001):
		XTDD=-W*W*XT
		XTD=XTD+H*XTDD
		XT=XT+H*XTD
		T=T+H
	XTH=XT
	XTDH=XTD
	return(XTH,XTDH)


ArrayT=[]
ArrayW=[]
ArrayWH=[]
ArrayERRW=[]
ArraySP33=[]
ArraySP33P=[]
		
SDRE=1;
CHOICE=1;
HP=.001;
W=1.;
A=1.;
TS=.1;
ORDER=3;
PHIS=.1;
SIGX=1.;
T=0.;
S=0.;
H=.001;
IDNP=np.identity(ORDER);
Q=np.matrix([[0,0,0],[0,0,0],[0,0,PHIS*TS]])
RMAT=np.matrix([[SIGX**2]])
P=np.matrix([[SIGX**2,0,0],[0,2.**2,0],[0,0,2.**2]])

XTD=A*W;
HMAT=np.matrix([[1,0,0]]);

for CHOICE in [0,1,2,3]:
	S = 0.0
	T = 0.0
	XTH=0.
	XTDH=0.
	WH=2.
	XT=0.
	P=np.matrix([[SIGX**2,0,0],[0,2.**2,0],[0,0,2.**2]])

	while (T<=20.0):
		XTOLD=XT;
		XTDOLD=XTD;
		XTDD=-W*W*XT;
		XT=XT+H*XTD;
		XTD=XTD+H*XTDD;
		T=T+H;
		XTDD=-W*W*XT;
		XT=.5*(XTOLD+XT+H*XTD);
		XTD=.5*(XTDOLD+XTD+H*XTDD);
		S=S+H;
		if (S>=(TS-.00001)):
			S=0.;
			if (CHOICE==0):
				F=np.matrix([[0,1,0],[-WH**2,0,-2*WH*XTH],[0,0,0]])
			elif (CHOICE==1):
				F=np.matrix([[0,1,0],[0,0,-WH*XTH],[0,0,0]])
			elif (CHOICE==2):
				F=np.matrix([[0,1,0],[-WH**2,0,0],[0,0,0]])
			else:
				F=np.matrix([[0,1,0],[-WH**2,0,-2*WH*XTH],[0,0,0]])
			PHI=IDNP+F*TS;
			M=PHI*P*PHI.transpose()+Q;
			K = M*HMAT.transpose()*inv(HMAT*M*HMAT.transpose() + RMAT);
			P = (IDNP - K*HMAT)*M;
			XTNOISE=GAUSS_PY(SIGX);
			XTMEAS=XT+XTNOISE;
			[XTB,XTDB]=projectc18l1(T,TS,XTH,XTDH,HP,WH);
			RES=XTMEAS-XTB;
			XTH=XTB+K[0,0]*RES;
			XTDH=XTDB+K[1,0]*RES;
			WH=WH+K[2,0]*RES;
			WH=abs(WH);
			ERRW=W-WH;
			SP33=math.sqrt(P[2,2]);
			SP33P=-SP33;
			ArrayT.append(T)
			ArrayW.append(W)
			ArrayWH.append(WH)
	plt.plot(ArrayT,ArrayWH,label='SDRE'+str(CHOICE), linewidth=0.6)
	if (CHOICE == 1):
		plt.plot(ArrayT,ArrayW,label='W', linewidth=0.6)
	ArrayT = []
	ArrayWH = []
	ArrayW = []

plt.figure(1)
plt.grid(True)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Frequency (R/S)')
plt.legend()

plt.show()
