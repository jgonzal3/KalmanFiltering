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

def projectc18l2(TP,TS,XP,XDP,BETA,HP):
	T=0.;
	X=XP;
	XD=XDP;
	H=HP;
	while (T<=(TS-.0001)):
		XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2;
		XD=XD+H*XDD;
		X=X+H*XD;
		T=T+H;
	XB=X;
	XDB=XD;
	return (XB, XDB)

ArrayT=[]
ArrayBETA=[]
ArrayBETAH=[]
ArrayERRBETA=[]
ArrayW=[]
ArrayWH=[]
ArrayERRW=[]
ArraySP33=[]
ArraySP33P=[]

EKF=0;
PHIS=100;
CHOICE=2;
SIGNOISE=25.;
X=150000.;
XD=-6000.;
BETA=500.;
XH=150025.;
XDH=-6150.;
BETAH=800.;
ORDER=3;
TS=.1;
TF=30.;
T=0.;
S=0.;
H=.001;
HP=.001;

HMAT=np.matrix([[1,0,0]]);
IDNP=np.identity(ORDER);
Q=np.matrix([[0,0,0],[0,0,0],[0,0,PHIS*TS]])
RMAT=np.matrix([[SIGNOISE**2]])
P=np.matrix([[SIGNOISE**2,0,0],[0,20000.,0],[0,0,(BETA-BETAH)**2]])

while (T<=TF): 
	XOLD=X;
	XDOLD=XD;
	XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2;
	X=X+H*XD;
	XD=XD+H*XDD;
	T=T+H;
	XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2;
	X=.5*(XOLD+X+H*XD);
	XD=.5*(XDOLD+XD+H*XDD);
	S=S+H;
	if (S>=(TS-.00001)):
		S=0.;
		RHOH=.0034*math.exp(-XH/22000.);
		if (EKF==1):
			F21=-32.2*RHOH*XDH*XDH/(44000.*BETAH);
			F22=RHOH*32.2*XDH/BETAH;
			F23=-RHOH*32.2*XDH*XDH/(2.*BETAH*BETAH);
			F=np.matrix([[0,1,0],[F21, F22, F23],[0,0,0]])
		else:
			if (CHOICE==1):
				F22=(32.2*RHOH*XDH*XDH/(2.*BETAH)-BETAH)/XDH;
				F=np.matrix([[0,1,0],[0, F22, 1],[0,0,0]])
			else:
				F22=(32.2*RHOH*XDH*XDH/(2.*BETAH)+BETAH)/XDH;
				F=np.matrix([[0,1,0],[0, F22, -1],[0,0,0]])
		PHI=IDNP+TS*F;
		M=PHI*P*PHI.transpose()+Q;
		K = M*HMAT.transpose()*inv(HMAT*M*HMAT.transpose() + RMAT);
		P = (IDNP - K*HMAT)*M;
		XNOISE=GAUSS_PY(SIGNOISE);
		[XB,XDB]=projectc18l2(T,TS,XH,XDH,BETAH,HP);
		RES=X+XNOISE-XB;
		XH=XB+K[0,0]*RES;
		XDH=XDB+K[1,0]*RES;
		BETAH=BETAH+K[2,0]*RES;
		ERRX=X-XH;
		SP11=math.sqrt(P[0,0]);
		ERRXD=XD-XDH;
		SP22=math.sqrt(P[1,1]);
		ERRBETA=BETA-BETAH;
		SP33=math.sqrt(P[2,2]);
		SP11P=-SP11;
		SP22P=-SP22;
		SP33P=-SP33;
		ArrayT.append(T)
		ArrayBETA.append(BETA)
		ArrayBETAH.append(BETAH)
		ArrayERRBETA.append(ERRBETA)
		ArraySP33.append(SP33)
		ArraySP33P.append(SP33P)

plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayBETA,label='BETA'+str(CHOICE), linewidth=0.6)
plt.plot(ArrayT,ArrayBETAH,label='BETAH', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Frequency (R/S)')
plt.legend()
plt.show()		 
