import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

WN=6.28*.1
W=6.28*1.

TF=20.
SIGX=.1
TS=.01
Z=.7
A=-Z*WN
B=WN*math.sqrt(1.-Z*Z)
A2=.1
X1=.25
B2=1.25
X1D=0.
X2 = X1D
B1=1.
T=0.
S=0.
H=.001
X2DDOLD=0.

t=[]
x1=[]
x1_hat=[]
x1dot=[]
x1dot_hat=[]
xs=[]
xddot_hat=[]
x1_hat_ERR=[]
sp11=[]
sp11p=[]
xdot_hat_ERR=[]
sp22=[]
sp22p=[]
xddot_hat_ERR=[]
x2ddold =[]

PHIS=0.
TS=.01
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=.1

phi_11 = math.exp(A*TS)*(-A*math.sin(B*TS)+B*math.cos(B*TS))/B
phi_12 =  math.exp(A*TS)*math.sin(B*TS)/B
phi_21 = -WN*WN*math.exp(A*TS)*math.sin(B*TS)/B
phi_22 = math.exp(A*TS)*(A*math.sin(B*TS)+B*math.cos(B*TS))/B

g_11=-(math.exp(A*TS)*(A*math.sin(B*TS)-B*math.cos(B*TS))+B)/(B*(A*A+B*B))
g_21=-math.exp(A*TS)*math.sin(B*TS)/B

# New PHI matrix
PHI = np.matrix([[phi_11, phi_12],[phi_21, phi_22]])
G = np.matrix([[g_11], [g_21]])
P = np.matrix([[99999999., 0],[0,99999999.]])
I = np.matrix([[1, 0],[0, 1]])
Q = np.matrix([[0, 0] ,[0, 0]])
HMAT = np.matrix([[1, 0]])
R = np.matrix([[SIGMA_NOISE**2]])
RMAT = np.matrix([[SIGMA_NOISE**2]])

PHIT=PHI.transpose()
HT=HMAT.transpose()


X1H = X1
X1DH = X1D
while (T <= 20):
	S=S+H
	X1OLD=X1
	X1DOLD=X1D
	X2=A2*math.sin(W*T)
	X2D=A2*W*math.cos(W*T)
	X2DD=-A2*W*W*math.sin(W*T)
	X1DD=-2.*Z*WN*X1D-WN*WN*X1-X2DD
	X1=X1+H*X1D
	X1D=X1D+H*X1DD
	T=T+H
	X2=A2*math.sin(W*T)
	X2D=A2*W*math.cos(W*T)
	X2DD=-A2*W*W*math.sin(W*T)
	X1DD=-2.*Z*WN*X1D-WN*WN*X1-X2DD
	X1=.5*(X1OLD+X1+H*X1D)
	X1D=.5*(X1DOLD+X1D+H*X1DD)
	if(S>=(TS-.00001)):
		S=0.0
		PHIP=PHI*P;
		PHIPPHIT=PHIP*PHIT;
		M=PHIPPHIT+Q;
		HM=HMAT*M
		HMHT=HM*HT
		HMHTR=HMHT+RMAT
		HMHTRINV=inv(HMHTR);
		MHT=M*HT;
		K=MHT*HMHTRINV;
		KH=K*HMAT;
		IKH=I-KH;
		P=IKH*M;
		XNOISE = GAUSS_PY(SIGMA_NOISE)
		XMEAS=X1+B1+X2+B2+XNOISE
		XS=XMEAS-X2-B1-B2
		RES=XS-PHI[0,0]*X1H-PHI[0,1]*X1DH-G[0,0]*X2DDOLD
		X1HOLD=X1H
		X1H=PHI[0,0]*X1H+PHI[0,1]*X1DH+G[0,0]*X2DDOLD+K[0,0]*RES
		X1DH=PHI[1,0]*X1HOLD+PHI[1,1]*X1DH+G[1,0]*X2DDOLD+K[1,0]*RES
		ERRX1=X1-X1H
		SP11=math.sqrt(P[0,0])
		SP11P = -SP11
		ERRX1D=X1D-X1DH
		SP22=math.sqrt(P[1,1])
		X2DDOLD=-A2*W*W*math.sin(W*T)
		t.append(T)
		x1.append(X1)
		xs.append(XS)
		x1_hat.append(X1H)
		x1dot.append(X1D)
		x1dot_hat.append(X1DH)
		x1_hat_ERR.append(ERRX1)
		sp11.append(SP11)
		sp11p.append(SP11P)
		xdot_hat_ERR.append(ERRX1D)
		sp22.append(SP22)
		x2ddold.append(X2DDOLD)

plt.figure(1)
plt.grid(True)
plt.plot(t,x1_hat,label='x1_hat',linewidth=0.6)
plt.plot(t,x1,label='x1',linewidth=0.6)
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,20)
plt.ylim(-0.2,1)

plt.figure(2)
plt.grid(True)
#plt.plot(t,xs)
plt.plot(t,x1_hat_ERR,label='x1_hat_error',linewidth=0.6)
plt.plot(t,sp11,label='sp11',linewidth=0.6)
plt.plot(t,sp11p,label='sp11p',linewidth=0.6)
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Error in x1')
plt.xlim(0,20)
plt.ylim(-.05,0.05)

plt.figure(3)
plt.grid(True)
plt.plot(t,x1dot,label='x1dot',linewidth=0.6)
plt.plot(t,x1dot_hat,label='x1dot_hat',linewidth=0.6)
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('True Derivative and estimate')
plt.xlim(0,20)
plt.ylim(-4,4)


plt.figure(4)
plt.grid(True)
plt.plot(t,xdot_hat_ERR,label='x1dot_hat_Error',linewidth=0.6)

plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Error in deerivative')
plt.xlim(0,20)
plt.ylim(-4,4)


plt.show()

	