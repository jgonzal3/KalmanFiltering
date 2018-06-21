import math
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

def GAUSS(SIG):
	SUM=0
	for j in range(1,7):
		# THE NEXT STATEMENT PRODUCES A UNIF. DISTRIBUTED NUMBER FROM -0.5 and 0.5
		IRAN=random.uniform(-0.5, 0.5)
		SUM=SUM+IRAN
		# In this case we have to multiply the resultant random variable by the square root of 2,
	X=math.sqrt(2)*SUM*SIG
	return (X)
	
def GAUSS_PY(SIG):
	X=(random.uniform(-3.0,3.0))*SIG
	return (X)

def f1(x1,x2,beta,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2, beta,t):
	dx2dt = (0.0034*G*x2*x2*math.exp(-x1/22000.0)/(2.0*beta))-G
	return (dx2dt)

def PROJECT(TS,XP,XDP,BETAP,HP):
	T=0.
	X=XP
	XD=XDP
	BETA=BETAP
	H=HP
	while (T<=(TS-.0001)):
		XDD=.0034*32.2*XD*XD*math.exp(-X/22000.)/(2.*BETA)-32.2
		XD=XD+H*XDD
		X=X+H*XD
		T=T+H
	XH=X
	XDH=XD
	XDDH=XDD
	return [XH,XDH, XDDH]
	

t=[]
x1=[]
xs=[]
x1_hat=[]
x1dot=[]
x1dot_hat=[]
x1ddot=[]
x1ddot_hat=[]
x1_hat_ERR=[]
sp11=[]
sp11n=[]
x1dot_hat_ERR=[]
sp22=[]
sp22n=[]
x1ddot_hat_ERR=[]
sp33=[]
sp33n=[]
beta_hat=[]
res=[]


PHIS=322*322/30
TS=.1
SIGMA_NOISE=25.
X1=200000.
X1D=-6000.
X1DD = 0
BETA=500.
BETAH=800.
X1H=200025.
X1DH=-6150.
X1DDH=0.0
TS=0.1
TF=30.
T=0.0
H=0.001
G=32.2
S=0

p_11 = SIGMA_NOISE*SIGMA_NOISE	# Error in the altitude
p_22 = 150*150					# Error in the velocity
p_33 = 322*322					# Error in BETA parameter

P = np.matrix([[p_11, 0, 0],[0, p_22, 0], [0, 0, p_33]])
PHI = np.matrix([[1, TS, 0.5*TS*TS],[0, 1, TS], [0, 0, 1]])
I = np.matrix([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
Q = np.matrix([[TS**5/20, TS**4/8, TS**3/6],[TS**4/8, TS**3/3, TS*TS/2] ,[TS**3/6, TS*TS/2, TS] ])
HMAT = np.matrix([[1, 0 , 0]])
R = np.matrix([[SIGMA_NOISE**2]])
rk = 0

dt= H
while (T <= 30.0):
	if (rk == 1):
		k11 = dt*f1(X1,X1D,BETA,T)
		k21 = dt*f2(X1,X1D,BETA,T)
		k12 = dt*f1(X1+0.5*k11,X1D+0.5*k21,BETA,T+0.5*dt)
		k22 = dt*f2(X1+0.5*k11,X1D+0.5*k21,BETA,T+0.5*dt)
		k13 = dt*f1(X1+0.5*k12,X1D+0.5*k22,BETA,T+0.5*dt)
		k23 = dt*f2(X1+0.5*k12,X1D+0.5*k22,BETA,T+0.5*dt)
		k14 = dt*f1(X1+k13,X1D+k23,BETA,T+dt)
		k24 = dt*f2(X1+k13,X1D+k23,BETA,T+dt)
		X1DD = f2(X1,X1D,BETA,T)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X1D = X1D + (k21+2*k22+2*k23+k24)/6
		T = T+dt
	else:
		X1OLD=X1
		X1DOLD=X1D
		X1DD = 0.0034*32.2*X1D*X1D*math.exp(-X1/22000.)/(2.*BETA)-32.2
		X1  = X1+dt*X1D
		X1D = X1D+dt*X1DD
		T=T+dt
		X1DD=.0034*32.2*X1D*X1D*math.exp(-X1/22000.)/(2.*BETA)-32.2
		X1=.5*(X1OLD+X1+dt*X1D)
		X1D=.5*(X1DOLD+X1D+dt*X1DD)
	S = S+H
	if(S>=(TS-.00001)):
		S=0.
		M=PHI*P*PHI.transpose()+PHIS*Q
		K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + R))
		P=(I-K*HMAT)*M	
		XS = X1 + GAUSS_PY(SIGMA_NOISE)
		RES=XS-X1H-TS*X1DH-.5*TS*TS*X1DDH
		X1H=X1H+TS*X1DH+.5*TS*TS*X1DDH+K[0,0]*RES
		X1DH=X1DH+TS*X1DDH+K[1,0]*RES
		X1DDH=X1DDH+K[2,0]*RES
		RHOH=.0034*math.exp(-X1H/22000.)
		BETAH=16.1*RHOH*X1DH*X1DH/(X1DDH+32.2)
		ERRX1=X1-X1H
		SP11=math.sqrt(P[0,0])
		ERRX1D=X1D-X1DH
		SP22=math.sqrt(P[1,1])
		ERRX1DD=X1DD-X1DDH
		SP33=math.sqrt(P[2,2])
		SP11N=-SP11
		SP22N=-SP22
		SP33N=-SP33
		t.append(T)
		x1.append(X1)
		xs.append(XS)
		res.append(RES)
		x1_hat.append(X1H)
		x1dot.append(X1D)
		x1dot_hat.append(X1DH)
		x1ddot.append(X1DD)
		x1ddot_hat.append(X1DDH)
		x1_hat_ERR.append(ERRX1)
		sp11.append(SP11)
		sp11n.append(SP11N)
		x1dot_hat_ERR.append(ERRX1D)
		sp22.append(SP22)
		sp22n.append(SP22N)
		x1ddot_hat_ERR.append(ERRX1DD)
		sp33.append(SP33)
		sp33n.append(SP33N)
		beta_hat.append(BETAH)

				
'''
Adding process noise increases errors in estimate of altitude
but reduces the hangoff error 
'''

plt.figure(1)
plt.grid(True)
plt.plot(t,x1_hat_ERR,label='x-hat', linewidth=0.6)
plt.plot(t,sp11,label='sp11', linewidth=0.6)
plt.plot(t,sp11n,label='sp11n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-50,50)

plt.figure(2)
plt.grid(True)
plt.plot(t,x1dot_hat_ERR,label='xd-hat', linewidth=0.6)
plt.plot(t,sp22,label='sp22', linewidth=0.6)
plt.plot(t,sp22n,label='sp22n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-200,200)

plt.figure(3)
plt.grid(True)
plt.plot(t,x1ddot_hat_ERR,label='xdd-hat', linewidth=0.6)
plt.plot(t,sp33,label='sp33', linewidth=0.6)
plt.plot(t,sp33n,label='sp33n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-200,200)

plt.figure(4)
plt.grid(True)
plt.plot(t,x1ddot,label='ac actual', linewidth=0.6)
plt.plot(t,x1ddot_hat,label='ac predicted', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-400,400)

plt.figure(5)
plt.grid(True)
plt.plot(t,[BETA for x in t],label='beta actual', linewidth=0.6)
plt.plot(t,beta_hat,label='beta predicted', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-4000,4000)
plt.show()
