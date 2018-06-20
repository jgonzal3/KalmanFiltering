import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
import random

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

def f1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,t):
	dx2dt = (0.0034*G*x2*x2*math.exp(-x1/22000.0)/(2.0*BETAH))-G
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
	return [XH,XDH]

t=[]
x1=[]
xs=[]
res =[]
x1_hat=[]
x1dot=[]
x1dot_hat=[]
xs=[]
xddot_hat=[]
x1_hat_ERR=[]
sp11=[]
sp11n=[]
x1dot_hat_ERR=[]
sp22=[]
sp22n=[]
beta_hat_ERR = []
sp33=[]
sp33n=[]
xddot_hat_ERR=[]
x2ddold =[]
k1 =[]
k2 =[]

TS=.1
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=25.
X1=200000.
X1D=-6000.
BETA=500.
BETAH=800.
XH=200025.
XDH=-6150.
TS=0.1
TF=30.

'''
As the  object descends in altitude, there is more drag, and the object becomes
more  observable  from a  filtering point of view. However, because there is no
process noise the filter gains will go to zero. This means that the filter will
stop paying attention to the measurements (i.e., when the ballistic coefficient 
is most observable)  and hangoff error will result, as  can be seen in Fig. 8.5. 
Finally, process noise was added to the extended Kalman filter.
'''
PHIS=3000.
T=0.
S=0.
H = 0.001
G = 32.2
rk =0

p_11 = SIGMA_NOISE*SIGMA_NOISE	# Error in the altitude
p_22 = 150*150					# Error in the velocity
p_33 = 300*300					# Error in BETA parameter

P = np.matrix([[p_11, 0, 0],[0, p_22, 0], [0, 0, p_33]])
I = np.matrix([[1,0,0],[0,1,0], [0,0,1]])

HMAT = np.matrix([[1, 0, 0]])
R = np.matrix([[SIGMA_NOISE**2]])

H = TS
dt = H
while (T <= 30.0):
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
	S = S+H;
	#print (T, S)
	# this step is here so that the discritezation of the Riccati equations takes the correct
	# values of the output. Note that H=0.0001 but TS=0.1.
	if S>=(TS-.00001):
		S=0.;
		RHOH = 0.0034*math.exp(-XH/22000.0)
		F21  =-32.2*RHOH*XDH*XDH/(44000.0*BETAH)
		F22  = RHOH*32.2*XDH/BETAH
		F23  =-32.2*RHOH*XDH*XDH/(2.0*BETAH*BETAH)

		
		F = np.matrix([[0, 1, 0],[F21, F22, F23],[0, 0, 0]])
		#PHIK = I + TS*F
		F2 = F*F
		F3 = F*F*F
		F4= F*F*F*F
		
		ITERM =4
		if ITERM == 2:
			PHI = I + TS*F
		if ITERM == 3:
			PHI = I + TS*F + F2*(TS/2)**2
		if ITERM == 4:
			PHI = I + TS*F + F2*(TS/2)**2 + F3*(TS/6)**3
		if ITERM == 5:
			PHI = I + TS*F + F2*(TS/2)**2 + F3*(TS/6)**3 + F4*(TS/24)**4
			
		q_22 = F23*F23*TS*TS*TS/3.0
		q_23 = F23*TS*TS/2.0
		q_32 = q_23
		Q = np.matrix([[0, 0, 0],[0, q_22, q_23],[0, q_32, TS]])
			
		M=PHI*P*PHI.transpose()+PHIS*Q
		K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + R))
		P=(I-K*HMAT)*M	
		XNOISE = GAUSS_PY(SIGMA_NOISE)
		
		if (rk == 1):
			# Use integration Runge-Kutta to propagate XH and XDH
			T_ = 0
			while (T_< TS):
				k11 = dt*f1(XH,XDH,T_)
				k21 = dt*f2(XH,XDH,T_)
				k12 = dt*f1(XH+0.5*k11,XDH+0.5*k21,T_+0.5*dt)
				k22 = dt*f2(XH+0.5*k11,XDH+0.5*k21,T_+0.5*dt)
				k13 = dt*f1(XH+0.5*k12,XDH+0.5*k22,T_+0.5*dt)
				k23 = dt*f2(XH+0.5*k12,XDH+0.5*k22,T_+0.5*dt)
				k14 = dt*f1(XH+k13,XDH+k23,T_+dt)
				k24 = dt*f2(XH+k13,XDH+k23,T_+dt)
				XH = XH + (k11+2*k12+2*k13+k14)/6
				XDH = XDH + (k21+2*k22+2*k23+k24)/6
				XDB = XDH
				XB = XH
				T_ = T_+dt
		elif (rk == 2):
			# Use Euler integration to propagate XH and XDH	
			XDB= XDH+TS*f2(XH,XDH,T)
			XB = XH+TS*XDB
		else:
			HP = 0.001
			[XB, XDB] = PROJECT(TS,XH,XDH,BETAH,HP)
			 
			
		XS = X1+XNOISE
		RES= XS-XB
		XH=XB+K[0,0]*RES
		k1.append(K[0,0])
		XDH=XDB+K[1,0]*RES
		k2.append(K[1,0])
		BETAH = BETAH +K[2,0]*RES
		ERRX1=X1-XH
		SP11=math.sqrt(P[0,0])
		SP11N = -SP11
		ERRX1D=X1D-XDH
		SP22=math.sqrt(P[1,1])
		SP22N = -SP22
		ERRBETA=BETA-BETAH
		'''
		hangoff error are error does not go to zero.
		'''
		SP33=math.sqrt(P[2,2])
		SP33N = -SP33
		t.append(T)
		x1.append(X1)
		xs.append(XS)
		res.append(RES)
		x1_hat.append(XH)
		x1dot.append(X1D)
		x1dot_hat.append(XDH)
		x1_hat_ERR.append(ERRX1)
		sp11.append(SP11)
		sp11n.append(SP11N)
		x1dot_hat_ERR.append(ERRX1D)
		sp22.append(SP22)
		sp22n.append(SP22N)
		beta_hat_ERR.append(ERRBETA)
		sp33.append(SP33)
		sp33n.append(SP33N)
		
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
plt.plot(t,beta_hat_ERR,label='beta-hat', linewidth=0.6)
plt.plot(t,sp33,label='sp33', linewidth=0.6)
plt.plot(t,sp33n,label='sp33n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-400,400)
plt.show()
