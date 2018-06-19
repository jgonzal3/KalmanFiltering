import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

def GAUSS_PY(SIG):
	X=(random.uniform(-3.0,3.0))*SIG
	return (X)	

def f1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,t):
	dx2dt = (0.0034*G*x2*x2*math.exp(-x1/22000.0)/(2.0*BETA))-G
	return (dx2dt)

	
t=[]
x1=[]
xs=[]
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
xddot_hat_ERR=[]
x2ddold =[]
res = []

TS=.1
XH=0
XDH=0
XDDH=0
SIGMA_NOISE=25.
X1=200000.
X1D=-6000.
BETA=5000.
XH=200025.
XDH=-6150.
TS=0.1
TF=30.
PHIS=0.
T=0.
S=0.
H = 0.001
G = 32.2

p_11 = SIGMA_NOISE*SIGMA_NOISE
p_22 = 20000
p_12 = 0
p_21 = p_12
P = np.matrix([[p_11, p_12],[p_21, p_22]])

I = np.matrix([[1, 0],[0, 1]])
HMAT = np.matrix([[1, 0]])
R = np.matrix([[SIGMA_NOISE**2]])

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
		F21  =-32.2*RHOH*XDH*XDH/(44000.0*BETA)
		F22  = RHOH*32.2*XDH/BETA	
		phi_11=1.0
		phi_12=TS
		phi_21=F21*TS
		phi_22=1.0+F22*TS
		PHI = np.matrix([[phi_11, phi_12],[phi_21, phi_22]])
			
		q_11=PHIS*TS*TS*TS/3.0
		q_12=PHIS*(TS*TS/2.0+F22*TS*TS*TS/3.0)
		q_21=q_12
		q_22=PHIS*(TS+F22*TS*TS+F22*F22*TS*TS*TS/3.0)
		Q = np.matrix([[q_11, q_12],[q_21, q_22]])
			
		M=PHI*P*PHI.transpose()+PHIS*Q
		K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + R))
		P=(I-K*HMAT)*M	
		XNOISE = GAUSS_PY(SIGMA_NOISE)
				T_ = 0
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
		else:
			# Use Euler integration to propagate XH and XDH	
			XDB= XDH+TS*f2(XH,XDH,T)
			XB = XH+TS*XDB
			
		XS = X1+XNOISE
		RES= XS-XB
		XH=XB+K[0,0]*RES
		XDH=XDB+K[1,0]*RES
		ERRX1=X1-XH
		SP11=math.sqrt(P[0,0])
		SP11N = -SP11
		ERRX1D=X1D-XDH
		SP22=math.sqrt(P[1,1])
		SP22N = -SP22
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

plt.figure(1)
plt.grid(True)
plt.plot(t,x1_hat_ERR,label='x-hat', linewidth=0.6)
plt.plot(t,sp11,label='sp11', linewidth=0.6)
plt.plot(t,sp11n,label='sp11n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-1000,1000)

plt.figure(2)
plt.grid(True)
plt.plot(t,x1dot_hat_ERR,label='xd-hat', linewidth=0.6)
plt.plot(t,sp22,label='sp22', linewidth=0.6)
plt.plot(t,sp22n,label='sp22n', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-250,250)

plt.figure(3)
plt.grid(True)
plt.plot(t,res,label='residuals', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,30)
plt.legend()
plt.ylim(-100,100)

plt.show()

	
