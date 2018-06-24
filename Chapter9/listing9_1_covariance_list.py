import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Because our states have been chosen to be Cartesian, the radar 
# measurements r* and theta*  will  automatically be  non-linear 
# functions  of  those  states.  Therefore,  we  must  write the 
# linearised measurement equation.

# Because the system and measurement equations are  non-linear, a 
# first-order  approximation  is used  in the  continuous Riccati 
# equations for the systems dynamics matrix F and the measurement
# matrix H. The matrices are related to the non-linear system and
# measurement equations according to
#
# F = @f(x)/@x|
#  			  |x = x_hat
#
# H = @h(x)/@x |
#			   |x = x_hat
# where the symbol @ means the partial derivative
# 
# Because the dynamics of  the given  problem is lineal, xddot = 0 and 
# yddot = -g,  the  linearisation  of the states in F is not required. 
# However, as indicated  in the text, the  measurements in r and theta
# are non linear, and therefore is will be required to use the formula
# to linearise the variables.	

# theta = atan( YT - YR/XT - XR)
# r = sqrt((XT - XR)**2 + (YT - YR)**2)
# The states are: XT, XT_DOT, YT, YT_DOT

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
	dx2dt = 0.0
	return (dx2dt)

def f3(y1,y2,t):
	dy1dt = y2
	return (dy1dt)

def f4(y1,y2,t):
	dy2dt = -G
	return (dy2dt)
	
t=[]
x=[]
y=[]

xt_hat=[]
yt_hat=[]
xtdot=[]
xtdot_hat=[]
ytdot_hat=[]
ytdot_hat=[]
xt_hat_ERR=[]
yt_hat_ERR=[]
xtdot_hat_ERR=[]
ytdot_hat_ERR=[]

sp44=[]
sp44n=[]
sp11=[]
sp11n=[]
sp22=[]
sp22n=[]
sp33=[]
sp33n=[]


TS=1.
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
XTH=XT+1000.
XTDH=XTD-100.
YTH=YT-1000.
YTDH=YTD+100.
SIGMA_NOISE_THETA = 0.01
SIGMA_NOISE_R = 100.0

'''
As the  object descends in altitude, there is more drag, and the object becomes
more  observable  from a  filtering point of view. However, because there is no
process noise the filter gains will go to zero. This means that the filter will
stop paying attention to the measurements (i.e., when the ballistic coefficient 
is most observable)  and hangoff error will result, as  can be seen in Fig. 8.5. 
Finally, process noise was added to the extended Kalman filter.
'''
rk = 0
ITERM =3
H=0.001

p_11 = 1000.0**2				# Error in the downrange
p_22 = 100.0*22		 			# Error in the downrange velocity
p_33 = 1000.0**2				# Error in altitude
p_44 = 100.0**2					# Error in altitude velocity

P = np.matrix([[p_11, 0, 0, 0],[0, p_22, 0, 0], [0, 0, p_33, 0], [0, 0, 0, p_44]])
I = np.matrix([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]])
R = np.matrix([[SIGMA_NOISE_THETA**2, 0], [0, SIGMA_NOISE_R**2]])
Q = np.matrix([[TS**3/3,TS**2/2,0,0],[TS**2/2,TS,0,0],[0,0,TS**3/3,TS**2/2],[0,0,TS**2/2,TS]])
F = np.matrix([[0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1], [0, 0, 0, 0]])
F2 = F*F
if ITERM == 2:
	PHI = I + TS*F
if ITERM == 3:
	PHI = I + TS*F + F2*(TS/2)**2

for PHIS in [0.0, 10.0, 50.0, 10.0]:
	XTD=VT*math.cos(GAMDEG/57.3)
	YTD=VT*math.sin(GAMDEG/57.3)
	XTH=XT+1000.
	XTDH=XTD-100.
	YTH=YT-1000.
	YTDH=YTD+100.
	H = 0.001
	T=0.
	S=0.
	XT=0.
	YT=0.
	p_11 = 1000.0**2				# Error in the downrange
	p_22 = 100.0*22		 			# Error in the downrange velocity
	p_33 = 1000.0**2				# Error in altitude
	p_44 = 100.0**2					# Error in altitude velocity

	P = np.matrix([[p_11, 0, 0, 0],[0, p_22, 0, 0], [0, 0, p_33, 0], [0, 0, 0, p_44]])
	while (YT >= 0.0):
		if (rk == 1):
			k11 = H*f1(XT,XTD,T)
			k21 = H*f2(XT,XTD,T)
			k12 = H*f1(XT+0.5*k11,XTD+0.5*k21,T+0.5*H)
			k22 = H*f2(XT+0.5*k11,XTD+0.5*k21,T+0.5*H)
			k13 = H*f1(XT+0.5*k12,XTD+0.5*k22,T+0.5*H)
			k23 = H*f2(XT+0.5*k12,XTD+0.5*k22,T+0.5*H)
			k14 = H*f1(XT+k13,XTD+k23,T+H)
			k24 = H*f2(XT+k13,XTD+k23,T+H)
			XT  = XT + (k11+2*k12+2*k13+k14)/6
			XTD = XTD + (k21+2*k22+2*k23+k24)/6
			k11 = H*f3(YT,YTD,T)
			k21 = H*f4(YT,YTD,T)
			k12 = H*f3(YT+0.5*k11,YTD+0.5*k21,T+0.5*H)
			k22 = H*f4(YT+0.5*k11,YTD+0.5*k21,T+0.5*H)
			k13 = H*f3(YT+0.5*k12,YTD+0.5*k22,T+0.5*H)
			k23 = H*f4(YT+0.5*k12,YTD+0.5*k22,T+0.5*H)
			k14 = H*f3(YT+k13,YTD+k23,T+H)
			k24 = H*f4(YT+k13,YTD+k23,T+H)
			YT  = YT + (k11+2*k12+2*k13+k14)/6
			YTD = YTD + (k21+2*k22+2*k23+k24)/6
			T = T+H
		else:
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
		S = S+H;
		# this step is here so that the discritezation of the Riccati equations takes the correct
		# values of the output. Note that H=0.0001 but TS=0.1.
		if S>=(TS-.00001):
			dt = 0.001
			S=0.;
			if (rk == 1):
				# Use integration Runge-Kutta to propagate XH and XDH
				T_ = 0
				while (T_< TS):
					k11 = dt*f1(XTH,XTDH,T_)
					k21 = dt*f2(XTH,XTDH,T_)
					k12 = dt*f1(XTH+0.5*k11,XTDH+0.5*k21,T_+0.5*dt)
					k22 = dt*f2(XTH+0.5*k11,XTDH+0.5*k21,T_+0.5*dt)
					k13 = dt*f1(XTH+0.5*k12,XTDH+0.5*k22,T_+0.5*dt)
					k23 = dt*f2(XTH+0.5*k12,XTDH+0.5*k22,T_+0.5*dt)
					k14 = dt*f1(XTH+k13,XTDH+k23,T_+dt)
					k24 = dt*f2(XTH+k13,XTDH+k23,T_+dt)
					XTH = XTH + (k11+2*k12+2*k13+k14)/6
					XTDH = XTDH + (k21+2*k22+2*k23+k24)/6
					XTDB = XTDH
					XTB = XTH
					k11 = dt*f3(YTH,YTDH,T_)
					k21 = dt*f4(YTH,YTDH,T_)
					k12 = dt*f3(YTH+0.5*k11,YTDH+0.5*k21,T_+0.5*dt)
					k22 = dt*f4(YTH+0.5*k11,YTDH+0.5*k21,T_+0.5*dt)
					k13 = dt*f3(YTH+0.5*k12,YTDH+0.5*k22,T_+0.5*dt)
					k23 = dt*f4(YTH+0.5*k12,YTDH+0.5*k22,T_+0.5*dt)
					k14 = dt*f3(YTH+k13,YTDH+k23,T_+dt)
					k24 = dt*f4(YTH+k13,YTDH+k23,T_+dt)
					YTH = YTH + (k11+2*k12+2*k13+k14)/6
					YTDH = YTDH + (k21+2*k22+2*k23+k24)/6
					YTDB = YTDH
					YTB = YTH
					T_ = T_+dt
			else:
				# Simple Euler method to propagate the estimates
				XTB=XTH+TS*XTDH
				XTDB=XTDH
				YTB=YTH+TS*YTDH-.5*G*TS*TS
				YTDB=YTDH-G*TS
				
			RTB=math.sqrt((XTB-XR)**2+(YTB-YR)**2)
			THETB=math.atan((YTB-YR)/(XTB-XR))
			h11 = -(YTB - YR)/RTB**2
			h12 = 0.0
			h13 = (XTB - XR)/RTB**2
			h14 = 0.0
			h21 = (XTB - XR)/RTB
			h22 = 0.0
			h23 = (YTB - YR)/RTB
			h24 = 0.0
			HMAT = np.matrix([[h11,h12,h13,h14], [h21, h22,h23, h24]])
			M=PHI*P*PHI.transpose() +PHIS*Q
			K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + R))
			P=(I-K*HMAT)*M	
			THET=math.atan((YT-YR)/(XT-XR))
			RT=math.sqrt((XT-XR)**2+(YT-YR)**2)
			THETMEAS=THET+GAUSS_PY(SIGMA_NOISE_THETA)
			RTMEAS=RT+GAUSS_PY(SIGMA_NOISE_R)
			RES1=THETMEAS-THETB
			RES2=RTMEAS-RTB
			XTH=XTB + K[0,0]*RES1+K[0,1]*RES2
			XTDH=XTDB+K[1,0]*RES1+K[1,1]*RES2
			YTH=YTB + K[2,0]*RES1+K[2,1]*RES2
			YTDH=YTDB+K[3,0]*RES1+K[3,1]*RES2
			ERRXD=XTD-XTDH
			SP22=math.sqrt(P[1,1])
			t.append(T)
			sp22.append(SP22)
			'''
			hangoff error are error does not go to zero.
			'''
	plt.plot(t,sp22,label='PHIS='+str(PHIS), linewidth=0.6)
	t = []
	sp22 = []

'''
Adding process noise increases errors in estimate of altitude
but reduces the hangoff error 
'''
plt.figure(1)
plt.grid(True)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Downrange Velocity (Ft/Sec)')
plt.xlim(0,140)
plt.legend()
plt.ylim(0,90)
plt.show()
