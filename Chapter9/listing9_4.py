import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random

# THETADD = (-G*cos(THETA) - 2*RD*THETAD)/R
# RDD = (R**2*THETAD**2 -G*R*SIN(THETA)/R
# States to solve runge kutta
# X1= R
# X1D = RD
# X2 = X1D
# X2D = RDD

# X3 = THETA
# X3D = THETAD
# X4 = X3D
# X4D = THETADD

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

# theta = atan2( YT - YR,XT - XR)
# r = sqrt((XT - XR)**2 + (YT - YR)**2)
# The states are: XT, XT_DOT, YT, YT_DOT

# The use of atan2, it is required because it needs to  considered the 
# sign to get the proper value Arc tangent of two  numbers. ATAN2(y,x) 
# returns the arc tangent of the two numbers x and y. It is similar to 
# calculating the  arc tangent of y/x,  except  that the signs of both 
# arguments are used to determine the quadrant of the result.

# The result is an angle expressed in radians. To convert from radians 
# to degrees, use the DEGREES  function. In  terms of the standard arc 
# tangent function, atan2 can be expressed as follows: 
#
#			 arctan(y/x) 		if x > 0
#			 arctan(y/x) + pi	if x < 0 and y >= 0
#atan2(y,x)= arctan(y/x) - pi	if x < 0 and y < 0
#			 +pi/2				if x = 0 and y > 0
#			 -pi/2				if x = 0 and y < 0
#			 undifined 			if x = 0 and y = 0
 

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

def f1(x1,x2,x3,x4,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,x3,x4,t):
	G =32.2
	#print (x1,x2,x3,x4)
	dx2dt = (x1**2*x4**2 - G*x1*np.sin(x3))/x1
	return (dx2dt)
	
def f3(x1,x2,x3,x4,t):
	dx3dt = x4
	return (dx3dt)
	
def f4(x1,x2,x3,x4,t):
	G =32.2
	dx4dt = (-G*np.cos(x3) - 2*x2*x4)/x1
	return (dx4dt)

def g1(x1,x2,x3,x4,t):
	dx1dt = x2
	return (dx1dt)

def g2(x1,x2,x3,x4,t):
	G =32.2
	#print (x1,x2,x3,x4)
	dx2dt = 0
	return (dx2dt)
	
def g3(x1,x2,x3,x4,t):
	dx3dt = x4
	return (dx3dt)
	
def g4(x1,x2,x3,x4,t):
	G =32.2
	dx4dt = -G
	return (dx4dt)
	
def project3(TP,TS,THP,THDP,RP,RDP,HP):
	T=0.;
	G=32.2;
	TH=THP;
	THD=THDP;
	R=RP;
	RD=RDP;
	H=HP;
	while (T<=(TS-0.0001)):
		THDD=(-G*R*math.cos(TH)-2.*THD*R*RD)/R**2
		RDD=(R*R*THD*THD-G*R*math.sin(TH))/R
		THD=THD+H*THDD
		TH=TH+H*THD
		RD=RD+H*RDD
		R=R+H*RD
		T=T+H
	RH=R
	RDH=RD
	THH=TH
	THDH=THD
	return [THH,THDH,RH,RDH]
	
	
ArrayT=[]
ArrayERRXT=[]
ArrayERRXTD=[]
ArrayERRYT=[]
ArrayERRYTD=[]
ArraySP11P=[]
ArraySP11PP=[]
ArraySP22P=[]
ArraySP22PP=[]
ArraySP33P=[]
ArraySP33PP=[]
ArraySP44P=[]
ArraySP44PP=[]

ArrayERRR=[]
ArrayERRRD=[]
ArrayERRTH=[]
ArrayERRTHD=[]
ArraySP11PR=[]
ArraySP11PPR=[]
ArraySP22PR=[]
ArraySP22PPR=[]
ArraySP33PR=[]
ArraySP33PPR=[]
ArraySP44PR=[]
ArraySP44PPR=[]


TS=1.
ORDER=4
PHIS=0.0
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
H =0.001
HP=0.001
XTH=XT+1000.
YTH=YT-1000.
XTDH=XTD-100.
YTDH=YTD+100.
SIGMA_NOISE_TH = 0.01
SIGMA_NOISE_R = 100.0
#TH=math.atan2((YT-YR),(XT-XR))
#R=math.sqrt((XT-XR)**2+(YT-YR)**2)
#THD=((XT-XR)*YTD-(YT-YR)*XTD)/R**2
#RD=((XT-XR)*XTD+(YT-YR)*YTD)/R
#THH=math.atan2((YTH-YR),(XTH-XR))
#RH=math.sqrt((XTH-XR)**2+(YTH-YR)**2)
#THDH=((XTH-XR)*YTDH-(YTH-YR)*XTDH)/RH**2
#RDH=((XTH-XR)*XTDH+(YTH-YR)*YTDH)/RH

'''
As the  object descends in altitude, there is more drag, and the object becomes
more  observable  from a  filtering point of view. However, because there is no
process noise the filter gains will go to zero. This means that the filter will
stop paying attention to the measurements (i.e., when the ballistic coefficient 
is most observable)  and hangoff error will result, as  can be seen in Fig. 8.5. 
Finally, process noise was added to the extended Kalman filter.
'''
ITERM =2
rk = 1
rkp = 0
PHIS = 0.0
''' THe marrix P is componse of the square of the errors in the estimate  which 
is error = (real value - estimated value)'''
PX=np.matrix([[(XT-XTH)**2,0.0],[0,(XTD-XTDH)**2]])
PY=np.matrix([[(YT-YTH)**2,0.0],[0,(YTD-YTDH)**2]])

'''Identity matrix'''
I = np.identity(2)
F = np.matrix([[0.0, 1.0],[0.0, 0.0]])
F2 = F*F
F3 = F*F*F
ITERM = 2
if ITERM == 2:
	PHI = I+TS*F
if ITERM == 3:
	PHI = I + TS*F + TS**2*F2/2
if ITERM == 4:
	PHI = I + TS*F +TS**2*F2/2 + TS**3*F3/6
	
Q = np.matrix([[TS**3/3,TS**2/2],[TS**2/2,TS]])	
# this lines are not correct because the only know value is the SIGTH and SIGR
# We need to use this it to recalculate every time RMAT is used:
# SIGYT = (SIN(THETH)*SIGR)**2+(RTH*COS(THETH)*SIGTH)**2
# SIGXT = (COS(THETH)*SIGR)**2+(RTH*SIN(THETH)*SIGTH)**2

HMAT=np.matrix([[1.0,0.0]])
dt = H
add_noise = 1

while (YT >= 0.0):
	if (rk == 0):
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
	else:
		# Propagate using Runge-Kutta procedure.
		k11 = dt*g1(XT,XTD,YT,YTD,T)
		k21 = dt*g2(XT,XTD,YT,YTD,T)
		k31 = dt*g3(XT,XTD,YT,YTD,T)
		k41 = dt*g4(XT,XTD,YT,YTD,T)
		k12 = dt*g1(XT+0.5*k11,XTD+0.5*k21,YT+0.5*k31,YTD+0.5*k41,T+0.5*dt)
		k22 = dt*g2(XT+0.5*k11,XTD+0.5*k21,YT+0.5*k31,YTD+0.5*k41,T+0.5*dt)
		k32 = dt*g3(XT+0.5*k11,XTD+0.5*k21,YT+0.5*k31,YTD+0.5*k41,T+0.5*dt)
		k42 = dt*g4(XT+0.5*k11,XTD+0.5*k21,YT+0.5*k31,YTD+0.5*k41,T+0.5*dt)
		k13 = dt*g1(XT+0.5*k12,XTD+0.5*k22,YT+0.5*k32,YTD+0.5*k42,T+0.5*dt)
		k23 = dt*g2(XT+0.5*k12,XTD+0.5*k22,YT+0.5*k32,YTD+0.5*k42,T+0.5*dt)
		k33 = dt*g3(XT+0.5*k12,XTD+0.5*k22,YT+0.5*k32,YTD+0.5*k42,T+0.5*dt)
		k43 = dt*g4(XT+0.5*k12,XTD+0.5*k22,YT+0.5*k32,YTD+0.5*k42,T+0.5*dt)
		k14 = dt*g1(XT+0.5*k13,XTD+0.5*k23,YT+0.5*k33,YTD+0.5*k43,T+0.5*dt)
		k24 = dt*g2(XT+0.5*k13,XTD+0.5*k23,YT+0.5*k33,YTD+0.5*k43,T+0.5*dt)
		k34 = dt*g3(XT+0.5*k13,XTD+0.5*k23,YT+0.5*k33,YTD+0.5*k43,T+0.5*dt)
		k44 = dt*g4(XT+0.5*k13,XTD+0.5*k23,YT+0.5*k33,YTD+0.5*k43,T+0.5*dt)
		XT  = XT  + (k11+2*k12+2*k13+k14)/6
		XTD = XTD + (k21+2*k22+2*k23+k24)/6
		YT  = YT  + (k31+2*k32+2*k33+k34)/6
		YTD = YTD + (k41+2*k42+2*k43+k44)/6
		T = T+dt
	S=S+H
	
# this step is here  so that the  discritezation of the  Riccati equations  takes 
# the correct values of the output. Note that H=0.0001 but TS=1. for this example
	if S>=(TS-.00001):
		S=0.
		# Calculate the theta hat and rt hat from xt hat and yt hat.
		# This has to be done because we only have information of the errors for
		# THETA and RT. If we had information about the variance of XT and YT
		# then, this step wouldn't be necessary.
		THETH=math.atan2((YTH-YR),(XTH-XR))
		RTH=math.sqrt((XTH-XR)**2+(YTH-YR)**2)
		# then, using this values and the fact that we only have the variance of the
		# error in theta and r, we need to calculate the variance of xt and yt using
		# the formulas provided
		SIGMA_NOISE_YT = (math.sin(THETH)*SIGMA_NOISE_R)**2+(RTH*math.cos(THETH)*SIGMA_NOISE_TH)**2
		SIGMA_NOISE_XT = (math.cos(THETH)*SIGMA_NOISE_R)**2+(RTH*math.sin(THETH)*SIGMA_NOISE_TH)**2
		# With these two values, we can estimate the R matrix for the Riccati equations
		RMATX=np.matrix([[SIGMA_NOISE_XT]])
		RMATY=np.matrix([[SIGMA_NOISE_YT]])
		MX=PHI*PX*PHI.transpose() + PHIS*Q
		MY=PHI*PY*PHI.transpose() + PHIS*Q
		KX = MX*HMAT.transpose()*(inv(HMAT*MX*HMAT.transpose() + RMATX))
		KY = MY*HMAT.transpose()*(inv(HMAT*MY*HMAT.transpose() + RMATY))
		PX=(I-KX*HMAT)*MX
		PY=(I-KY*HMAT)*MY
		THET=math.atan2((YT-YR),(XT-XR))
		RT=math.sqrt((XT-XR)**2+(YT-YR)**2)
		# If we need to compare the solution with the solution given in the book using MATLAB code.
		if (add_noise == 1):
			THETNOISE = GAUSS_PY(SIGMA_NOISE_TH)
			RTNOISE = GAUSS_PY(SIGMA_NOISE_R)
		else:
			THETNOISE = 0.0
			RTNOISE = 0.0
		THETMEAS=THET+THETNOISE
		RTMEAS=RT+RTNOISE
		XTMEAS=RTMEAS*math.cos(THETMEAS)+XR
		RES1=XTMEAS-XTH-TS*XTDH
		# We must be able to propagate the XT and XTD from the  present sampling time  
		# to the next sampling time. This could be done exactly  with the fundamental 
		# matrix when we were working in the Cartesian system because the fundamental 
		# matrix was exact.
		# XTH = XTH + Vx*T
		XTH=XTH+TS*XTDH+KX[0,0]*RES1
		XTDH=XTDH+KX[1,0]*RES1
		YTMEAS=RTMEAS*math.sin(THETMEAS)+YR
		RES2=YTMEAS-YTH-TS*YTDH+.5*TS*TS*G
		# We must be able to propagate the YT and YTD  from the present sampling time to 
		# next sampling time.This  could be done  exactly  with the  fundamental  matrix
		# when we were working in the Cartesian   system  because the fundamental matrix
		# was exact. 
		# YTH = YTH + Vy*T - 1/2*G*T^2
		YTH=YTH+TS*YTDH-.5*TS*TS*G+KY[0,0]*RES2
		YTDH=YTDH-TS*G+KY[1,0]*RES2

		ERRXT=XT-XTH
		SP11P=np.sqrt(PX[0,0])
		ERRXTD=XTD-XTDH
		SP22P=np.sqrt(PX[1,1])
		ERRYT=YT-YTH
		SP33P=np.sqrt(PY[0,0])
		ERRYTD=YTD-YTDH
		SP44P=np.sqrt(PY[1,1])
		SP11PP=-SP11P
		SP22PP=-SP22P
		SP33PP=-SP33P
		SP44PP=-SP44P
		ArrayT.append(T)
		ArrayERRXT.append(ERRXT)
		ArrayERRXTD.append(ERRXTD)
		ArrayERRYT.append(ERRYT)
		ArrayERRYTD.append(ERRYTD)
		ArraySP11P.append(SP11P)
		ArraySP11PP.append(SP11PP)
		ArraySP22P.append(SP22P)
		ArraySP22PP.append(SP22PP)
		ArraySP33P.append(SP33P)
		ArraySP33PP.append(SP33PP)
		ArraySP44P.append(SP44P)
		ArraySP44PP.append(SP44PP)
'''
Adding process noise increases errors in estimate of altitude
but reduces the hangoff error 
'''
plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayERRXT,label='xt', linewidth=0.6)
plt.plot(ArrayT,ArraySP11P,label='sp11p', linewidth=0.6)
plt.plot(ArrayT,ArraySP11PP,label='sp11pp', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Downrange (Ft)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-200,200)

plt.figure(2)
plt.grid(True)
plt.plot(ArrayT,ArrayERRXTD,label='xdot', linewidth=0.6)
plt.plot(ArrayT,ArraySP22P,label='sp22p', linewidth=0.6)
plt.plot(ArrayT,ArraySP22PP,label='sp22pp', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Downrange Velocity (Ft/Sec)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-20,20)

plt.figure(3)
plt.grid(True)
plt.plot(ArrayT,ArrayERRYT,label='yt', linewidth=0.6)
plt.plot(ArrayT,ArraySP33P,label='sp33p', linewidth=0.6)
plt.plot(ArrayT,ArraySP33PP,label='sp33pp', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-600,600)

plt.figure(4)
plt.grid(True)
plt.plot(ArrayT,ArrayERRYTD,label='Downrange', linewidth=0.6)
plt.plot(ArrayT,ArraySP44P,label='sp44p', linewidth=0.6)
plt.plot(ArrayT,ArraySP44PP,label='sp44pp', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude Velocity (Ft/Sec)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-20, 20)
plt.show()

		
