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

def g1(x1,x2,x3,x4,t):
	dx1dt = x2
	return (dx1dt)

def g2(x1,x2,x3,x4,t):
	G =32.2
	#print (x1,x2,x3,x4)
	dx2dt = (x1**2*x4**2 - G*x1*np.sin(x3))/x1
	return (dx2dt)
	
def g3(x1,x2,x3,x4,t):
	dx3dt = x4
	return (dx3dt)
	
def g4(x1,x2,x3,x4,t):
	G =32.2
	dx4dt = (-G*np.cos(x3) - 2*x2*x4)/x1
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
	
def projectRK(TP,TS,THP,THDP,RP,RDP,HP):
	G=32.2;
	T = TP
	T_ = 0
	THH=THP;
	THDH=THDP;
	RH=RP;
	RDH=RDP;
	dt=HP;
	while (T_< TS):
		k11 = dt*g1(RH,RDH,THH,THDH,T)
		k21 = dt*g2(RH,RDH,THH,THDH,T)
		k31 = dt*g3(RH,RDH,THH,THDH,T)
		k41 = dt*g4(RH,RDH,THH,THDH,T)
		k12 = dt*g1(RH+0.5*k11,RDH+0.5*k21,THH+0.5*k31,THDH+0.5*k41,T+0.5*dt)
		k22 = dt*g2(RH+0.5*k11,RDH+0.5*k21,THH+0.5*k31,THDH+0.5*k41,T+0.5*dt)
		k32 = dt*g3(RH+0.5*k11,RDH+0.5*k21,THH+0.5*k31,THDH+0.5*k41,T+0.5*dt)
		k42 = dt*g4(RH+0.5*k11,RDH+0.5*k21,THH+0.5*k31,THDH+0.5*k41,T+0.5*dt)
		k13 = dt*g1(RH+0.5*k12,RDH+0.5*k22,THH+0.5*k32,THDH+0.5*k42,T+0.5*dt)
		k23 = dt*g2(RH+0.5*k12,RDH+0.5*k22,THH+0.5*k32,THDH+0.5*k42,T+0.5*dt)
		k33 = dt*g3(RH+0.5*k12,RDH+0.5*k22,THH+0.5*k32,THDH+0.5*k42,T+0.5*dt)
		k43 = dt*g4(RH+0.5*k12,RDH+0.5*k22,THH+0.5*k32,THDH+0.5*k42,T+0.5*dt)
		k14 = dt*g1(RH+0.5*k13,RDH+0.5*k23,THH+0.5*k33,THDH+0.5*k43,T+0.5*dt)
		k24 = dt*g2(RH+0.5*k13,RDH+0.5*k23,THH+0.5*k33,THDH+0.5*k43,T+0.5*dt)
		k34 = dt*g3(RH+0.5*k13,RDH+0.5*k23,THH+0.5*k33,THDH+0.5*k43,T+0.5*dt)
		k44 = dt*g4(RH+0.5*k13,RDH+0.5*k23,THH+0.5*k33,THDH+0.5*k43,T+0.5*dt)
		RH   = RH + (k11+2*k12+2*k13+k14)/6
		RDH  = RDH + (k21+2*k22+2*k23+k24)/6
		THH  = THH + (k31+2*k32+2*k33+k34)/6
		THDH = THDH + (k41+2*k42+2*k43+k44)/6
		T_ = T_+dt
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
SIGMA_NOISE_THETA = 0.01
SIGMA_NOISE_R = 100.0
TH=math.atan2((YT-YR),(XT-XR))
R=math.sqrt((XT-XR)**2+(YT-YR)**2)
THD=((XT-XR)*YTD-(YT-YR)*XTD)/R**2
RD=((XT-XR)*XTD+(YT-YR)*YTD)/R
THH=math.atan2((YTH-YR),(XTH-XR))
RH=math.sqrt((XTH-XR)**2+(YTH-YR)**2)
THDH=((XTH-XR)*YTDH-(YTH-YR)*XTDH)/RH**2
RDH=((XTH-XR)*XTDH+(YTH-YR)*YTDH)/RH

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
rkp = 1
''' THe marrix P is componse of the square of the errors in the estimate  which 
is error = (real value - estimated value)'''
P=np.matrix([[(TH-THH)**2,0.0,0.0,0.0],[0,(THD-THDH)**2,0.0,0.0],[0.0,0.0,(R-RH)**2,0.0],[0.0,0.0,0.0,(RD-RDH)**2]])
'''Identity matrix'''
I = np.identity(4)
Q   =np.matrix([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])	
A   =np.matrix([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
RMAT=np.matrix([[SIGMA_NOISE_THETA**2,0.0],[0.0,SIGMA_NOISE_R**2]])
HMAT=np.matrix([[1.0,0,0,0],[0,0,1.0,0]])
dt = H
add_noise = 1

while (YT >= 0.0):
	if (rk == 0):
		THOLD=TH
		THDOLD=THD
		ROLD=R
		RDOLD=RD
		THDD=(-G*R*math.cos(TH)-2.*THD*R*RD)/R**2
		RDD=(R*R*THD*THD-G*R*math.sin(TH))/R
		TH=TH+H*THD
		THD=THD+H*THDD
		R=R+H*RD
		RD=RD+H*RDD
		T=T+H
		THDD=(-G*R*math.cos(TH)-2.*THD*R*RD)/R**2
		RDD=(R*R*THD*THD-G*R*math.sin(TH))/R
		TH=.5*(THOLD+TH+H*THD)
		THD=.5*(THDOLD+THD+H*THDD)
		R=.5*(ROLD+R+H*RD)
		RD=.5*(RDOLD+RD+H*RDD)
	else:
		k11 = dt*g1(R,RD,TH,THD,T)
		k21 = dt*g2(R,RD,TH,THD,T)
		k31 = dt*g3(R,RD,TH,THD,T)
		k41 = dt*g4(R,RD,TH,THD,T)
		k12 = dt*g1(R+0.5*k11,RD+0.5*k21,TH+0.5*k31,THD+0.5*k41,T+0.5*dt)
		k22 = dt*g2(R+0.5*k11,RD+0.5*k21,TH+0.5*k31,THD+0.5*k41,T+0.5*dt)
		k32 = dt*g3(R+0.5*k11,RD+0.5*k21,TH+0.5*k31,THD+0.5*k41,T+0.5*dt)
		k42 = dt*g4(R+0.5*k11,RD+0.5*k21,TH+0.5*k31,THD+0.5*k41,T+0.5*dt)
		k13 = dt*g1(R+0.5*k12,RD+0.5*k22,TH+0.5*k32,THD+0.5*k42,T+0.5*dt)
		k23 = dt*g2(R+0.5*k12,RD+0.5*k22,TH+0.5*k32,THD+0.5*k42,T+0.5*dt)
		k33 = dt*g3(R+0.5*k12,RD+0.5*k22,TH+0.5*k32,THD+0.5*k42,T+0.5*dt)
		k43 = dt*g4(R+0.5*k12,RD+0.5*k22,TH+0.5*k32,THD+0.5*k42,T+0.5*dt)
		k14 = dt*g1(R+0.5*k13,RD+0.5*k23,TH+0.5*k33,THD+0.5*k43,T+0.5*dt)
		k24 = dt*g2(R+0.5*k13,RD+0.5*k23,TH+0.5*k33,THD+0.5*k43,T+0.5*dt)
		k34 = dt*g3(R+0.5*k13,RD+0.5*k23,TH+0.5*k33,THD+0.5*k43,T+0.5*dt)
		k44 = dt*g4(R+0.5*k13,RD+0.5*k23,TH+0.5*k33,THD+0.5*k43,T+0.5*dt)
		R   = R   + (k11+2*k12+2*k13+k14)/6
		RD  = RD  + (k21+2*k22+2*k23+k24)/6
		TH  = TH  + (k31+2*k32+2*k33+k34)/6
		THD = THD + (k41+2*k42+2*k43+k44)/6
		T = T+dt
	S=S+H
	
	# this step is here so that the discritezation of the Riccati equations takes the correct
	# values of the output. Note that H=0.0001 but TS=0.1.
	if S>=(TS-.00001):
		S=0.;
		f21 = G*math.sin(THH)/RH
		f22 = -2.0*RDH/RH
		f23 = (G*math.cos(THH) + 2.0*THDH*RDH)/RH**2
		f24 = -2.0*THDH/RH
		f41 = -G*math.cos(THH)
		f42 = 2.0*RH*THDH
		f43 = THDH**2
		f44 = 0.0
		F = TS*np.matrix([[0.0, 1.0, 0.0, 0.0],[f21,f22,f23,f24],[0.0, 0.0, 0.0, 1.0], [f41,f42,f43,f44]])
		F2 = F*F
		F3 = F*F*F
		ITERM = 2
		
		if ITERM == 2:
			PHI = I+TS*F
		if ITERM == 3:
			PHI = I + TS*F + TS**2*F2/2
		if ITERM == 4:
			PHI = I + TS*F +TS**2*F2/2 + TS**3*F3/6

		M=PHI*P*PHI.transpose() + Q
		K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
		P=(I-K*HMAT)*M
		
# We must be able to propagate the states from the present sampling time to the
# next sampling time.This  could be done  exactly  with the  fundamental matrix
# when we were working in the Cartesian  system  because the fundamental matrix
# was exact. In the polar system the fundamental matrix is approximate,  and so
# it is best to numerically integrate the non-linear differential equations for
# a sampling interval to get the projected states.

		if (rkp == 0):
			[THB,THDB,RB,RDB]=project3(T,TS,THH,THDH,RH,RDH,HP)
		else:
			[THB,THDB,RB,RDB]=projectRK(T,TS,THH,THDH,RH,RDH,HP)
			
		if (add_noise == 1):
			THNOISE = GAUSS_PY(SIGMA_NOISE_THETA)
			RNOISE = GAUSS_PY(SIGMA_NOISE_R)
		else:
			THNOISE = 0.0
			RNOISE = 0.0
		RES1=TH+THNOISE-THB
		RES2=R+RNOISE- RB
		THH  = THB  + K[0,0]*RES1 + K[0,1]*RES2
		THDH = THDB + K[1,0]*RES1 + K[1,1]*RES2
		RH   = RB   + K[2,0]*RES1 + K[2,1]*RES2
		RDH  = RDB  + K[3,0]*RES1 + K[3,1]*RES2
		ERRTH=TH-THH
		SP11PR=math.sqrt(P[0,0])
		ERRTHD=THD-THDH
		SP22PR=math.sqrt(P[1,1])
		ERRR=R-RH
		SP33PR=math.sqrt(P[2,2])
		ERRRD=RD-RDH
		SP44PR=math.sqrt(P[3,3])
		SP11PPR=-SP11PR
		SP22PPR=-SP22PR
		SP33PPR=-SP33PR
		SP44PPR=-SP44PR
		XT=R*math.cos(TH)+XR
		YT=R*math.sin(TH)+YR
		XTD=RD*math.cos(TH)-R*THD*math.sin(TH)
		YTD=RD*math.sin(TH)+R*THD*math.cos(TH)
		XTH=RH*math.cos(THH)+XR
		YTH=RH*math.sin(THH)+YR
		XTDH=RDH*math.cos(THH)-RH*THDH*math.sin(THH)
		YTDH=RDH*math.sin(THH)+RH*THDH*math.cos(THH)
		
# If we define the transformation matrix A relating the polar total differentials 
# to the Cartesian total differentials, we get

		A11 =-RH*math.sin(THH)
		A12 = 0.0
		A13 = math.cos(THH)
		A14 = 0.0
		A21 =-RDH*math.sin(THH)-RH*THDH*math.cos(THH)
		A22 =-RH*math.sin(THH)
		A23 =-THDH*math.sin(THH)
		A24 = math.cos(THH)
		A31 = RH*math.cos(THH)
		A32 = 0.0
		A33 = math.sin(THH)
		A34 = 0.0
		A41 = RDH*math.cos(THH)-RH*THDH*math.sin(THH)
		A42 = RH*math.cos(THH)
		A43 = THDH*math.cos(THH)
		A44 = math.sin(THH)
		
		A = np.matrix([[A11,A12,A13,A14],[A21,A22,A23,A24],[A31,A32,A33,A34],[A41,A42,A43,A44]])
		PNEW=A*P*A.transpose();
		'''
		hangoff error are error does not go to zero.
		'''
		ERRXT=XT-XTH
		SP11P=np.sqrt(PNEW[0,0])
		ERRXTD=XTD-XTDH
		SP22P=np.sqrt(PNEW[1,1])
		ERRYT=YT-YTH
		SP33P=np.sqrt(PNEW[2,2])
		ERRYTD=YTD-YTDH
		SP44P=np.sqrt(PNEW[3,3])
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
		ArrayERRR.append(ERRR)
		ArrayERRRD.append(ERRRD)
		ArrayERRTH.append(ERRTH)
		ArrayERRTHD.append(ERRTHD)
		ArraySP11PR.append(SP11PR)
		ArraySP11PPR.append(SP11PPR)
		ArraySP22PR.append(SP22PR)
		ArraySP22PPR.append(SP22PPR)
		ArraySP33PR.append(SP33PR)
		ArraySP33PPR.append(SP33PPR)
		ArraySP44PR.append(SP44PR)
		ArraySP44PPR.append(SP44PPR)
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

plt.figure(5)
plt.grid(True)
plt.plot(ArrayT,ArrayERRR,label='r', linewidth=0.6)
plt.plot(ArrayT,ArraySP33PR,label='sp33pr', linewidth=0.6)
plt.plot(ArrayT,ArraySP33PPR,label='sp33ppr', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of range (Ft)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-100,100)

plt.figure(6)
plt.grid(True)
plt.plot(ArrayT,ArrayERRRD,label='rdot', linewidth=0.6)
plt.plot(ArrayT,ArraySP44PR,label='sp44pr', linewidth=0.6)
plt.plot(ArrayT,ArraySP44PPR,label='sp44ppr', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of range velocity (Ft/Sec)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-70,70)

plt.figure(7)
plt.grid(True)
plt.plot(ArrayT,ArrayERRTH,label='theta', linewidth=0.6)
plt.plot(ArrayT,ArraySP11PR,label='sp33pr', linewidth=0.6)
plt.plot(ArrayT,ArraySP11PPR,label='sp33ppr', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of theta (Rads)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-0.02,0.02)

plt.figure(8)
plt.grid(True)
plt.plot(ArrayT,ArrayERRTHD,label='thetad', linewidth=0.6)
plt.plot(ArrayT,ArraySP22PR,label='sp44pr', linewidth=0.6)
plt.plot(ArrayT,ArraySP22PPR,label='sp44ppr', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of theta rate (Rads/Sec)')
plt.xlim(0,140)
plt.legend()
plt.ylim(-0.001, 0.001)

plt.show()
		
