import math
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

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

# XDD = 0
# YDD = -G
# States to solve runge kutta
# X1= X
# X1D = XD
# X2 = X1D
# X2D = XDD

# X3 = Y
# X3D = YD
# X4 = X3D
# X4D = YDD

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

def rk(t0, r0, v0, h, RDR):

	x1=[]
	x2=[]
	x3=[]
	x4=[]
	t=[]
	
	TS = 0.1
	S=0
	YTH = 0
	
	XT  = r0[0]
	YT  = r0[1]
	XTD = v0[0]
	YTD = v0[1]
	XR  = RDR[0] 
	YR  = RDR[1]
	T = t0
	G = 32.2
	
	RT = math.sqrt((XT-XR)**2+(YT-YR)**2)
	X1 = math.sqrt((XT-XR)**2+(YT-YR)**2)
	X2 = ((XT-XR)*XTD+(YT-YR)*YTD)/RT
	X3 = math.atan2((YT-YR),(XT-XR))
	X4 = ((XT-XR)*YTD-(YT-YR)*XTD)/RT**2
	dt= 0.01
	while (YTH >= 0.0):
		k11 = dt*f1(X1,X2,X3,X4,T)
		k21 = dt*f2(X1,X2,X3,X4,T)
		k31 = dt*f3(X1,X2,X3,X4,T)
		k41 = dt*f4(X1,X2,X3,X4,T)
		k12 = dt*f1(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k22 = dt*f2(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k32 = dt*f3(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k42 = dt*f4(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k13 = dt*f1(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k23 = dt*f2(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k33 = dt*f3(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k43 = dt*f4(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k14 = dt*f1(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		k24 = dt*f2(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		k34 = dt*f3(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		k44 = dt*f4(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X2 = X2 + (k21+2*k22+2*k23+k24)/6
		X3 = X3 + (k31+2*k32+2*k33+k34)/6
		X4 = X4 + (k41+2*k42+2*k43+k44)/6
		T = T+dt
		S = S+dt
		if(S>=(TS-.00001)):
			S=0.
			XTH=X1*math.cos(X3)+XR
			YTH=X1*math.sin(X3)+YR
			XTDH=X2*np.cos(X3)-X1*X4*(np.sin(X3))
			YTDH=X2*np.sin(X3)+X1*X4*(np.cos(X3))
			x1.append(XTH)
			x2.append(YTH)
			x3.append(XTDH)
			x4.append(YTDH)
			t.append(T)
	return(x1,x2,x3,x4,t)
		
def rk_cartesian(t0, r0, v0, h, RDR):

	x1=[]
	x2=[]
	x3=[]
	x4=[]
	t=[]
	
	TS = 0.1
	S=0
	YTH = 0
	
	X1  = r0[0]
	X3  = r0[1]
	X2 = v0[0]
	X4 = v0[1]
	XR  = RDR[0] 
	YR  = RDR[1]
	T = t0
	G = 32.2
	
	dt= 0.01
	while (YTH >= 0.0):
		k11 = dt*g1(X1,X2,X3,X4,T)
		k21 = dt*g2(X1,X2,X3,X4,T)
		k31 = dt*g3(X1,X2,X3,X4,T)
		k41 = dt*g4(X1,X2,X3,X4,T)
		k12 = dt*g1(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k22 = dt*g2(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k32 = dt*g3(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k42 = dt*g4(X1+0.5*k11,X2+0.5*k21,X3+0.5*k31,X4+0.5*k41,T+0.5*dt)
		k13 = dt*g1(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k23 = dt*g2(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k33 = dt*g3(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k43 = dt*g4(X1+0.5*k12,X2+0.5*k22,X3+0.5*k32,X4+0.5*k42,T+0.5*dt)
		k14 = dt*g1(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		k24 = dt*g2(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		k34 = dt*g3(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		k44 = dt*g4(X1+0.5*k13,X2+0.5*k23,X3+0.5*k33,X4+0.5*k43,T+0.5*dt)
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X2 = X2 + (k21+2*k22+2*k23+k24)/6
		X3 = X3 + (k31+2*k32+2*k33+k34)/6
		X4 = X4 + (k41+2*k42+2*k43+k44)/6
		T = T+dt
		S = S+dt
		if(S>=(TS-.00001)):
			S=0.
			XTH=X1
			YTH=X3
			XTDH=X2
			YTDH=X4
			x1.append(XTH)
			x2.append(YTH)
			x3.append(XTDH)
			x4.append(YTDH)
			t.append(T)
	return(x1,x2,x3,x4,t)
	
GAMDEG = 45.0
VT = 3000				  # Initial velocity
t0 = 0.0				  # Initial time
r0 = [0., 0.]			# Initial location in Cartesian coordinates 
v0 = [VT*math.cos(GAMDEG/57.3), VT*math.sin(GAMDEG/57.3)] 	# Initial velocity in Cartesian coordinates
RDR = [100000., 0.]		# Location of the radar in Cartesian coordinates
h = 0.001

x1,x2,x3,x4,t = rk(t0, r0, v0, h, RDR)
y1,y2,y3,y4,t = rk_cartesian(t0, r0, v0, h, RDR)

plt.figure(1)
plt.grid(True)
plt.plot(x1,x2,label='trajectory', linewidth=0.6)
plt.plot(y1,y2,label='trajectory', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,300000)
plt.legend()
plt.ylim(0, 75000)

plt.figure(2)
plt.grid(True)
plt.plot(t,x4,label='Altitude velocity (Ft/s)', linewidth=0.6)
plt.plot(t,y4,label='Altitude velocity (Ft/s)', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,140)
plt.legend()
plt.ylim(-2300,2300)

plt.figure(3)
plt.grid(True)
plt.plot(t,x3,label='Downrange velocity (Ft/s) ', linewidth=0.6)
plt.plot(t,y3,label='Downrange velocity (Ft/s) ', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Estimate and True Signal')
plt.xlim(0,140)
plt.legend()
plt.ylim(0,2300)

plt.show()
