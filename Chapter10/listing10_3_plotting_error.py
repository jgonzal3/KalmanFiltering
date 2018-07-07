import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random

def project3(TS,Ti,XP,XDP,WP,HP):
	T=0.;
	X=XP;
	XD=XDP;
	H=HP;
	W = WH
	while (T<=(TS-0.0001)):
		XDD=-W*W*X
		XD=XD+H*XDD
		X=X+H*XD
		T=T+H
	XH=X
	XDH=XD
	return [XH,XDH]
	
def projectRK(TS,T,X,XD,W,HP):
	T_ = 0.0
	dt = HP
	while (T_<TS):
		k11 = dt*g1(X,XD,W,T)
		k21 = dt*g2(X,XD,W,T)
		k12 = dt*g1(X+0.5*k11,XD+0.5*k21,W,T+0.5*dt)
		k22 = dt*g2(X+0.5*k11,XD+0.5*k21,W,T+0.5*dt)
		k13 = dt*g1(X+0.5*k12,XD+0.5*k22,W,T+0.5*dt)
		k23 = dt*g2(X+0.5*k12,XD+0.5*k22,W,T+0.5*dt)
		k14 = dt*g1(X+0.5*k13,XD+0.5*k23,W,T+0.5*dt)
		k24 = dt*g2(X+0.5*k13,XD+0.5*k23,W,T+0.5*dt)
		X  = X  + (k11+2*k12+2*k13+k14)/6
		XD = XD + (k21+2*k22+2*k23+k24)/6
		T_ = T_+dt
	XH=X
	XDH=XD
	return [XH,XDH]
	
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

def g1(x1,x2,W,t):
	dx1dt = x2
	return (dx1dt)

def g2(x1,x2,W,t):
	#print (x1,x2,x3,x4)
	dx2dt = -W*W*x1
	return (dx2dt)

ArrayT=[]
ArrayX=[]
ArrayXH=[]
ArrayA =[]
ArrayW =[]
ArrayXD =[]
ArrayERRA=[]
ArrayERRW=[]
ArrayERRPHI=[]

ArraySP11P=[]
ArraySP11PP=[]
ArraySP22P=[]
ArraySP22PP=[]
ArraySP33P=[]
ArraySP33PP=[]
ArraySP44P=[]
ArraySP44PP=[]

# Although the state-space equation is linear,  the measurement equation 
# is non-linear for this particular formulation of the problem. In fact, 
# information concerning the  sinusoidal nature of  the signal is buried  
# in the measurement equation,  and  we need to  use the Tylor series to 
#

# 		  @x    @x   @x     Df 
# Dx* = [ --    --   -- ] [ Do ] + v
#    	 @phi  @ome  @A	    DA
#
# with x = A*sin(wt) or A*sin(phi) 


TS=0.1
PHIS=0.0

H =0.001
HP=0.001

# Initial conditions for states
W = 1.0
A = 1.0
ITERM =2
rk = 0
prk = 0
add_noise =1
SIGMA_NOISE_X = 1.0

'''Identity matrix'''
I = np.identity(3)

RMAT=np.matrix([[SIGMA_NOISE_X]])
dt = H

for MC in range (0,1):
	ArrayT  =[]
	ArrayX  =[]
	ArrayXH =[]
	ArrayA  =[]
	ArrayW  =[]
	ArrayWH =[]
	ArrayXD =[]
	ArrayXDH=[]
	ArrayERRW=[]
	ArrayERRX=[]
	ArrayERRXD=[]

	ArraySP22P=[]
	ArraySP22PP=[]
	ArraySP33P=[]
	ArraySP33PP=[]

	T = 0.0
	S =0.0
	WH = 2
	X = 0.0
	XD = A*W
	XH =0.0
	XDH =0.0
	''' THe matrix P is compose of the square of the errors in the estimate  which 
	is error = (real value - estimated value)'''
	P=np.matrix([[SIGMA_NOISE_X**2, 0.0, 0.0],[0.0, 2.0**2, 0],[0.0, 0.0, 2.0**2]])
	while (T <= 20.0):
		if (rk == 0):
			XOLD=X
			XDOLD=XD
			XDD=-W*W*X
			X=X+H*XD
			XD=XD+H*XDD
			T=T+H
			XDD=-W*W*X
			X=.5*(XOLD+X+H*XD)
			XD=.5*(XDOLD+XD+H*XDD)
		else:
			# Propagate using Runge-Kutta procedure.
			k11 = dt*g1(X,XD,W,T)
			k21 = dt*g2(X,XD,W,T)
			k12 = dt*g1(X+0.5*k11,XD+0.5*k21,W,T+0.5*dt)
			k22 = dt*g2(X+0.5*k11,XD+0.5*k21,W,T+0.5*dt)
			k13 = dt*g1(X+0.5*k12,XD+0.5*k22,W,T+0.5*dt)
			k23 = dt*g2(X+0.5*k12,XD+0.5*k22,W,T+0.5*dt)
			k14 = dt*g1(X+0.5*k13,XD+0.5*k23,W,T+0.5*dt)
			k24 = dt*g2(X+0.5*k13,XD+0.5*k23,W,T+0.5*dt)
			X  = X  + (k11+2*k12+2*k13+k14)/6
			XD = XD + (k21+2*k22+2*k23+k24)/6
			T = T+dt
		S=S+H
		
	# this step is here  so that the  discritezation of the  Riccati equations  takes 
	# the correct values of the output. Note that H=0.0001 but TS=1. for this example
		if S>=(TS-.00001):
			S=0.
			F = np.matrix([[0.0,    1.0,  0.0],
						   [-WH*WH, 0.0, -2*WH*XH],
						   [0.0,    0.0,  0.0]])
			F2 = F*F
			F3 = F*F*F
			ITERM = 2
			if ITERM == 2:
				PHIMAT = I+TS*F
			if ITERM == 3:
				PHIMAT = I + TS*F + TS**2*F2/2
			if ITERM == 4:
				PHIMAT = I + TS*F +TS**2*F2/2 + TS**3*F3/6 
				
				
			Q = np.matrix([[0.0, 0.0, 				       0.0], 
				    	   [0.0, 1.333*WH*WH*XH*XH*TS**3, -WH*XH*TS**2],
						   [0.0, -WH*XH*TS**2, 	           TS]])
			# Predicted states
			HMAT=np.matrix([[1.0, 0.0, 0.0]])
			M=PHIMAT*P*PHIMAT.transpose() + PHIS*Q
			K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
			P=(I-K*HMAT)*M
		# Need to propagate the estimated  states to obtain an approximation of  the new predicted
		# state. 
		# If we need to compare the solution with the solution given in the book using MATLAB code.
			if (add_noise == 1):
				XNOISE = GAUSS_PY(SIGMA_NOISE_X)
			else:
				XNOISE = 0.0
		# Add noise to the real value
			XMEAS=X+XNOISE
		
		# We can select to use runge kutta or Euler to propagate the states during the
		# discrete interval of TS.
			if prk == 0:
				# Use Euler to propagate
				[XB,XDB] =  project3(TS,T,XH,XDH,WH,HP)
			else:
				# Use runge kutta order 4 to propagate the states and obtain X_hat/XD_hat
				[XB,XDB] = projectRK(TS,T,XH,XDH,WH,HP)
				
			RES=XMEAS-XB
			XH = XB + K[0,0]*RES
			XDH = XDB + K[1,0]*RES
			WH	 = WH + K[2,0]*RES
			
			ERRX = X-XH
			SP11P  = np.sqrt(P[0,0])
			ERRXD = XD-XDH
			SP22P  = np.sqrt(P[1,1])
			ERRW   = W-WH
			SP33P  = np.sqrt(P[2,2])
			SP11PP=-SP11P
			SP22PP=-SP22P
			SP33PP=-SP33P
			ArrayT.append(T)
			ArrayW.append(WH)
			ArrayWH.append(WH)
			ArrayXDH.append(XDH)
			ArrayXD.append(XD)
			ArrayX.append(X)
			ArrayXH.append(XH)
			ArrayERRW.append(ERRW)
			ArrayERRX.append(ERRX)
			ArrayERRXD.append(ERRXD)
			ArraySP22P.append(SP22P)
			ArraySP22PP.append(SP22PP)
			ArraySP11P.append(SP11P)
			ArraySP11PP.append(SP11PP)
	plt.figure(1)		
	plt.plot(ArrayT,ArrayERRX,label='simulated', linewidth=0.6)
	plt.plot(ArrayT,ArraySP11P,label='sp11p', linewidth=0.6)
	plt.plot(ArrayT,ArraySP11PP,label='sp11pp', linewidth=0.6)
	plt.figure(2)
	plt.plot(ArrayT,ArrayERRXD,label='simulated', linewidth=0.6)
	plt.plot(ArrayT,ArraySP22P,label='sp22p', linewidth=0.6)
	plt.plot(ArrayT,ArraySP22PP,label='sp22pp', linewidth=0.6)
'''
Adding process noise increases errors in estimate of altitude but reduces the hangoff error 
'''

plt.figure(1)
plt.grid(True)
plt.xlabel('Time (Sec)')
plt.ylabel('X Signal')
plt.xlim(0,20)
plt.legend()
plt.ylim(-1.0,1.0)

plt.figure(2)
plt.grid(True)
plt.xlabel('Time (Sec)')
plt.ylabel('X dot Signal')
plt.xlim(0,20)
plt.legend()
plt.ylim(-3.0,3.0)

plt.show()

