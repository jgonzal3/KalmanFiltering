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

def g1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)

def g2(x1,x2,t):
	#print (x1,x2,x3,x4)
	dx2dt = -W*W*x1
	return (dx2dt)

ArrayT=[]
ArrayX=[]
ArrayA =[]
ArrayW =[]
ArrayPHI =[]
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
PHIS1=0.0
PHIS2=0.0
T=0.
S=0.
H =0.001
HP=0.001

# Initial conditions for states
W = -1.0
A = 1.0
PHI =0.0
X = 0.0
XD = A*W

# Initial conditions for predicted states
AH=3
WH = 2
PHIH = 0.0
SIGMA_NOISE_X = 1.0

ITERM =2
rk = 1
rkp = 0

''' THe marrix P is componse of the square of the errors in the estimate  which 
is error = (real value - estimated value)'''
P=np.matrix([[ 0.0, 0.0, 	   0.0 	 ],
			  [0.0, (W-WH)**2, 0.0	 ],
			  [0.0, 0.0, 	  (A-AH)**2]])

'''Identity matrix'''
I = np.identity(3)
F = np.matrix([[0.0, 1.0, 0.0],
			   [0.0, 0.0, 0.0],
			   [0.0, 0.0, 0.0]])
F2 = F*F
F3 = F*F*F
ITERM = 2
if ITERM == 2:
	PHIMAT = I+TS*F
if ITERM == 3:
	PHIMAT = I + TS*F + TS**2*F2/2
if ITERM == 4:
	PHIMAT = I + TS*F +TS**2*F2/2 + TS**3*F3/6
	
QPHI1 = PHIS1*np.matrix([[TS**3/3, TS**2/2, 0.0],
						 [TS**2/2, TS,      0.0],
						 [0.0,     0.0,     0.0]])	
QPHI2 = PHIS2*np.matrix([[0.0, 0.0, 0.0],
						 [0.0, 0.0, 0.0],
						 [0.0, 0.0,  TS]])
Q = QPHI1 + QPHI2
# this lines are not correct because the only know value is the SIGTH and SIGR
# We need to use this it to recalculate every time RMAT is used:
# SIGYT = (SIN(THETH)*SIGR)**2+(RTH*COS(THETH)*SIGTH)**2
# SIGXT = (COS(THETH)*SIGR)**2+(RTH*SIN(THETH)*SIGTH)**2

RMAT=np.matrix([[SIGMA_NOISE_X]])
dt = H
add_noise = 1

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
		k11 = dt*g1(X,XD,T)
		k21 = dt*g2(X,XD,T)
		k12 = dt*g1(X+0.5*k11,XD+0.5*k21,T+0.5*dt)
		k22 = dt*g2(X+0.5*k11,XD+0.5*k21,T+0.5*dt)
		k13 = dt*g1(X+0.5*k12,XD+0.5*k22,T+0.5*dt)
		k23 = dt*g2(X+0.5*k12,XD+0.5*k22,T+0.5*dt)
		k14 = dt*g1(X+0.5*k13,XD+0.5*k23,T+0.5*dt)
		k24 = dt*g2(X+0.5*k13,XD+0.5*k23,T+0.5*dt)
		X  = X  + (k11+2*k12+2*k13+k14)/6
		XD = XD + (k21+2*k22+2*k23+k24)/6
		T = T+dt
	S=S+H
	
# this step is here  so that the  discritezation of the  Riccati equations  takes 
# the correct values of the output. Note that H=0.0001 but TS=1. for this example
	if S>=(TS-.00001):
		S=0.
# Need to propagate the states, but because the fundamental matrix is  exact in this
# application, we can also use it to exactly propagate the state estimates in the 
# extended Kalman filter over the sampling interval. By using the fundamental matrix 
# we can propagate the states from time k-1 to time k;
		PHIB = PHIH + WH*TS
		WB = WH
		AB = AH
		# Predicted states
		HMAT=np.matrix([[AB*math.cos(PHIB), 0.0, math.sin(PHIB)]])
		M=PHIMAT*P*PHIMAT.transpose() + Q
		K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
		P=(I-K*HMAT)*M

		# If we need to compare the solution with the solution given in the book using MATLAB code.
		if (add_noise == 1):
			XNOISE = GAUSS_PY(SIGMA_NOISE_X)
		else:
			THETNOISE = 0.0
			RTNOISE = 0.0
		XMEAS=X+XNOISE
		RES=XMEAS-AB*math.sin(PHIB)
		PHIH = PHIB + K[0,0]*RES
		WH	 = WH   + K[1,0]*RES
		AH 	 = AH   + K[2,0]*RES

		ERRPHI = W*T-PHIH
		SP11P  = np.sqrt(P[0,0])
		ERRW   = W-WH
		SP22P  = np.sqrt(P[1,1])
		ERRA   = A-AH
		SP33P  = np.sqrt(P[2,2])
		SP11PP=-SP11P
		SP22PP=-SP22P
		SP33PP=-SP33P
		
		ArrayT.append(T)
		ArrayW.append(WH)
		ArrayA.append(AH)
		ArrayPHI.append(PHIH)
		ArrayX.append(X)
		ArrayERRA.append(ERRA)
		ArrayERRW.append(ERRW)
		ArrayERRPHI.append(ERRPHI)
		ArraySP11P.append(SP11P)
		ArraySP11PP.append(SP11PP)
		ArraySP22P.append(SP22P)
		ArraySP22PP.append(SP22PP)
		ArraySP33P.append(SP33P)
		ArraySP33PP.append(SP33PP)
'''
Adding process noise increases errors in estimate of altitude
but reduces the hangoff error 
'''

plt.figure(1)
plt.grid(True)
plt.plot(ArrayT,ArrayA,label='actual', linewidth=0.6)
plt.plot(ArrayT,[A for i in ArrayT],label='estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Amplitud ')
plt.xlim(0,20)
plt.legend()
plt.ylim(0,3)

plt.figure(2)
plt.grid(True)
plt.plot(ArrayT,ArrayW,label='actual', linewidth=0.6)
plt.plot(ArrayT,[W for i in ArrayT],label='estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('OMEGA (Rad/sec)')
plt.xlim(0,20)
plt.legend()
plt.ylim(-2.0,2.0)

plt.figure(3)
plt.grid(True)
plt.plot(ArrayT,ArrayX,label='actual', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('A (Rad/sec)')
plt.xlim(0,20)
plt.legend()
plt.ylim(-2.0,2.0)
plt.show()

# The conclusion reached is that the extended  Kalman filter we  have formulated
# in this listing does not appear to be  working satisfactorily if the filter is 
# not properly initialized. Is it possible that, for this problem, the frequency 
# of the sinusoid is unobservable?

