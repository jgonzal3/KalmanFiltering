import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
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
PHIS=0.0

H =0.001
HP=0.001

# Initial conditions for states
W = -1.0
A = 1.0
PHI =0.0
X = 0.0
XD = A*W

# Initial conditions for predicted states
WH = 2
PHIH = 0.0
SIGMA_NOISE_X = 0.1

ITERM =2
rk = 1
rkp = 0

''' THe matrix P is compose of the square of the errors in the estimate  which 
is error = (real value - estimated value)'''
P=np.matrix([[ 0.0, 0.0],
			 [0.0, (W-WH)**2]])

'''Identity matrix'''
I = np.identity(2)
F = np.matrix([[0.0, 1.0],
			   [0.0, 0.0]])
F2 = F*F
F3 = F*F*F
ITERM = 2
if ITERM == 2:
	PHIMAT = I+TS*F
if ITERM == 3:
	PHIMAT = I + TS*F + TS**2*F2/2
if ITERM == 4:
	PHIMAT = I + TS*F +TS**2*F2/2 + TS**3*F3/6
	
Q = PHIS*np.matrix([[TS**3/3, TS**2/2],
					 [TS**2/2, TS]])	

# this lines are not correct because the only know value is the SIGTH and SIGR
# We need to use this it to recalculate every time RMAT is used:
# SIGYT = (SIN(THETH)*SIGR)**2+(RTH*COS(THETH)*SIGTH)**2
# SIGXT = (COS(THETH)*SIGR)**2+(RTH*SIN(THETH)*SIGTH)**2

RMAT=np.matrix([[SIGMA_NOISE_X]])
dt = H
add_noise = 1

for MC in range (0,10):
	ArrayT=[]
	ArrayX=[]
	ArrayA =[]
	ArrayW =[]
	ArrayPHI =[]
	#ArrayERRA=[]
	ArrayERRW=[]

	#ArrayERRPHI=[]

	#ArraySP11P=[]
	#ArraySP11PP=[]
	ArraySP22P=[]
	ArraySP22PP=[]
	#ArraySP33P=[]
	#ArraySP33PP=[]
	#ArraySP44P=[]
	#ArraySP44PP=[]
	T = 0.0
	S =0.0
	WH = -2
	PHIH = 0.0
	X = 0
	XD = A*W
	P=np.matrix([[ 0.0, 0.0],[0.0, (W-WH)**2]])
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
			# Predicted states
			HMAT=np.matrix([[A*math.cos(PHIB), 0.0]])
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
			RES=XMEAS-A*math.sin(PHIB)
			PHIH = PHIB + K[0,0]*RES
			WH	 = WH   + K[1,0]*RES

			ERRPHI = W*T-PHIH
			SP11P  = np.sqrt(P[0,0])
			ERRW   = W-WH
			SP22P  = np.sqrt(P[1,1])
			SP11PP=-SP11P
			SP22PP=-SP22P
			
			ArrayT.append(T)
			ArrayW.append(WH)
			ArrayPHI.append(PHIH)
			ArrayX.append(X)
			ArrayERRW.append(ERRW)
			ArrayERRPHI.append(ERRPHI)
			ArraySP22P.append(SP22P)
			ArraySP22PP.append(SP22PP)
			
	plt.plot(ArrayT,ArrayERRW,label='MC ='+str(MC), linewidth=0.6)
	plt.plot(ArrayT,ArraySP22P, linewidth=0.6)
	plt.plot(ArrayT,ArraySP22PP, linewidth=0.6)
	
'''
Adding process noise increases errors in estimate of altitude
but reduces the hangoff error 
'''

plt.figure(1)
plt.grid(True)

plt.xlabel('Time (Sec)')
plt.ylabel('OMEGA (Rad/sec)')
plt.xlim(0,20)
plt.ylim(-0.02,0.02)

plt.show()

# The conclusion reached is that the extended  Kalman filter we  have formulated
# in this listing does not appear to be  working satisfactorily if the filter is 
# not properly initialized. Is it possible that, for this problem, the frequency 
# of the sinusoid is unobservable?

