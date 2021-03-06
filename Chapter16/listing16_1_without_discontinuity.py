import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

t=[]
x=[]
x_hat=[]
xd=[]
xdd=[]
xddd=[]
xdddd=[]
xd_hat=[]
xs=[]
xdd_hat=[]
xddd_hat=[]
xdddd_hat=[]
x_hat_ERR=[]
sp11=[]
sp11P=[]
xd_hat_ERR=[]
sp22=[]
sp22P=[]
xdd_hat_ERR=[]
sp33=[]
sp33P=[]
xddd_hat_ERR=[]
sp44=[]
sp44P=[]
ArrayRES = []

PHIS=0000.
TS=0.1
SIGMA_NOISE=1000.0
P0 = 9999999999999.0
add_noise = 1
g = 32.2

# ***************************************************************************************
#
# The higher-order polynomial Kalman filter is less accurate than  the lower-order filter 
# in estimating the lower-order  derivatives of  the signal when only a few  measurements 
# are available.
# Of course, the advantage of a higher-order polynomial Kalman filter is that we are able
# to estimate higher-order derivatives of the signal.
#
# ***************************************************************************************
A0=400000.
A1=-6000.
A2=-32.2/2.
T =0.0
u = 32.2
ITERM = 2
I = np.identity(ITERM)

F = np.matrix([[0.0,  1.0],  [0.0, 0.0]])
P = np.matrix([[P0, 0.0],[0.0 ,P0]])
HMAT = np.matrix([[1.0, 0.0]])
XH = np.matrix([[0.0],[0.0]])
RMAT = np.matrix([[SIGMA_NOISE**2]])

# The reason for the  discontinuity is  that at the time the data rate changed the
# fundamental  matrix was in error.  It predicted  ahead 0.1 s rather than 0.01 s.
# Although this error only occurred at 15 s, its effects  were felt throughout the
# remainder of the flight. Logic was added to the Kalman-filter simulation so that
# the fundamental matrix would be valid for both sampling times.

while ( T < 30.0):
	if (T <= 15.0):
		TSOLD = 0.1
		TS = TSOLD
	else:
		TSOLD = 0.01
	
	Q = np.matrix([[TS*TS*TS/3.0,0.5*TS*TS],[0.5*TS*TS, TS]])
	G = np.matrix([[-0.5*TS*TS],[-TS]])
	PHI = I+TS*F
	
	# Estimate the Riccatti matrices and solve the Riccati differential equation
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
	P=(I-K*HMAT)*M	
	if add_noise == 1:
		XNOISE = GAUSS_PY(SIGMA_NOISE)
	else:
		XNOISE = 0.0
	# Instead of using a single value for each variable, a matrix is defined
	X=np.matrix([[A0+A1*T+A2*T*T],[A1+2.0*A2*T]])
	XS=X[0,0]+XNOISE
	# The is the residual.
	RES = XS - HMAT*PHI*XH - HMAT*G*u
	XH = PHI*XH + G*u + K*(XS - HMAT*PHI*XH - HMAT*G*g)
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	# Theoretical value of the residuals estimated in the book
	XHERR=X-XH
	SP11P=-SP11
	SP22P=-SP22

	t.append(T)
	ArrayRES.append(RES[0,0])
	xs.append(XS)
	x.append(X[0,0])
	x_hat.append(XH[0,0])
	xd.append(X[1,0])
	xd_hat.append(XH[1,0])
	x_hat_ERR.append(XHERR[0,0])
	xd_hat_ERR.append(XHERR[1,0])
	sp11.append(SP11)
	sp11P.append(SP11P)
	sp22.append(SP22)
	sp22P.append(SP22P)
	T= T+TSOLD
	TS = TSOLD

plt.figure(1)
plt.grid(True)
plt.plot(t,x_hat_ERR,label='XH error', linewidth=0.6)
plt.plot(t,sp11,label='sp11', linewidth=0.6)
plt.plot(t,sp11P,label='sp11p', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.ylim(-1500,1500)
plt.xlim(0,30)

plt.figure(2)
plt.grid(True)
plt.plot(t,xd_hat_ERR,label='Xdot error', linewidth=0.6)
plt.plot(t,sp22,label='sp22', linewidth=0.6)
plt.plot(t,sp22P,label='sp22p', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity Estimate and True Signal')
plt.xlim(0,30)
plt.ylim(-200,200)
plt.show()

