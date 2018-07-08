import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

t=[]
x=[]
x_hat=[]
xdot=[]
xddot=[]
xdddot=[]
xddddot=[]
xdot_hat=[]
xs=[]
xddot_hat=[]
xdddot_hat=[]
xddddot_hat=[]
x_hat_ERR=[]
sp11=[]
sp11P=[]
xdot_hat_ERR=[]
sp22=[]
sp22P=[]
xddot_hat_ERR=[]

PHIS=0.
TS=1
XH=0.0
XDH=1.0
XDDH=0.0
SIGMA_NOISE=100.0
W = 1.0
A = 1.0
WH = 2.0

# ***************************************************************************************
# The initial covariance matrix is infinite.  Essentially, an infinite  covariance matrix 
# means that the filter  does  not realize  it is correctly initialized.  The  estimation 
# results  are virtually unchanged with  perfect initialization  when the signal-to-noise 
# ratio is 0.01.
# ***************************************************************************************
P0 = 99999999.0
add_noise = 0

# ***************************************************************************************
#
# The higher-order polynomial Kalman filter is less accurate than  the lower-order filter 
# in estimating the lower-order  derivatives of  the signal when only a few  measurements 
# are available.
# Of course, the advantage of a higher-order polynomial Kalman filter is that we are able
# to estimate higher-order derivatives of the signal.
#
# ***************************************************************************************

ITERM = 5
I = np.identity(ITERM)
if ITERM == 2:
	F = np.matrix([[0.0,  1.0],  [0.0, 0.0]])
	P = np.matrix([[P0, 0.0],[0.0 ,P0]])
	HMAT = np.matrix([[1.0, 0.0]])
	PHI = I+TS*F
if ITERM == 3:
	P = np.matrix([[P0, 0.0, 0.0],[0.0, P0, 0.0],[0.0, 0.0, P0]])
	F = np.matrix([[0.0,  1.0, 0.0], [0.0, 0.0, 1.0],[0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0]])
	PHI = I + TS*F + TS**2*F*F/2
if ITERM == 4:
	P = np.matrix([[P0, 0.0, 0.0,0.0],[0.0, P0, 0.0,0.0],[0.0, 0.0, P0,0.0],[0.0, 0.0, 0.0,P0]])
	F = np.matrix([[0.0,  1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0,1.0], [0.0, 0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0], [0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0 , 0.0]])
	PHI = I + TS*F +TS**2*F*F/2 + TS**3*F*F*F/6
if ITERM == 5:
	P = np.matrix([[P0, 0.0, 0.0, 0.0, 0.0],[0.0, P0, 0.0, 0.0, 0.0],[0.0, 0.0, P0, 0.0, 0.0],[0.0, 0.0, 0.0, P0, 0.0],[0.0, 0.0, 0.0, 0.0, P0]])
	F = np.matrix([[0.0,  1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
	HMAT = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0]])
	XH = np.matrix([[0.0],[0.0],[0.0], [0.0], [0.0]])
	PHI = I + TS*F +TS**2*F*F/2 + TS**3*F*F*F/6 + TS**4*F*F*F*F/24
	
Q = PHIS*np.identity(ITERM)
RMAT = np.matrix([[SIGMA_NOISE**2]])

for T in [0,1,2,3,4,5]:
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*HMAT.transpose()*(inv(HMAT*M*HMAT.transpose() + RMAT))
	P=(I-K*HMAT)*M	
	if add_noise == 1:
		XNOISE = GAUSS_PY(SIGMA_NOISE)
	else:
		XNOISE = 0.0
	X=T**2.0+15.0*T-3.0
	XD=2.0*T+15.0
	XDD = 2.0
	XDDD = 0.0
	XDDDD = 0.0
	XS=X+XNOISE
	XH = PHI*XH + K*(XS - HMAT*PHI*XH)
  # T is not the actual time, but the current measurement. 
	t.append(T+1)
	x.append(X)
	xs.append(XS)
	x_hat.append(XH[0,0])
	xdot.append(XD)
	xdot_hat.append(XH[1,0])
	xddot.append(XDD)
	xddot_hat.append(XH[2,0])
	xdddot.append(XDDD)
	xdddot_hat.append(XH[3,0])
	xddddot.append(XDDDD)
	xddddot_hat.append(XH[4,0])

plt.figure(1)
plt.grid(True)
plt.plot(t,x,label='X real', linewidth=0.6)
plt.plot(t,x_hat,label='X estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X Estimate and True Signal')
plt.xlim(1,6)

plt.figure(2)
plt.grid(True)
plt.plot(t,xdot,label='Xdot real', linewidth=0.6)
plt.plot(t,xdot_hat,label='Xdot estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X dot Estimate and True Signal')
plt.xlim(1,6)

plt.figure(3)
plt.grid(True)
plt.plot(t,xddot,label='Xdotdot real', linewidth=0.6)
plt.plot(t,xddot_hat,label='Xdotdot estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X dot dot Estimate and True Signal')
plt.xlim(1,6)

plt.figure(4)
plt.grid(True)
plt.plot(t,xdddot,label='Xdotdot real', linewidth=0.6)
plt.plot(t,xdddot_hat,label='Xdotdotdot estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X tripple Estimate and True Signal')
plt.xlim(1,6)

plt.figure(5)
plt.grid(True)
plt.plot(t,xddddot,label='Xdotdot real', linewidth=0.6)
plt.plot(t,xddddot_hat,label='Xdotdotdot estimated', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('X four dot Estimate and True Signal')
plt.xlim(1,6)

plt.show()
