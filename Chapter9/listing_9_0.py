import math
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

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
import math
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

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

t=[]
x=[]
xs=[]
y=[]
x_hat=[]
y_hat=[]
xdot=[]
ydot=[]
xdot_hat = []
ydot_hat =[]
x1ddot_hat=[]
x1_hat_ERR=[]
sp11=[]
sp11n=[]
x1dot_hat_ERR=[]
sp22=[]
sp22n=[]
x1ddot_hat_ERR=[]
sp33=[]
sp33n=[]
beta_hat=[]
res=[]


PHIS=322*322/30
TS=.1
X=0.
X1D=-6000.
X1DD = 0
Y=0.0
BETAH=800.
X1H=200025.
X1DH=-6150.
X1DDH=0.0
TS=0.1
TF=140.
T=0.0
G=32.2
S=0
SIGMA_NOISE_THETA=0.001
SIGMA_NOISE_R=100

H = 1
V0 =  3000
RAD = 180/math.pi
THETA0 = 45/RAD
XR = -150000
YR = 0
XH = 0
YH = 0

while (T< TF):
	
	Y = V0*T*math.sin(THETA0) - 0.5*G*T*T
	X = V0*T*math.cos(THETA0) 
	R = math.sqrt((X-XR)**2 + (Y-YR)**2)
	THETA = math.atan((Y-YR)/(X-XR+0.00001))
	t.append(T)
	x.append(X)
	y.append(Y)
	
	RNOISE = GAUSS_PY(SIGMA_NOISE_R)
	THETANOISE = GAUSS_PY(SIGMA_NOISE_THETA)
	#XNOISE=0
	#YNOISE=0
	if T == 0:
		RH =0.0
		THETAH = THETA0
	else:
		RH=R+RNOISE
		THETAH=THETA+THETANOISE
	XHOLD = XH
	YHOLD = YH
	XH = RH*math.cos(THETAH) + XR
	YH = RH*math.sin(THETAH) + YR
	x_hat.append(XH)
	y_hat.append(YH)
	T = T + H
	XDH = (XH - XHOLD)/H
	YDH = (YH - YHOLD)/H
	xdot_hat.append(XDH)
	ydot_hat.append(YDH)

VX = [V0*math.cos(THETA0) for k in t]
VY = [V0*math.sin(THETA0) - G*k for k in t]

plt.figure(1)
plt.grid(True)
plt.plot(x,y,label='trajectory', linewidth=0.6)
plt.plot(x_hat,y_hat,label='trajectory', linewidth=0.6)
plt.xlabel('Downrange (Ft)')
plt.ylabel('Altitude (Ft)')
plt.xlim(0,300000)
plt.legend()
plt.ylim(0,100000)

plt.figure(2)
plt.grid(True)
plt.plot(t,xdot_hat,label='Velocity Simulated', linewidth=0.6)
plt.plot(t, VX,label='Velocity Actual', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity (Ft/s)')
plt.xlim(2,140)
plt.legend()
plt.ylim(0,5000)

plt.figure(3)
plt.grid(True)
plt.plot(t,ydot_hat,label='Velocity Simulated', linewidth=0.6)
plt.plot(t,VY,label='Velocity Actual', linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Velocity (Ft/s)')
plt.xlim(2,140)
plt.legend()
plt.ylim(-5000,5000)

plt.show()
