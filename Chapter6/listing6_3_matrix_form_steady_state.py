import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY
'''
In other words, in the steady state the continuous second-order polynomial
Kalman filter is simply a third-order transfer function whose poles follow a
Butterworth distribution with a natural frequency that depends on the ratio of
the spectral densities of the process noise to the measurement noise. Having more
process noise or less measurement noise will tend to increase the bandwidth of
the Kalman filter.
'''

def f1(p,t):
	#PDOT = -P*HT*RINV*H*P + P*FT * F*P + Q *Riccati differential equation)
	
	dpdt =  NI*p*H.transpose()*inv(R)*H*p + p*F.transpose() + F*p + Q
	return (dpdt)

def runge(p0, t0, size, H):

	row, col = p0.shape
	IDN = np.identity(row)
	
	p = p0
	t = t0*IDN
	dt = H*IDN
	
	T =[]
	P =[]
	
	for j in range(0,size):
		k11 = dt*f1(p,t)
		k12 = dt*f1(p+0.5*k11,t+0.5*dt)
		k13 = dt*f1(p+0.5*k12,t+0.5*dt)
		k14 = dt*f1(p+k13,t+dt)
		p = p + (k11+2*k12+2*k13+k14)/6
		t = t+dt
		T.append(t)
		P.append(p)
	return (T,P)

	
PHIS=10.
TS=.1
SIGMA_NOISE=1.
SIGMA2 = SIGMA_NOISE*SIGMA_NOISE
PHIN=SIGMA2*TS
T0 = 0.0

p0 = np.matrix([[100.0,0,0],[0,100.0,0],[0,0,100.0]])
F = np.matrix([[0,1,0],[0,0,1], [0,0,0]])
NI = np.matrix([[-1,0,0],[0,-1,0],[0,0,-1]])
Q = PHIS*np.matrix([[0,0,0],[0,0,0],[0,0,1]])
H = np.matrix([[1,0,0]])
R = np.matrix([[PHIN]])

K=[]
K1c=[]
K2c=[]
K3c=[]	
K1d=[]
K2d=[]
K3d=[]
t=[]


seconds = 10
h = 0.001
runs = int(seconds/h)

T,P = runge(p0,T0, runs, h)

# From the elements of the P matrix, extrace the Kalmam Gains based on the 
# formula:
#                 K = P*HT*RINV

P11 = 2*(PHIS)**(1/6)*(PHIN)**(5/6)
P12 = 2*(PHIS)**(1/3)*(PHIN)**(2/3)
P13 = math.sqrt(PHIS*PHIN)
P22 = 3*math.sqrt(PHIS*PHIN)
P23 = 2*(PHIS)**(2/3)*(PHIN)**(1/3)
P33 = 2*(PHIS)**(5/6)*(PHIN)**(1/6)

K11 = P11/PHIN
K22 = P12/PHIN
K33 = P13/PHIN

K1STD = [K11 for x in range(0,10000)]
K2STD = [K22 for x in range(0,10000)]
K3STD = [K33 for x in range(0,10000)]

for p in P:
	k= p*H.transpose()*inv(R)
	K.append(k)

for el in K:
	K1d.append(TS*el[0,0])
	K2d.append(TS*el[1,0])
	K3d.append(TS*el[2,0])
	
for el in K:
	K1c.append(el[0,0])
	K2c.append(el[1,0])
	K3c.append(el[2,0])

t = [h*x for x in range(0,runs)]

plt.figure(1)
plt.grid(True)
plt.plot(t,K1c,label='k1c',linewidth=0.6)
plt.plot(t,K1STD,label='K1 steady state')
plt.xlabel('Time (Sec)')
plt.ylabel('First Kalam Gain ')
plt.legend()
plt.xlim(0,10)
plt.ylim(0,12)

plt.figure(2)
plt.grid(True)
plt.plot(t,K2c, label='k2c',linewidth=0.6)
plt.plot(t,K2STD,label='K2 steady state')
plt.xlabel('Time (Sec)')
plt.ylabel('Second Kalam Gain ')
plt.xlim(0,10)
plt.ylim(0,60)

plt.figure(3)
plt.grid(True)
plt.plot(t,K3c, label='k3c',linewidth=0.6)
plt.plot(t,K3STD,label='K3 steady state')
plt.xlabel('Time (Sec)')
plt.ylabel('Third Kalam Gain ')
plt.xlim(0,10)
plt.ylim(0,40)
plt.show()

