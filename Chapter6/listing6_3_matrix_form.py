import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY
'''
Integrating two-state nonlinear matrix Riccati differential equation yields
good match for firs and second diagonal element of covariance matrix.
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

	
PHIS=0.
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
K1=[]
K2=[]
K3=[]	
t=[]


seconds = 10
h = 0.001
runs = int(seconds/h)

T,P = runge(p0,T0, runs, h)

# From the elements of the P matrix, extrace the Kalmam Gains based on the 
# formula:
#                 K = P*HT*RINV

for p in P:
	k= p*H.transpose()*inv(R)
	K.append(k)

for el in K:
	K1.append(TS*el[0,0])
	K2.append(TS*el[1,0])
	K3.append(TS*el[2,0])

t = [h*x for x in range(0,runs)]

plt.figure(1)
plt.grid(True)
plt.plot(t,K1,label='k1c',linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('First Kalam Gain ')
plt.legend()
plt.xlim(0,10)
plt.ylim(0,1.2)

plt.figure(2)
plt.grid(True)
plt.plot(t,K2, label='k2c',linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Second Kalam Gain ')
plt.xlim(0,10)
plt.ylim(0,8)

plt.figure(3)
plt.grid(True)
plt.plot(t,K3, label='k3c',linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Third Kalam Gain ')
plt.xlim(0,10)
plt.ylim(0,5)
plt.show()

