import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def f1(p,t):
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
PHIN=SIGMA_NOISE*SIGMA_NOISE*TS

p0 = np.matrix([[100.0,0],[0,100.0]])
F = np.matrix([[0,1],[0,0]])
NI = np.matrix([[-1,0],[0,-1]])
Q = PHIS*np.matrix([[0,0],[0,1]])
H = np.matrix([[1,0]])
R = np.matrix([[PHIN]])

d,d1 = p0.shape

K=[]
K1=[]
K2=[]
K3=[]	
t=[]

t0 = 0.0
seconds = 10
h = 0.001
runs = int(seconds/h)

T,P = runge(p0,t0, runs, h)
for p in P:
	k= p*H.transpose()*inv(R)
	K.append(k)

for el in K:
	K1.append(TS*el[0,0])
	if (d == 2 or d == 3):
		K2.append(TS*el[1,0])
	if (d == 3):
		K3.append(TS*el[2,0])

t = [h*x for x in range(0,runs)]

plt.figure(1)
plt.grid(True)
plt.plot(t,K1)
plt.xlabel('Time (Sec)')
plt.ylabel('Kalam Gain 1')
plt.xlim(0.1,10)
plt.ylim(0,1.2)

if (d == 2):
	plt.figure(2)
	plt.grid(True)
	plt.plot(t,K2)
	plt.xlabel('Time (Sec)')
	plt.ylabel('Kalam Gain 2')
	plt.xlim(0.1,10)
	plt.ylim(0,10)

if (d == 3):
	plt.figure(3)
	plt.grid(True)
	plt.plot(t,K3)
	plt.xlabel('Time (Sec)')
	plt.ylabel('Kalam Gain 3')
	plt.xlim(0,10)
	plt.ylim(0,10)
plt.show()
