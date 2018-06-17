import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

def f1(p,t):
	dpdt =  NI*p*H.transpose()*inv(R)*H*p + p*F.transpose() + F*p + Q
	return (dpdt)

def runge(p0, t0, runs, H):

	row, col = p0.shape
	ONES = np.identity(row)
	
	p = p0
	t = t0*ONES
	dt = H*ONES
	
	T =[]
	P =[]
	
	for j in range(0,runs):
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
PHIN=SIGMA_NOISE*SIGMA_NOISE*TS

F = np.matrix([[0]])
NI = np.matrix([[-1]])
Q = np.matrix([[PHIS]])
H = np.matrix([[1]])
R = np.matrix([[PHIN]])
p0 = np.matrix([[100.]])

K1d=[]
K1c=[]
P1c = []
K=[]
t=[]

t0 = 0.0
seconds = 10.
h = 0.001
runs = int(seconds/h)

# Solving the differential equation using runge-kutta 
T,P = runge(p0,t0, runs, h)
for p in P:
	k= p*H.transpose()*inv(R)
	K.append(k)

for el in K:
	K1d.append(TS*el[0,0])

for el in K:
	K1c.append(el[0,0])
	
for el in P:
	P1c.append(el[0,0])
	
t = [h*x for x in range(0,runs)]

KSTD = [math.sqrt(PHIS/PHIN) for x in range(0,10000)]
PSTD = [math.sqrt(PHIS*PHIN) for x in range(0,10000)]

plt.figure(1)
plt.grid(True)
plt.plot(t,K1c,linewidth=0.6,label='integration')
plt.plot(t,KSTD,linewidth=0.6,label='steady')
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Measurement and True Signal')
plt.xlim(0,10)
plt.ylim(9,10.5)

plt.figure(2)
plt.grid(True)
plt.plot(t, P1c, linewidth=0.6, label='covariance integration')
plt.plot(t, PSTD, linewidth=0.6, label='covariance steady state')
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Covariance')
plt.xlim(0,10)
plt.ylim(0,1.2)

plt.show()
