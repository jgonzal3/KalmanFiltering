import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

''' Integrating one-state covariance nonlinear 
Riccati differential equation
'''

TS = 0.1
SIGMA_NOISE = 1.
PHIN = SIGMA_NOISE*SIGMA_NOISE*TS
PHIS = 0

def f1(p,t):
	dpdt = -p*p/PHIN + PHIS
	return (dpdt)

def runge(size, delta):

	P= []
	T =[]
	p = 100
	t  = 0
	dt = delta

	for j in range(0,size):
		k11 = dt*f1(p,t)
		k12 = dt*f1(p+0.5*k11,t+0.5*dt)
		k13 = dt*f1(p+0.5*k12,t+0.5*dt)
		k14 = dt*f1(p+k13,t+dt)
		p = p + (k11+2*k12+2*k13+k14)/6
		#print(p)
		t = t+dt
		T.append(t)
		P.append(p)
	return (T,P)
	
t,p = runge(20000,0.001)

kc = [p/PHIN for p in p]
kd = [x/TS for x in kc]

plt.figure(1)
plt.grid(True)
plt.plot(t, kc, linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Kalamn Gain')
plt.xlim(0,10)
plt.ylim(0,5)
	
plt.figure(2)
plt.grid(True)
plt.plot(t, p, linewidth=0.6)
plt.xlabel('Time (Sec)')
plt.ylabel('Covariance')
plt.xlim(0,10)
plt.ylim(0,0.5)
plt.show()
