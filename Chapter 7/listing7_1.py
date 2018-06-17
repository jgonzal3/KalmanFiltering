import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

G = 32.2
# x1 = x
# x2 = x1'
# x2' = x1' = x''
# original equation:
# x'' = A*exp(-x)(x')^2 - g
# x1 = x
# x1' = x2
# x2' = A*exp(-x1)(x2)^2 - g

def f1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,t):
	global BETA
	dx2dt = (0.0034*G*x2*x2*math.exp(-x1/22000.0)/(2.0*BETA))-G
	return (dx2dt)

def runge(size, delta):

	X1= []
	X2 = []
	T =[]
	x1 = 200000.0
	x2 = -6000.0
	t  = 0
	dt = delta

	for j in range(0,size):
		k11 = dt*f1(x1,x2,t)
		k21 = dt*f2(x1,x2,t)
		k12 = dt*f1(x1+0.5*k11,x2+0.5*k21,t+0.5*dt)
		k22 = dt*f2(x1+0.5*k11,x2+0.5*k21,t+0.5*dt)
		k13 = dt*f1(x1+0.5*k12,x2+0.5*k22,t+0.5*dt)
		k23 = dt*f2(x1+0.5*k12,x2+0.5*k22,t+0.5*dt)
		k14 = dt*f1(x1+k13,x2+k23,t+dt)
		k24 = dt*f2(x1+k13,x2+k23,t+dt)
		x1 = x1 + (k11+2*k12+2*k13+k14)/6
		x2 = x2 + (k21+2*k22+2*k23+k24)/6
		t = t+dt
		T.append(t)
		X1.append(x1)
		X2.append(x2)
	return (T,X1,X2)

BETA_ = [500.0,1000.0,5000.0,9999999.0]

for beta in BETA_:
	BETA = beta
	t,x,v = runge(30000,0.001)
	plt.figure(1)
	plt.plot(t,x,label='Beta ='+ str(BETA), linewidth=0.6)
	plt.figure(2)
	plt.plot(t,v,label='Beta ='+str(BETA), linewidth=0.6)
	
plt.figure(1)
plt.grid(True)
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Distance')
plt.xlim(0,30)

plt.figure(2)
plt.grid(True)
plt.legend()
plt.xlabel('Time (Sec)')
plt.ylabel('Speed')
plt.xlim(0,30)
plt.show()

plt.show()
