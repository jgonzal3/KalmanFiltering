import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

x1 = 0.25
x2 = 0.
t  = 0
Z  = 0.7
A2 = 0.1
B1 = 1.0
B2 = 1.25
W =  6.28
WN = 6.28*0.1
dt = 0.01
S=0

X1= []
X2 = []
SUSPENSION = []
DISTANCE = []
T =[]

def f1(x1,x2,t):
	dx1dt = x2
	return (dx1dt)

def f2(x1,x2,t):
	dx2dt = -2*Z*WN*x2 - WN*WN*x1 + A2*W*W*math.sin(W*t)
	return (dx2dt)

def runge(size):
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
    return(T, X1,X2)

T, X1, X2 = runge(2000)
D = [x1 + A2*math.sin(W*t)+B1+B2 for x1,t in zip(X1,T)]
S = [x1+B2 for x1 in X1]
X2_S = [A2*math.sin(W*t) for t in T]

plt.figure(1)
plt.plot(T,D,linewidth=0.8)
plt.plot(T,X2_S,linewidth=0.8)
plt.grid(True)
plt.ylim(-.3, 3)

plt.figure(1)
plt.plot(T,S,linewidth=0.8)
plt.grid(True)
plt.ylim(-.3, 3)

plt.show()
