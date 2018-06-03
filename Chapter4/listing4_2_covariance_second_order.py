import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

'''
Errors in estimates of first, second and third state of a second-order polynomial Kalman
filter are fairly insensitive to the initial covariance matrix.'''
 
xn    = []
k1    = []
k1gm  = []
k2    = []
k2gm  = []
k3    = []
k3gm  = []
sp11  = []
sp11gm= []
sp22  = []
sp22gm= []
sp33  = []
sp33gm= []
	
TS=1.
SIGMA_NOISE=1.

PHI = np.matrix([[1, TS, 0.5*TS*TS],[0, 1, TS] ,[0, 0, 1] ])
P = np.matrix([[9999999., 0, 0],[0,9999999. , 0], [0, 0, 9999999.]])
M = np.matrix([[0, 0, 0],[0, 0 ,0],[0, 0, 0]])
H = np.matrix([[1, 0 , 0]])
I = np.matrix([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
R = np.matrix([[SIGMA_NOISE**2]])

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for P0 in [999999.0, 100.0, 1.0, 0.1, 0]:

	P = np.matrix([[P0, 0, 0],[0, P0, 0], [0, 0, P0]])
	
	for XN in range(1,101):
		M=PHI*P*PHI.transpose()
		# Kalman gaing given by the solution of Riccati equation
		K = M*H.transpose()*(inv(H*M*H.transpose() + R))  
		P=(I-K*H)*M;
		# Formula for the variance of the errors in the estimates using recursive least square approximation
		if(XN<3):
			P11GM=P0
			P22GM=P0
			P33GM=P0
		else:
			# P11GM=(3*(3*k^2-3*k+2)/(k*(k+1)*(k+2)))*SIGMA^2
			# P22GM=(12*(16*k^2-30*k+11)/(k*(k^2-1)*(k^2-4)*TS*TS))*SIGMA^2
			# P33GM=(720/(k*(k^2-1)*(k^2-4)*TS*TS*TS*TS))*SIGMA^2
			P11GM=(3*(3*XN*XN-3*XN+2)/(XN*(XN+1)*(XN+2)))*SIGMA_NOISE**2
			P22GM=(12*(16*XN*XN-30*XN+11)/(XN*(XN*XN-1)*(XN*XN-4)*TS*TS))*SIGMA_NOISE**2
			P33GM=(720/(XN*(XN*XN-1)*(XN*XN-4)*TS*TS*TS*TS))*SIGMA_NOISE**2
		SP11=math.sqrt(P[0,0])
		SP22=math.sqrt(P[1,1])
		SP33=math.sqrt(P[2,2])
		SP11GM=math.sqrt(P11GM)
		SP22GM=math.sqrt(P22GM)
		SP33GM=math.sqrt(P33GM)
		# Formula for the Kalman gain using recursive least square approximation
		# K1GM=3*(3*k^2-3*k+2)/(k*(k+1)*(k+2))
		# K2GM=18*(2*k-1)/(k*(k+1)*(k+2)*TS)
		# K3GM=60/(k*(k+1)*(k+2)*TS*TS)
		K1GM=3*(3*XN*XN-3*XN+2)/(XN*(XN+1)*(XN+2))
		K2GM=18*(2*XN-1)/(XN*(XN+1)*(XN+2)*TS)
		K3GM=60/(XN*(XN+1)*(XN+2)*TS*TS)
		K1=K[0,0]
		K2=K[1,0]
		K3=K[2,0]
		if XN > 2:
      # Only collect terms above n=2
			xn.append(XN)
			sp11.append(SP11)
			sp22.append(SP22)
			sp33.append(SP33)
	ax1.plot(xn,sp11)
	ax2.plot(xn,sp22)
	ax3.plot(xn,sp33)
	sp11  = []
	sp22  = []	
	sp33  = []
	xn    = []
	
plt.figure(1)	
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,1.0)
ax2.yaxis.set_ticks(np.arange(0, 0.5, 0.1))
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of First State')

plt.figure(2)
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,0.5)
ax3.yaxis.set_ticks(np.arange(0, 1.0, 0.1))
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of Second State')

plt.figure(3)
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,0.5)
ax3.yaxis.set_ticks(np.arange(0, 0.5, 0.1))
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of Third State')
plt.show()
 
