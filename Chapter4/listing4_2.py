import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


K_G = open('kalman_gains2.txt','w') 
SP_G = open('SP_errors2.txt','w') 
 
K_G.write('XN,K1,K1GM,K2,K2GM,K3,K3GM') 
SP_G.write('XN,SP11,SP11GM,SP22,SP22GM,SP33,SP33GM')
 
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

for XN in range(1,101):
	M=PHI*P*PHI.transpose()
	# Kalman gaing given by the solution of Riccati equation
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))  
	P=(I-K*H)*M;
	# Formula for the variance of the errors in the estimates using recursive least square approximation
	if(XN<3):
		P11GM=9999999999.
		P22GM=9999999999.
		P33GM=9999999999.
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
	xn.append(XN)
	k1.append(K1)
	k1gm.append(K1GM)
	k2.append(K2)
	k2gm.append(K2GM)
	k3.append(K3)
	k3gm.append(K3GM)

	sp11.append(SP11)
	sp11gm.append(SP11GM)
	sp22.append(SP22)
	sp22gm.append(SP22GM)
	sp33.append(SP33)
	sp33gm.append(SP33GM)
	K_G.write( str(XN)+','+str(K1)  +','+str(K1GM)  +','+str(K2)  +','+ str(K2GM) +','+str(K3)  +','+ str(K3GM)+'\n') 
	SP_G.write(str(XN)+','+str(SP11)+','+str(SP11GM)+','+str(SP22)+','+str(SP22GM)+','+str(SP33)+','+str(SP33GM) +'\n')
	
	

fig1, ax1 = plt.subplots()
plt.grid(True)
plt.plot(xn,sp11)
plt.plot(xn,sp11gm)
ax1.yaxis.set_ticks(np.arange(0, 1, 0.1))
plt.xlim(0,100)
plt.ylim(0,1)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of First State')

fig2, ax2 = plt.subplots()
plt.grid(True)
plt.plot(xn,sp22)
plt.plot(xn,sp22gm)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of Second State')
ax2.yaxis.set_ticks(np.arange(0, 1, 0.05))
plt.xlim(0,100)
plt.ylim(0,0.5)

fig3, ax3 = plt.subplots()
plt.grid(True)
plt.plot(xn,k1)
plt.plot(xn,k1gm)
plt.xlabel('Number of Measurements')
plt.ylabel('First Kalman Gain')
ax3.yaxis.set_ticks(np.arange(0, 1, 0.1))
plt.xlim(0,100)
plt.ylim(0,1)

fig4, ax4 = plt.subplots()
plt.grid(True)
plt.plot(xn,k2)
plt.plot(xn,k2gm)
plt.xlabel('Number of Measurements')
plt.ylabel('Second Kalman Gain')
ax4.yaxis.set_ticks(np.arange(0, 1, 0.01))
plt.xlim(0,100)
plt.ylim(0,0.1)

fig5, ax5 = plt.subplots()
plt.grid(True)
plt.plot(xn,k3)
plt.plot(xn,k3gm)
plt.xlabel('Number of Measurements')
plt.ylabel('Third Kalman Gain')
ax5.yaxis.set_ticks(np.arange(0, 1, 0.01))
plt.xlim(0,100)
plt.ylim(0,0.1)

fig6, ax6 = plt.subplots()
plt.grid(True)
plt.plot(xn,sp33)
plt.plot(xn,sp33gm)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of Third State')
ax6.yaxis.set_ticks(np.arange(0, 1, 0.01))
plt.xlim(0,100)
plt.ylim(0,0.1)

plt.show()
K_G.close() 
SP_G.close() 
