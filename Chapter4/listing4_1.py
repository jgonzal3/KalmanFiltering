import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


K_G = open('kalman_gains.txt','w') 
SP_G = open('SP_errors.txt','w') 
 
K_G.write('XN,K1,K1GM,K2,K2GM') 
SP_G.write('XN,SP11,SP11GM,SP22,SP22GM')
 
xn    = []
k1    = []
k1gm  = []
k2    = []
k2gm  = []
sp11  = []
sp11gm= []
sp22  = []
sp22gm= []
	
TS=1.
SIGMA_NOISE=1.

PHI = np.matrix([[1, TS],[0,1]])
P = np.matrix([[9999999., 0],[0,99999999.]])
M = np.matrix([[0, 0],[0,0]])
H = np.matrix([[1, 0]])
I = np.matrix([[1, 0],[0,1]])
R = np.matrix([[SIGMA_NOISE**2]])

for XN in range(1,101):
	M=PHI*P*PHI.transpose()
	# Kalman gaing given by the solution of Riccati equation
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))  
	P=(I-K*H)*M;
	# Formula for the variance of the errors in the estimates using recursive least square approximation
	if(XN<2):
		P11GM=9999999999.
		P22GM=9999999999.
	else:
		#P11=2*(2*k-1)*SIGMA^2/(k*(k+1))	
		#P22=12*SIGMA^2/(k*(k*k-1)*TS*TS)
		P11GM=2*(2*XN-1)*SIGMA_NOISE*SIGMA_NOISE/(XN*(XN+1))
		P22GM=12*SIGMA_NOISE*SIGMA_NOISE/(XN*(XN*XN-1)*TS*TS)
	SP11=math.sqrt(P[0,0])
	SP22=math.sqrt(P[1,1])
	SP11GM=math.sqrt(P11GM)
	SP22GM=math.sqrt(P22GM)
	# Formula for the Kalman gain using recursive least square approximation
	# K1GM=2*(2*k-1)/(k*(k+1))
	# K2GM=6/(k*(k+1)*TS)
	K1GM=2*(2*XN-1)/(XN*(XN+1))
	K2GM=6/(XN*(XN+1)*TS)
	K1=K[0,0]
	K2=K[1,0]
	xn.append(XN)
	k1.append(K1)
	k1gm.append(K1GM)
	k2.append(K2)
	k2gm.append(K2GM)
	sp11.append(SP11)
	sp11gm.append(SP11GM)
	sp22.append(SP22)
	sp22gm.append(SP22GM)
	K_G.write( str(XN)+','+str(K1)  +','+str(K1GM)  +','+str(K2)  +','+ str(K2GM) +'\n') 
	SP_G.write(str(XN)+','+str(SP11)+','+str(SP11GM)+','+str(SP22)+','+str(SP22GM) +'\n')
	
plt.figure(1)
plt.grid(True)
plt.plot(xn,sp11)
plt.plot(xn,sp11gm)
plt.xlim(0,100)
plt.ylim(0,1)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of First State')

plt.figure(2)
plt.grid(True)
plt.plot(xn,sp22)
plt.plot(xn,sp22gm)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of Second State')
plt.xlim(0,100)
plt.ylim(0,0.1)

plt.figure(3)
plt.grid(True)
plt.plot(xn,k1)
plt.plot(xn,k1gm)
plt.xlabel('Number of Measurements')
plt.ylabel('First Kalman Gain')
plt.xlim(0,100)
plt.ylim(0,1)

plt.figure(4)
plt.grid(True)
plt.plot(xn,k2)
plt.plot(xn,k2gm)
plt.xlabel('Number of Measurements')
plt.ylabel('Second Kalman Gain')
plt.xlim(0,100)
plt.ylim(0,0.1)
plt.show()
K_G.close() 
SP_G.close() 
