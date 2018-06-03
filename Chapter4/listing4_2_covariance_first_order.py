import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


'''
Errors in estimates of first state and second state of a first-order polynomial Kalman filter:
fairly insensitive to the initial covariance matrix.
'''

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
M = np.matrix([[0, 0],[0,0]])
H = np.matrix([[1, 0]])
I = np.matrix([[1, 0],[0,1]])
R = np.matrix([[SIGMA_NOISE**2]])

fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for P0 in [999999.0, 100.0, 1.0, 0.1, 0]:
	P = np.matrix([[P0, 0],[0,P0]])

	for XN in range(1,101):
		M=PHI*P*PHI.transpose()
		# Kalman gaing given by the solution of Riccati equation
		K = M*H.transpose()*(inv(H*M*H.transpose() + R))  
		P=(I-K*H)*M;
		# Formula for the variance of the errors in the estimates using recursive least square approximation
		if(XN<2):
			P11GM=P0
			P22GM=P0
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
		sp11.append(SP11)
		sp11gm.append(SP11GM)
		sp22.append(SP22)
		sp22gm.append(SP22GM)
		xn.append(XN)		
	ax2.plot(xn,sp11)
	ax3.plot(xn,sp22)
  # Clear the lists before starting the new covariance value
	sp11  = []
	sp11gm= []	
	sp22  = []
	sp22gm= []
	xn    = []

plt.figure(1)	
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,1.0)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of First State')

plt.figure(2)
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,0.5)
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of Second State')
plt.show()
K_G.close() 
SP_G.close()

plt.show()
K_G.close() 
SP_G.close() 
