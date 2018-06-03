import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

''' Errors in estimates of first state of a zeroth-order polynomial Kalman
filter: fairly insensitive to the initial covariance matrix.'''

K_G = open('kalman_gains_0.txt','w') 
SP_G = open('SP_errors_0.txt','w') 
 
K_G.write('XN,K1,K1GM') 
SP_G.write('XN,SP11,SP11GM')
 
xn    = []
k1    = []
k1gm  = []
sp11 = []
sp11gm = []
	
TS=1.
SIGMA_NOISE=1.
P0 = 999999999.0
PHI = 1
P = P0
H = 1
I = 1
R = SIGMA_NOISE**2

fig2, ax2 = plt.subplots()

for P in [99999999.0, 100, 1, 0.1, 0]:
	for XN in range(1,101):
		M=PHI*P*PHI
		# Kalman gaing given by the solution of Riccati equation
		K = M*H/(H*M*H + R)  
		P=(I-K*H)*M;
		# Formula for the variance of the errors in the estimates using recursive least square approximation
		P11GM=SIGMA_NOISE*SIGMA_NOISE/XN
		SP11=math.sqrt(P)
		SP11GM=math.sqrt(P11GM)
		# Formula for the Kalman gain using recursive least square approximation
		# K1GM=1/k
		K1GM=1/XN
		K1=K
		xn.append(XN)
		k1.append(K1)
		k1gm.append(K1GM)
		sp11.append(SP11)
		sp11gm.append(SP11GM)
		#print(str(XN) +','+str(M) +','+str(K)+','+str(P)+','+str(K1GM)+'\n')
		#K_G.write( str(XN)+','+str(K1)  +','+str(K1GM) +'\n') 
		#SP_G.write(str(XN)+','+str(SP11)+','+str(SP11GM)+'\n')
	plt.plot(xn,sp11)
	xn = []
	sp11 = []

plt.grid(True)
plt.xlim(0,100)
plt.ylim(0,1)
ax2.yaxis.set_ticks(np.arange(0, 1, 0.1))
plt.xlabel('Number of Measurements')
plt.ylabel('Error in Estimate of First State')

plt.show()
K_G.close() 
SP_G.close() 
