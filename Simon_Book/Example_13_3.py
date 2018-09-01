import numpy as np
#import numpy.linalg.cholesky as chsky
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Local calls to own modules
from gauss import GAUSS_PY

# state Altitude is represented with x1
# state Velocity is represented with x2
# state Ballistic Coeficient is represented with x3


def std(myArray):
	Array = myArray[1:]
	N = len(Array)
	mu = sum(Array)/N
	norm = math.sqrt(sum([(x-mu)*(x-mu) for x in Array]))
	norm2 = norm*norm
	var = norm2/N
	sigma = math.sqrt(var)
	return (sigma)

	
def g1(X,t):
	X2 = X[1]
	dX1dt = X2 
	return (dX1dt)
		
def g2(X,t):
	X1=X[0]
	X2=X[1]
	X3=X[2]
	R0 = 0.0034
	g = 32.2
	k = 22000
	
	dX2dt = RO*(math.exp(-X1/k)*X2*X2)/(2*X3) - g
	return (dX2dt)
		
def g3(x,t):
	dX3dt = 0
	return (dX3dt)

def rk4_X(TS,X,HP,TP):
	t = TP
	dt = HP
	T_SIM = 0.0
	X1 =X[0]
	X2 =X[1]
	X3 = X[2]
	
	while (T_SIM<=TS-.000001):
		T_SIM = T_SIM+dt
		x_next = [X1,X2,X3]
		k11 = dt*g1(x_next,t)
		k21 = dt*g2(x_next,t)
		k31 = dt*g3(x_next,t)
		
		x_next = [X1+0.5*k11,X2+0.5*k21,X3+0.5*k31]
		k12 = dt*g1(x_next,t+0.5*dt )
		k22 = dt*g2(x_next,t+0.5*dt)
		k32 = dt*g3(x_next,t+0.5*dt)
		
		x_next = [X1+0.5*k12,X2+0.5*k22,X3+0.5*k32]
		k13 = dt*g1(x_next,t+0.5*dt)
		k23 = dt*g2(x_next,t+0.5*dt)
		k33 = dt*g3(x_next,t+0.5*dt)
		
		x_next = [X1+k13,X2+k23,X3+k33]
		k14 = dt*g1(x_next,t+dt)
		k24 = dt*g2(x_next,t+dt)
		k34 = dt*g3(x_next,t+dt)
		
		X1 = X1 + (k11+2*k12+2*k13+k14)/6
		X2 = X2 + (k21+2*k22+2*k23+k24)/6
		X3 = X3 + (k31+2*k32+2*k33+k34)/6
		
	X1a = X1
	X2a = X2
	X3a = X3
	return ([X1a,X2a, X3a])
		
Arrayt = []

ArrayX1 = []
ArrayX2 = []
ArrayX3 = []
ArrayX1H = []
ArrayX2H = []
ArrayX3H = []

ArraySP11 = []
ArraySP11P = []
ArraySP33 = []
ArraySP33P = []

errorArrayX12=[]
errorArrayX1=[]
errorArrayX22=[]
errorArrayX2=[]
errorArrayX32=[]
errorArrayX3=[]

t=0.
HP=0.0004
TS=0.00005

RO = 0.0034
G = 32.2
K_C = 22000
	
X1 = 100000.
X2 = -6000.0
X3 = 2000.0
Xhat= [X1+10.0,X2-100.0,X3+500.0]
X= [X1,X2,X3]
PlotStep =200

ControlNoise = 0.01; #standard dev of uncertainty in control inputs
MeasNoise = 100.0; # standard deviation of measurement noise
xdotNoise = [500.0, 20000.0, 250000.0];
Q = np.matrix([[0, 0, 0],[0, 0, 0],[0, 0, 0]]) # Process noise covariance
R = np.matrix([[MeasNoise]]) # Measurement noise covariance
P = np.matrix([[xdotNoise[0], 0, 0],[0, xdotNoise[1], 0,],[0, 0, xdotNoise[2]]])
H = np.matrix([[1.0, 0., 0.]])

dt=  0.0004
tf = 16
add_noise = 1
EULER = 2
i=0
size = int(tf/dt)+1
count=0
# Begin simulation loop
ArrayX1.append(X[0])
ArrayX2.append(X[1])
ArrayX3.append(X[2])
ArrayX1H.append(Xhat[0])
ArrayX2H.append(Xhat[1])
ArrayX3H.append(Xhat[2])
errorArrayX1.append((X[0]-Xhat[0]))
errorArrayX2.append((X[1]-Xhat[1]))
errorArrayX3.append((X[2]-Xhat[2]))
ArraySP11.append(P[0,0])
ArraySP33.append(P[1,1])
ArraySP11P.append(-P[0,0])
ArraySP33P.append(-P[1,1])
f.write(str(X[1])+','+str(Xhat[1])+'\n')

for t in [X*dt for X in range(1,size+1)]:
   # Simulate the system (rectangular integration).
	if EULER == 1:
		X1D=g1(X,t)
		X2D=g2(X,t)
		X3D=g3(X,t)
		X1 = X[0] + dt*X1D 
		X2 = X[1] + dt*X2D 
		X3 = X[2] + dt*X3D 
	else:
		[X1,X2,X3] = rk4_X(TS,X,HP,t)
		
	X = [X1, X2, X3]

	X_MAT = np.matrix(X)

   # Simulate the measurement.
	if add_noise == 1:
		V =  np.matrix([[math.sqrt(MeasNoise)*np.random.randn()]])
	else:
		V = np.matrix([[math.sqrt(0.0)*np.random.randn()]])
		
	z = H * X_MAT.transpose() + V
   # Simulate the filter.
	A21 = -RO*math.exp(-Xhat[0]/K_C)*Xhat[1]*Xhat[1]/(2*K_C*Xhat[2])
	A22 =  RO*math.exp(-Xhat[0]/K_C)*(Xhat[1]/Xhat[2])
	A23 = -RO*math.exp(-Xhat[0]/K_C)*Xhat[1]*Xhat[1]/(2*Xhat[2]*Xhat[2])

	F = np.matrix([[0., 1., 0.],
				   [A21, A22, A23],
				   [0., 0., 0.]])
	
	# RICATTI EQUATION FOR CONTINUES TIME PROCESS
	#                                
 	# PDOT = F*P + P*F' + Q - P*H'*inv(R)*H*P
	# where F,H are from the linearisation equation:
	#		xdot = F*x + B*u
	#		y    = H*x
	#       Q is the process noise covariance
	#	    R is the measurement noise covariance
	#		P is the error estimation covariance	
	Pdot = F * P + P * F.transpose() - P * H.transpose()*inv(R) * H * P;

	# Solve Riccati equation using simple Euler integration rule
	P = P + Pdot * dt;	
	K = P * H.transpose()*inv(R)
	xhatdot = [g1(Xhat,t),g2(Xhat,t),g3(Xhat,t)]
	
	# Convert xhat to a Matrix 
	Xhat_MAT  = np.matrix(Xhat).transpose()
	# Find Kalman Gain for the filter
	xhatdot_MAT = np.matrix(xhatdot).transpose() + K * (z - H * Xhat_MAT);
	# Predicted value propagated
	Xhat_MAT = Xhat_MAT + xhatdot_MAT * dt
	# Converts the matrix to a vector
	Xhat = [Xhat_MAT[0,0], Xhat_MAT[1,0], Xhat_MAT[2,0]]
	t=t+dt
	
	i = i + 1;
	if i == PlotStep:
		i = 0;
		ArrayX1.append(X[0])
		ArrayX2.append(X[1])
		ArrayX3.append(X[2])
		ArrayX1H.append(Xhat[0])
		ArrayX2H.append(Xhat[1])
		ArrayX3H.append(Xhat[2])
		errorArrayX1.append((X[0]-Xhat[0]))
		errorArrayX2.append((X[1]-Xhat[1]))
		errorArrayX3.append((X[2]-Xhat[2]))
		ArraySP11.append(math.sqrt(P[0,0]))
		ArraySP33.append(math.sqrt(P[1,1]))
		ArraySP11P.append(-math.sqrt(P[0,0]))
		ArraySP33P.append(-math.sqrt(P[1,1]))

size = len(errorArrayX1)
Arrayt = [PlotStep*dt*i for i in range(0,size)]
AltErr = std(errorArrayX1)
VelErr = std(errorArrayX2)
BetaErr = std(errorArrayX3)

print('Continuous EKF RMS altitude estimation error = '+ str(AltErr))
print('Continuous EKF RMS velocity estimation error = '+ str(VelErr))
print('Continuous EKF RMS ballistic coefficient estimation error = '+ str(BetaErr))

plt.figure(1)
plt.grid(True)
plt.plot(Arrayt,errorArrayX1,linewidth=0.6)
plt.plot(Arrayt,ArraySP11,linewidth=0.6)
plt.plot(Arrayt,ArraySP11P,linewidth=0.6)
plt.ylabel('Error position (feet))')
plt.ylim(-30,30)

plt.figure(2)
plt.grid(True)
plt.plot(Arrayt,errorArrayX2, linewidth=0.6)
plt.plot(Arrayt,ArraySP33,linewidth=0.6)
plt.plot(Arrayt,ArraySP33P,linewidth=0.6)
plt.ylabel('Error velocity (feet/s)')
plt.ylim(-40,40)

plt.figure(3)
plt.grid(True)
plt.plot(Arrayt,errorArrayX3,linewidth=0.6)
plt.ylabel('Error Ballistic Coeficient')

plt.figure(4)
plt.grid(True)
plt.plot(Arrayt,ArrayX1, linewidth=0.6)
plt.ylabel('True position (feet))')

plt.figure(5)
plt.grid(True)
plt.plot(Arrayt,ArrayX2, linewidth=0.6)
plt.ylabel('True velocity (feet/s)')

plt.figure(6)
plt.grid(True)
plt.plot(Arrayt,errorArrayX3,linewidth=0.6)
plt.ylabel('True Ballistic Coeficient')
plt.show()
