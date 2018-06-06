import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Local calls to own modules
from gauss import GAUSS_PY

thetdeg=[]
bias=[]
biash=[]
sf=[]
sfh=[]
xk=[]
xkh=[]
biaserr=[]
sp11=[]
sp11p=[]
sferr=[]
sp22=[]
sp22p=[]
xkerr=[]
sp33=[]
sp33p=[]
actnoise=[]
sigr=[]
sigrp=[]

SIGTH=.000001;
S=0.;
BIAS=.00001*32.2
SF=.000005
XK=.000001/32.2
SIGTH=.000001
G=32.2
BIASH=0.
SFH=0.
XKH=0
SIGMA_NOISE=.000001
PHIS=0


#Because the systems dynamics matrix can be seen from the preceding equation to be zero or
#	| 0 0 0 |
#F =| 0 0 0 |
#	| 0 0 0 |
#
# Then, the discrete fundamental matrix must be the identity matrix because
#
#     					| 1 0 0 | 	| 0 0 0 |		| 1 0 0 |
# PHIK = I +Ft + Ft^2 = | 0 1 0 | + | 0 0 0 |*TS =  | 0 1 0 | 
#	  					| 0 0 1 |	| 0 0 0 |		| 0 0 1 |
#
# RK = E[(vk)^2] = g^2*sin^2[(THETAK)]*SIGMA^2
#
#

PHI = np.matrix([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
P = np.matrix([[99999999., 0, 0],[0,99999999. , 0], [0, 0, 99999999.]])
I = np.matrix([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
Q = np.matrix([[0, 0, 0],[0, 0, 0] ,[0, 0, 0] ])

for THETDEG in range(0, 182, 2):
	THET=THETDEG/57.3
	THETNOISE = GAUSS_PY(SIGTH)
	THETS=THET+THETNOISE
	H = np.matrix([[1, G*math.cos(THETS) , (G*math.cos(THETS))**2]])
	R=(G*math.sin(THETS)*SIGTH)**2
	M=PHI*P*PHI.transpose()+PHIS*Q
	K = M*H.transpose()*(inv(H*M*H.transpose() + R))
	P=(I-K*H)*M
	Z=BIAS+SF*G*math.cos(THETS)+XK*(G*math.cos(THETS))**2-G*math.cos(THET)+G*math.cos(THETS)
	RES=Z-BIASH-SFH*G*math.cos(THETS)-XKH*(G*math.cos(THETS))**2
	BIASH=BIASH+K[0,0]*RES
	SFH=SFH+K[1,0]*RES
	XKH=XKH+K[2,0]*RES
	# Need to use np.sqrt because sometimes the values are negatives. Using math.sqrt will not work.
	SP11=np.sqrt(P[0,0])
	SP22=np.sqrt(P[1,1])
	SP33=np.sqrt(P[2,2])
	BIASERR=BIAS-BIASH
	SFERR=SF-SFH
	XKERR=XK-XKH
	SP11P=-SP11
	SP22P=-SP22
	SP33P=-SP33
	ACTNOISE=G*math.cos(THETS)-G*math.cos(THET)
	SIGR=np.sqrt(R)
	SIGRP=-SIGR
	thetdeg.append(THETDEG)
	bias.append(BIAS)
	biash.append(BIASH)
	sf.append(SF)
	sfh.append(SFH)
	xk.append(XK)
	xkh.append(XKH)
	biaserr.append(BIASERR)
	sp11.append(SP11)
	sp11p.append(SP11P)
	sferr.append(SFERR)
	sp22.append(SP22)
	sp22p.append(SP22P)
	xkerr.append(XKERR)
	sp33.append(SP33)
	sp33p.append(SP33P)
	actnoise.append(ACTNOISE)
	sigr.append(SIGR)
	sigrp.append(SIGRP)

plt.figure(1)
plt.grid(True)
plt.plot(thetdeg,bias,linewidth=0.6)
plt.plot(thetdeg,biash,linewidth=0.6)
plt.xlabel('measurement angle (deg)')
plt.ylabel('actual measurement noise (rad)')
plt.xlim(0, 180)
plt.ylim(-.06, .06)
plt.show()
ds

plt.figure(1)
plt.grid(True)
plt.plot(thetdeg,actnoise,linewidth=0.6)
plt.plot(thetdeg,sigr)
plt.plot(thetdeg,sigrp)
plt.xlabel('measurement angle (deg)')
plt.ylabel('actual measurement noise (rad)')
plt.xlim(0, 180)
plt.ylim(-.00006, .00006)

plt.figure(2)
plt.grid(True)
plt.plot(thetdeg,biaserr,linewidth=0.6)
plt.plot(thetdeg,sp11)
plt.plot(thetdeg,sp11p)
plt.xlabel('measurement angle (deg)')
plt.ylabel('error in estimate of bias (ft/sec^2)')
plt.xlim(0, 180)
plt.ylim(-.00005 ,.00005)

plt.figure(3)
plt.grid(True)
plt.plot(thetdeg,sferr,linewidth=0.6)
plt.plot(thetdeg,sp22)
plt.plot(thetdeg,sp22p)
plt.xlabel('measurement angle (deg)')
plt.ylabel('error in estimate of scale factor')
plt.xlim(0,180)
plt.ylim(-.00005, .00005)

plt.figure(4)
plt.grid(True)
plt.plot(thetdeg,xkerr,linewidth=0.6)
plt.plot(thetdeg,sp33)
plt.plot(thetdeg,sp33p)
plt.xlabel('measurement angle (deg)')
plt.ylabel('error in estimate of g-sensitive drift (sec^2/ft)')
plt.xlim(0 ,180)
plt.ylim(-.000005, .000005)

plt.show()
