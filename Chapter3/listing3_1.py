import numpy as np
import matplotlib.pyplot as plt
import math

# Local calls to own modules
from gauss import GAUSS_PY
from estimation import ESTIMATION_1,ESTIMATION_2,ESTIMATION_3

TS=0.1
SIGNOISE=1.
A0=1.
A1=0.
XH=0.
XN=0
L = int(10/TS)
xherr = []
xs =[]
xh = []
t =[]
sp11 = []
act =[]

# Generate the noise list so it is possible to test the same fortran code 
XNOISE_g = [GAUSS_PY(SIGNOISE) for k in range(0,L)]

for k in range(0,L):
	ACT=A0+A1*TS*k
	XS=ACT+XNOISE_g[k]  # signal plus noise
	K_f=1./(k+1)        # gain of filter
	RES=XS-XH	   # residuals
	XH=XH+K_f*RES   # x_k predicted
	SP11=SIGNOISE/math.sqrt(k+1)
	XHERR=ACT-XH
	EPS=0.5*A1*TS*(k)
	xherr.append(XHERR)
	xs.append(XS)
	xh.append(XH)
	t.append(TS*k)
	sp11.append(SP11)
	act.append(ACT)
	print('%.2f %.4f %.4f %.4f %.4f %.4f' % (TS*k,ACT,XS,XH,XHERR,EPS))

plt.figure(1)
plt.grid(True)
plt.plot(t,xh,dashes=[6, 2],label='Pedicted')
plt.plot(t,act,dashes=[6, 2],label='Measured')
plt.legend()
plt.show()	
