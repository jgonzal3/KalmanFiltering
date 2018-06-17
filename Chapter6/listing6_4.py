import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

M1 = []
M2 = []
M3 = []
W =[]

w0=10.
for w in range(1,100):
	XMAG1=1/math.sqrt(1+(w/w0)**2)
	TOP1=1+2*(w/w0)**2
	BOT1=(1.-(w*w/(w0*w0)))**2+2*(w/w0)**2
	XMAG2=math.sqrt(TOP1/(BOT1+.00001))
	TOP2=(1-2*w*w/(w0*w0))**2+(2*w/w0)**2
	TEMP1=(1-2*w*w/(w0*w0))**2
	TEMP2=(2*w/w0-(w/w0)**3)**2
	XMAG3=math.sqrt(TOP2/(TEMP1+TEMP2+.00001))
	print (w,XMAG1,XMAG2,XMAG3)
	M1.append(XMAG1)
	M2.append(XMAG2)
	M3.append(XMAG3)
	W.append(w)

plt.figure(1)
plt.grid(True)
plt.semilogx(W,M1,label='Magnitud Zero Order',linewidth=0.6)
plt.semilogx(W,M2,label='Magnitud First Order',linewidth=0.6)
plt.semilogx(W,M3,label='Magnitud Second Order',linewidth=0.6)
plt.xlabel('Frequency (rads)')
plt.ylabel('Magnitud of Transfere funcion ')
plt.legend()
plt.show()
