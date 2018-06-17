import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

M1 = []
M2 = []
M3 = []
W =[]

W_=[5.0, 10.0, 25.0]

for w0 in W_:
	for w in range(1,100):
		TOP2=(1-2*w*w/(w0*w0))**2+(2*w/w0)**2
		TEMP1=(1-2*w*w/(w0*w0))**2
		TEMP2=(2*(w/w0)-(w/w0)**3)**2
		XMAG3=math.sqrt(TOP2/(TEMP1+TEMP2+.00001))
		M3.append(XMAG3)
		W.append(w)
	plt.semilogx(W,M3,linewidth=0.6)
	M3 = []
	W=[]

plt.figure(1)
plt.grid(True)
plt.xlabel('Frequency (rads)')
plt.ylabel('Magnitud of Transfere funcion ')
plt.show()
