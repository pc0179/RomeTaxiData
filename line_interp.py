import numpy as np
import matplotlib.pyplot as plt

x2 = -10
y2 = -10
x1 = 10
y1 = 5
t1 = 0
t2 = 5
T =4.804

def mid_point(x1,y1,t1,x2,y2,t2,T):
	dt = t2-t1
	dT = T-t1
	xT = dT*(x2-x1)/dt + x1
	yT = dT*(y2-y1)/dt + y1

	return xT,yT

xT,yT =mid_point(x1,y1,t1,x2,y2,t2,T)

plt.plot(x1,y1,'*b')
plt.plot(x2,y2,'*r')
plt.plot(xT,yT,'ok')
plt.show()


