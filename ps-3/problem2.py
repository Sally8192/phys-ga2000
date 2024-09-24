import numpy as np
import matplotlib.pyplot as plt
from random import random

random()

h = 1 

Bi209 = 0
Pb209 = 0
Ti209 = 0
Bi213 = 10000

pPb = 1 - 2**(-h/3.3/60)
pTi = 1 - 2**(-h/2.2/60)
pBi = 1 - 2**(-h/46/60)

Bi209_list = []
Pb209_list = []
Ti209_lsit = []
Bi213_list = []

t = np.arange(0,2e4,h)
for ti in t:
	Bi209_list.append(Bi209)
	Pb209_list.append(Pb209)
	Ti209_lsit.append(Ti209)
	Bi213_list.append(Bi213)
	
	for i in range(Pb209):
		if random()<pPb:
			Pb209-=1
			Bi209+=1
	
	for i in range(Ti209):
		if random()<pTi:
			Ti209-=1
			Pb209+=1
	
	for i in range(Bi213):
		if random()<pBi:
			Bi213 -=1
			if random()<0.9791:
				Pb209+=1
			else:
				Ti209+=1
				
plt.plot(t,Bi209_list,label='Bi209')
plt.plot(t,Pb209_list,label='Pb209')
plt.plot(t,Ti209_lsit,label='Ti209')
plt.plot(t,Bi213_list,label='Bi213')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Number of Atoms')
plt.title('Decay of Isotopes')
plt.savefig("Decay.png")
plt.show()