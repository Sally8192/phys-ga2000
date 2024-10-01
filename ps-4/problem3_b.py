import math
import numpy as np
import matplotlib.pyplot as plt

def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        H0 = 1
        H1 = 2 * x
        for i in range(2, n + 1):
            Hn = 2 * x * H1 - 2 * (i - 1) * H0
            H0 = H1
            H1 = Hn
        return Hn
    
n = 30
x = np.linspace(-10, 10, 1000)
psi_n = (1 / (2**n * math.factorial(n) * math.sqrt(math.pi))) * np.exp(-x**2 / 2) * H(n, x)

plt.figure(figsize=(8, 6))
plt.plot(x, psi_n)
plt.xlabel('x')
plt.ylabel('Wavefunction (ψn(x))')
plt.title('Harmonic Oscillator Wavefunction for n = 30')
plt.savefig('quantum_b.png')
plt.show()