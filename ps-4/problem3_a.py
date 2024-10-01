import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

# Function to calculate Hermite polynomial Hn(x) recursively
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


def psi_n(n, x):
    prefactor = 1 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
    return prefactor * np.exp(-x**2 / 2) * H(n, x)


x_values = np.linspace(-4, 4, 400)
plt.figure(figsize=(8, 6))

for n in range(4):
    psi = psi_n(n, x_values)
    plt.plot(x_values, psi, label=f'n = {n}')

plt.xlabel('x')
plt.ylabel('Wavefunction (ψn(x))')
plt.title('Harmonic Oscillator Wavefunctions')
plt.legend()
plt.savefig('quantum_a.png')
plt.show()
