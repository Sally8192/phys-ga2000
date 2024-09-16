import matplotlib.pyplot as plt
import numpy as np

N = 1000
x = np.linspace(-2, 2, N)
y = x
x, y = np.meshgrid(x, y)
Z = np.zeros([N, N], dtype=complex)  # Initialize Z with complex dtype
c = x + 1j * y

for i in range(N):
    for j in range(N):
        z = 0
        nn = 100
        while nn > 0:
            if abs(z) > 2.0:
                break
            z = z**2 + c[i][j]
            nn -= 1
        Z[i][j] = z

plt.figure(figsize=(9, 9))
plt.title("Mandelbrot Set")
plt.imshow(np.abs(Z), origin="lower")  # Display the magnitude of Z
plt.colorbar()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('mandelbrot.png')
plt.show()