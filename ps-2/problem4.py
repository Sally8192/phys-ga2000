import matplotlib.pyplot as plt
import numpy as np

# Parameters
N = 1000  # Grid size
max_iter = 100  # Maximum number of iterations

# Create a grid of complex numbers
x = np.linspace(-2.0, 2.0, N)
y = np.linspace(-2.0, 2.0, N)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros(C.shape, dtype=complex)

# Initialize the escape time array
escape_time = np.zeros(C.shape, dtype=int)

# Iterate over each point
for i in range(N):
    for j in range(N):
        z = 0
        c = C[i, j]
        iteration = 0
        while abs(z) <= 2 and iteration < max_iter:
            z = z**2 + c
            iteration += 1
        escape_time[i, j] = iteration

# Plotting
plt.figure(figsize=(10, 10))
plt.imshow(escape_time, extent=(-2, 2, -2, 2), cmap='inferno', origin='lower')
plt.colorbar(label='Number of iterations')
plt.title('Mandelbrot Set')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.savefig('mandelbrot.png')
plt.show()
