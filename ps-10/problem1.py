import numpy as np
from banded import banded  
from vpython import curve, rate, scene, color
import matplotlib.pyplot as plt 

h = 1e-18 * 10  # Time step (seconds)
hbar = 1.0546e-36  # Reduced Planck's constant (J·s)
L = 1e-8  # Box length (meters)
M = 9.109e-31  # Electron mass (kg)
N = 1000  # Grid slices
a = L / N  # Spatial step size

# Crank-Nicolson coefficients
a1 = 1 + h * hbar / (2 * M * a ** 2) * 1j
a2 = -h * hbar * 1j / (4 * M * a ** 2)
b1 = 1 - h * hbar / (2 * M * a ** 2) * 1j
b2 = h * hbar * 1j / (4 * M * a ** 2)

# Initial wavefunction psi(x, 0) = exp(-(x - x0)**2 / (2 * sigma**2)) * exp(1j * k * x)
x0 = L / 2
sigma = 1e-10  # Width of the Gaussian wave packet
k = 5e10  # Wave number
x = np.linspace(0, L, N+1)

# Initialize wavefunction vector
ksi = np.zeros(N+1, complex)

# Define initial wavefunction
def ksi0(x):
    return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k * x)

# Set initial wavefunction
ksi[:] = ksi0(x)

# Boundary conditions: psi = 0 at x=0 and x=L
ksi[0] = ksi[N] = 0

# Matrix A for Crank-Nicolson (tridiagonal matrix)
A = np.zeros((3, N), complex)
A[0, :] = a2
A[1, :] = a1
A[2, :] = a2

# Set up visual curve for animation
ksi_c_real = curve(color=color.orange, radius=0.005)  # Real part in orange
ksi_c_real.x = x - L / 2  # Center the x-axis around L / 2

# Function to save static plots
def save_plot(ksi, time_step):
    plt.figure()
    plt.plot(x, np.real(ksi), label='Re(ψ(x,t))', color='blue')
    plt.title(f"Wavefunction at t = {time_step:.2e} s")
    plt.xlabel('x (m)')
    plt.ylabel('ψ(x,t)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"wfn_t{time_step:.2e}.png")  # Save as PNG file with unique name
    plt.close()

# Animation loop with saving static plots at specific intervals
for time_step in range(100):
    rate(30)  # Control the frame rate (30 frames per second)

    # Update the visual curve with the real part of the wavefunction
    ksi_c_real.y = np.real(ksi) * 1e-8  # Real part of psi (scaled for better visualization)

    # Save static plots every 5 steps (you can change the interval)
    if time_step % 5 == 0:  # Save plot every 5 steps with time step as part of the filename
        save_plot(ksi, time_step * h)  # Save plot

    # Time evolution: solve for the next time step
    for i in range(1, N):
        # Create the right-hand side vector v = B * psi
        v = b1 * ksi[1:N] + b2 * (ksi[2:N+1] + ksi[0:N-1])

        # Use the banded solver to solve for the next psi
        ksi[1:N] = banded(A, v, 1, 1)
