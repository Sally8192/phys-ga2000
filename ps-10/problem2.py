import numpy as np
import matplotlib.pyplot as plt
from dcst import dst, idst  # Assuming the DCST package is available
from vpython import curve, rate, color

# Constants
L = 1e-8  # Box length (meters)
M = 9.109e-31  # Electron mass (kg)
sigma = 1e-10  # Width of the Gaussian wave packet (meters)
k = 5e10  # Wave number (m^-1)
hbar = 1.0546e-34  # Reduced Planck's constant (J·s)
N = 1000  # Number of slices (grid points)
x = np.linspace(0, L, N+1)  # Grid points (0 to L)
x0 = L / 2  # Center of the wave packet

# Initial wavefunction at t=0
psi_real = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.cos(k * x)  # Real part of the initial wavefunction
psi_imag = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.sin(k * x)  # Imaginary part

# Perform DST on real and imaginary parts to get alpha_k and eta_k
alpha_k = dst(psi_real)  # Real part coefficients
eta_k = dst(psi_imag)  # Imaginary part coefficients

# Function to compute the real part of the wavefunction at time t
def real_wavefunction(t, alpha_k, eta_k, x, N):
    """ Calculate the real part of the wavefunction at time t """
    psi_real_t = np.zeros_like(x, dtype=complex)
    for k in range(1, N):
        term = alpha_k[k] * np.cos((np.pi**2 * hbar * k**2 * t) / (2 * M * L**2)) - \
               eta_k[k] * np.sin((np.pi**2 * hbar * k**2 * t) / (2 * M * L**2))
        psi_real_t += term * np.sin(np.pi * k * x / L)
    return np.real(psi_real_t) / N

# Function to compute the imaginary part of the wavefunction at time t
def imaginary_wavefunction(t, alpha_k, eta_k, x, N):
    """ Calculate the imaginary part of the wavefunction at time t """
    psi_imag_t = np.zeros_like(x, dtype=complex)
    for k in range(1, N):
        term = alpha_k[k] * np.sin((np.pi**2 * hbar * k**2 * t) / (2 * M * L**2)) + \
               eta_k[k] * np.cos((np.pi**2 * hbar * k**2 * t) / (2 * M * L**2))
        psi_imag_t += term * np.sin(np.pi * k * x / L)
    return np.imag(psi_imag_t) / N

# Function to save a graph at a specific time
def save_plot(ksi_real, ksi_imag, time_step):
    plt.figure()
    plt.plot(x, ksi_real, label='Re(ψ(x,t))', color='blue')
    plt.plot(x, ksi_imag, label='Im(ψ(x,t))', color='red')
    plt.title(f"Wavefunction at t = {time_step:.2e} s")
    plt.xlabel('x (m)')
    plt.ylabel('ψ(x,t)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"wft_t{time_step:.2e}.png")  # Save as PNG file with unique name
    plt.close()

# Setup for animation using vpython
ksi_c_real = curve(color=color.cyan, radius=0.005)  # Curve for visualizing real part
ksi_c_imag = curve(color=color.red, radius=0.005)  # Curve for visualizing imaginary part
ksi_c_real.x = x - L / 2  # Center the x-axis around L / 2
ksi_c_imag.x = x - L / 2

# Animation loop: evolve the wavefunction over time and animate
t = 0  # Starting time
h = 1e-18  # Time step
total_time = 1e-15  # Total time for the animation

# Only save one graph at t = 10^-16s, so we check if time is close to it
save_at_t = 1e-16
plot_saved = False

while t < total_time:  # Run until total_time is reached
    rate(30)  # Adjust frame rate (30 frames per second)
    
    # Calculate real and imaginary parts of the wavefunction at time t
    ksi_real = real_wavefunction(t, alpha_k, eta_k, x, N)
    ksi_imag = imaginary_wavefunction(t, alpha_k, eta_k, x, N)
    
    # Update the visual curves with the real and imaginary parts of the wavefunction
    ksi_c_real.y = ksi_real * 1e-8  # Scale for better visualization
    ksi_c_imag.y = ksi_imag * 1e-8  # Scale for better visualization
    
    # Save a static plot at t = 10^-16s
    if not plot_saved and abs(t - save_at_t) < h:
        save_plot(ksi_real, ksi_imag, t)
        plot_saved = True
    
    # Save static plots less frequently (every 100 steps for instance)
    if int(t / h) % 100 == 0:
        save_plot(ksi_real, ksi_imag, t)
    
    # Increment time step
    t += h  # Increase time by one time step

# After the loop, you can have a final plot for the last time step
ksi_real_final = real_wavefunction(t, alpha_k, eta_k, x, N)
ksi_imag_final = imaginary_wavefunction(t, alpha_k, eta_k, x, N)
save_plot(ksi_real_final, ksi_imag_final, t)

