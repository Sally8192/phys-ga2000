import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi

class rksolve:
    
    def __init__(self, f):
        self.f = f
        self.initial_conditions = None
        self.solution = None
        
    def iterate(self, a, b, N=1000):
        f = self.f
        r0 = np.array(self.initial_conditions, float)  
        
        h = (b - a) / N
        tpoints = np.arange(a, b, h)
        solution = np.empty(tpoints.shape + r0.shape, float)
        
        r = r0
        for i, t in enumerate(tpoints):
            solution[i] = r
            k1 = h * f(r, t)
            k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
            k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
            k4 = h * f(r + k3, t + h)
            r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        self.h = h
        self.solution = solution
        self.t = tpoints

def trajectory(m=1):
    ro = 1.22  # Air density (kg/m^3)
    C = 0.47   # Drag coefficient
    g = 9.81   # Acceleration due to gravity (m/s^2)
    R = 8e-2   # Radius of the cannonball (m)
    
    def f(r, t):
        r = np.array(r) 
        x, y, vx, vy = r
        v = sqrt(vx**2 + vy**2)
        F_fr = 1/2 * pi * R**2 * ro * C * v**2
        
        Dr = [vx, vy]
        
        Dvx = -F_fr / m * vx / v
        Dvy = -F_fr / m * vy / v - g
        Dv = [Dvx, Dvy]
        
        return np.array(Dr + Dv) 
    
    prob = rksolve(f)
    r0 = [0, 0]  # Initial position (x0, y0)
    v0e = 100 * np.exp(1j * np.radians(30))  # Initial velocity with 30-degree angle
    v0 = [v0e.real, v0e.imag]  # Convert complex number to real and imaginary parts
    prob.initial_conditions = r0 + v0
    prob.iterate(0, 10, N=1000)
    
    x = prob.solution[:, 0]
    y = prob.solution[:, 1]
    
    plt.plot(x[y > 0], y[y > 0], label=f'Mass {m} kg')
    
    return x[np.abs(y) < 0.2][-1]  


m_range = np.arange(1, 3.5, 0.5)  
x_ground = [trajectory(m) for m in m_range]

# Plot the trajectories
plt.title('Trajectories of Cannonball for Different Masses')
plt.legend()
plt.xlabel('Horizontal distance (m)')
plt.ylabel('Vertical distance (m)')
plt.savefig('cannonball_trajectories.png')
plt.show()

# Plot mass vs distance traveled
plt.plot(m_range, x_ground)
plt.title('Distance Traveled as a Function of Mass')
plt.xlabel('Mass (kg)')
plt.ylabel('Distance Traveled (m)')
plt.savefig('mass_vs_distance.png') 
plt.show()


m = 1  # Mass is fixed at 1 kg
distance_traveled = trajectory(m)

# Print the result for m = 1 kg
print(f"Mass {m} kg: Distance traveled = {distance_traveled:.2f} meters")