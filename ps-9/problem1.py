import numpy as np
import matplotlib.pyplot as plt

class rksolve:
    
    def __init__(self,f):
        self.f = f
        self.initial_conditions = None
        self.solution = None
        
    def iterate(self,a,b,N=1000):
        f = self.f
        r0 = np.array(self.initial_conditions, float)
        
        h = (b-a)/N
        
        tpoints = np.arange(a,b,h)
        solution = np.empty(tpoints.shape + r0.shape, float)
        
        # Initial conditions
        r = r0
        for i, t in enumerate(tpoints):
            solution[i] = r
            k1 = h*f(r,t)
            k2 = h*f(r+0.5*k1,t+0.5*h)
            k3 = h*f(r+0.5*k2,t+0.5*h)
            k4 = h*f(r+k3,t+h)
            r += (k1 + 2*k2 + 2*k3 + k4)/6
        
        self.solution = solution
        self.t = tpoints

# Problem A and B (Simple Harmonic Oscillator)

omega = 1

def f(r, t):
    x = r[0]
    x_d = r[1]
    fx = x_d
    fx_d = -omega**2 * x
    return np.array([fx, fx_d], float)

# Solve for the simple harmonic oscillator with initial condition x=1, dx/dt=0
prob = rksolve(f)
prob.initial_conditions = [1, 0]
prob.iterate(0, 50, N=500)
x = prob.solution[:, 0]
plt.plot(prob.t, x, label='Amplitude x = 1')

# Solve for the simple harmonic oscillator with initial condition x=2, dx/dt=0
prob = rksolve(f)
prob.initial_conditions = [2, 0]
prob.iterate(0, 50, N=500)
x = prob.solution[:, 0]
plt.plot(prob.t, x, label='Amplitude x = 2')

plt.legend()
plt.xlabel("Time (t)")
plt.ylabel("Displacement (x)")
plt.title("Simple Harmonic Oscillator")
# Save the plot to a file
plt.savefig('simple_harmonic_oscillator.png')
plt.show()

# Problem C (Anharmonic Oscillator)

def f_anharmonic(r, t):
    x = r[0]
    x_d = r[1]
    fx = x_d
    fx_d = -omega**2 * x**3
    return np.array([fx, fx_d], float)

# Solve for the anharmonic oscillator with initial condition x=1, dx/dt=0
prob = rksolve(f_anharmonic)
prob.initial_conditions = [1, 0]
prob.iterate(0, 50, N=1000)
x = prob.solution[:, 0]
plt.plot(prob.t, x, label='Amplitude x = 1')

# Solve for the anharmonic oscillator with initial condition x=1.1, dx/dt=0
prob = rksolve(f_anharmonic)
prob.initial_conditions = [1.1, 0]
prob.iterate(0, 50, N=1000)
x = prob.solution[:, 0]
plt.plot(prob.t, x, label='Amplitude x = 1.1')

plt.legend()
plt.xlabel("Time (t)")
plt.ylabel("Displacement (x)")
plt.title("Anharmonic Oscillator")
# Save the plot to a file
plt.savefig('anharmonic_oscillator.png')
plt.show()

# Problem D (Phase Space Plot for Anharmonic Oscillator)

prob = rksolve(f_anharmonic)
prob.initial_conditions = [1, 0]
prob.iterate(0, 50, N=1000)
x = prob.solution[:, 0]
x_d = prob.solution[:, 1]
plt.plot(x, x_d, label='Amplitude x = 1')

# Solve for the anharmonic oscillator with initial condition x=1.1, dx/dt=0
prob = rksolve(f_anharmonic)
prob.initial_conditions = [1.1, 0]
prob.iterate(0, 50, N=1000)
x = prob.solution[:, 0]
x_d = prob.solution[:, 1]
plt.plot(x, x_d, label='Amplitude x = 1.1')

plt.legend()
plt.xlabel("Displacement (x)")
plt.ylabel("Velocity (dx/dt)")
plt.title("Phase Space Plot for Anharmonic Oscillator")
# Save the plot to a file
plt.savefig('phase_space_anharmonic_oscillator.png')
plt.show()

# Problem E (Van der Pol Oscillator)

def f_vdp(r, t, mu, omega=1):
    x = r[0]
    x_d = r[1]
    fx = x_d
    fx_d = -omega**2 * x + mu * (1 - x**2) * x_d
    return np.array([fx, fx_d], float)

# Solve for the van der Pol oscillator with mu=1 and initial condition x=1, dx/dt=0
mu_values = [1, 2, 4]
for mu in mu_values:
    prob = rksolve(lambda r, t: f_vdp(r, t, mu))
    prob.initial_conditions = [1, 0]
    prob.iterate(0, 20, N=500)
    x = prob.solution[:, 0]
    x_d = prob.solution[:, 1]
    plt.plot(x, x_d, label=f'µ = {mu}')

plt.legend()
plt.xlabel("Displacement (x)")
plt.ylabel("Velocity (dx/dt)")
plt.title("Phase Space Plot for Van der Pol Oscillator")
# Save the plot to a file
plt.savefig('phase_space_vdp_oscillator.png')
plt.show()
