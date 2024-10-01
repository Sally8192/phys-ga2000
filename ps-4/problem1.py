import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import roots_hermite


#1 The function to calculate gaussian quadrature was taken from the textbook
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

#1a)
def cv(T, N):
    theta_D = 428
    def f(x):
        return np.power((T/theta_D), 3)*np.power(x, 4)*np.exp(x)/(np.power((np.exp(x) - 1), 2))

    x, w = gaussxwab(N, 0, theta_D/T)
    integral = 0.0

    for k in range(N):
        integral += w[k]*f(x[k])
    return integral

#b)
int_list = []
for i in np.arange(5, 500, 1):
    int_val = cv(i, 50) * 9 * 10**(-3) * 6.022*10**(28) * 1.381*10**(-23)
    int_list.append(int_val)


plt.plot(np.arange(5, 500, 1), int_list)
plt.xlabel("Temperature (K)")
plt.ylabel("Specific Heat (J/K)")
plt.title("Heat Capacity")
plt.savefig("heat_capacity.png")
plt.show()

#c)
N_list = []
for i in np.arange(10, 80, 10):
    int_val = cv(50, i)* 9 * 10**(-3) * 6.022*10**(28) * 1.381*10**(-23)
    N_list.append(int_val)

plt.plot(np.arange(10, 80, 10), N_list)
plt.xlabel("Number of Sampled Points N")
plt.ylabel("Specific Heat at T = 50K (J/K)")
plt.title("Convergence of an Integral with Increase in Sampled Points\n")
plt.savefig("convergence.png")
plt.show()