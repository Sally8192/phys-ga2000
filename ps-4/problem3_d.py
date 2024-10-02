import numpy as np
from numpy import ones,copy,cos,tan,pi,linspace

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*tan(a)))

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

from math import factorial

def H(n,x):
    h=[np.ones(np.shape(x)),2*x]
    if n<2:
        return h[n]
    else:
        for i in range(2,n+1):
            h.append(2*x*h[i-1]-2*(i-1)*h[i-2])
    return h[n]



def inv_coeff_2(n):
    return (2**n)*factorial(n)*np.sqrt(np.pi)


from scipy.special import roots_hermite
x,w=roots_hermite(7)
var_5_gh = np.sum(w*(x**2)*(np.abs(H(5,x))**2))
print(np.sqrt(var_5_gh/inv_coeff_2(5)))