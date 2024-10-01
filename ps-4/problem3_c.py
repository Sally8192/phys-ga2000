from math import *
from numpy import *
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
    
def psi(n,x):
    
    n = int(n)
    coeff = 1/sqrt( 2**n*factorial(n)*sqrt(pi) )
    ex = exp(-x**2/2)
    herm = H(n,x)
    
    return coeff*ex*herm

def gaussxw(N):
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gquad(function, b):
    xp, wp = gaussxwab(N,a,b)
    s = 0.0
    for k in range(N):
        s += wp[k]*function(xp[k])
    return s

def dfdz(z):
    coeff = ((z/(1-z**2))**2)*(1+z**2)/((1-z**2)**2)
    wf = psi(n,(z/(1-z**2)))
    return coeff*abs(wf)**2

N = 100 
n = 5
a = -1

val = gquad(dfdz,1)
print(f"The uncertainty for n = {n} is {round(sqrt(val),3)}")