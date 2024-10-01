
from numpy import *
import matplotlib.pyplot as plt

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

m = 1 
N = 200 # points
a = 0
amplitudes = linspace(0,2,N)
def gquad(function, b):
    xp, wp = gaussxwab(N,a,b)
    s = 0.0
    for k in range(N):
        s += wp[k]*function(xp[k])
    return s

def dTdt(x):
    return sqrt(8*m)/sqrt(amp**4-x**4)
    
S = list()
for amp in amplitudes:
    S.append(gquad(dTdt,amp))

plt.figure(figsize=(7,4))
plt.title("Period of an anharmonic oscillator vs. Amplitude")
plt.xlabel("Amplitude (m)")
plt.ylabel("Period (s)")
plt.plot(amplitudes,S)
plt.savefig('oscillator.png')
plt.show()
