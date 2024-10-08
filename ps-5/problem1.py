
# Exercise 5.15
# f(x) = 1+0.5*tanh(2x)
# f'(x) = 0.5*d/dx(tanh(2x)) = 0.5*2*(1-tanh^2(2x)) = 1-tanh^2(2x) = 1/(cosh^2(2x))
# Calculate f'(x) and compare to true value using central difference derivatives

from math import tanh, cosh
from numpy import linspace
from matplotlib.pyplot import plot, show, legend, xlabel, ylabel, title, savefig

a = -2.0  
b = 2.0  
N = 500   
h = 1e-8  

def f(x):
    return 1 + 0.5 * tanh(2 * x)

def fprime_true(x):
    return 1 / cosh(2 * x) ** 2

xpoints = linspace(a, b, N)

dpoints_estimated = []
dpoints_true = []


for x in xpoints:
    df = (f(x + 0.5 * h) - f(x - 0.5 * h)) / h
    dpoints_estimated.append(df)
    dpoints_true.append(fprime_true(x))

plot(xpoints, dpoints_estimated, "ko", label="Calculated Derivative", markersize=4)  # Black dots for estimated
plot(xpoints, dpoints_true, label="True Derivative", color='r')  # Blue line for true derivative

xlabel("x values")
ylabel("Derivative f'(x)")
title("Numerical vs. Analytical Derivative of f(x)")
legend()

savefig("derivative_plot.png")

show()
