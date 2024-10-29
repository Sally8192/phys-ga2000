import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brent

# Define the function to minimize
f = lambda x: (x - 0.3)**2 * np.exp(x)

def brent_method(f, a, b, tol=1e-7):
    # Initial values
    fa = f(a)
    fb = f(b)
    if fa > fb:
        a, b = b, a
        fa, fb = fb, fa
    
    c = a  # Initial midpoint
    fc = fa
    mflag = True  # Flag to indicate if the last step was a bisection

    b_list = []  # To store estimates of the root
    err_list = []  # To store error estimates

    while abs(b - a) > tol:
        # Save the current estimate
        b_list.append(b)
        err_list.append(abs(b - a))

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Bisection method
            s = (a + b) / 2
        
        # Check if s is in the range [a, b]
        if (s < (3 * a + b) / 4 or s > (a + 3 * b) / 4):
            s = (a + b) / 2  # Use bisection if s is out of range

        fs = f(s)

        # Update a, b, c based on the function values
        if fs < fb:
            if fs < fb:
                c = b
                b = s
                fb = fs
            else:
                a = s
                fa = fs
        else:
            if fs < fc:
                a = c
                c = s
                fa = fc
            else:
                a = s
                fa = fs
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    
    # Final estimates
    b_list.append(b)
    err_list.append(abs(b - a))
    return (a + b) / 2, b_list, err_list  # Return the midpoint as the minimum

def plot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list if err > 0]  # Only log of positive errors
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axs[0], axs[1]

    # Plot root estimates
    ax0.scatter(range(len(b_list)), b_list, marker='o', facecolor='red', edgecolor='k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha=0.5)
    ax1.plot(range(len(err_list)), log_err, '.-')
    ax1.set_xlabel('Number of iterations')
    ax0.set_ylabel(r'$x_{min}$')
    ax1.set_ylabel(r'$\log{\delta}$')
    plt.savefig('convergence.png')
    plt.show()

if __name__ == "__main__":
    # Define the interval for optimization
    a, b = -0.5, 1.0
    tol = 1e-7

    # Perform Brent's optimization
    min_b, b_list, err_list = brent_method(f, a, b)
    print(f'Brent Method Minimum: x = {min_b}, f(x) = {f(min_b)}')

    # Compare with scipy.optimize.brent
    min_scipy = brent(f, brack=(a, b), tol=tol)
    print(f'Scipy Brent Minimum: x = {min_scipy}, f(x) = {f(min_scipy)}')

    # Plot the convergence
    plot(b_list, err_list)
