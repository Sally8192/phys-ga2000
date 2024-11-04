import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

def nLL(beta, X_data, Y_data):
    # Return the negative log likelihood, which we can then minimize
    p = lambda x, beta: 1 / (1 + np.exp(-beta[0] + x * beta[1]))
    p_i = p(X_data, beta)
    epsilon = 1e-8  # For numerical stability
    ll_sum = np.sum(Y_data * np.log(p_i + epsilon) + (1 - Y_data) * np.log(1. - p_i + epsilon))
    return -ll_sum

def minimize_nLL(X, Y):
    x0 = np.array([-5, 0])
    res = minimize(nLL, x0=x0, args=(X, Y), method='BFGS')
    return res

def plot_res(X, Y):
    # Minimize negative log likelihood
    res = minimize_nLL(X, Y)
    beta0_hat, beta1_hat = res.x
    hessian_inv = res.hess_inv  # This is the covariance matrix
    
    # Standard errors
    beta0_hat_err = np.sqrt(hessian_inv[0, 0])
    beta1_hat_err = np.sqrt(hessian_inv[1, 1])
    
    # Covariance matrix
    covariance_matrix = hessian_inv

    print("Estimated Parameters:")
    print(f"β0: {beta0_hat:.3f} ± {beta0_hat_err:.2f}")
    print(f"β1: {beta1_hat:.3f} ± {beta1_hat_err:.2f}")
    print("Covariance Matrix:")
    print(covariance_matrix)

    # Plot data
    fig, ax = plt.subplots()
    ax.scatter(X, Y, marker='o', facecolor='r', edgecolor='k')
    ax.set_xlabel('age')
    ax.set_ylabel('ans')
    
    # Plot fit
    p = lambda x, beta: 1 / (1 + np.exp(-beta[0] + x * beta[1]))
    X_fit = np.linspace(min(X), max(X), 100)
    Y_fit = p(X_fit, [beta0_hat, beta1_hat])
    params = (r'$\beta_0$ = {} +/-'.format(np.around(beta0_hat, 3)) + 
              f'{beta0_hat_err:,.2f}' + '\n' + 
              r'$\beta_1$ = {} +/-'.format(np.around(beta1_hat, 3)) + 
              f'{beta1_hat_err:,.2f}')
    ax.plot(X_fit, Y_fit, 'b--', label=params)
    ax.legend()
    plt.savefig('logistic_curve_fit.png', dpi=96)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('survey.csv')
    X, Y = np.array(df['age']), np.array(df['recognized_it'])
    plot_res(X, Y)
