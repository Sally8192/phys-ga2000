from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import timeit

def read_file(filename):
    # Open FITS file and extract the relevant data
    hdu_list = fits.open(filename)
    logwave = hdu_list['LOGWAVE'].data
    flux = hdu_list['FLUX'].data
    return logwave, flux

def plot_galaxies(n, logwave, flux):
    plt.figure(figsize=(10, 6))
    plt.xlabel(r'log$_{10}(\lambda [\AA])$')
    plt.ylabel(r'flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
    for i in range(n):
        plt.plot(logwave, flux[i], alpha=0.5)
    plt.title('Spectra of First {} Galaxies'.format(n))
    plt.savefig('wave_v_spectrum.png', dpi=96)
    plt.show()

def prep_flux(flux):
    '''Preprocessing of the flux data'''
    row_sum = flux.sum(axis=1)  # sum of flux along each row (each galaxy)
    normed_flux = flux / row_sum[:, np.newaxis]  # normalize each spectrum
    row_ave = np.mean(normed_flux, axis=1)  # average of each row (spectrum)
    normed_centered_flux = normed_flux - row_ave[:, np.newaxis]  # subtract mean from each spectrum
    return normed_centered_flux, row_sum, row_ave

def PCA(logwave, flux):
    Y, row_sum, row_ave = prep_flux(flux)  # Y is normed_centered_flux

    # Construct the covariance matrix using the residuals
    start01 = timeit.default_timer()
    C = np.transpose(Y) @ Y  # Covariance matrix (Nwave x Nwave)
    eigval, eigvec = linalg.eig(C)
    dt01 = timeit.default_timer() - start01

    # Ensure eigenvalues and eigenvectors are real
    eigvec_real = np.real(eigvec)

    # Plot first 5 eigenvectors from covariance method
    n = 5
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(logwave, eigvec_real[:, i], alpha=0.5)
    plt.xlabel(r'log$_{10}(\lambda [\AA])$')
    plt.ylabel('eigenvectors [arbs]')
    plt.title('Eigenvectors from Covariance Matrix')
    plt.savefig('eigenvectors_method01.png', dpi=96)
    plt.show()

    # SVD implementation for PCA
    start02 = timeit.default_timer()
    U, W, V_T = linalg.svd(Y, full_matrices=False)
    C_svd = np.transpose(V_T) @ np.diag(W) @ np.transpose(U) @ U @ np.diag(W) @ V_T
    eigval_svd, eigvec_svd = linalg.eig(C_svd)
    dt02 = timeit.default_timer() - start02

    # Ensure eigenvalues and eigenvectors are real
    eigvec_svd_real = np.real(eigvec_svd)

    # Plot first 5 eigenvectors from SVD method
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(logwave, eigvec_svd_real[:, i], alpha=0.5)
    plt.xlabel(r'log$_{10}(\lambda [\AA])$')
    plt.ylabel('eigenvectors [arbs]')
    plt.title('Eigenvectors from SVD Method')
    plt.savefig('eigenvectors_method02.png', dpi=96)
    plt.show()

    print('Covariance Method Time: {}s'.format(dt01))
    print('SVD Method Time: {}s'.format(dt02))

    return eigvec_svd_real, row_sum, row_ave, Y

def plot_principal_components(c_0, c_1, c_2):
    '''Plot the first three principal components (c0 vs c1 and c0 vs c2)'''
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    
    # Plot c0 vs c1
    axs[0].scatter(c_0, c_1, color='red', alpha=0.5)
    axs[0].set_ylabel(r'$c_{1}$')
    axs[0].set_title(r'$c_{0}$ vs $c_{1}$')

    # Plot c0 vs c2
    axs[1].scatter(c_0, c_2, color='red', alpha=0.5)
    axs[1].set_xlabel(r'$c_{0}$')
    axs[1].set_ylabel(r'$c_{2}$')
    axs[1].set_title(r'$c_{0}$ vs $c_{2}$')

    plt.tight_layout()
    plt.savefig('principal_components.png', dpi=96)
    plt.show()

def calculate_squared_residuals(logwave, flux_without_offset, eig_vec_svd, max_n):
    '''Calculate squared residuals for varying number of principal components Nc'''
    residuals = []
    for i in range(1, max_n + 1):
        weights = np.dot(flux_without_offset, eig_vec_svd[:, :i])
        approx_spectra = np.dot(weights, eig_vec_svd[:, :i].T)
        
        # Calculate squared residuals
        squared_residual = np.mean((flux_without_offset - approx_spectra) ** 2, axis=1)
        residuals.append(np.mean(squared_residual))
    
    return residuals

def plot_squared_residuals(n_components, squared_residuals):
    '''Plot squared residuals as a function of Nc (number of principal components)'''
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), squared_residuals, marker='o', color='blue')
    plt.xlabel('Number of Principal Components (Nc)')
    plt.ylabel('Mean Squared Residuals')
    plt.title('Mean Squared Residuals vs Number of Principal Components')
    plt.grid(True)
    plt.savefig('squared_residuals_vs_Nc.png', dpi=96)
    plt.show()

if __name__ == "__main__":
    logwave, flux = read_file('specgrid.fits')
    plot_galaxies(5, logwave, flux)
    
    flux_without_offset, row_sum, row_ave = prep_flux(flux)
    eig_vec_svd, row_sum, row_ave, Y = PCA(logwave, flux)

    # Extract the first three principal components
    c_i = Y @ eig_vec_svd
    c_0, c_1, c_2 = c_i[:, 0], c_i[:, 1], c_i[:, 2]

    # Plot the principal components c0 vs c1 and c0 vs c2
    plot_principal_components(c_0, c_1, c_2)

    # Calculate and plot the squared residuals for Nc = 1, 2, ..., 20
    max_n = 20
    squared_residuals = calculate_squared_residuals(logwave, flux_without_offset, eig_vec_svd, max_n)
    plot_squared_residuals(max_n, squared_residuals)
    rms_residual_n20 = np.sqrt(squared_residuals[-1])  # squared_residuals[-1] corresponds to Nc = 20
    print(f"Root-Mean Squared Residual for Nc = 20: {rms_residual_n20}")
