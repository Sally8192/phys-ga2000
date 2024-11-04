import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    return np.loadtxt(filename)

def plot_data(dates, values, title, label):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, label=label)
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Closing Value')
    plt.legend()
    plt.grid()

def perform_fft(data):
    # Calculate the Fourier coefficients using rfft
    return np.fft.rfft(data)

def zero_out_coefficients(fft_data, percentage):
    # Zero out all but the first 'percentage' of the coefficients
    n = len(fft_data)
    cutoff_index = int(n * percentage)
    fft_data[cutoff_index:] = 0
    return fft_data

def inverse_fft(fft_data):
    # Calculate the inverse Fourier transform
    return np.fft.irfft(fft_data)

if __name__ == "__main__":
    dow_data = read_data('dow.txt')
    dates = np.arange(len(dow_data))  # Assuming daily data with simple integer x-axis
    
    plot_data(dates, dow_data, 'Dow Jones Industrial Average', 'Original Data')
    
    plt.savefig('dow_jones_original.png', dpi=200)
    plt.show()

    fft_coeffs = perform_fft(dow_data)
    
    # Set all but the first 10% of the coefficients to zero
    fft_coeffs_reduced_10 = zero_out_coefficients(fft_coeffs.copy(), 0.1)
    
    # Inverse FFT and plot
    reconstructed_10 = inverse_fft(fft_coeffs_reduced_10)
    plt.plot(dates[:len(reconstructed_10)], reconstructed_10, label='Reconstructed (10% Coefficients)', color='red', alpha=0.7)
    
    plt.legend()
    plt.title('Dow Jones with 10% Fourier Coefficients Retained')
    plt.savefig('dow_jones_reconstructed_10_percent.png', dpi=200)
    plt.show()
    
    # Set all but the first 2% of the coefficients to zero and run again
    fft_coeffs_reduced_2 = zero_out_coefficients(fft_coeffs.copy(), 0.02)
    reconstructed_2 = inverse_fft(fft_coeffs_reduced_2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, dow_data, label='Original Data')
    plt.plot(dates[:len(reconstructed_2)], reconstructed_2, label='Reconstructed (2% Coefficients)', color='green', alpha=0.7)
    
    plt.legend()
    plt.title('Dow Jones with 2% Fourier Coefficients Retained')
    plt.savefig('dow_jones_reconstructed_2_percent.png', dpi=200)
    plt.show()
