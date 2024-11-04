import numpy as np
import scipy
import matplotlib.pyplot as plt

# piano = np.loadtxt("piano.txt")
# trumpet = np.loadtxt("trumpet.txt")

def plot_fourier(dataname):
    signal = np.loadtxt(f"{dataname}.txt")
    plt.plot(signal, linewidth = 0.05)
    plt.xlabel("t")
    plt.ylabel("magnitude")
    plt.title(f"{dataname} data")
    plt.savefig(f"{dataname}.png")
    plt.clf()
    
    sample_period = 1.0 / 44100
    N = len(signal)
    signal_freq = scipy.fft.fft(signal)[:10000]
    freqs = scipy.fft.fftfreq(N, sample_period)[:10000]
    signal_freq_abs = 2 / N *np.absolute(signal_freq) # magnitude
    plt.plot(freqs, signal_freq_abs)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude")
    plt.title(f"fourier transform of {dataname} data")
    plt.savefig(f"{dataname}-fourier.png")
    plt.clf()
    freq_peaks = []
    for i in range(10000):
        if(signal_freq_abs[i] > 500):
            freq_peaks.append(freqs[i])
    print(f"peak frequency of {dataname}:")
    print(freq_peaks)

def q1():
    plot_fourier("piano")
    plot_fourier("trumpet")

q1()