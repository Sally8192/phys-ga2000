import matplotlib.pyplot as plt
import numpy as np 

mean = 0
std = 3
x_range = np.linspace(-10, 10, 100)

def gaussian(x, mu, sigma):
    value = 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2)
    return value 

plt.xlabel("Values over the range [-10, 10]")
plt.ylabel("Normalized Gaussian value")
plt.title("Gaussian Curve")
plt.plot(x_range, gaussian(x_range, mean, std))
plt.savefig('gaussian.png')
plt.show()
