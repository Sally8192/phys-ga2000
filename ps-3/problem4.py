import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def generate_y(N, num_samples):
    """Generate y as the average of N exponential random variables."""
    x = np.random.exponential(scale=1.0, size=(num_samples, N))
    y = np.mean(x, axis=1)  # Calculate the mean for each sample
    return y

# Parameters
num_samples = 10000  # Choose a large enough number for accuracy
N_values_distribution = [1, 5, 10, 30, 100, 200, 500, 1000]  # Specific N values for distribution plot

# Plot distributions for different N
plt.figure(figsize=(15, 10))
for N in N_values_distribution:
    y = generate_y(N, num_samples)
    plt.hist(y, bins=30, density=True, alpha=0.6, label=f'N={N}')

# Overlay normal distributions for comparison
x = np.linspace(0, 2, 1000)
for N in N_values_distribution:
    plt.plot(x, stats.norm.pdf(x, 1, np.sqrt(1/N)), 'k--', alpha=0.3)  # Normal distribution overlay

plt.title('Distribution of y for Different N')
plt.xlabel('y')
plt.ylabel('Density')
plt.xlim(0, 2)  
plt.legend()
plt.grid()

N_values = np.arange(1, 1001, 1)  # Using a smooth range from 1 to 1000

# Calculate mean, variance, skewness, and kurtosis for a smooth range of N
results = []
for N in N_values:
    y = generate_y(N, num_samples)
    mean_y = np.mean(y)
    var_y = np.var(y)
    skew_y = stats.skew(y)
    kurt_y = stats.kurtosis(y)
    results.append([N, mean_y, var_y, skew_y, kurt_y])

results_df = pd.DataFrame(results, columns=['N', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])

# Estimate when skewness and kurtosis drop below 1%
threshold_skewness = 0.01 * results_df['Skewness'].iloc[0]
threshold_kurtosis = 0.01 * results_df['Kurtosis'].iloc[0]

# Find the first N values where skewness and kurtosis are below the thresholds
skewness_N = results_df.loc[results_df['Skewness'] < threshold_skewness, 'N'].iloc[0]
kurtosis_N = results_df.loc[results_df['Kurtosis'] < threshold_kurtosis, 'N'].iloc[0]

# Print out the estimation results
print(f'First N value for skewness < 1% of N=1: {skewness_N}')
print(f'First N value for kurtosis < 1% of N=1: {kurtosis_N}')

plt.savefig('clt_distribution.png')  


plt.figure(figsize=(15, 10))


plt.subplot(2, 2, 1)
plt.plot(results_df['N'], results_df['Mean'], color='blue', label='Mean')
plt.axhline(y=1, color='red', linestyle='--', label='Expected Mean = 1')
plt.title('Mean of y vs N')
plt.xlabel('N')
plt.ylabel('Mean')
plt.legend()
plt.grid()


plt.subplot(2, 2, 2)
plt.plot(results_df['N'], results_df['Variance'], color='orange', label='Variance')
plt.axhline(y=1, color='red', linestyle='--', label='Expected Variance = 1/N')
plt.title('Variance of y vs N')
plt.xlabel('N')
plt.ylabel('Variance')
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(results_df['N'], results_df['Skewness'], color='green', label='Skewness')
plt.axhline(y=threshold_skewness, color='purple', linestyle='--', label='1% of Skewness at N=1')
plt.title('Skewness of y vs N')
plt.xlabel('N')
plt.ylabel('Skewness')
plt.legend()
plt.grid()


plt.subplot(2, 2, 4)
plt.plot(results_df['N'], results_df['Kurtosis'], color='brown', label='Kurtosis')
plt.axhline(y=threshold_kurtosis, color='purple', linestyle='--', label='1% of Kurtosis at N=1')
plt.title('Kurtosis of y vs N')
plt.xlabel('N')
plt.ylabel('Kurtosis')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('clt_statistics.png')  # Save the statistics plot
plt.show()
