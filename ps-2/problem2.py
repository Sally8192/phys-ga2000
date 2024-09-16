import numpy as np

# Define the number 1.0
one_float32 = np.float32(1.0)
one_float64 = np.float64(1.0)

# Calculate the machine epsilon
epsilon_float32 = np.finfo(np.float32).eps
epsilon_float64 = np.finfo(np.float64).eps

# Smallest value that can be added to 1.0
smallest_add_float32 = one_float32 + epsilon_float32 - one_float32
smallest_add_float64 = one_float64 + epsilon_float64 - one_float64

print(f"Smallest value that can be added to 1 (32-bit): {smallest_add_float32}")
print(f"Smallest value that can be added to 1 (64-bit): {smallest_add_float64}")