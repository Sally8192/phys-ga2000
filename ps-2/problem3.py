import numpy as np
import timeit

# For loop implementation
def madelung_constant(L):
    M = 0
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                if (i == 0 and j == 0 and k == 0):
                    continue
                distance = np.sqrt(i**2 + j**2 + k**2)
                M += (-1)**(i + j + k) / distance
    return abs(M)

# Vectorized implementation
def madelung(L):
    # Generate arrays of indices
    i, j, k = np.meshgrid(np.arange(-L, L + 1), np.arange(-L, L + 1), np.arange(-L, L + 1), indexing='ij')

    # Mask the (0, 0, 0) point
    mask = (i != 0) | (j != 0) | (k != 0)
    
    distance = np.sqrt(i**2 + j**2 + k**2)
    sign = (-1.0) ** (i + j + k)

    # Calculate the Madelung constant
    result = np.sum(sign[mask] / distance[mask])

    return abs(result)

# Timing the for loop implementation
start1 = timeit.default_timer()
result_loop = madelung_constant(100)
stop1 = timeit.default_timer()

print("By using a for loop:")
print(f"The value of Madelung constant is {result_loop}")
print(f"Runtime for using a loop (s): {stop1 - start1}")
print()

# Timing the vectorized implementation
start2 = timeit.default_timer()
result_vectorized = madelung(100)
stop2 = timeit.default_timer()

print("Without using a for loop:")
print(f"The value of Madelung constant is {result_vectorized}")
print(f"Runtime without using a loop (s): {stop2 - start2}")
print()

# Determining which method is faster
if (stop2 - start2) < (stop1 - start1):
    print("Program without a for-loop is faster.")
else:
    print("Program with for-loop is faster.")
    print()
