import numpy as np
import time
import matplotlib.pyplot as plt

matrix_sizes = [10, 30, 50, 70, 90, 110, 130] 
computation_times_explicit = [] 
computation_times_dot = []      

for size in matrix_sizes:
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    start_time = time.time()
    result_explicit = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result_explicit[i, j] += A[i, k] * B[k, j]
    end_time = time.time()
    computation_time_explicit = end_time - start_time
    computation_times_explicit.append(computation_time_explicit)
    

    start_time = time.time()
    result_dot = np.dot(A, B)
    end_time = time.time()
    computation_time_dot = end_time - start_time
    computation_times_dot.append(computation_time_dot)


N_cubed = [size ** 3 for size in matrix_sizes]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(matrix_sizes, computation_times_explicit, label='Explicit Function', color='tab:blue')
ax1.plot(matrix_sizes, computation_times_dot, label='dot() Method', color='tab:orange')
ax1.set_xlabel('Matrix Size (N x N)')
ax1.set_ylabel('Computation Time (s)')
ax1.grid(True)


ax2 = ax1.twinx()
ax2.plot(matrix_sizes, N_cubed, label='N^3', linestyle='--', color='tab:red')
ax2.set_ylabel('N^3', color='tab:red')


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left')


plt.title('Computation Time vs. Matrix Size for Matrix Multiplication')
plt.savefig('matrix_multiplication.png')

plt.show()