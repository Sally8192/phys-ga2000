import numpy as np
import matplotlib.pyplot as plt


total_atoms = 1000
half_life_208Tl = 3.053 * 60  

random_numbers = np.random.rand(total_atoms)

decay_times = -half_life_208Tl * np.log(1 - random_numbers)

sorted_decay_times = np.sort(decay_times)


max_time = np.max(sorted_decay_times)
time_array = np.linspace(0, max_time, num=1000)


undecayed_atoms = np.zeros_like(time_array)


for i, t in enumerate(time_array):
    undecayed_atoms[i] = np.sum(sorted_decay_times > t)


plt.figure(figsize=(10, 6))
plt.plot(time_array, undecayed_atoms, label='Undecayed Atoms')
plt.xlabel('Time (s)')
plt.ylabel('Number of Undecayed Atoms')
plt.title('Decay of 208Tl Over Time')

plt.legend()
plt.savefig("Decay2.png")
plt.show()