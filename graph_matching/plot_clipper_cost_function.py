import numpy as np
import matplotlib.pyplot as plt
import math


def plot_clipper_cost_function(sig, eps, name):
    # Define the function to calculate sp
    def calculate_sp(euclidean_diff, sig, eps):
        sp = math.exp(-0.5 * euclidean_diff * euclidean_diff / (sig * sig))
        return sp if euclidean_diff < eps else 0.0

    # Adjust the x-limits to a bit more than epsp
    x_max = eps + 0.2

    # Generate a new range of euclidean_diff values
    euclidean_diff_values = np.linspace(0, x_max, 400)
    sp_values = [calculate_sp(ed, sig, eps) for ed in euclidean_diff_values]

    # Plot the function with adjusted x-limits
    plt.figure(name, figsize=(10, 6))
    plt.plot(euclidean_diff_values, sp_values, label=f'sig={sig}, eps={eps}')
    plt.xlabel(f'{name} Difference')
    plt.ylabel('sp')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, x_max)
    plt.show()


sigp = 0.2
epsp = 1.0
# plot_clipper_cost_function(sigp, epsp, "Points")

sign = 0.1
epsn = 0.3
plot_clipper_cost_function(sign, epsn, "Normals")