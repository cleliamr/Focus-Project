import numpy as np
import matplotlib.pyplot as plt

# define constants
t = np.linspace(0, 9, 500)
Hz = 1 # rotations per second
rot_freq = 60 # number of seconds to return to old rot. axis
alpha_degrees = 60
alpha_pi = alpha_degrees / 180
beta_degrees = 70
beta_pi = beta_degrees / 180

"""
# y- rot. axis
current1 = np.sin(np.pi * Hz * t) # blue
current2 = np.sin(np.pi * (Hz * t - 1)) # green
current3 = np.sin(np.pi * (Hz * t - alpha_pi)) # red
current4 = np.sin(np.pi * (Hz * t - (1 + alpha_pi)) # yellow

# x- rot. axis
current1 = np.sin(np.pi * (Hz * t - beta_pi)) # blue
current2 = np.sin(np.pi * Hz * t) # green
current3 = np.sin(np.pi * Hz * t) + 0.05 # red
current4 = np.sin(np.pi * (Hz * t - beta_pi)) + 0.05 # yellow
"""


def plot_current_single(current1, t):
    fig, ax = plt.subplots()

    plt.plot(t, current1, label='Current in Coil', color='r')

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()


def plot_current_all(current1, current2, current3, current4, t):
    fig, ax = plt.subplots()

    plt.plot(t, current1, label='Current in Coil 1', color='r')
    plt.plot(t, current2, label='Current in Coil 2', color='g')
    plt.plot(t, current3, label='Current in Coil 3', color='b')
    plt.plot(t, current4, label='Current in Coil 4', color='y')

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

