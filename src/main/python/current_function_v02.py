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
current3 = np.sin(np.pi * Hz * t) # red
current4 = np.sin(np.pi * (Hz * t - beta_pi)) # yellow
"""


def current_flow_4S(time_steps, span_of_animation, Hz, rot_freq, I_max):
    t = np.linspace(0, span_of_animation - 1, time_steps)

    current1 = []
    current2 = []
    current3 = []

    for i in range(len(t)):
        current1.append(current_magnitude(rot_freq, t[i], Hz) * I_max)
        current2.append(current_magnitude(rot_freq, t[i] - (rot_freq / 3 - Hz / 2), Hz) * I_max)
        current3.append(current_magnitude(rot_freq, t[i] - (2 * rot_freq / 3 - Hz), Hz) * I_max)

    return current1, current2, current3


def current_magnitude(rot_freq, t, Hz):
    t_state = t % (rot_freq - 3 * (Hz / 2)) # calculate state of time in periodic cycle

    if t_state < (rot_freq / 6):
        current_magnitude = np.sin(t_state * np.pi / 20)

    elif t_state < ((2 * rot_freq / 3) - Hz):
        current_magnitude = 1

    elif t_state < ((5 * rot_freq / 6) - Hz):
        current_magnitude = np.cos((t_state - ((2 * rot_freq / 3) - Hz))  * np.pi / 20)

    else:
        current_magnitude = 0

    return current_magnitude * np.sin(t_state * Hz * np.pi)

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

