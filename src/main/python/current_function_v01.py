import numpy as np
import matplotlib.pyplot as plt

def current_mag_flex(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice):
    t = np.linspace(0, span_of_animation - 1, time_steps)

    current1 = np.zeros(time_steps)
    current2 = np.zeros(time_steps)
    current3 = np.zeros(time_steps)
    current4 = np.zeros(time_steps)

    if model_choice == "3S":
        for i in range(len(t)):
            current1[i] = current_magnitude_3S(rot_freq, t[i], Hz) * I_max
            current2[i] = current_magnitude_3S(rot_freq, t[i] - (rot_freq / 3 - Hz / 2), Hz) * I_max
            current3[i] = current_magnitude_3S(rot_freq, t[i] - (2 * rot_freq / 3 - Hz), Hz) * I_max
        current = np.array([current1, current2, current3])
    else:
        for i in range(len(t)):
            current1[i] = np.sin(np.pi * Hz * t[i]) * I_max
            current2[i] = np.sin(np.pi * (Hz * t[i] - 1)) * I_max
            current3[i] = np.sin(np.pi * (Hz * t[i] - 1/3)) * I_max
            current4[i] = np.sin(np.pi * (Hz * t[i] - (1 + 1/3))) * I_max
        current = np.array([current1, current2, current3, current4])

    return current


def current_magnitude_3S(rot_freq, t, Hz):
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


def plot_current_all(current1, current2, current3, t):
    fig, ax = plt.subplots()

    plt.plot(t, current1, label='Current in Coil 1', color='r')
    plt.plot(t, current2, label='Current in Coil 2', color='g')
    plt.plot(t, current3, label='Current in Coil 3', color='b')

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

