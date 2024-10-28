import numpy as np
import matplotlib.pyplot as plt

def current_mag_flex(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice):
    t = np.linspace(0, span_of_animation, time_steps)

    current1 = np.zeros(time_steps)
    current2 = np.zeros(time_steps)
    current3 = np.zeros(time_steps)
    current4 = np.zeros(time_steps)

    if model_choice == "3S":
        for i in range(len(t)):
            current1[i] = current_magnitude_3S(rot_freq, t[i], Hz) * I_max
            current2[i] = current_magnitude_3S(rot_freq, t[i] - (rot_freq / 3 - Hz / 2), Hz) * I_max
            current3[i] = current_magnitude_3S(rot_freq, t[i] - (2 * rot_freq / 3 - Hz), Hz) * I_max
        current_mag = np.array([current1, current2, current3])
    else:
        for i in range(len(t)):
            current1[i], current2[i], current3[i], current4[i] = current_magnitude_4S(rot_freq, t[i], Hz, I_max)
        current_mag = np.array([current1, current2, current3, current4])

    return current_mag

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

def current_magnitude_4S(rot_freq, t, Hz, I_max):
    t_state = t % rot_freq

    if t_state < rot_freq / 6:
        current1 = np.sin(np.pi * Hz * t)
        current2 = np.sin(np.pi * (Hz * t - 1))
        current3 = np.sin(np.pi * (Hz * t - 1 / 3))
        current4 = np.sin(np.pi * (Hz * t - (1 + 1 / 3)))

    elif t_state <  rot_freq / 3:
        offset = (t_state - (rot_freq / 3)) // 2
        current1 = np.sin(np.pi * Hz * t)
        current2 = np.sin(np.pi * (Hz * t - 1 + (offset/20)))
        current3 = np.sin(np.pi * (Hz * t - (1/3) + (offset/30)))
        current4 = np.sin(np.pi * (Hz * t - (4/3) + (5 * offset/40)))

    elif t_state < rot_freq / 2:
        current1 = np.sin(np.pi * Hz * t)
        current2 = np.sin(np.pi * (Hz * t - (1/2)))
        current3 = current1
        current4 = current2

    elif t_state < 2 * rot_freq / 3:
        offset = (t_state - (rot_freq / 3)) // 2
        current1 = np.sin(np.pi * Hz * t)
        current2 = np.sin(np.pi * (Hz * t - (1/2) + (offset/60)))
        current3 = np.sin(np.pi * (Hz * t - (offset/30)))
        current4 = np.sin(np.pi * (Hz * t + (offset/20)))

    elif t_state < 5 * rot_freq / 6:
        current1 = np.sin(np.pi * Hz * t)
        current2 = np.sin(np.pi * (Hz * t - (1/3)))
        current3 = current2
        current4 = current1

    else:
        offset = (t_state - (5 *rot_freq / 6)) // 2
        current1 = np.sin(np.pi * Hz * t)
        current2 = np.sin(np.pi * (Hz * t - (1/2 + offset/20)))
        current3 = np.sin(np.pi * (Hz * t - 1 / 3))
        current4 = np.sin(np.pi * (Hz * t - (4 * offset / 30)))

    return current1 * I_max, current2 * I_max, current3 * I_max, current4 * I_max

def plot_current_single(current1, t):
    fig, ax = plt.subplots()

    plt.plot(t, current1, label='Current in Coil', color='r')

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()


def plot_current_all(current1, current2, current3, current4, span_of_animation, time_steps):
    t = np.linspace(0, span_of_animation, time_steps)
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
