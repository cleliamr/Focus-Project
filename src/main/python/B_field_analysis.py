import numpy as np
import matplotlib.pyplot as plt

# Takes B-field at point defined in config and analyses few parameters
def B_field_analysis(B_field):
    # multiply to calc in Milliteslas
    B_field_new = np.squeeze(B_field[:])
    print(B_field_new)
    # define array that saves mag of B-field
    B_field_mag = np.zeros(len(B_field_new))
    for i in range(len(B_field_new)):
        B_field_mag[i] = np.sqrt(B_field_new[i,0]**2 + B_field_new[i,1]**2 + B_field_new[i,2]**2)
    print(B_field_mag)
    # print certain key values
    print("Minimal Magnitude of B at point (in Milliteslas): ", B_field_mag.min())
    print("Maximal Magnitude of B at point (in Milliteslas): ", B_field_mag.max())
    print("Average Magnitude of B at point (in Milliteslas): ", B_field_mag.mean())

    # plot the behaviour of the B-field over time
    plot_B_over_time(B_field_new)

def plot_B_over_time(B_field):
    t = np.linspace(0, (len(B_field) // 10) - 1, len(B_field))
    fig, ax = plt.subplots()

    plt.plot(t, B_field[:,0], label='Bx', color='r')
    plt.plot(t, B_field[:,1], label='By', color='g')
    plt.plot(t, B_field[:,2], label='Bz', color='b')

    ax.set(xlabel='time (s)', ylabel='B-field strength (T)',
           title='B over time (in Milliteslas)')
    ax.grid()

    fig.savefig("test.png")
    plt.show()