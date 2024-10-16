import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_solenoid_points(N_turns, L, R, shift_distance, points_per_turn):
    # Calculate total points to plot
    total_points = N_turns * points_per_turn

    # Parametric equation for the helical shape
    z = np.linspace(0, L, total_points)  # z-axis positions along the solenoid length
    theta = np.linspace(0, 2 * np.pi * N_turns, total_points)  # Angular positions

    # Helix coordinates in cylindrical form
    x_helix = R * np.cos(theta)  # x-coordinates (circle)
    y_helix = R * np.sin(theta)  # y-coordinates (circle)

    # Solenoid 1: Along the Z-axis (standard solenoid)
    solenoid1 = np.column_stack((x_helix, y_helix, z + shift_distance))

    # Solenoid 2: Along the X-axis, swap x and z, shift in x
    solenoid2 = np.column_stack((z + shift_distance, y_helix, x_helix))

    # Solenoid 3: Along the Y-axis, swap y and z, shift in y
    solenoid3 = np.column_stack((x_helix, z + shift_distance, y_helix))

    return solenoid1, solenoid2, solenoid3


def plot_solenoids(solenoid1, solenoid2, solenoid3):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each solenoid with different colors
    ax.plot(solenoid1[:, 0], solenoid1[:, 1], solenoid1[:, 2], lw=1, color='b', label='Solenoid 1 (Z-axis)')
    ax.plot(solenoid2[:, 0], solenoid2[:, 1], solenoid2[:, 2], lw=1, color='r', label='Solenoid 2 (X-axis)')
    ax.plot(solenoid3[:, 0], solenoid3[:, 1], solenoid3[:, 2], lw=1, color='g', label='Solenoid 3 (Y-axis)')

    # Set labels and title
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    ax.set_zlabel('Z-axis (meters)')
    ax.set_title('3D Visualization of Three Solenoids')

    # Set aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    # Add a legend to differentiate the solenoids
    ax.legend()

    # Show the plot
    plt.show()



