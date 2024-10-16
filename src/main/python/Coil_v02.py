import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters for the solenoids
N_turns = 100  # Number of turns
L = 0.1  # Length of each solenoid (meters)
R = 0.01  # Radius of each solenoid (meters)
shift_distance = 0.2  # Distance of solenoids from origin (meters)
points_per_turn = 50  # Points per turn (rendered)

def generate_solenoid_points(N_turns, L, R, shift_distance, points_per_turn):
    # Total points to plot along the solenoid
    total_points = N_turns * points_per_turn

    # Parametric equations for the helical shape of the solenoid
    z = np.linspace(0, L, total_points)  # Along the solenoid's length
    theta = np.linspace(0, 2 * np.pi * N_turns, total_points)  # Angular positions

    # Coordinates of a standard helix (solenoid along z-axis)
    x_helix = R * np.cos(theta)
    y_helix = R * np.sin(theta)

    # Shift each solenoid along different axes and rotate them
    # First, set the base solenoid along the positive z-axis (Solenoid 1)
    solenoid1 = np.column_stack((x_helix, y_helix, z - shift_distance))

    # Solenoid 2: Rotate solenoid1 by 90 degrees around the y-axis (placing it along the x-axis)
    solenoid2 = np.column_stack((z - shift_distance, y_helix, x_helix))

    return solenoid1, solenoid2

def plot_solenoids(solenoid1, solenoid2, solenoid3, solenoid4):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each solenoid with different colors
    ax.plot(solenoid1[:, 0], solenoid1[:, 1], solenoid1[:, 2], lw=1, color='b', label='Solenoid 1 (z-axis)')
    ax.plot(solenoid2[:, 0], solenoid2[:, 1], solenoid2[:, 2], lw=1, color='r', label='Solenoid 2 (x-axis)')
    ax.plot(solenoid3[:, 0], solenoid3[:, 1], solenoid3[:, 2], lw=1, color='g', label='Solenoid 3 (60° rotation)')
    ax.plot(solenoid4[:, 0], solenoid4[:, 1], solenoid4[:, 2], lw=1, color='m', label='Solenoid 4 (60° rotation)')

    # Set labels and title
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    ax.set_zlabel('Z-axis (meters)')
    ax.set_title('3D Visualization of Four Solenoids with Adjusted Angles')

    # Set aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    # Add a legend to differentiate the solenoids
    ax.legend()

    # Show the plot
    plt.show()

# Generate the solenoids
solenoid1, solenoid2, solenoid3, solenoid4 = generate_solenoid_points(N_turns, L, R, shift_distance, points_per_turn)

# Plot the solenoids
plot_solenoids(solenoid1, solenoid2, solenoid3, solenoid4)
