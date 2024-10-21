import numpy as np
import matplotlib.pyplot as plt

# Function to generate points for the solenoids
def generate_solenoid_points(N_turns, L, R, shift_distance, points_per_turn):
    # Calculate total points to plot
    total_points = N_turns * points_per_turn

    # Parametric equation for the helical shape
    z = np.linspace(0, L, total_points)  # Solenoid length centered at the origin
    theta = np.linspace(0, 2 * np.pi * N_turns, total_points)  # Angular positions

    # Helix coordinates in cylindrical form
    x_helix = R * np.cos(theta)  # x-coordinates (circle)
    y_helix = R * np.sin(theta)  # y-coordinates (circle)

    # solenoid 1:
    z_shifted = (np.sqrt(2) / 2) * (z + shift_distance + x_helix)
    x_shifted = (np.sqrt(2) / 2) * (z + shift_distance - x_helix)

    # Solenoid 1: Along the x = z axis
    solenoid1 = np.column_stack((x_shifted, y_helix, z_shifted))

    #solenoid 3:
    x_shifted = (x_shifted + np.sqrt(3) * y_helix) / 2
    y_shifted = (y_helix + np.sqrt(3) * x_shifted) / 2

    # Solenoid 3: rotated sol. 1 by 60 degrees
    solenoid3 = np.column_stack((x_shifted, y_shifted, z_shifted))

    # solenoid 2:
    z_shifted = (np.sqrt(2) / 2) * (z + shift_distance + x_helix)
    x_shifted = -(np.sqrt(2) / 2) * (z + shift_distance - x_helix)

    # Solenoid 2: Along the x = -z axis
    solenoid2 = np.column_stack((x_shifted, y_helix, z_shifted))

    # solenoid 4:
    x_shifted = (x_shifted + np.sqrt(3) * y_helix) / 2
    y_shifted = (y_helix + np.sqrt(3) * x_shifted) / 2

    # Solenoid 4: rotated sol. 2 by 60 degrees
    solenoid4 = np.column_stack((x_shifted, y_shifted, z_shifted))

    return solenoid1, solenoid2, solenoid3, solenoid4

# Plot the solenoids
def plot_solenoids(solenoid1, solenoid2, solenoid3, solenoid4):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each solenoid with different colors
    ax.plot(solenoid1[:, 0], solenoid1[:, 1], solenoid1[:, 2], lw=1, color='b', label='Solenoid 1')
    ax.plot(solenoid2[:, 0], solenoid2[:, 1], solenoid2[:, 2], lw=1, color='r', label='Solenoid 2')
    ax.plot(solenoid3[:, 0], solenoid3[:, 1], solenoid2[:, 2], lw=1, color='y', label='Solenoid 3')
    ax.plot(solenoid4[:, 0], solenoid4[:, 1], solenoid2[:, 2], lw=1, color='g', label='Solenoid 4')

    # Set labels and title
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    ax.set_zlabel('Z-axis (meters)')
    ax.set_title('Four Solenoids Model')

    # Set aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
