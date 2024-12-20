import numpy as np
import matplotlib.pyplot as plt

# Function to generate points for the solenoids
def generate_solenoid_points_flex(N_turns, L, R, shift_distance, points_per_turn, model_choice, angle, angle_adj, angle_opp):
    # Calculate total points to plot
    total_points = N_turns * points_per_turn

    # Parametric equation for the helical shape
    z = np.linspace(0, L, total_points)  # Solenoid length centered at the origin
    theta = np.linspace(0, 2 * np.pi * N_turns, total_points)  # Angular positions

    # Helix coordinates in cylindrical form
    x_helix = R * np.cos(theta)  # x-coordinates (circle)
    y_helix = R * np.sin(theta)  # y-coordinates (circle)

    # Solenoid Base: Along the Z-axis (standard solenoid)
    solenoid_base = np.column_stack((x_helix, y_helix, z + shift_distance))
    if model_choice == "4S":
        # define solenoid points for Coils
        solenoid1 = rotate_vector(solenoid_base, 'y', angle_opp / 2)
        solenoid2 = rotate_vector(solenoid_base, 'y', -angle_opp / 2)
        solenoid3 = rotate_vector(solenoid1, 'z', angle_adj / 2)
        solenoid4 = rotate_vector(solenoid2, 'z', angle_adj / 2)
        solenoid_points = solenoid1, solenoid2, solenoid3, solenoid4

        plot_solenoids_flex(solenoid1, solenoid2, solenoid3, solenoid4)

    elif model_choice == "3S":
        solenoid1 = solenoid_base
        solenoid2 = rotate_vector(solenoid_base, 'y', angle)
        solenoid3 = rotate_vector(solenoid_base, 'x', -angle)
        solenoid_points = solenoid1, solenoid2, solenoid3

        plot_solenoids_flex(solenoid1, solenoid2, solenoid3)

    else:
        solenoid1 = solenoid_base
        solenoid2 = rotate_vector(solenoid_base, 'y', angle)
        solenoid_points = solenoid1, solenoid2

        plot_solenoids_flex(solenoid1, solenoid2)

    return solenoid_points

def rotate_vector(vector, axis, theta):
    """
    Rotates a 3D vector around the x, y, or z axis by a specified angle.

    Parameters:
    vector : array-like (3 elements)
        The 3D vector to be rotated.
    axis : str
        The axis of rotation: 'x', 'y', or 'z'.
    theta : float
        The angle of rotation in radians.

    Returns:
    rotated_vector : numpy array (3 elements)
        The rotated 3D vector.
    """
    vector = np.array(vector)

    # Rotation matrix based on the axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Perform the rotation
    rotated_vector = np.dot(vector, rotation_matrix.T)

    return rotated_vector

# plotting of solenoids
def plot_solenoids_flex(*solenoids):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'y']
    labels = ['Solenoid 1', 'Solenoid 2', 'Solenoid 3', 'Solenoid 4']
    solenoids = np.array(solenoids)
    for i in range(len(solenoids)):
        ax.plot(solenoids[i][:, 0], solenoids[i][:, 1], solenoids[i][:, 2], lw=1, color=colors[i], label=labels[i])

    # Set labels and title
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    ax.set_zlabel('Z-axis (meters)')
    ax.set_title('Solenoid Model')

    # Set aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()