import numpy as np
import matplotlib.pyplot as plt

def calculate_current_flex(N_turns, L, points_per_turn, model_choice, angle, angle_adj, angle_opp):
    # Calculate total points to plot
    total_points = N_turns * points_per_turn

    # Parametric equation for the helical shape
    z = np.linspace(0, L, total_points)  # z-axis positions along the solenoid length
    theta = np.linspace(0, 2 * np.pi * N_turns, total_points)  # Angular positions

    # Helix derivative of coordinates in cylindrical form
    dx_helix = np.sin(theta)  # x-coordinates (circle)
    dy_helix = -np.cos(theta)  # y-coordinates (circle)

    # Solenoid 1: Along the Z-axis (standard solenoid)
    current_base = np.column_stack((dx_helix, dy_helix, [0]*total_points))
    if model_choice == "4S":
        current1 = rotate_vector(current_base, 'y', angle_opp / 2)
        current2 = rotate_vector(current_base, 'y', -angle_opp / 2)
        current3 = rotate_vector(current1, 'z', angle_adj)
        current4 = rotate_vector(current2, 'z', angle_adj)

        current = current1, current2, current3, current4

    elif model_choice == "3S":
        current1 = current_base
        current2 = rotate_vector(current_base, 'y', angle)
        current3 = rotate_vector(current_base, 'x', -angle)

    else:
        current1 = current_base
        current2 = rotate_vector(current_base, 'y', angle)

        current = current1, current2

    return current

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

def plot_solenoid_currents(solenoid_points, current_directions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through each solenoid
    for i, (points, currents) in enumerate(zip(solenoid_points, current_directions)):
        # Extract the x, y, z coordinates for each solenoid's points
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]

        # Extract the x, y, z components of the current direction for each point
        u = currents[:, 0]  # x-component of current direction
        v = currents[:, 1]  # y-component of current direction
        w = currents[:, 2]  # z-component of current direction

        # Plot the current direction as arrows at each point along the solenoid
        ax.quiver(x_points, y_points, z_points, u, v, w, length=1, normalize=True)

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Current Direction Along Solenoids")
    plt.show()