import numpy as np

def calculate_current_flex(N_turns, L, points_per_turn, current_mag, model_choice):
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
        current1 = rotate_vector(current_base, 'y', np.pi / 4) * current_mag[0]
        current2 = rotate_vector(current_base, 'y', -np.pi / 4) * current_mag[1]
        current3 = rotate_vector(current1, 'z', np.pi / 3) * current_mag[2]
        current4 = rotate_vector(current2, 'z', np.pi / 3) * current_mag[3]

        current = current1, current2, current3, current4

    else:
        current1 = np.column_stack((dx_helix, dy_helix, [0] * total_points)) * current_mag[0]
        current2 = np.column_stack(([0] * total_points, dy_helix, dx_helix)) * current_mag[1]
        current3 = np.column_stack((dx_helix, [0] * total_points, dy_helix)) * current_mag[2]

        current = current1, current2, current3

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