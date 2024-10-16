import numpy as np

def calculate_current(N_turns, L, points_per_turn, current1_mag, current2_mag, current3_mag):
    # Calculate total points to plot
    total_points = N_turns * points_per_turn

    # Parametric equation for the helical shape
    z = np.linspace(0, L, total_points)  # z-axis positions along the solenoid length
    theta = np.linspace(0, 2 * np.pi * N_turns, total_points)  # Angular positions

    # Helix derivative of coordinates in cylindrical form
    dx_helix = np.sin(theta)  # x-coordinates (circle)
    dy_helix = -np.cos(theta)  # y-coordinates (circle)

    # Solenoid 1: Along the Z-axis (standard solenoid)
    current1 = np.column_stack((dx_helix, dy_helix, [0]*total_points)) * -current1_mag

    # Solenoid 2: Along the X-axis, swap x and z, shift in x
    current2 = np.column_stack(([0]*total_points, dy_helix, dx_helix)) * current2_mag

    # Solenoid 3: Along the Y-axis, swap y and z, shift in y
    current3 = np.column_stack((dx_helix, [0]*total_points, dy_helix)) * current3_mag

    return current1, current2, current3