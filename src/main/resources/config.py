import numpy as np

# Constants: Coil
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
N_turns = 50  # Number of turns
L = 0.1  # 10 cm length
R = 0.025  # 2.5 cm radius
points_per_turn = 10  # points rendered
shift_distance = 0.2 # distance to focus point
I_max = 10  # maximal current
Grid_density = 0.1  # defines the Grid_density
Hz = 1 # rotations per second
rot_freq = 60 # number of seconds to return to old rot. axis
Grid_size = 0.2 # describes size of grid (x2)
angle_opp = np.pi / 2 # describes angle between opposite solenoids (4S model)
angle_adj = np.pi / 2 # describes angle between adjacent solenoids (4S model)
angle = np.pi / 2 # angle between solenoids (3S model)
dl = 2 * np.pi * R / points_per_turn # defines dl for Biot-Savart Law

# Constants: Animation
time_steps = 20 # Number of frames for the animation
output_folder = "frames"  # Folder to save the frames
video_filename = "magnetic_field_animation.mp4"  # Output video filename
span_of_animation = 2

# Constants: Perm. Magnet
magnet_center = np.array([0, 0, 0.16]) # Permanent magnet coordinates
magnet_dimensions = np.array([0.01, 0.01, 0.05]) # Permanent magnets dimensions
magnet_moment = np.array([1.4/mu_0, 0, 0]) # Magnetization of magnet
cube_size = 0.001

# Constants Canc. field
canc_vert_distance = 0.16 # vertical distance to origin / focus point
canc_hor_distance = 0.42 # distance between magnets
canc_magnet_dimensions = np.array([0.05, 0.015, 0.015])
canc_magnet_moment = np.array([0, 0, -1.4/mu_0]) # Magnetization of magnet
canc_cube_size = 0.005 # FEM smallest magnet size