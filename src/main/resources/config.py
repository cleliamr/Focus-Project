import numpy as np

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
N_turns = 10  # Number of turns
L = 0.1  # 10 cm length
R = 0.01  # 1 cm radius
points_per_turn = 50  # points rendered
shift_distance = 0.2 # distance to focus point
I = 10  # current
Grid_density = 0.05  # defines the Grid_density
time_steps = 100  # Number of frames for the animation
output_folder = "frames"  # Folder to save the frames
video_filename = "magnetic_field_animation.mp4"  # Output video filename
span_of_animation = 20
Hz = 1 # rotations per second
rot_freq = 60 # number of seconds to return to old rot. axis
Grid_size = 0.01