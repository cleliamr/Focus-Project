from Vector_Field_Animation import generate_animation_frames
from Vector_Field_Animation import create_video_from_frames
from Coil_v02 import generate_solenoid_points_4S
from Coil_v02 import plot_solenoids_4S
from config import mu_0, N_turns, L, R, points_per_turn, shift_distance, I, Grid_density, Grid_size, time_steps, output_folder, video_filename, span_of_animation, Hz, rot_freq


# Generate frames and create the video
generate_animation_frames(mu_0, N_turns, L, R, points_per_turn, shift_distance, I, Grid_density, time_steps, output_folder, span_of_animation, Hz, rot_freq, Grid_size)
create_video_from_frames(time_steps, output_folder, video_filename)

"""
solenoid1, solenoid2, solenoid3, solenoid4 = generate_solenoid_points_4S(N_turns, L, R, shift_distance, points_per_turn)
plot_solenoids_4S(solenoid1, solenoid2, solenoid3, solenoid4)
"""