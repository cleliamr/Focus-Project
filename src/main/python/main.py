from Vector_Field_Animation import setup_animation_frames, create_video_from_frames, run_multiprocessing
from B_field_analysis import B_field_analysis
from config import time_steps
import numpy as np
from Permanent_Magnet_model import generate_animation_frames_pmodel, create_video_from_frames_pmodel
"""
# run the multiprocessing
if __name__ == '__main__':
    # define model choice
    model_choice = 0
    while model_choice not in ("4S", "3S", "2S"):
        model_choice = input("Choose between '4S', '3S' and '2S' Model: ")

    # Generate setup
    solenoid_points, current, current_mag, x, y, z = setup_animation_frames(model_choice)

    # run multiprocessing
    B_field = run_multiprocessing(time_steps, solenoid_points, current, current_mag, x, y, z)

    # output analysis of B-field over time
    B_field_analysis(B_field, x ,y ,z)

    # create video / animation from frames
    create_video_from_frames()
"""

generate_animation_frames_pmodel()
create_video_from_frames_pmodel()
