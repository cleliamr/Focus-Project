from Vector_Field_Animation import setup_animation_frames, create_video_from_frames, run_multiprocessing
from config import time_steps

from Permanent_Magnet_model import generate_animation_frames_pmodel, create_video_from_frames_pmodel

# run the multiprocessing
if __name__ == '__main__':
    # define model choice
    model_choice = 0
    while model_choice not in ("4S", "3S", "2S"):
        model_choice = input("Choose between '4S', '3S' and '2S' Model: ")

    # Generate setup and create the video
    solenoid_points, current, current_mag, x, y, z = setup_animation_frames(model_choice)

    # run multiprocessing
    B_over_time = run_multiprocessing(time_steps, solenoid_points, current, current_mag, x, y, z)
    print(B_over_time)

    # create video / animation from frames
    create_video_from_frames()

"""
generate_animation_frames_pmodel()
create_video_from_frames_pmodel()
"""