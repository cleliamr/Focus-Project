import numpy as np
from mayavi import mlab
import imageio.v2 as imageio
import os
from src.main.python.current_function_v01 import current_mag_flex
from src.main.python.Coil_v02 import generate_solenoid_points_flex
from src.main.python.Current_v02 import calculate_current_flex


# creating Grid, defining render density
def setup_plot(Grid_density, Grid_size):
    x, y, z = np.mgrid[-Grid_size:Grid_size:Grid_density , -Grid_size:Grid_size:Grid_density, -Grid_size:Grid_size:Grid_density]
    return x, y, z

# calculating number of points in solenoid
def solenoid_amnt_points(N_turns, points_per_turn):
    return N_turns * points_per_turn

# calculating magnitude of r
def r_magnitude(r):
    return np.linalg.norm(r)

# take 3d current and radius and return 3d B-field
def Biot_Savart_Law(mu_0, current, r):
    r_mag = np.linalg.norm(r)  # Magnitude of r vector
    if r_mag == 0:
        return np.array([0, 0, 0])  # To avoid division by zero
    return mu_0 * np.cross(current, r) / ((r_mag ** 3) * 4 * np.pi)

# take B-field created by solenoids and add them up, return as Bx, By, Bz
def superpositioning_of_Vector_fields(B_fields):
    # Initialize sums for each component (x, y, z) as the first field components
    B_x_sum = B_fields[0][..., 0]
    B_y_sum = B_fields[0][..., 1]
    B_z_sum = B_fields[0][..., 2]

    # Loop over the rest of the fields and add the components
    for B in B_fields[1:]:
        B_x_sum += B[..., 0]
        B_y_sum += B[..., 1]
        B_z_sum += B[..., 2]

    return B_x_sum, B_y_sum, B_z_sum

# calculating B-field for each solenoid
def calculate_B_field(solenoid, current, x, y, z, N_turns, points_per_turn, mu_0):
    # Initialize B1_single as a 3D array (same shape as x, y, z) to store 3D vectors
    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))  # (nx, ny, nz, 3) for 3D vectors

    for i in range(solenoid_amnt_points(N_turns, points_per_turn)):
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    # Calculate the vector distance r from the solenoid point to the grid point
                    r = np.array(solenoid[i]) - np.array([x[j, k, l], y[j, k, l], z[j, k, l]])
                    B[j, k, l] += Biot_Savart_Law(mu_0, current[i], r) # Sum the contributions from each solenoid point

    return B

# Plot and save each frame
def plot_magnetic_field(x, y, z, Bx, By, Bz, step, output_folder):
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    mlab.figure(bgcolor=(1, 1, 1))  # Create a white background figure
    quiver = mlab.quiver3d(x, y, z, Bx, By, Bz, scalars=B_magnitude, scale_factor=50, colormap='jet')
    mlab.view(azimuth=45, elevation=45, distance=1)
    mlab.colorbar(quiver, title="Field Magnitude", orientation='vertical')
    mlab.title(f"Magnetic Field at Step {step}", size=0.2)

    # Find the vector magnitude at the origin
    origin_index = np.argmin(np.abs(x) + np.abs(y) + np.abs(z))  # Closest index to origin
    origin_magnitude = B_magnitude.flatten()[origin_index]  # Vector magnitude at origin

    # Get the x, y, z coordinates of the origin
    origin_coords = (x.flatten()[origin_index], y.flatten()[origin_index], z.flatten()[origin_index])

    # Add custom text along with the vector magnitude at the origin
    text = f"Magnitude in Milliteslas: {round(origin_magnitude, 6) * 1000}"
    mlab.text3d(origin_coords[0], origin_coords[1], origin_coords[2], text, scale=0.01, color=(0, 0, 0))

    # Save the frame as an image
    frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
    mlab.savefig(frame_filename)
    mlab.close()  # Close the figure for the next frame

# Generate animation frames
def generate_animation_frames(mu_0, N_turns, L, R, points_per_turn, shift_distance, I_max, Grid_density, time_steps, output_folder, span_of_animation, Hz, rot_freq, Grid_size, model_choice):
    # Create folder to store frames if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create grid for vector field
    x, y, z = setup_plot(Grid_density, Grid_size)

    # Generate Solenoid points according to model choice
    solenoid_points = generate_solenoid_points_flex(N_turns, L, R, shift_distance, points_per_turn, model_choice)

    # Calculate current magnitudes according to model choice
    current_mag = np.array(current_mag_flex(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice))

    # Loop through time steps and update the magnetic field
    for step in range(time_steps):
        # define current for each solenoid at point in time
        current = calculate_current_flex(N_turns, L, points_per_turn, current_mag[:,step], model_choice)

        # Calculate B-field for each solenoid
        B_fields = []
        for i in range(len(solenoid_points)):
            B_fields.append(calculate_B_field(solenoid_points[i], current[i], x, y, z, N_turns, points_per_turn, mu_0))

        # Sum the magnetic fields
        Bx, By, Bz = superpositioning_of_Vector_fields(B_fields)

        # Plot and save the magnetic field at this time step
        plot_magnetic_field(x, y, z, Bx, By, Bz, step, output_folder)

# Create video from saved frames
def create_video_from_frames(time_steps, output_folder, video_filename):
    images = []
    for step in range(time_steps):
        frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
        images.append(imageio.imread(frame_filename))

    # Save frames as a video
    imageio.mimsave(video_filename, images, fps=5)  # Adjust fps for desired speed
    print(f"Video saved as {video_filename}")
