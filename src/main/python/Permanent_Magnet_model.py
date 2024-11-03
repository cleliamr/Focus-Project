import numpy as np
from mayavi import mlab
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from config import mu_0, time_steps, video_filename, Hz, magnet_center, magnet_moment, magnet_dimensions, Grid_density, output_folder, Grid_size

# creating Grid, defining render density
def setup_plot(Grid_density, Grid_size):
    x, y, z = np.mgrid[-Grid_size:Grid_size:Grid_density , -Grid_size:Grid_size:Grid_density, -Grid_size:Grid_size:Grid_density]
    return x, y, z

def plot_B_over_time(Bx_over_time, By_over_time, Bz_over_time, time_steps):
    t = np.linspace(0, (time_steps // 10) - 1, time_steps)
    fig, ax = plt.subplots()

    plt.plot(t, Bx_over_time, label='Bx', color='r')
    plt.plot(t, By_over_time, label='By', color='g')
    plt.plot(t, Bz_over_time, label='Bz', color='b')

    ax.set(xlabel='time (s)', ylabel='B-field strength (A)',
           title='B over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

# define the magnetisation orientation for each time point
def magnet_moment_position(magnet_moment, time_steps, Hz, magnet_volume):
    magnet_orientations = np.zeros((time_steps, 3))
    t = np.linspace(0, (time_steps // 10) - 1, time_steps)

    theta = Hz * 2 * np.pi * t

    for i in range(time_steps):
        magnet_orientations[i] = rotate_vector(magnet_moment, 'z', theta[i]) * magnet_volume

    return magnet_orientations

# define the magents volume
def calc_magnet_volume(magnet_dimensions):
    magnet_volume = magnet_dimensions[0] * magnet_dimensions[1] * magnet_dimensions[2]
    return magnet_volume
# Function to calculate magnetic field at a point due to a single magnet
def calculate_magnetic_field(r, magnet_moment):
    r_mag = np.linalg.norm(r)

    if r_mag < 0.2:
        return np.array([0.0, 0.0, 0.0])

    B = (mu_0 / (4 * np.pi)) * (3 * np.dot(magnet_moment, r) * r / r_mag ** 5 - magnet_moment / r_mag ** 3)
    return B

# based on dipole assumption calculate the magnetic field strength at each point
def calculate_B_field_in_room(magnet_center, magnet_moment, x, y, z):
    # Initialize B1_single as a 3D array (same shape as x, y, z) to store 3D vectors
    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))  # (nx, ny, nz, 3) for 3D vectors

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                # Calculate the vector distance r from the solenoid point to the grid point
                r = np.array(magnet_center) - np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                B[i, j, k] = calculate_magnetic_field(r, magnet_moment)

    return B[..., 0], B[..., 1], B[..., 2]

# Plot and save each frame
def plot_magnetic_field(x, y, z, Bx, By, Bz, step, output_folder):
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))  # Create a white background figure
    quiver = mlab.quiver3d(x, y, z, Bx, By, Bz, scalars=B_magnitude, scale_factor=10, colormap='jet')
    mlab.view(azimuth=45, elevation=45, distance=5)
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
    mlab.savefig(frame_filename, size=(1920, 1080))
    mlab.close()  # Close the figure for the next frame

# Create video from saved frames
def create_video_from_frames_pmodel():
    images = []
    for step in range(time_steps):
        frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
        images.append(imageio.imread(frame_filename))

    # Save frames as a video
    imageio.mimsave(video_filename, images, fps=10)  # Adjust fps for desired speed
    print(f"Video saved as {video_filename}")

# vector rotation
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

# Generate animation frames
def generate_animation_frames_pmodel():
    # Create folder to store frames if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create grid for vector field
    x, y, z = setup_plot(Grid_density, Grid_size)

    # caculate magnets volume
    magnet_volume = calc_magnet_volume(magnet_dimensions)

    # define magnetisation orientation over time
    magnet_orientations = magnet_moment_position(magnet_moment, time_steps, Hz, magnet_volume)

    Bx_over_time = np.zeros(time_steps)
    By_over_time = np.zeros(time_steps)
    Bz_over_time = np.zeros(time_steps)

    # check for single point grid:
    single_point = False
    if Grid_density > (Grid_size * 4):
        single_point = True
    # Loop through time steps and update the magnetic field
    for step in range(time_steps):
        # Calculate B-field for each solenoid
        Bx, By, Bz = calculate_B_field_in_room(magnet_center, magnet_orientations[step], x, y, z)
        if single_point:
            Bx_over_time[step] = Bx
            By_over_time[step] = By
            Bz_over_time[step] = Bz

            # Plot and save the magnetic field at this time step
            plot_magnetic_field(x, y, z, Bx, By, Bz, step, output_folder)

    plot_B_over_time(Bx_over_time, By_over_time, Bz_over_time, time_steps)
