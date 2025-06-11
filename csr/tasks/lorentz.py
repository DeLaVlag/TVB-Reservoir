import numpy as np
import matplotlib.pyplot as plt
from csr.plottools.heatmaps import *


# Define the system of ODEs
def _lorenz_system(x, y, z):
    dx_dt = 10 * (y - x)
    dy_dt = x * (28 - z) - y
    dz_dt = x * y - (8 / 3) * z
    return dx_dt, dy_dt, dz_dt

def _apply_constraints(x, y, z):
    x = max(0, x)
    y = max(0, y)
    z = max(0, z)
    return x, y, z


# Forward Euler method
def forward_euler(x0, y0, z0, h, num_steps):
    # Initialize arrays to store the solutions
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)

    # Set initial conditions
    x[0], y[0], z[0] = x0, y0, z0

    # Iterate using the Forward Euler method
    for i in range(1, num_steps):
        dx_dt, dy_dt, dz_dt = _lorenz_system(x[i - 1], y[i - 1], z[i - 1])
        x[i] = x[i - 1] + h * dx_dt
        y[i] = y[i - 1] + h * dy_dt
        z[i] = z[i - 1] + h * dz_dt

        # Apply constraints to ensure positivity// maybe not necessary as negatives get sumtracted
        # x[i], y[i], z[i] = _apply_constraints(x[i], y[i], z[i])

    return x, y, z

def _plot_dyn(x, y, z):
    # Plot the results
    fig = plt.figure(figsize=(12, 8))

    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_title('Lorenz System (Forward Euler Method)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 2D projections
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.plot(x, y, lw=0.5)
    ax1.set_title('X vs Y')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.plot(x, z, lw=0.5)
    ax2.set_title('X vs Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')

    ax3.plot(y, z, lw=0.5)
    ax3.set_title('Y vs Z')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')

    plt.tight_layout()
    plt.show()

def run_dyn(x0, y0, z0, dt, num_steps):

    # x0 = np.random.uniform(-20, 40)
    # y0 = np.random.uniform(-25, 50)
    # z0 = np.random.uniform(0, 50)
    # dt = 0.005  # Step size
    # num_steps = 10000  # Number of steps

    # Solve the system using Forward Euler method
    x, y, z = forward_euler(x0, y0, z0, dt, num_steps)

    return x, y, z

def shift_lorenz_trajectory(x, y, z, phase_shift_pis, h, char_frequency=1):
    # Compute the time offset in terms of steps
    time_shift = phase_shift_pis * np.pi / (2 * np.pi / char_frequency)
    steps_shift = int(time_shift / h)

    # Circularly shift the trajectories
    x_shifted = np.roll(x, -steps_shift)
    y_shifted = np.roll(y, -steps_shift)
    z_shifted = np.roll(z, -steps_shift)

    return x_shifted, y_shifted, z_shifted


def normalize_weights(arr, new_min_max):
    """
    Normalize each column of a 2D array to its respective new range.

    Parameters:
    arr (numpy array): The input 2D array to be normalized.
    new_min_max (list of tuples): A list containing tuples of the form (new_min, new_max) for each column.

    Returns:
    numpy array: The normalized 2D array.
    """
    normalized_arr = np.zeros_like(arr, dtype=float)
    for i in range(arr.shape[1]):
        col = arr[:, i]
        arr_min = np.min(col)
        arr_max = np.max(col)
        new_min, new_max = new_min_max[i]
        normalized_arr[:, i] = (col - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min

    return normalized_arr

def normalize_lorenz_signals(x, y, z, method="minmax"):
    """
    Normalize Lorenz system signals.

    Parameters:
    - x, y, z: Lorenz signals.
    - method: "minmax", "zscore", or "unit".

    Returns:
    - x_norm, y_norm, z_norm: Normalized signals.
    """
    def normalize(signal, method):
        if method == "minmax":
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        elif method == "zscore":
            return (signal - np.mean(signal)) / np.std(signal)
        elif method == "unit":
            return signal / np.linalg.norm(signal)
        else:
            raise ValueError("Unknown normalization method: choose 'minmax', 'zscore', or 'unit'")

    x_norm = normalize(x, method)
    y_norm = normalize(y, method)
    z_norm = normalize(z, method)

    return x_norm, y_norm, z_norm


if __name__ == "__main__":

    # Random initial conditions within the specified ranges
    # x0 = np.random.uniform(-20, 40)
    # y0 = np.random.uniform(-25, 50)
    # z0 = np.random.uniform(0, 50)

    x0 = np.random.uniform(0, 30)
    y0 = np.random.uniform(0, 30)
    z0 = np.random.uniform(0, 30)


    h = 0.004  # Step size
    num_steps = 1500  # Number of steps

    # Solve the system using Forward Euler method
    x, y, z = run_dyn(x0, y0, z0, h, num_steps)
    # _plot_dyn(x, y, z)

    stacked = np.vstack((x, y, z))
    plot_target_dyn(stacked, stacked, "trainer and teacher sig")
    plt.show()

