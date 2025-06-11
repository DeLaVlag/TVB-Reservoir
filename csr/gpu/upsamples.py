import numpy as np
from numpy.fft import fft, ifft


def fft_upsample(pred_partly, n_inner_steps):
    """
    Perform FFT-based upsampling on the second dimension of pred_partly.

    Args:
        pred_partly (np.ndarray): Input array with shape (nsims, timesteps, features).
        n_inner_steps (int): Upsampling factor.

    Returns:
        np.ndarray: Upsampled array.
    """
    nsims, timesteps, features = pred_partly.shape
    new_timesteps = timesteps * n_inner_steps  # New size after upsampling

    # FFT along the timesteps axis (axis=1)
    freq_data = fft(pred_partly, axis=1)

    # Zero-pad in the frequency domain to increase resolution
    pad_width = (new_timesteps - timesteps) // 2
    freq_data_padded = np.pad(freq_data,
                              ((0, 0), (pad_width, pad_width), (0, 0)),
                              mode='constant')

    # Perform inverse FFT to get the upsampled time-domain data
    upsampled = ifft(freq_data_padded, axis=1).real  # Keep only the real part

    return upsampled


import matplotlib.pyplot as plt


def plot_upsampled(original, upsampled, whattoplot, n_samples=3):
    """
    Plot the original and upsampled arrays for comparison.

    Args:
        original (np.ndarray): Original array of shape (nsims, timesteps, features).
        upsampled (np.ndarray): Upsampled array of shape (nsims, new_timesteps, features).
        n_samples (int): Number of simulations (nsims) to plot. Defaults to 3.
    """
    nsims, timesteps, features = original.shape
    _, new_timesteps, _ = upsampled.shape

    # Limit the number of samples to plot
    n_samples = min(n_samples, nsims)

    # Create a figure for plotting
    plt.figure(figsize=(15, 5 * n_samples))

    for sim in range(n_samples):
        for feature in range(features):
            plt.subplot(n_samples, features, sim * features + feature + 1)

            # Time axes for original and upsampled
            time_original = np.linspace(0, timesteps - 1, timesteps)
            time_upsampled = np.linspace(0, timesteps - 1, new_timesteps)

            # Plot original and upsampled data
            plt.plot(time_original, original[sim, :, feature], 'o-', label='Original')
            plt.plot(time_upsampled, upsampled[sim, :, feature], '-', label='Upsampled', alpha=0.7)

            # Labels and title
            plt.title(f"Lorentz Dimension {feature + 1}")
            plt.xlabel("Time")
            plt.ylabel(whattoplot)
            plt.legend()

    plt.tight_layout()

def fft_upsample_with_window(pred_partly, n_inner_steps):
    """
    Perform FFT-based upsampling with windowing.

    Args:
        pred_partly (np.ndarray): Input array with shape (nsims, timesteps, features).
        n_inner_steps (int): Upsampling factor.

    Returns:
        np.ndarray: Upsampled array.
    """
    nsims, timesteps, features = pred_partly.shape
    new_timesteps = timesteps * n_inner_steps  # New size after upsampling

    # Initialize output
    upsampled = np.zeros((nsims, new_timesteps, features))

    for sim in range(nsims):
        for feature in range(features):
            signal = pred_partly[sim, :, feature]

            # Apply a Hanning window to reduce edge effects
            windowed_signal = signal * np.hanning(len(signal))

            # FFT and zero-padding
            freq_data = fft(windowed_signal)
            pad_width = (new_timesteps - timesteps) // 2
            freq_data_padded = np.pad(freq_data, (pad_width, pad_width), mode='constant')

            # Inverse FFT and real part
            upsampled_signal = ifft(freq_data_padded).real

            # Store the upsampled signal
            upsampled[sim, :, feature] = upsampled_signal

    return upsampled

from scipy.interpolate import interp1d

def spline_upsample(pred_partly, n_inner_steps):
    """
    Perform spline interpolation-based upsampling.

    Args:
        pred_partly (np.ndarray): Input array with shape (nsims, timesteps, features).
        n_inner_steps (int): Upsampling factor.

    Returns:
        np.ndarray: Upsampled array.
    """
    nsims, timesteps, features = pred_partly.shape
    new_timesteps = timesteps * n_inner_steps
    upsampled = np.zeros((nsims, new_timesteps, features))

    for sim in range(nsims):
        for feature in range(features):
            signal = pred_partly[sim, :, feature]
            time_original = np.linspace(0, timesteps - 1, timesteps)
            time_upsampled = np.linspace(0, timesteps - 1, new_timesteps)

            # Spline interpolation
            interpolator = interp1d(time_original, signal, kind='cubic')
            upsampled[sim, :, feature] = interpolator(time_upsampled)

    return upsampled



if __name__ == "__main__":

    # Example data
    nsims, timesteps, features = 5, 20, 3
    pred_partly = np.random.rand(nsims, timesteps, features)  # Random example data
    n_inner_steps = 4

    # Perform FFT upsampling
    # upsampled_data = fft_upsample(pred_partly, n_inner_steps)
    # upsampled_data = fft_upsample_with_window(pred_partly, n_inner_steps)
    upsampled_data = spline_upsample(pred_partly, n_inner_steps)

    # Verify the new shape
    print("Original shape:", pred_partly.shape)
    print("Upsampled shape:", upsampled_data.shape)

    plot_upsampled(pred_partly, upsampled_data, "testing")
    plt.show()
