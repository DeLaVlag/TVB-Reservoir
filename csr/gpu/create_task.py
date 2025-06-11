import numpy as np
import matplotlib.pyplot as plt


def generate_sinusoidal_signals(time_steps=2500, num_signals=3, pshifts=0):
    """
    Generate sinusoidal signals with random initial conditions.

    Parameters:
        time_steps (int): Number of time steps for each signal.
        num_signals (int): Number of sinusoidal signals to generate.

    Returns:
        np.ndarray: Array of shape (num_signals, time_steps) containing the signals.
    """
    signals = np.zeros((num_signals, time_steps))
    t = np.linspace(0, 2 * np.pi, time_steps)  # Time points

    for i in range(num_signals):
        freq = np.random.uniform(10, 10)  # Random frequency
        phase = np.random.uniform(pshifts, pshifts)  # Random phase
        # phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        amplitude = np.random.uniform(.5, .5)  # Random amplitude
        signals[i] = amplitude * np.sin(freq * t + phase) + 3

    return signals


def plot_signals(signals, time_points):
    """
    Plot the generated sinusoidal signals.

    Parameters:
        signals (np.ndarray): Array of shape (num_signals, time_steps) containing the signals.
        time_points (np.ndarray): Time steps array for the x-axis.
    """
    num_signals = signals.shape[0]
    plt.figure(figsize=(10, 6))

    for i in range(num_signals):
        plt.plot(time_points, signals[i], label=f"Signal {i + 1}")

    plt.title("Generated Sinusoidal Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    # Example usage
    signals, time_points = generate_sinusoidal_signals()
    plot_signals(signals, time_points)
