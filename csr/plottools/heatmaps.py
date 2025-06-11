import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools, os
from scipy.fft import fft, fftfreq


def generic_heatmap(data, thattoplot, ntimesteps):
    # print("Heatmap input shape:", data.shape)

    # Determine the best square shape
    n_sims = data.shape[0]
    dim = int(np.floor(np.sqrt(n_sims)))  # Largest possible dimension
    elements_to_plot = dim * dim  # Total elements that can be plotted in a square
    discarded_elements = n_sims - elements_to_plot

    # print(f"Reshaping to: ({dim}, {dim})")
    # print(f"Discarded elements: {discarded_elements}")

    # Reshape to the largest possible square
    reshaped_pcis = data[:elements_to_plot].reshape(dim, dim)

    # Range is generic
    param1_values = np.linspace(0, 16, dim)  # Adjust 0-7 range as needed
    param2_values = np.linspace(0, 16, dim)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(reshaped_pcis,
                xticklabels=np.round(param2_values, 2),  # Adjust rounding as needed
                yticklabels=np.round(param1_values, 2),
                cmap="viridis", annot=True, fmt=".2f")

    plt.xlabel("Parameter 0-6")
    plt.ylabel("Parameter 0-6")
    plt.title(f"{thattoplot} of Pspace")
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs("./plots", exist_ok=True)
    # print("curdir", os.getcwd())

    plt.savefig(f'./plots/{thattoplot}_nsims_{n_sims}_train_{ntimesteps}.png')

    # return discarded_elements

def mse_heatmap(mses):

    print("msesheatmap", mses.shape)
    # dim = int(np.sqrt(mses.shape))
    dim = 8

    # Example data (replace with your parameter and MSE values)
    param1_values = np.linspace(0, 7, dim)  # Values for parameter 1
    param2_values = np.linspace(0, 7, dim)  # Values for parameter 1

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(mses.reshape(8,8),
                xticklabels=np.round(param1_values, dim),
                yticklabels=np.round(param2_values, dim),
                cmap="viridis", annot=True, fmt=".2f")

    plt.xlabel("Parameter 0-6")
    plt.ylabel("Parameter 0-6")
    plt.title("PCI of Pspace")
    plt.tight_layout()
    # plt.show()


def mse_heatmap_(mses):
    # Define ranges for 6 parameters (3 for x-axis, 3 for y-axis)
    param_x1 = [0.1, 0.2, 0.3]
    param_x2 = [10, 20]
    param_x3 = [0.01, 0.02]

    param_y1 = [5, 15]
    param_y2 = [0.001, 0.005]
    param_y3 = [100, 200]

    # Generate all combinations of 3 parameters for x and y
    x_combinations = list(itertools.product(param_x1, param_x2, param_x3))  # X-axis
    y_combinations = list(itertools.product(param_y1, param_y2, param_y3))  # Y-axis

    # Generate a results matrix based on combinations
    # For simplicity, random values are used. Replace with actual results.
    # results = np.random.rand(len(y_combinations), len(x_combinations))

    # Format tick labels
    xtick_labels = [f"{x[0]}-{x[1]}-{x[2]}" for x in x_combinations]
    ytick_labels = [f"{y[0]}-{y[1]}-{y[2]}" for y in y_combinations]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mses.reshape(8,8), cmap="viridis", annot=True, cbar=True,
                xticklabels=xtick_labels, yticklabels=ytick_labels)

    plt.xlabel("Combinations of Param_x1, Param_x2, Param_x3")
    plt.ylabel("Combinations of Param_y1, Param_y2, Param_y3")
    plt.title("Heatmap of Parameter Combinations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()

def plotpred(Ys, Ypreds, ntimesteps, whattoplot):

    n_work_items = Ys.shape[1]
    # print("Ys.shape", Ys.shape)
    # print("Ypreds.shape", Ypreds.shape)

    # Select 4 signals (e.g., the first 4 signals)
    selected_signals = [0, 1, 2, 3]  # Indices of the 4 signals to plot

    # Create a 4x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns

    for row, signal_idx in enumerate(selected_signals):  # Loop over the 4 selected signals
        for col in range(3):  # Loop over the 3 outputs
            ax = axes[row, col]  # Access the specific subplot

            # Extract the signal and prediction for this output
            Y_signal = Ys[signal_idx, :, col]  # True signal
            Y_pred_signal = Ypreds[signal_idx, :, col]  # Predicted output

            # Plot the true signal and the prediction
            ax.plot(Y_signal, label="True", color="blue")
            ax.plot(Y_pred_signal, label="Predicted", linestyle="--", color="red")

            # Add titles and labels
            if row == 0:  # Add column titles to the first row
                ax.set_title(f"Output {col + 1}", fontsize=12)
            if col == 0:  # Add row labels for the first column
                ax.set_ylabel(f"{whattoplot} {signal_idx + 1}", fontsize=12)

            ax.legend(fontsize=8)
            ax.grid(True)

    # Adjust layouut
    # plt.title(f"{whattoplot}")
    plt.suptitle(f"{whattoplot}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./plots/Prediction_4x3_{whattoplot}_nsims_{n_work_items}_train_{ntimesteps}.png')

def plotpred_teach_train(Ys, Ypreds, Ysomehing,  ntimesteps, whattoplot, delta_t=10):

    n_work_items = Ys.shape[1]
    # print("Y_Ypred[0,0,:,2]", Y_Ypred[0,0,:,2])

    # Select 4 signals (e.g., the first 4 signals)
    selected_signals = [0, 1, 2, 3]  # Indices of the 4 signals to plot
    # selected_signals = [0]  # Indices of the 4 signals to plot

    # Create a 4x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns

    for row, signal_idx in enumerate(selected_signals):  # Loop over the 4 selected signals
        for col in range(3):  # Loop over the 3 outputs
            ax = axes[row, col]  # Access the specific subplot

            # Extract the signal and prediction for this output
            Y_signal = Ys[signal_idx, :, col]  # True signal
            Y_pred_signal = Ypreds[signal_idx, :, col]  # Predicted output
            Y_some = Ysomehing[signal_idx, :, col]

            original_indices = np.arange(len(Y_signal))  # Original indices
            shifted_indices = original_indices[:len(Y_pred_signal)] - delta_t

            # Plot the true signal and the prediction
            ax.plot(original_indices, Y_signal, label="Input", color="blue")
            ax.plot(shifted_indices, Y_pred_signal, label="Predicted", linestyle="--", color="red")
            ax.plot(original_indices, Y_some, label="Teacher", linestyle="dotted", color="green")

            # center_x = 40
            # range_x = 30  # Half the range to show on either side
            # ax.set_xlim(center_x - range_x, center_x + range_x)
            # ax.axvline(center_x, color="red", linestyle="--", label="Center (-40)")

            # Add titles and labels
            if row == 0:  # Add column titles to the first row
                ax.set_title(f"Lorentz Dim {col + 1}", fontsize=12)
            if col == 0:  # Add row labels for the first column
                ax.set_ylabel(f"{whattoplot}", fontsize=12)

            ax.legend(fontsize=8)
            ax.grid(True)

    # Adjust layouut
    # plt.title(f"{whattoplot}")
    plt.suptitle(f"{whattoplot}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./plots/Prediction_4x3_{whattoplot}_nsims_{n_work_items}_train_{ntimesteps}.png')

def plot_predictions_vs_true_fullsims(y_true, y_pred, whattoplot):
    """
    Plot predictions (y_pred) vs true values (y_true) for each simulation
    but only for the first output (index 0) in a square layout of subplots.

    Args:
        y_true (torch.Tensor): True values, shape (nsims, ntimesteps, noutputs)
        y_pred (torch.Tensor): Predicted values, shape (nsims, ntimesteps, noutputs)
        nsims (int): Number of simulations.
        ntimesteps (int): Number of timesteps.
    """

    nsims, ntimesteps, _ = y_true.shape
    # Calculate the number of rows and columns for square subplot layout
    ncols = int(np.ceil(np.sqrt(nsims)))  # Columns = square root of nsims
    nrows = int(np.floor(np.sqrt(nsims)))  # Rows = floor of square root of nsims

    # Adjust if needed to ensure the grid fits all subplots
    if nrows * ncols < nsims:
        nrows += 1

    # Create a new figure for plotting
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))

    # Flatten the axes array to easily loop over them
    axes = axes.flatten()

    # Loop over each simulation and plot
    for sim_idx in range(nsims):
        ax = axes[sim_idx]
        ax.plot(range(ntimesteps), y_true[sim_idx, :, 0], label='True', color='blue')
        ax.plot(range(ntimesteps), y_pred[sim_idx, :, 0], label='Pred', color='red')

        ax.set_title(f'{whattoplot[sim_idx]} {sim_idx + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('1st pred')
        ax.legend()

    # Hide unused subplots if nsims is less than nrows * ncols
    for i in range(nsims, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()


def plot_target_dyn(target_dyn, teach_dyn, whattoplot):
    """
    Plots time series data with shape (3, 2500).

    Parameters:
        target_dyn (numpy.ndarray): A 2D array with shape (3, 2500).
    """
    # Validate input shape
    # if target_dyn.shape != (3, 2500):
    #     raise ValueError(f"Expected shape (3, 2500), but got {target_dyn.shape}")

    # Create a figure and axis for each time series
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    time = range(target_dyn.shape[1])  # Time axis (0 to 2499)

    # Plot each time series
    for i in range(3):
        axs[i].plot(time, target_dyn[i], label=f"Input {i + 1}", color=f"C{i}")
        axs[i].plot(time, teach_dyn[i], label=f"Teacher {i + 1}", color=f"C{i+1}")
        axs[i].set_ylabel(f"Lorentz Dim {i + 1}")
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    # Set shared x-axis label
    axs[-1].set_xlabel("Timesteps")

    # Add a title
    fig.suptitle(f"{whattoplot}", fontsize=14)
    # plt.show()

def plot_tavgs_notstacked(tavgs, whattoplot, inregions=[10, 17, 30], outregions=[16, 21, 23]):
    plt.figure(figsize=(12, 8))
    for i in range(tavgs.shape[1]):  # Loop through all time series
        if i in inregions:
            # Plot regions in `inregions` with a different style
            plt.plot(
                tavgs[:, i] + 0 * 1.5,
                label=f'Region {i} (Input)',
                linewidth=2.5,  # Thicker line
                linestyle='--',  # Dashed line
                color='red' if i == inregions[0] else None  # Optional color
            )
        elif i in outregions:
            # Plot regions in `inregions` with a different style
            plt.plot(
                tavgs[:, i] + 0 * 1.5,
                label=f'Region {i} (Output)',
                linewidth=2.5,  # Thicker line
                linestyle=':',  # Dashed line
                color='red' if i == inregions[0] else None  # Optional color
            )
        else:
            # Plot other regions normally
            plt.plot(
                tavgs[:, i] + 0 * 1.5,
                label=f'Region {i}' if i < 10 else None,
                alpha=0.7  # Optional transparency
            )
    plt.title(whattoplot)
    plt.xlabel("Time")
    plt.ylabel("Activity (offset per region)")
    plt.legend(loc='upper right', fontsize='small', ncol=2)  # Adjust legend if needed
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./plots/{whattoplot}.png')


def plot_tavgs(tavgs, whattoplot, inregions=[10, 17, 30], outregions=[16, 21, 23]):
    plt.figure(figsize=(12, 8))
    for i in range(tavgs.shape[1]):  # Loop through all time series
        if i in inregions:
            # Plot regions in `inregions` with a different style
            plt.plot(
                tavgs[:, i] + i * 1.5,
                label=f'Region {i} (Input)',
                linewidth=2.5,  # Thicker line
                linestyle='--',  # Dashed line
                color='red' if i == inregions[0] else None  # Optional color
            )
        elif i in outregions:
            # Plot regions in `inregions` with a different style
            plt.plot(
                tavgs[:, i] + i * 1.5,
                label=f'Region {i} (Output)',
                linewidth=2.5,  # Thicker line
                linestyle=':',  # Dashed line
                color='red' if i == inregions[0] else None  # Optional color
            )
        else:
            # Plot other regions normally
            plt.plot(
                tavgs[:, i] + i * 1.5,
                label=f'Region {i}' if i < 10 else None,
                alpha=0.7  # Optional transparency
            )
    plt.title(whattoplot)
    plt.xlabel("Time")
    plt.ylabel("Activity (offset per region)")
    plt.legend(loc='upper right', fontsize='small', ncol=2)  # Adjust legend if needed
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./plots/{whattoplot}.png')


def plot_frequency_spectra(Ypred, Y, dt):
    """
    Plots the frequency spectrum of predictions vs. teacher signals for multiple simulations.

    Args:
        Ypred (torch.Tensor): Predicted signals, shape (nsims, ntimesteps, noutputs).
        Y (torch.Tensor): Teacher signals, shape (nsims, ntimesteps, noutputs).
        dt (float): Time step for signal sampling.
        nsims (int): Number of simulations to process.
    """
    # Calculate the number of rows and columns for the square subplot layout
    nsims = Y.shape[0]

    ncols = int(np.ceil(np.sqrt(nsims)))
    nrows = int(np.floor(np.sqrt(nsims)))
    if nrows * ncols < nsims:
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    dt = .001
    sampling_rate = 1 / dt  # Sampling rate from time step

    for i in range(nsims):
        signalYpred = Ypred[i, :, 0]  # Predicted signal for output 0
        signalY = Y[i, :, 0]  # Teacher signal for output 0

        # Perform FFT and calculate power spectrum
        N = len(signalYpred)
        signal_fftYpred = fft(signalYpred)
        signal_fftY = fft(signalY)
        signal_powerYpred = np.abs(signal_fftYpred) ** 2
        signal_powerY = np.abs(signal_fftY) ** 2

        # Compute corresponding frequencies
        freqs = fftfreq(N, d=1 / sampling_rate)

        # Focus on positive frequencies
        positive_freqs = freqs[1:N // 2]
        positive_powerYpred = signal_powerYpred[1:N // 2]
        positive_powerY = signal_powerY[1:N // 2]

        # Find the dominant frequency
        dominant_frequencyYpred = positive_freqs[np.argmax(positive_powerYpred)]
        dominant_frequencyY = positive_freqs[np.argmax(positive_powerY)]

        # Plot the frequency spectrum
        ax = axes[i]
        ax.plot(positive_freqs, positive_powerYpred, label="Prediction", color='red')
        ax.plot(positive_freqs, positive_powerY, label="Teacher", color='blue')
        ax.set_title(f"Simulation {i + 1}\nDominant Freq (Pred): {dominant_frequencyYpred:.2f} Hz\n"
                     f"Dominant Freq (Teacher): {dominant_frequencyY:.2f} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.grid(True)
        ax.legend()

        # Print stats for the current simulation
        print(f"Simulation {i + 1}:")
        print(f"  Dominant Frequency (Prediction): {dominant_frequencyYpred:.2f} Hz")
        print(f"  Dominant Frequency (Teacher): {dominant_frequencyY:.2f} Hz")
        print(f"  Prediction Power - Mean: {np.mean(positive_powerYpred):.2e}, Std: {np.std(positive_powerYpred):.2e}")
        print(f"  Teacher Power - Mean: {np.mean(positive_powerY):.2e}, Std: {np.std(positive_powerY):.2e}")
        print("-" * 40)

    # Hide unused subplots
    for i in range(nsims, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

if __name__ == "__main__":

    def test_mse_heat():
        mses = np.random.rand(8, 8)
        mse_heatmap(mses)

    def test_pci_heat():
        pcis = np.random.rand(111)
        generic_heatmap(pcis)
        plt.show()

    def test_plotpred():
        Y = np.random.rand(16, 200, 3)  # True values
        Y_pred = np.random.rand(16, 200, 3)  # Predicted values

        plotpred(Y, Y_pred)

    def test_plottargdyn():
        target_dyn = np.random.rand(3, 2500)

        # Plot the data
        plot_target_dyn(target_dyn)

    # test_plotpred()

    # test_pci_heat()

    test_plottargdyn()