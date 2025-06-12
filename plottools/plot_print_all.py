from scipy.stats import wilcoxon, shapiro

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pretty_print_best_params(best_indices, best_mse_tloop0, best_mse_tloop1, params, whattoplot):
    """
    Pretty prints the best parameters and their corresponding MSE and tloop.

    Args:
    - best_indices (1D NumPy array): Indices of the best parameters.
    - best_mse_tloop0 (1D NumPy array): Array of MSE values (in percentages).
    - best_mse_tloop1 (1D NumPy array): Array of corresponding tloop values.
    - params (2D NumPy array): Parameter values, where rows correspond to simulations and columns to parameters.
    """
    # Flatten the arrays to 1D if necessary
    best_indices = best_indices.flatten()
    best_mse_tloop0 = best_mse_tloop0.flatten()
    best_mse_tloop1 = best_mse_tloop1.flatten()

    print(f"\n{whattoplot}:")
    print("-" * 50)
    print("i: bestprms [g, noise, ex_I, delta, Jmsw, eta, speed] have MSE xx% at tloop if avail")

    for i, idx in enumerate(best_indices):
        # Round params to two decimal places
        rounded_params = np.round(params[i], 2)
        mse = best_mse_tloop0[i]
        # tloop = best_mse_tloop1[idx]
        tloop = 0

        print(f"{i}: prms[{idx}] {list(rounded_params)} have MSE {mse:.5f} at tloop {tloop}")

    print("-" * 50)

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
    # plt.savefig(f'./plots/Prediction_4x3_{whattoplot}_nsims_{n_work_items}_train_{ntimesteps}.png')


def plot_3d_simulation_par(data_pcis, data_dfas, data_lyas, data_mses, whattosave, whattosave2):
    """
    Plots 3D scatter plots for PCIs, DFAs, LYAs, and MSEs in a single figure with 4 subplots.

    Parameters:
    data_pcis (numpy.ndarray): Array of shape (nsims, 4).
    data_dfas (numpy.ndarray): Array of shape (nsims, 4).
    data_lyas (numpy.ndarray): Array of shape (nsims, 4).
    data_mses (numpy.ndarray): Array of shape (nsims, 4).
    """
    fig = plt.figure(figsize=(14, 10))
    ft = 16
    lft = 12

    # Define subplots
    titles = ['PCIs', 'DFAs', 'LYAs', 'MSEs']
    datasets = [data_pcis, data_dfas, data_lyas, data_mses]
    colorbar_titles = ["PCI", r'$\alpha$', r'$\lambda$', "MSE"]

    for i, (data, title) in enumerate(zip(datasets, titles), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')

        # Extract X, Y, Z, and color values
        X, Y, Z, C = data[:, 0], data[:, 1], data[:, 2] * 1, data[:, 3]

        # Scatter plot
        sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', marker='o', alpha=0.3, s=25)

        if whattosave == "MBR":

            # Labels for MBR
            ax.set_xlabel('I', labelpad=10, fontsize=ft)
            ax.set_ylabel('J', labelpad=10, fontsize=ft)
            ax.set_zlabel(r'$\eta$', labelpad=10, fontsize=ft, ha='right', va='center')

        elif whattosave == "LB":
            # Labels for LB
            ax.set_xlabel('rNMDA', labelpad=10, fontsize=14)
            ax.set_ylabel(r'$a_{ee}$', labelpad=10, fontsize=14)
            ax.set_zlabel(r'$V_{T}$', labelpad=10, fontsize=14)

        ax.tick_params(axis='both', labelsize=lft)

        ax.set_title(title, fontsize=ft, loc='left', pad=0)
        # ax.title.set_y(15.02)
        # ax.title.set_position([0, 15])

        # Add colorbar with adjustments
        cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.15, shrink=0.8, aspect=20)
        cbar.set_label(colorbar_titles[i - 1], fontsize=ft)
        cbar.ax.tick_params(labelsize=lft)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save figure
    # plt.savefig(f'../plots/allCIs_params_{whattosave}_ctomesize{whattosave2}.png')

    # For HPC
    # plt.savefig(f'/p/project1/vbt/vandervlag1/liquidinterferencelearning/plots/allCIs_params_{whattosave}_ctomesize{whattosave2}.png')
    # plt.savefig(f'./plots/allCIs_params_{whattosave}_ctomesize{whattosave2}.svg', format="svg")
    plt.savefig(
        f'/home/michiel/Documents/Repos/LiquidInterferenceLearning/csr/gpu/plots/'
        f'allCIs_params_{whattosave}_ctomesize{whattosave2}.svg',
        format="svg")

    # plt.show()  # Uncomment to display the plot


def merged_histograms(data_pcis, data_dfas, data_lyas, data_mses, whattosave):
    """
    Plots a combined figure with two rows:
    - First row: Histograms of PCI, DFA, and LYA for active vs resting states.
    - Second row: Histograms of I, J, η distributions categorized by PCI, DFA, and LYA values.
    """

    ft = 14

    x = data_pcis[:, 0]  # I
    y = data_pcis[:, 1]  # η
    # z = np.round(data_pcis[:, 2], 1)  # J
    z = (data_pcis[:, 2])  # J
    mse = data_mses[:, 3]  # MSE values

    pcis = data_pcis[:, 3]
    dfas = data_dfas[:, 3]
    lyas = data_lyas[:, 3]

    mpcis = np.median(pcis[pcis < 3])
    # mpcis2 = np.median(pcis)
    mdfas = np.median(dfas)
    mlyas = np.median(lyas)

    mepcis = np.mean(pcis[pcis < 3])
    medfas = np.mean(dfas)
    melyas = np.mean(lyas)

    print(mpcis, mdfas, mlyas)
    print(mepcis, medfas, melyas)

    fig, axes = plt.subplots(2, 3, figsize=(15, 12), sharey=False)


    if whattosave == "MBR":

        param_labels = ["I", "J", "η"]
        # Masks for active vs resting states
        mask_cons = (x < 0.) & (y > 15) & (z < -5)
        mask_uncons = (x > 0.) & (y > 2.5) & (y < 15) & (z < -5)

        # Masks based on relative PCI, DFA, LYA values
        medfas = .5
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        # mask_highpci_highdfa_lowlya = (pcis >= .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_highpci_lowdfa_highlya = (pcis >= .75) & (dfas < 0.5) & (lyas > 0.3)
        # mask_lowpci_highdfa_lowlya = (pcis < .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_lowpci_lowdfa_highlya = (pcis < .75) & (dfas < 0.5) & (lyas > 0.3)
        colors = ['red', 'blue', 'green', 'purple']

    elif whattosave == "LB":

        param_labels = ["r_nmda", "a_ee", "V_T (mV)"]
        print(param_labels)

        mask_cons =   (x <= 0.0505) & (y >= 2.55) & (z <= -45e-3)
        mask_uncons = (x >  0.0505) & (y <  2.55) & (z >  -45e-3)

        # Masks based on relative PCI, DFA, LYA values
        # mask_highpci_highdfa_lowlya = (pcis >= .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_highpci_lowdfa_highlya = (pcis >= .75) & (dfas <  0.5) & (lyas >  0.3)
        # mask_lowpci_highdfa_lowlya =  (pcis  < .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_lowpci_lowdfa_highlya =  (pcis  < .75) & (dfas <  0.5) & (lyas >  0.3)

        medfas = .5
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        cmap = plt.get_cmap('viridis', 4)
        colors = [cmap(i) for i in range(4)]

        # setting and rounding of the ticks
        ticks = np.linspace(z.min(), z.max(), num=7)
        axes[1, 2].set_xticks(ticks)
        axes[1, 2].set_xticklabels([f"{tick:.3f}" for tick in ticks])

    # First row: PCI, DFA, LYA histograms
    metric_labels = ["PCI", "DFA", "LYA"]  # Swapped order
    metric_data = [pcis, dfas, lyas]

    labels = ["Active state", "Resting state"]
    masks = [mask_cons, mask_uncons]
    param_data = [x, y, z]

    y_max = 0
    for i in range(3):
        fixed_bins = np.histogram_bin_edges(metric_data[i], bins=7)
        hist_values, _ = np.histogram(metric_data[i][mask_cons], bins=fixed_bins)
        hist_values_uncons, _ = np.histogram(metric_data[i][mask_uncons], bins=fixed_bins)
        y_max = max(y_max, hist_values.max(), hist_values_uncons.max())

    for i, ax in enumerate(axes[0]):
        fixed_bins = np.histogram_bin_edges(metric_data[i], bins=7)
        bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2
        bar_width = (fixed_bins[1] - fixed_bins[0]) / 3

        for j, (mask, color, label) in enumerate(zip(masks, colors, labels)):
            hist_values, _ = np.histogram(metric_data[i][mask], bins=fixed_bins, density=False)
            ax.bar(bin_centers + j * bar_width - 1.5 * bar_width, hist_values, width=bar_width, color=color, alpha=0.7,
                   label=label)

        ax.set_xlabel(metric_labels[i], fontsize=ft)
        ax.set_title(f"Distribution of {metric_labels[i]}", fontsize=ft)
        ax.grid(True)

        ax.tick_params(axis='x', labelsize=ft)
        ax.tick_params(axis='y', labelsize=ft)

    axes[0, 0].set_ylabel("Count", fontsize=ft)
    axes[0, 0].legend(fontsize=ft)

    # Second row: I, J, η histograms
    labels = [">PCI >DFA <LYA", ">PCI <DFA >LYA", "<PCI >DFA <LYA", "<PCI <DFA >LYA"]
    masks = [mask_highpci_highdfa_lowlya,
             mask_highpci_lowdfa_highlya,
             mask_lowpci_highdfa_lowlya,
             mask_lowpci_lowdfa_highlya]

    # fixed_bins = np.linspace(-10, 10, 8)
    # bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2
    # bar_width = (fixed_bins[1] - fixed_bins[0]) / 5
    # fixed_bins = 7

    y_max = 0
    for i in range(3):
        fixed_bins = np.histogram_bin_edges(param_data[i], bins=6)
        for mask in masks:
            hist_values, _ = np.histogram(param_data[i][mask], bins=fixed_bins, density=False)
            y_max = max(y_max, hist_values.max())

    for i, ax in enumerate(axes[1]):
        fixed_bins = np.histogram_bin_edges(param_data[i], bins=6)
        bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2
        bar_width = (fixed_bins[1] - fixed_bins[0]) / 5

        for j, (mask, color, label) in enumerate(zip(masks, colors, labels)):
            hist_values, _ = np.histogram(param_data[i][mask], bins=fixed_bins, density=False)
            ax.bar(bin_centers + j * bar_width - 1.5 * bar_width, hist_values, width=bar_width, color=color, alpha=0.7,
                   label=label)

        ax.set_xlabel(param_labels[i], fontsize=ft)
        ax.set_title(f"Distribution of {param_labels[i]}", fontsize=ft)
        ax.grid(True)

        ax.tick_params(axis='x', labelsize=ft)
        ax.tick_params(axis='y', labelsize=ft)

    # Adding extra space at the top of the first row
    extra_space0 = 0  # Adjust this value as needed for the first three subplots
    for i in range(3):
        current_ylim = axes[0, i].get_ylim()
        axes[0, i].set_ylim(current_ylim[0], current_ylim[1] + extra_space0)

    # Adding extra space at the top of the second row
    extra_space1 = 0  # Adjust this value as needed for the last three subplots
    for i in range(3, 6):
        current_ylim = axes[1, i - 3].get_ylim()
        axes[1, i - 3].set_ylim(current_ylim[0], current_ylim[1] + extra_space1)

    axes[1, 0].set_ylabel("Count", fontsize=ft)
    axes[1, 0].legend(fontsize=ft)

    plt.tight_layout()
    # plt.show()

    # plt.savefig(f'./plots/PCIDFALYA_MSE_histogram_{whattosave}.png')
    # plt.savefig(f'/home/michiel/Documents/Repos/LiquidInterferenceLearning/csr/gpu/plots/PCIDFALYA_histogram_{whattosave}.png')


def mse_histogram(data_pcis, data_dfas, data_lyas, data_mses, whattosave):
    """
    Plots a combined figure with two subplots:
    - First subplot: Histogram of MSE for active vs resting states.
    - Second subplot: Histogram of MSE for different metric-based masks.
    """

    ft = 14

    x = data_pcis[:, 0]  # I
    y = data_pcis[:, 1]  # η
    z = data_pcis[:, 2]  # J
    mse = data_mses[:, 3]  # MSE values

    pcis = data_pcis[:, 3]
    dfas = data_dfas[:, 3]
    lyas = data_lyas[:, 3]

    mpcis = np.median(pcis[pcis < 3])
    # mpcis2 = np.median(pcis)
    mdfas = np.median(dfas)
    mlyas = np.median(lyas)

    mepcis = np.mean(pcis[pcis < 3])
    medfas = np.mean(dfas)
    melyas = np.mean(lyas)

    unique_values = np.unique(pcis[pcis < 3])

    # Calculate the arithmetic mean of the unique values
    average_unique_values = np.mean(unique_values)
    print("average_unique_values", average_unique_values)

    print(mpcis, mdfas, mlyas)
    print(mepcis, medfas, melyas)

    if whattosave == "MBR":

        param_labels = ["I", "J", "η"]
        # Masks for active vs resting states
        mask_cons = (x < 0.) & (y > 15) & (z < -5)
        mask_uncons = (x > 0.) & (y > 2.5) & (y < 15) & (z < -5)

        # Masks based on relative PCI, DFA, LYA values
        mask_highpci_highdfa_lowlya = (pcis >= .75) & (dfas >= 0.5) & (lyas <= 0.3)  # & (pcis < 3)
        mask_highpci_lowdfa_highlya = (pcis >= .75) & (dfas < 0.5) & (lyas > 0.3)  # & (pcis < 3)
        mask_lowpci_highdfa_lowlya = (pcis < .75) & (dfas >= 0.5) & (lyas <= 0.3)  # & (pcis < 3)
        mask_lowpci_lowdfa_highlya = (pcis < .75) & (dfas < 0.5) & (lyas > 0.3)  # & (pcis < 3)

        medfas = .5
        # mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        # mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        # mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        # mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        colors = ['red', 'blue', 'green', 'purple']

    elif whattosave == "LB":

        param_labels = ["rnmda", "a_ee", "V_T (mV)"]
        print(param_labels)

        mask_cons = (x <= 0.0505) & (y >= 2.55) & (z <= -45e-3)
        mask_uncons = (x > 0.0505) & (y < 2.55) & (z > -45e-3)

        # Masks based on relative PCI, DFA, LYA values
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= 0.5) & (lyas <= 0.3)  # & (pcis < 3)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < 0.5) & (lyas > 0.3)  # & (pcis < 3)
        mask_lowpci_highdfa_lowlya =  (pcis <  mpcis) & (dfas >= 0.5) & (lyas <= 0.3)  # & (pcis < 3)
        mask_lowpci_lowdfa_highlya =  (pcis <  mpcis) & (dfas < 0.5) & (lyas > 0.3)

        medfas = .5
        # mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        # mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        # mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        # mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        cmap = plt.get_cmap('viridis', 4)
        colors = [cmap(i) for i in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    # First subplot: MSE histogram for active vs resting states
    # colors = ['red', 'blue']
    # cmap = plt.get_cmap('viridis', 2)
    # colors = [cmap(i) for i in range(2)]
    labels = ["Active state", "Resting state"]
    masks = [mask_cons, mask_uncons]
    fixed_bins = np.histogram_bin_edges(mse, bins=7)
    bar_width = (fixed_bins[1] - fixed_bins[0]) / len(masks)

    for i, (mask, color, label) in enumerate(zip(masks, colors, labels)):
        hist_values, _ = np.histogram(mse[mask], bins=fixed_bins, density=False)
        axes[0].bar((fixed_bins[:-1] + fixed_bins[1:]) / 2 + i * bar_width, hist_values, width=bar_width, color=color,
                    alpha=0.7, label=label)

    axes[0].set_xlabel("MSE", fontsize=ft)
    axes[0].set_ylabel("Count", fontsize=ft)
    axes[0].set_title("(A) MSE Distribution: Active vs Resting States", fontsize=ft)
    axes[0].legend(fontsize=ft)
    axes[0].grid(True)
    axes[0].tick_params(axis='x', labelsize=ft)
    axes[0].tick_params(axis='y', labelsize=ft)

    # Second subplot: MSE histogram for metric-based masks
    # colors = ['red', 'blue', 'green', 'purple']
    # cmap = plt.get_cmap('viridis', 4)
    # colors = [cmap(i) for i in range(4)]
    labels = [">PCI >DFA <LYA", ">PCI <DFA >LYA", "<PCI >DFA <LYA", "<PCI <DFA >LYA"]
    masks = [mask_highpci_highdfa_lowlya, mask_highpci_lowdfa_highlya, mask_lowpci_highdfa_lowlya,
             mask_lowpci_lowdfa_highlya]
    bar_width = (fixed_bins[1] - fixed_bins[0]) / len(masks)

    for i, (mask, color, label) in enumerate(zip(masks, colors, labels)):
        hist_values, _ = np.histogram(mse[mask], bins=fixed_bins, density=False)
        axes[1].bar((fixed_bins[:-1] + fixed_bins[1:]) / 2 + i * bar_width, hist_values, width=bar_width, color=color,
                    alpha=0.7, label=label)

    axes[1].set_xlabel("MSE", fontsize=ft)
    axes[1].set_title("(B) MSE Distribution: Metric-Based Masks", fontsize=ft)
    axes[1].legend(fontsize=ft)
    axes[1].grid(True)
    axes[1].tick_params(axis='x', labelsize=ft)
    axes[1].tick_params(axis='y', labelsize=ft)

    plt.tight_layout()
    plt.savefig(f'/home/michiel/Documents/Repos/LiquidInterferenceLearning/csr/gpu/plots/MSE_histogram_{whattosave}.png')


def tranposed_histograms(data_pcis, data_dfas, data_lyas, data_mses, whattosave):
    """
    Plots a combined figure with two rows:
    - First row: Histograms of PCI, DFA, and LYA for active vs resting states.
    - Second row: Histograms of I, J, η distributions categorized by PCI, DFA, and LYA values.
    """

    ft = 14

    x = data_pcis[:, 0]  # I
    y = data_pcis[:, 1]  # η
    # z = np.round(data_pcis[:, 2], 1)  # J
    z = (data_pcis[:, 2])  # J
    mse = data_mses[:, 3]  # MSE values

    pcis = data_pcis[:, 3]
    dfas = data_dfas[:, 3]
    lyas = data_lyas[:, 3]

    mpcis = np.median(pcis[pcis < 3])
    # mpcis2 = np.median(pcis)
    mdfas = np.median(dfas)
    mlyas = np.median(lyas)

    mepcis = np.mean(pcis[pcis < 3])
    medfas = np.mean(dfas)
    melyas = np.mean(lyas)

    print(mpcis, mdfas, mlyas)
    print(mepcis, medfas, melyas)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey=False) # 6, 8


    if whattosave == "MBR":

        param_labels = ["I", "J", "η"]
        # Masks for active vs resting states
        mask_cons = (x < 0.) & (y > 15) & (z < -5)
        mask_uncons = (x > 0.) & (y > 2.5) & (y < 15) & (z < -5)

        # Masks based on relative PCI, DFA, LYA values
        medfas = .5
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        # mask_highpci_highdfa_lowlya = (pcis >= .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_highpci_lowdfa_highlya = (pcis >= .75) & (dfas < 0.5) & (lyas > 0.3)
        # mask_lowpci_highdfa_lowlya = (pcis < .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_lowpci_lowdfa_highlya = (pcis < .75) & (dfas < 0.5) & (lyas > 0.3)
        colors = ['red', 'blue', 'green', 'purple']

    elif whattosave == "LB":

        param_labels = ["r_nmda", "a_ee", "V_T (mV)"]
        print(param_labels)

        mask_cons =   (x <= 0.0505) & (y >= 2.55) & (z <= -45e-3)
        mask_uncons = (x >  0.0505) & (y <  2.55) & (z >  -45e-3)

        # Masks based on relative PCI, DFA, LYA values
        # mask_highpci_highdfa_lowlya = (pcis >= .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_highpci_lowdfa_highlya = (pcis >= .75) & (dfas <  0.5) & (lyas >  0.3)
        # mask_lowpci_highdfa_lowlya =  (pcis  < .75) & (dfas >= 0.5) & (lyas <= 0.3)
        # mask_lowpci_lowdfa_highlya =  (pcis  < .75) & (dfas <  0.5) & (lyas >  0.3)

        medfas = .5
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        cmap = plt.get_cmap('viridis', 4)
        colors = [cmap(i) for i in range(4)]

        # setting and rounding of the ticks
        ticks = np.linspace(z.min(), z.max(), num=7)
        axes[2, 1].set_xticks(ticks)
        axes[2, 1].set_xticklabels([f"{tick:.2f}" for tick in ticks])

    # First row: PCI, DFA, LYA histograms
    metric_labels = ["PCI", "DFA", "LYA"]
    metric_data = [pcis, dfas, lyas]

    labels = ["Active state", "Resting state"]
    masks = [mask_cons, mask_uncons]
    param_data = [x, y, z]

    y_max = 0
    for i in range(3):
        fixed_bins = np.histogram_bin_edges(metric_data[i], bins=7)
        hist_values, _ = np.histogram(metric_data[i][mask_cons], bins=fixed_bins)
        hist_values_uncons, _ = np.histogram(metric_data[i][mask_uncons], bins=fixed_bins)
        y_max = max(y_max, hist_values.max(), hist_values_uncons.max())

    for i, ax in enumerate(axes[:, 0]):
        # First column: PCI, DFA, LYA histograms
        fixed_bins = np.histogram_bin_edges(metric_data[i], bins=7)
        bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2
        bar_width = (fixed_bins[1] - fixed_bins[0]) / 3

        for j, (mask, color, label) in enumerate(zip(masks, colors, labels)):
            hist_values, _ = np.histogram(metric_data[i][mask], bins=fixed_bins, density=False)
            ax.bar(bin_centers + j * bar_width - 1.5 * bar_width, hist_values, width=bar_width, color=color, alpha=0.7,
                   label=label)

        ax.set_xlabel(metric_labels[i], fontsize=ft)
        ax.set_title(f"Distribution of {metric_labels[i]}", fontsize=ft)
        ax.grid(True)

        ax.tick_params(axis='x', labelsize=ft)
        ax.tick_params(axis='y', labelsize=ft)

    axes[0, 0].set_ylabel("Count", fontsize=ft)
    axes[0, 0].legend(fontsize=ft)

    # Define the correct labels and masks for the second column
    labels_col2 = [">PCI >DFA <LYA", ">PCI <DFA >LYA", "<PCI >DFA <LYA", "<PCI <DFA >LYA"]
    masks_col2 = [mask_highpci_highdfa_lowlya,
                  mask_highpci_lowdfa_highlya,
                  mask_lowpci_highdfa_lowlya,
                  mask_lowpci_lowdfa_highlya]

    for i, ax in enumerate(axes[:, 1]):
        # Second column: I, J, η histograms
        fixed_bins = np.histogram_bin_edges(param_data[i], bins=6)
        bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2
        bar_width = (fixed_bins[1] - fixed_bins[0]) / 5

        for j, (mask, color, label) in enumerate(zip(masks_col2, colors, labels_col2)):
            hist_values, _ = np.histogram(param_data[i][mask], bins=fixed_bins, density=False)
            ax.bar(bin_centers + j * bar_width - 1.5 * bar_width, hist_values, width=bar_width, color=color, alpha=0.7,
                   label=label)

        ax.set_xlabel(param_labels[i], fontsize=ft)
        ax.set_title(f"Distribution of {param_labels[i]}", fontsize=ft)
        ax.grid(True)

        ax.tick_params(axis='x', labelsize=ft)
        ax.tick_params(axis='y', labelsize=ft)

    axes[0, 1].set_ylabel("Count", fontsize=ft)
    axes[0, 1].legend(fontsize=ft)

    # Add space between plots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Add alphabetical capital letters to columns
    fig.text(0.1, 0.97, 'A', fontsize=24, fontweight='bold')
    fig.text(0.6, 0.97, 'B', fontsize=24, fontweight='bold')

    plt.tight_layout()
    # plt.show()

    # plt.savefig(f'./plots/PCIDFALYA_MSE_histogram_{whattosave}.png')
    plt.savefig(f'/home/michiel/Documents/Repos/LiquidInterferenceLearning/csr/gpu/plots/PCIDFALYA_histogram_T_{whattosave}.png')

def merged_mse_histogram(data_pcis, data_dfas, data_lyas, data_mses, whattosave, ax_row):
    ft = 16

    x = data_pcis[:, 0]  # I
    y = data_pcis[:, 1]  # η
    z = data_pcis[:, 2]  # J
    mse = data_mses[:, 3]  # MSE values

    pcis = data_pcis[:, 3]
    dfas = data_dfas[:, 3]
    lyas = data_lyas[:, 3]

    mpcis = np.median(pcis[pcis < 3])
    # mpcis2 = np.median(pcis)
    mdfas = np.median(dfas)
    mlyas = np.median(lyas)

    mepcis = np.mean(pcis[pcis < 3])
    medfas = np.mean(dfas)
    melyas = np.mean(lyas)

    unique_values = np.unique(pcis[pcis < 3])
    print(mpcis, mdfas, mlyas)
    print(mepcis, medfas, melyas)

    if whattosave == "MBR":

        param_labels = ["I", "J", "η"]
        # Masks for active vs resting states
        # original ranges:
        # - 20, 10, External Current I: -20, 20
        # -6, 30, Mean Synaptic weight J: 0, 30
        # -10, 10 eta
        # mask_cons = (x <= 0. ) & (y >= 15) & (z <= -5)
        # mask_uncons = (x > 0.) & (y < 15) & (z < -5)

        print("x",np.median(x))
        print("y",np.median(y))
        print("z",np.median(z))

        mask_cons = (x >= 0.) & (x <= 10) & (y >= 15) & (y < 20) & (z <= -3) & (z >= -8)
        mask_uncons = (x < 0.) & (x <= -10) & (y >= 5) & (y < 10) & (z <= -5) & (z >= -10)

        medfas = .5
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)

        # colors = ['red', 'blue', 'green', 'purple']
        ax_row[0].set_title("Active vs. Resting States", fontsize=ft)
        ax_row[1].set_title("Metric-Based Masks", fontsize=ft)

        ax_row[0].set_ylabel("Count MBR", fontsize=ft)

    elif whattosave == "LB":

        param_labels = ["rnmda", "a_ee", "V_T (mV)"]
        print(param_labels)

        print("x",np.median(x))
        print("y",np.median(y))
        print("z",np.median(z))
        # print("z",z)
        # ranges
        # 1e-3, 1e-1,  # rnmda
        # 0, 5.1,  # aee Excitatory-to-Excitatory Strength
        # -65e-3, -25-3,  # VT
        # mask_cons = (x <= 0.28) & (y >= 2.55) & (z <= -45e-3) & (z >= -60e-3)
        # mask_uncons = (x > 0.28) & (y < 2.55) & (z > -45e-3) & (z <= -30e-3)

        # mask_cons = (x <= 0.75) & (y >= 2.55) & (z <= 0) & (z >= -.38)
        # mask_uncons = (x > 0.75) & (y < 2.55) & (z > 0) & (z <= .4)

        mask_uncons = (x < 0.76) & (y > 2.55) & (z < -.1) #& (z >= -2)
        mask_cons = (x > 0.76) & (y < 2.55) & (z > -.1) #& (z <= 2)

        # mpcis = 1.
        medfas = .5
        # melyas = .1
        mask_highpci_highdfa_lowlya = (pcis >= mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_highpci_lowdfa_highlya = (pcis >= mpcis) & (dfas < medfas) & (lyas > melyas)
        mask_lowpci_highdfa_lowlya = (pcis < mpcis) & (dfas >= medfas) & (lyas <= melyas)
        mask_lowpci_lowdfa_highlya = (pcis < mpcis) & (dfas < medfas) & (lyas > melyas)


        ax_row[0].set_xlabel("MSE", fontsize=ft)
        ax_row[1].set_xlabel("MSE", fontsize=ft)

        ax_row[0].set_ylabel("Count LB", fontsize=ft)

    cmap = plt.get_cmap('viridis', 4)
    colors = [cmap(i) for i in range(4)]

    # First subplot
    labels = ["Active state", "Resting state"]
    masks = [mask_cons, mask_uncons]
    fixed_bins = np.histogram_bin_edges(mse, bins=7)
    bar_width = (fixed_bins[1] - fixed_bins[0]) / len(masks)

    print(f"Total values for {whattosave}: {len(mse)}")
    for i, (mask, color, label) in enumerate(zip(masks, colors, labels)):
        hist_values, _ = np.histogram(mse[mask], bins=fixed_bins, density=False)
        print(f"Values for {label}: {np.sum(mask)}")
        ax_row[0].bar((fixed_bins[:-1] + fixed_bins[1:]) / 2 + i * bar_width, hist_values, width=bar_width, color=color,
                    alpha=0.7, label=label)

    ax_row[0].legend(fontsize=ft)
    ax_row[0].grid(True)
    ax_row[0].tick_params(axis='x', labelsize=ft)
    ax_row[0].tick_params(axis='y', labelsize=ft)

    if whattosave == 'MBR':
        ax_row[0].text(-0.1, 1.1, 'A', transform=ax_row[0].transAxes, fontsize=ft, fontweight='bold')

    # Second subplot
    labels = [">PCI >DFA <LYA", ">PCI <DFA >LYA", "<PCI >DFA <LYA", "<PCI <DFA >LYA"]
    masks = [mask_highpci_highdfa_lowlya, mask_highpci_lowdfa_highlya, mask_lowpci_highdfa_lowlya,
             mask_lowpci_lowdfa_highlya]
    bar_width = (fixed_bins[1] - fixed_bins[0]) / len(masks)

    for i, (mask, color, label) in enumerate(zip(masks, colors, labels)):
        hist_values, _ = np.histogram(mse[mask], bins=fixed_bins, density=False)
        print(f"Values for {label}: {np.sum(mask)}")
        ax_row[1].bar((fixed_bins[:-1] + fixed_bins[1:]) / 2 + i * bar_width, hist_values, width=bar_width, color=color,
                    alpha=0.7, label=label)

    ax_row[1].legend(fontsize=ft)
    ax_row[1].grid(True)
    ax_row[1].tick_params(axis='x', labelsize=ft)
    ax_row[1].tick_params(axis='y', labelsize=ft)

    if whattosave == 'MBR':
        ax_row[1].text(-0.1, 1.1, 'B', transform=ax_row[1].transAxes, fontsize=ft, fontweight='bold')

    # Add labels "C" and "D" to the second row of subplots
    if whattosave=='LB':
        if ax_row[0].get_subplotspec().rowspan.stop == 2:
            ax_row[0].text(-0.1, 1.1, 'C', transform=ax_row[0].transAxes, fontsize=ft, fontweight='bold')
            ax_row[1].text(-0.1, 1.1, 'D', transform=ax_row[1].transAxes, fontsize=ft, fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'/home/michiel/Documents/Repos/LiquidInterferenceLearning/csr/gpu/plots/merged_MSE_histogram_MBR_LB.svg',
                format="svg")


def significance_paired(data_with_input, data_no_input):

    # Extract paired metric (e.g., MSE)
    mse_input = data_with_input[:, 3]
    mse_no_input = data_no_input[:, 3]

    # Compute difference
    mse_diff = mse_input - mse_no_input

    # Test for normality of differences
    p_normal = shapiro(mse_diff).pvalue

    if p_normal > 0.05:
        stat, pval = ttest_rel(mse_input, mse_no_input)
        test_name = "Paired t-test"
    else:
        stat, pval = wilcoxon(mse_input, mse_no_input)
        test_name = "Wilcoxon signed-rank test"

    print(f"{test_name}: p = {pval:.4e}, stat = {stat:.4f}")


def clean_pairwise_data(with_input, without_input):
    """
    Removes rows where either input array has NaN or inf in any column.

    Parameters:
        with_input (ndarray): Array of shape (N, D) with Lorenz input.
        without_input (ndarray): Array of shape (N, D) without input.

    Returns:
        cleaned_with (ndarray): Cleaned version of with_input.
        cleaned_without (ndarray): Cleaned version of without_input.
        valid_idx (ndarray): Indices of rows kept.
    """
    with_input = np.asarray(with_input)
    without_input = np.asarray(without_input)

    # Boolean masks of valid (finite) rows in both arrays
    valid_with = np.all(np.isfinite(with_input), axis=1)
    valid_without = np.all(np.isfinite(without_input), axis=1)

    # Only keep rows that are valid in BOTH arrays
    valid_idx = valid_with & valid_without

    cleaned_with = with_input[valid_idx]
    cleaned_without = without_input[valid_idx]

    return cleaned_with, cleaned_without


if __name__ == "__main__":

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    what2save = ['MBR', 'LB']


    for i in range(2):

        if what2save[i] == 'MBR':

            data_pcis = np.load("../data/all_p_MB_LorDyn0.npy")
            data_dfas = np.load("../data/all_d_MB_LorDyn0.npy")
            data_lyas = np.load("../data/all_l_MB_LorDyn0.npy")
            data_mses = np.load("../data/all_m_MB_LorDyn0.npy")

            # base line
            data_pcis_n = np.load("../data/all_p_MB_noin.npy")
            data_dfas_n = np.load("../data/all_d_MB_noin.npy")
            data_lyas_n = np.load("../data/all_l_MB_noin.npy")
            data_mses_n = np.load("../data/all_m_MB_noin.npy")

            data_pcis_c, data_pcis_n_c = clean_pairwise_data(data_pcis, data_pcis_n)
            data_dfas_c, data_dfas_n_c = clean_pairwise_data(data_dfas, data_dfas_n)
            data_lyas_c, data_lyas_n_c = clean_pairwise_data(data_lyas, data_lyas_n)
            data_mses_c, data_mses_n_c = clean_pairwise_data(data_mses, data_mses_n)

            print("sigi_pcis_MBR")
            significance_paired(data_pcis_c, data_pcis_n_c)
            print("sigi_dfas_MBR")
            significance_paired(data_dfas_c, data_dfas_n_c)
            print("sigi_lyas_MBR")
            significance_paired(data_lyas_c, data_lyas_n_c)
            print("sigi_mses_MBR")
            significance_paired(data_mses_c, data_mses_n_c)

            plot_3d_simulation_par(data_pcis, data_dfas, data_lyas, data_mses, whattosave="MBR", whattosave2=7)


        elif what2save[i] == 'LB':

            data_pcis = np.load("../data/all_p_LB_LorDyn0.npy")
            data_dfas = np.load("../data/all_d_LB_LorDyn0.npy")
            data_lyas = np.load("../data/all_l_LB_LorDyn0.npy")
            data_mses = np.load("../data/all_m_LB_LorDyn0.npy")

            # base line
            data_pcis_n = np.load("../data/all_p_LB_noin.npy")
            data_dfas_n = np.load("../data/all_d_LB_noin.npy")
            data_lyas_n = np.load("../data/all_l_LB_noin.npy")
            data_mses_n = np.load("../data/all_m_LB_noin.npy")

            data_pcis_c, data_pcis_n_c = clean_pairwise_data(data_pcis, data_pcis_n)
            data_dfas_c, data_dfas_n_c = clean_pairwise_data(data_dfas, data_dfas_n)
            data_lyas_c, data_lyas_n_c = clean_pairwise_data(data_lyas, data_lyas_n)
            data_mses_c, data_mses_n_c = clean_pairwise_data(data_mses, data_mses_n)

            print("sigi_pcis_LB")
            significance_paired(data_pcis_c, data_pcis_n_c)
            print("sigi_dfas_LB")
            significance_paired(data_dfas_c, data_dfas_n_c)
            print("sigi_lyas_LB")
            significance_paired(data_lyas_c, data_lyas_n_c)
            print("sigi_mses_LB")
            significance_paired(data_mses_c, data_mses_n_c)


            plot_3d_simulation_par(data_pcis, data_dfas, data_lyas, data_mses, whattosave="LB", whattosave2=6)

        #plot the mse of the above data in a signle graph
        merged_mse_histogram(data_pcis, data_dfas, data_lyas, data_mses, what2save[i], axes[i])

    plt.show()
