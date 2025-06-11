import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
import os


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_simulation_par_highlight(data, label, whattosave, whattosave2, highlight_condition=None):
    """
    Plots a 3D scatter plot of the selected parameters with optional highlighting.

    Parameters:
    data (numpy.ndarray): Input data of shape (nsims, 4).
    label (str): Label for the plot.
    highlight_condition (function): A function that returns a boolean mask for highlighting.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z, C = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    if highlight_condition is not None:
        mask = highlight_condition(data)
        ax.scatter(X[~mask], Y[~mask], Z[~mask], c='gray', alpha=0.2, marker='o')
        ax.scatter(X[mask], Y[mask], Z[mask], c='red', edgecolors='black', marker='o')
    else:
        sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', marker='o')
        fig.colorbar(sc, ax=ax, label='CI', fraction=0.03, pad=0.04)

    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('n')
    ax.set_title(label)

    plt.savefig(f'../plots/{label}_params_{whattosave}_ctomesize{whattosave2}.png')


def highlight_condition_pci(data):
    return (data[:, 3] >= 0.44) & (data[:, 3] <= 0.67)


def highlight_condition_dfa(data):
    return data[:, 3] > 0.5


def highlight_condition_lya(data):
    return (data[:, 3] > 0) & (data[:, 3] <= 0.1)


def plot_3d_simulation_par_highlight2(data_pcis, data_dfas, data_lyas, data_mses, whattosave, whattosave2):
    """
    Plots 3D scatter plots for PCIs, DFAs, and LYAs with highlighting.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={'projection': '3d'})

    scatter_configs = [
        (data_pcis, 'PCIs', highlight_condition_pci),
        (data_dfas, 'DFAs', highlight_condition_dfa),
        (data_lyas, 'LYAs', highlight_condition_lya),
        (data_mses, 'MSEs', None)
    ]

    for ax, (data, label, highlight) in zip(axes.flat, scatter_configs):
        X, Y, Z, C = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        if highlight is not None:
            mask = highlight(data)
            ax.scatter(X[~mask], Y[~mask], Z[~mask], c='gray', alpha=0.2, marker='o')
            ax.scatter(X[mask], Y[mask], Z[mask], c='red', edgecolors='black', marker='o')
        else:
            sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', marker='o')
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)

        ax.set_xlabel('J')
        ax.set_ylabel('I')
        ax.set_zlabel('n')
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(f'.q/plots/all_params_{whattosave}_ctomesize{whattosave2}.png')


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

    # Define subplots
    titles = ['PCIs', 'DFAs', 'LYAs', 'MSEs']
    datasets = [data_pcis, data_dfas, data_lyas, data_mses]

    for i, (data, title) in enumerate(zip(datasets, titles), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')

        # Extract X, Y, Z, and color values
        X, Y, Z, C = data[:, 0], data[:, 1], data[:, 2] * 1000, data[:, 3]

        # Scatter plot
        sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', marker='o', alpha=0.3, s=25)

        # Labels for LB
        ax.set_xlabel('rNMDA', labelpad=10, fontsize=14)
        ax.set_ylabel('a_ee', labelpad=10, fontsize=14)
        ax.set_zlabel('V_T', labelpad=10, fontsize=14)

        # Labels for MBR
        #ax.set_xlabel('I', labelpad=10)
        #ax.set_ylabel('J', labelpad=10)
        #ax.set_zlabel('eta', labelpad=10)

        ax.set_title(title, fontsize=14)

        # Add colorbar with adjustments
        cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.15, shrink=0.8, aspect=20)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save figure
    # plt.savefig(f'../plots/allCIs_params_{whattosave}_ctomesize{whattosave2}.png')

    # For HPC
    # plt.savefig(f'/p/project1/vbt/vandervlag1/liquidinterferencelearning/plots/allCIs_params_{whattosave}_ctomesize{whattosave2}.png')
    plt.savefig(f'./plots/allCIs_params_{whattosave}_ctomesize{whattosave2}.svg', format="svg")

    # plt.show()  # Uncomment to display the plot

def plot_3d_scatter(data, label, whattosave, whattosave2):
    """
    Plots a 3D scatter plot of the selected parameters.

    Parameters:
    data (numpy.ndarray): Input data of shape (nsims, 4).
    label (str): Label for the plot.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z, and color values
    X, Y, Z, C = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Scatter plot
    sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', marker='o')

    # Labels and colorbar
    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('n')
    ax.set_title(label)
    fig.colorbar(sc, ax=ax, label='CI')

    plt.savefig(f'../plots/{label}_params_{whattosave}_ctomesize{whattosave2}.png')

    # plt.savefig(f'/p/project1/vbt/vandervlag1/liquidinterferencelearning/plots/{label}s_params_{whattosave}_ctomesize{whattosave2}.png') # hpc


    # plt.show()

def plot_3d_simulation_ser(data_pcis, data_dfas, data_lyas, data_mses, whattosave, whattosave2):
    """
    Plots 3D scatter plots for PCIs, DFAs, and LYAs.

    Parameters:
    data_pcis (numpy.ndarray): Array of shape (nsims, 4).
    data_dfas (numpy.ndarray): Array of shape (nsims, 4).
    data_lyas (numpy.ndarray): Array of shape (nsims, 4).
    """
    plot_3d_scatter(data_pcis, 'PCIs', whattosave, whattosave2)
    plot_3d_scatter(data_dfas, 'DFAs', whattosave, whattosave2)
    plot_3d_scatter(data_lyas, 'LYAs', whattosave, whattosave2)
    plot_3d_scatter(data_mses, 'MSEs', whattosave, whattosave2)



def plot_surface3d(ax, data, label, z_idx):
    """
    Plots a 2D slice of the 3D data on a 3D axis.

    Parameters:
    ax (Axes3D): Matplotlib 3D axis.
    data (numpy.ndarray): Input data of shape (J, I, n, ...).
    label (str): Label for the plot.
    z_idx (int): Index along the 3rd dimension (n) to plot.
    """
    # Select the 2D slice for the z index
    data_slice = data[:, :, z_idx]  # Extract the 2D slice
    X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

    # Plot the surface
    ax.contourf(X, Y, data_slice.T, cmap='viridis')  # Transpose for proper alignment
    ax.set_xlabel(f'J ({label})')
    ax.set_ylabel(f'I ({label})')
    ax.set_title(f'{label} (n={z_idx})')

def plot_3d_simulation(data_pcis, data_dfas, data_lyas):
    """
    Plots 3D plots of the simulation data for PCIs, DFAs, and LYAs.

    Parameters:
    data_pcis (numpy.ndarray): Array of shape (J, I, n, ...).
    data_dfas (numpy.ndarray): Array of shape (J, I, n, ...).
    data_lyas (numpy.ndarray): Array of shape (J, I, n, ...).
    """
    J, I, n, _ = data_pcis.shape  # Assuming PCIs, DFAs, LYAs are 1D

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(15, 5))

    # Plot PCIs
    ax_pcis = fig.add_subplot(131, projection='3d')
    plot_surface3d(ax_pcis, data_pcis[:, :, :, 0], 'PCIs', 0)

    # Plot DFAs
    ax_dfas = fig.add_subplot(132, projection='3d')
    plot_surface3d(ax_dfas, data_dfas[:, :, :, 0], 'DFAs', 0)

    # Plot LYAs
    ax_lyas = fig.add_subplot(133, projection='3d')
    plot_surface3d(ax_lyas, data_lyas[:, :, :, 0], 'LYAs', 0)


def plot_combined_3d(data_pcis, data_dfas, data_lyas, z_idx=0):
    """
    Plots PCIs, DFAs, and LYAs in a single 3D plot with different colors.

    Parameters:
    data_pcis (numpy.ndarray): Array of shape (J, I, n, 1) for PCIs.
    data_dfas (numpy.ndarray): Array of shape (J, I, n, 1) for DFAs.
    data_lyas (numpy.ndarray): Array of shape (J, I, n, 1) for LYAs.
    z_idx (int): Index along the 3rd dimension (n) to plot.
    """
    J, I, n, _ = data_pcis.shape

    # Create a 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the meshgrid
    X, Y = np.meshgrid(np.arange(J), np.arange(I))

    # Extract 2D slices for PCIs, DFAs, and LYAs
    pcis_slice = data_pcis[:, :, z_idx, 0]  # Ensure 2D
    dfas_slice = data_dfas[:, :, z_idx, 0]  # Ensure 2D
    lyas_slice = data_lyas[:, :, z_idx, 0]  # Ensure 2D

    # Plot each dataset with a different color
    ax.plot_surface(X, Y, pcis_slice, cmap='Blues', alpha=0.7)
    ax.plot_surface(X, Y, dfas_slice, cmap='Greens', alpha=0.7)
    ax.plot_surface(X, Y, lyas_slice, cmap='Reds', alpha=0.7)

    # Set axis labels
    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('n')
    ax.set_title('Combined 3D Plot of PCIs, DFAs, and LYAs')

    # Add a legend for clarity
    ax.text2D(0.05, 0.95, "PCIs (Blue), DFAs (Green), LYAs (Red)", transform=ax.transAxes)

def plot_combined_3d_combined_ex(data_pcis, data_dfas, data_lyas, z_idx=0, whattosave=0, whattosave2=96):
    """
    Plots PCIs, DFAs, and LYAs in a single 3D plot with adjusted colorbars and transparent surfaces.

    Parameters:
    data_pcis (numpy.ndarray): Array of shape (J, I, n, 1) for PCIs.
    data_dfas (numpy.ndarray): Array of shape (J, I, n, 1) for DFAs.
    data_lyas (numpy.ndarray): Array of shape (J, I, n, 1) for LYAs.
    z_idx (int): Index along the 3rd dimension (n) to plot.
    """
    J, I, n, _ = data_pcis.shape

    # Create a 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the meshgrid
    X, Y = np.meshgrid(np.arange(J), np.arange(I))

    # Extract 2D slices for PCIs, DFAs, and LYAs
    pcis_slice = data_pcis[:, :, z_idx, 0]
    dfas_slice = data_dfas[:, :, z_idx, 0]
    lyas_slice = data_lyas[:, :, z_idx, 0]

    # Define colormaps
    cmap_pcis = 'Blues'
    cmap_dfas = 'Greens'
    cmap_lyas = 'Reds'

    # Plot PCIs
    pci_surface = ax.plot_surface(X, Y, pcis_slice, cmap=cmap_pcis, alpha=0.5)
    # Plot DFAs
    dfa_surface = ax.plot_surface(X, Y, dfas_slice, cmap=cmap_dfas, alpha=0.5)
    # Plot LYAs
    lya_surface = ax.plot_surface(X, Y, lyas_slice, cmap=cmap_lyas, alpha=0.5)

    # Set axis labels
    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('eta')
    ax.set_title('Combined 3D Plot of PCIs, DFAs, and LYAs')

    # Add colorbars with adjusted positions and labels above
    # cbar_pci = fig.colorbar(ScalarMappable(cmap=cmap_pcis), ax=ax, pad=0.15, aspect=10, shrink=0.5)
    # cbar_pci.set_label('PCIs (Blue)', labelpad=10)
    #
    # cbar_dfa = fig.colorbar(ScalarMappable(cmap=cmap_dfas), ax=ax, pad=0.2, aspect=10, shrink=0.5)
    # cbar_dfa.set_label('DFAs (Green)', labelpad=10)
    #
    # cbar_lya = fig.colorbar(ScalarMappable(cmap=cmap_lyas), ax=ax, pad=0.25, aspect=10, shrink=0.5)
    # cbar_lya.set_label('LYAs (Red)', labelpad=10)

    cbar_pci = fig.colorbar(ScalarMappable(cmap=cmap_pcis), ax=ax, pad=-0.05, shrink=0.5, aspect=10)
    cbar_pci.ax.text(0.5, 1.1, 'PCIs', ha='center', va='bottom', transform=cbar_pci.ax.transAxes)

    cbar_dfa = fig.colorbar(ScalarMappable(cmap=cmap_dfas), ax=ax, pad=-0.04, shrink=0.5, aspect=10)
    cbar_dfa.ax.text(0.5, 1.1, 'DFAs', ha='center', va='bottom', transform=cbar_dfa.ax.transAxes)

    cbar_lya = fig.colorbar(ScalarMappable(cmap=cmap_lyas), ax=ax, pad=0.05, shrink=0.5, aspect=10)
    cbar_lya.ax.text(0.5, 1.1, 'LYAs', ha='center', va='bottom', transform=cbar_lya.ax.transAxes)

    # plt.savefig(f'./plots/PCIDFASLYAs_params_{whattosave}_ctomesize{whattosave2}.png')


def plot_combined_3d_combined(data_pcis, data_dfas, data_lyas, whattosave=0, whattosave2=96):
    """
    Plots PCIs, DFAs, and LYAs in a single 3D plot with adjusted colorbars and transparent surfaces.

    Parameters:
    data_pcis (numpy.ndarray): Array of shape (nsims, parmres) for PCIs.
    data_dfas (numpy.ndarray): Array of shape (nsims, parmres) for DFAs.
    data_lyas (numpy.ndarray): Array of shape (nsims, parmres) for LYAs.
    whattosave (int): Parameter used for saving the plot (filename suffix).
    whattosave2 (int): Additional parameter for the filename suffix.
    """

    # Assuming the shape of the data arrays is (nsims, parmres)
    nsims = data_pcis.shape[0]  # Number of simulations
    parmres = data_pcis.shape[1]  # Number of parameters (or grid points)

    # Create a 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the x, y, z coordinates from the parameters
    x = data_pcis[:, 0]  # First parameter for x
    y = data_pcis[:, 1]  # Second parameter for y
    z = data_pcis[:, 2]  # Third parameter for z

    # Extract the results (you can also modify this to use different indexes for each data type)
    pcis = data_pcis[:, 3]  # Assuming the 4th column in data_pcis holds the PCI values
    dfas = data_dfas[:, 3]  # Assuming the 4th column in data_dfas holds the DFA values
    lyas = data_lyas[:, 3]  # Assuming the 4th column in data_lyas holds the LYA values

    # Define colormaps
    cmap_pcis = 'Blues'
    cmap_dfas = 'Greens'
    cmap_lyas = 'Reds'

    # Plot PCIs
    pci_surface = ax.scatter(x, y, z, c=pcis, cmap='Blues', marker='o', alpha=0.6)
    # Plot DFAs
    dfa_surface = ax.scatter(x, y, z, c=dfas, cmap='Greens', marker='^', alpha=0.6)
    # Plot LYAs
    lya_surface = ax.scatter(x, y, z, c=lyas, cmap='Reds', marker='s', alpha=0.6)

    # Set axis labels
    ax.set_xlabel('J (Mean synaptic weight)')
    ax.set_ylabel('I (Ex. current)')
    ax.set_zlabel('eta (F. rate feedback)')
    ax.set_title('Combined 3D Plot of PCIs, DFAs, and LYAs')

    # Add colorbars
    # cbar_pci = fig.colorbar(ScalarMappable(cmap='Blues'), ax=ax, pad=0.1, shrink=0.5, aspect=10)
    # cbar_pci.set_label('PCIs (Blue)', labelpad=10)
    #
    # cbar_dfa = fig.colorbar(ScalarMappable(cmap='Greens'), ax=ax, pad=0.1, shrink=0.5, aspect=10)
    # cbar_dfa.set_label('DFAs (Green)', labelpad=10)
    #
    # cbar_lya = fig.colorbar(ScalarMappable(cmap='Reds'), ax=ax, pad=0.1, shrink=0.5, aspect=10)
    # cbar_lya.set_label('LYAs (Red)', labelpad=10)

    cbar_pci = fig.colorbar(ScalarMappable(cmap=cmap_pcis), ax=ax, pad=-0.05, shrink=0.5, aspect=10)
    cbar_pci.ax.text(0.5, 1.1, 'PCIs', ha='center', va='bottom', transform=cbar_pci.ax.transAxes)

    cbar_dfa = fig.colorbar(ScalarMappable(cmap=cmap_dfas), ax=ax, pad=-0.04, shrink=0.5, aspect=10)
    cbar_dfa.ax.text(0.5, 1.1, 'DFAs', ha='center', va='bottom', transform=cbar_dfa.ax.transAxes)

    cbar_lya = fig.colorbar(ScalarMappable(cmap=cmap_lyas), ax=ax, pad=0.05, shrink=0.5, aspect=10)
    cbar_lya.ax.text(0.5, 1.1, 'LYAs', ha='center', va='bottom', transform=cbar_lya.ax.transAxes)

    print("here", os.path.dirname(os.path.abspath(__file__)))

    plt.savefig(f'../../plots/PCIDFASLYAs_params_{whattosave}_ctomesize{whattosave2}.png')
    # plt.savefig(f'/p/project1/vbt/vandervlag1/plots/PCIDFASLYAs_params_{whattosave}_ctomesize{whattosave2}.png') # hpc

import numpy as np
import matplotlib.pyplot as plt

def params_histogram(data_pcis, data_dfas, data_lyas, data_mses, whattosave):
    """
    Plots histograms of I, J, η distributions for conscious and unconscious states,
    and also adds a histogram for MSE distributions.
    """

    # Extract parameters
    x = data_pcis[:, 0]  # I
    y = data_pcis[:, 1]  # η
    z = data_pcis[:, 2]  # J
    mse = data_mses[:, 3]  # MSE values

    # Extract PCI, DFA, and Lyapunov values
    pcis = data_pcis[:, 3]
    dfas = data_dfas[:, 3]
    lyas = data_lyas[:, 3]

    # Apply filtering conditions
    # mask_conscious = (0.44 < pcis) & (pcis < 0.67) & (dfas > 0.5) & (lyas >= 0)
    # mask_unconscious = (0.12 < pcis) & (pcis < 0.31) & (dfas > 0.5) & (lyas >= 0)

    # mask_cons_hihgdfa_lowlya = (0.44 < pcis) & (pcis < 0.67) & (dfas >= 0.5) & (lyas <= 0.3)
    # mask_cons_lowdfa_highlya = (0.44 < pcis) & (pcis < 0.67) & (dfas < 0.5) & (lyas > 0.3)
    # mask_uncon_highdfa_lowlya = (0.12 < pcis) & (pcis < 0.31) & (dfas >= 0.5) & (lyas <= 0.3)
    # mask_uncon_lowdfa_highlya = (0.12 < pcis) & (pcis < 0.31) & (dfas < 0.5) & (lyas > 0.3)

    mask_cons_hihgdfa_lowlya = (0.44 < pcis)  & (pcis < 0.67) & (lyas <= 0.3)
    mask_cons_lowdfa_highlya = (0.44 < pcis)  & (pcis < 0.67) & (lyas > 0.3)
    mask_uncon_highdfa_lowlya = (0.12 < pcis) & (pcis < 0.31) & (lyas <= 0.3)
    mask_uncon_lowdfa_highlya = (0.12 < pcis) & (pcis < 0.31) & (lyas > 0.3)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)  # Added a fourth subplot for MSEs

    # Define parameter names and corresponding data
    param_labels = ["I", "J", "η", "MSE"]
    param_data = [x, y, z, mse]

    # histtype='step' for no overlap
    # density=True: Bars represent probabilities
    # density=False: Bars represent raw counts
    # fixed bins for no scaling
    fixed_bins = np.linspace(-10, 10, 11)  # 11 edges create 10 bins
    for i, ax in enumerate(axes):
        ax.hist(param_data[i][mask_cons_hihgdfa_lowlya], bins=fixed_bins, alpha=0.7, color='red',
                label="Conscious >DFA <LYA", histtype='step', density=False)
        ax.hist(param_data[i][mask_cons_lowdfa_highlya], bins=fixed_bins, alpha=0.7, color='blue',
                label="Conscious <DFA >LYA", histtype='step', density=False)
        ax.hist(param_data[i][mask_uncon_highdfa_lowlya], bins=fixed_bins, alpha=0.7, color='green',
                label="Unconscious >DFA <LYA", histtype='step', density=False)
        ax.hist(param_data[i][mask_uncon_lowdfa_highlya], bins=fixed_bins, alpha=0.7, color='purple',
                label="Unconscious <DFA >LYA", histtype='step', density=False)
        ax.set_xlabel(param_labels[i])
        ax.set_title(f"Distribution of {param_labels[i]}")
        ax.grid(True)

    axes[0].set_ylabel("Density")  # Only the first plot gets a y-label
    axes[0].legend()  # Show legend only once

    plt.savefig(f'./plots/PCIDFALYA_MSE_histogram_{whattosave}.png')




if __name__ == "__main__":

    # for top 2 functions
    # Create sample arrays of shape (10, 10, 10, 1)
    # data_pcis = np.random.rand(10, 10, 10, 1)
    # data_dfas = np.random.rand(10, 10, 10, 1)
    # data_lyas = np.random.rand(10, 10, 10, 1)
    #
    # # Plot the 3D data
    # plot_3d_simulation(data_pcis, data_dfas, data_lyas)
    # plot_combined_3d(data_pcis, data_dfas, data_lyas, z_idx=0)

    # print(data_pcis.shape)
    # plot_combined_3d_combined_ex(data_pcis, data_dfas, data_lyas, z_idx=0)


    nsims = 64
    parmres = 4  # Adjust the number of parameters as needed
    data_pcis = np.random.rand(nsims, parmres)
    data_dfas = np.random.rand(nsims, parmres)
    data_lyas = np.random.rand(nsims, parmres)
    # #
    print(data_pcis.shape)
    # plot_3d_simulation(data_pcis, data_dfas, data_lyas, whattosave=0)
    # plot_3d_simulation_par(data_pcis, data_dfas, data_lyas, data_lyas, whattosave=0, whattosave2=0)

    params_histogram(data_pcis, data_dfas, data_lyas, data_lyas, whattosave=0)


    plt.show()
