'''
Script for plotting the multiple graphs (figure 2) for the "Vast TVB parameter space exploration: A Modular Framework
for Accelerating the Multi-Scale Simulation of Human Brain Dynamics" manuscript
'''


import matplotlib.pyplot as plt
import numpy as np
import itertools
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle
import math
import os


def plot_multple(tavg_data, params, simtime, pcis, lzcs, dfas, lyas, loss, best_window_indices, fftprec, fftbest,
                 window_size, step_size, parname0, parname1, plot1st, plot2nd, n_regions, trainingloops):

    FR_exc = []
    FR_inh = []
    Ad_exc = []

    n_params = params.shape[0]
    idx =  int(np.sqrt(n_params))
    idx =  2
    labelticks_size = 6

    fig, axes = plt.subplots(idx, idx, figsize=(16, 8))

    # time_s from ms to sec
    cuttransient = 0
    # time_s = range(simtime, len(tavg_data[cuttransient:, 0, 0] * 1e-3)+simtime)
    # time_s = range(0, len(tavg_data[cuttransient:, 0, 0] * 1e-3))
    time_s = range(0, len(tavg_data[cuttransient:, 0, 0, 0]))
    print('tss', time_s)

    # for adex
    # modelmultplier = 1e3
    # for montbrio
    modelmultplier = 1
    darkyellow = (0.5, 0.5, 0)

    #noiseloop
    # for h in range(8):
    for i in range(0, idx):
        # clear_output(wait=True)

        plt.rcParams.update({'font.size': 14})

        for j in range(0, idx):

            sim_idx = i * idx + j

            # Find the best window for meansqaured error
            # trainingoffset = 50
            # best_window_start = (best_window_indices[sim_idx] * step_size) + trainingoffset
            # best_window_end = best_window_start + window_size
            #
            # # find best window for fft
            # fft_best_window_start = (fftbest[sim_idx] * step_size) + trainingoffset
            # fft_best_window_end = fft_best_window_start + window_size

            inregions = [10, 17, 30]
            outregions = [16, 21, 23]

            # inregions = [30]
            # outregions = [23]
            nodes_62 = [node for node in list(range(n_regions)) if node not in (inregions + outregions)]

            FR_exc_in = tavg_data[cuttransient:, 0, inregions, sim_idx] * modelmultplier  # from KHz to Hz; Excitatory firing rate
            FR_exc_out = tavg_data[cuttransient:, 0, outregions, sim_idx] * modelmultplier  # from KHz to Hz; Excitatory firing rate
            FR_rest = tavg_data[cuttransient:, 0, nodes_62, sim_idx] * modelmultplier  # from KHz to Hz; Excitatory firing rate

            FR_inh_in = tavg_data[cuttransient:, 1, inregions, sim_idx] * modelmultplier  # from KHz to Hz; Inhibitory firing rate
            FR_inh_out = tavg_data[cuttransient:, 1, outregions, sim_idx] * modelmultplier  # from KHz to Hz; Inhibitory firing rate

            # inoutregions = [10, 17, 30, 16, 21, 23]
            # FR_exc = tavg_data[cuttransient:, 0, inoutregions, i*idx+j] * modelmultplier  # from KHz to Hz; Excitatory firing rate
            # FR_inh = tavg_data[cuttransient:, 1, inoutregions, i*idx+j] * modelmultplier  # from KHz to Hz; Inhibitory firing rate

            # FR_exc = tavg_data[cuttransient:, 0, :, i*idx+j]  # from KHz to Hz; Excitatory firing rate
            # FR_inh = tavg_data[cuttransient:, 1, :, i*idx+j]  # from KHz to Hz; Inhibitory firing rate

            # '''plot traces'''
            # if plot1st:
            #     Li = axes[i, j].plot(time_s, FR_exc, color='darkred')  # [times, regions]
            # if plot2nd:
            #     Le = axes[i, j].plot(time_s, FR_inh, color='SteelBlue', alpha=.1)  # [times, regions]

            '''plot traces'''
            if plot1st:
                Li = axes[i, j].plot(time_s, FR_exc_in, color='darkred')  # [times, regions]
                # Li = axes[i, j].plot(time_s, FR_rest, color='darkgrey')  # [times, regions]
                # Li = axes[i, j].plot(time_s, FR_exc_out, color='green')  # [times, regions]
            if plot2nd:
                Le = axes[i, j].plot(time_s, FR_inh_in, color="darkgreen", alpha=.1)  # [times, regions]
                # Le = axes[i, j].plot(time_s, FR_inh_out, color='Purple', alpha=.1)  # [times, regions]

            # Plot vertical lines for the start and end of the best window
            # axes[i, j].axvline(x=best_window_start, color='blue', linestyle='--', label='Best Window Start')
            # axes[i, j].axvline(x=best_window_end, color='blue', linestyle='--', label='Best Window End')
            #
            # red_shades = ['#FF6666', '#800080', '#FFFF00', '#CC0000', '#990000']
            # for out in range(fft_best_window_start.shape[0]):
            #     color = red_shades[out % len(red_shades)]
            #     axes[i, j].axvline(x=fft_best_window_start[out], color=color, linestyle='--', label='Best Window Start')
            #     axes[i, j].axvline(x=fft_best_window_end[out], color=color, linestyle='--', label='Best Window End')

            axes[i, j].tick_params(axis='x', labelsize=labelticks_size)
            axes[i, j].tick_params(axis='y', labelsize=labelticks_size)

            # axes[1].plot(time_s,Ad_exc,color='goldenrod') # [times, regions]

            axes[idx-1, j].set_xlabel('Time (ms)', {"fontsize": labelticks_size})
            # axes[1].set_xlabel('Time (ms)')

            axes[i, 0].set_ylabel('F. rate (Hz)', {"fontsize": labelticks_size})
            # axes[1].set_ylabel('Firing rate (Hz)')

            # axes[0].set_ylim([-15,40])

            # axes[i, j].set_title('[g, b_e]:\n [PCI]:', loc='left', fontsize=labelticks_size)
            # axes[i, j].set_title(np.array2string(params[i*idx+j], separator=', '), fontsize=labelticks_size)

            # pcis = np.array(pcis)

            # title_part1 = '['+parname0+','+parname1+', '+parname2+']:       '
            title_part1 = '['+parname0+','+parname1+']:       '
            title_part2 = '[PCI]:        '
            title_part5 = '[LZc]:       '
            title_part7 = '[DFA]:       '
            title_part9 = '[Lya]:       '
            title_part11 = '[Loss]:       '
            title_part13 = '[FFT%]:       '
            title_part15 = '[Avg%]:       '
            # title_part3 = np.array2string(params[i * idx + j], separator=', ')
            title_part3 = np.array2string(params[i * idx + j], separator=', ') if isinstance(params[i * idx + j], (
            np.ndarray, list)) else str(params[i * idx + j])

            # title_part4 = pcis[i * idx + j]
            # array is fully outputted. therefore the 6:11 indicator. it fetches the results only
            # title_part4 = np.array2string(pcis[i * idx + j], precision=3, separator=', ')[6:10]
            # title_part6 = lzcs[i * idx + j]
            # title_part8 = dfas[i * idx + j]
            # title_part10 = lyas[i * idx + j]
            # title_part12 = loss[i * idx + j]
            # title_part14 = fftprec[i * idx + j]
            # title_part16 = np.mean(fftprec[i * idx + j])
            # title_part14 = np.array2string(fftprec[i * idx + j], precision=3, separator=', ')[6:10]
            full_title = (f'{title_part1}{title_part3}\n'
                          # f'{title_part2}{title_part4}      {title_part5}{title_part6}      '
                          # f'{title_part7}{title_part8:.2g}      {title_part9}{title_part10:.2g}\n'
                          # f'{title_part11}{title_part12:.2g} b ' f'{title_part13}{title_part14[0]:.0f} {title_part14[1]:.0f} {title_part14[2]:.0f}% r p y      '
                          # f'{title_part15}{title_part16:.2g}%'
                          )

            axes[i, j].set_title(full_title, loc='left', fontsize=labelticks_size)

    plt.tight_layout()
    plt.savefig(f'./plots/ts_params<_{n_params}_train{trainingloops}.png')
    # plt.show()


def plot_y_true(y_true):
    """
    Function to plot the y_true values over time.
    Assumes y_true shape is (timesteps, outputs, nsims).

    Args:
    y_true: numpy array of shape (timesteps, outputs, nsims)
    time_s: time array corresponding to the timesteps in seconds
    """

    timesteps, outputs, nsims = y_true.shape
    time_s = range(0, timesteps)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=outputs, figsize=(15, 5), sharex=True)

    # If there is only one output, wrap axes in a list for easier indexing
    if outputs == 1:
        axes = [axes]

    # Plot for each output and simulation
    for output_idx in range(outputs):
        for sim_idx in range(nsims):
            axes[output_idx].plot(time_s, y_true[:, output_idx, sim_idx], label=f'Sim {sim_idx + 1}')

        axes[output_idx].set_title(f'Output {output_idx + 1}')
        axes[output_idx].set_xlabel('Time (s)')
        axes[output_idx].set_ylabel('Signal')
        axes[output_idx].legend()

    plt.tight_layout()
    # plt.show()

def plot_y_true_in_sim_windows(y_true):
    """
    Function to plot y_true values over time in separate windows (subplots) for each simulation.
    Assumes y_true shape is (timesteps, outputs, nsims).

    Args:
    y_true: numpy array of shape (timesteps, outputs, nsims)
    time_s: time array corresponding to the timesteps in seconds
    """

    timesteps, outputs, nsims = y_true.shape
    time_s = range(0, timesteps)

    # Define the number of rows and columns for the grid (2D layout)
    n_cols = math.ceil(math.sqrt(nsims))  # Number of columns
    n_rows = math.ceil(nsims / n_cols)  # Number of rows based on the number of simulations

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 10), sharex=True, sharey=True)

    # Flatten axes array for easier iteration (if n_rows or n_cols > 1)
    axes = axes if nsims == 1 else axes.ravel()

    # Plot for each simulation in its corresponding subplot in the grid
    for sim_idx in range(nsims):
        i = sim_idx // n_cols  # Determine the row index
        j = sim_idx % n_cols  # Determine the column index

        for output_idx in range(outputs):
            FR_exc_in = y_true[:, output_idx, sim_idx]  # Excitatory firing rate or signal

            # Plot with specific colors and axes[i, j]
            axes[sim_idx].plot(time_s, FR_exc_in, color='darkred')  # [times, regions]
            axes[sim_idx].set_title(f'Sim {sim_idx + 1}')
            axes[sim_idx].set_ylabel('Lorenz')

        # Set x-axis label only for bottom row plots
        if i == n_rows - 1:
            axes[sim_idx].set_xlabel('Time (s)')

        axes[i].tick_params(axis='x', labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)

    # Hide any unused subplots if nsims isn't a perfect square
    for sim_idx in range(nsims, len(axes)):
        axes[sim_idx].axis('off')  # Turn off the empty plots

    plt.tight_layout()
    # plt.show()


def plot_lyas(lyas, best_indices, paramstoplot):

    # Plotting lyapunov

    ntimesteps, nsims, nregions = lyas.shape
    n_params = paramstoplot.shape[1]

    print("ntimesteps, nsims, nregions", ntimesteps, nsims, nregions)
    print("n_params", n_params)

    # y_min = np.min(lyas)
    # y_max = np.max(lyas)

    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    axes = axes.ravel()

    for idx, sim_idx in enumerate(best_indices):
    # for sim_idx in range(self.n_work_items):
      ax = axes[idx]
      # ax = axes[sim_idx]

      # Average Lyapunov exponent across regions for the current simulation
      avg_lya = np.mean(lyas[:, sim_idx, :], axis=1)

      # Scatter plot for each region at each timestep
      for region in range(nregions):
        ax.scatter(range(ntimesteps), lyas[:, sim_idx, region], s=10, alpha=0.5,)
                   # label=f'Region {region + 1}' if region < 5 else "")

      # Line plot for the average Lyapunov exponent across regions
      ax.plot(range(ntimesteps), avg_lya, color='blue', linewidth=2, label='Average')

      # ax.set_ylim(y_min, y_max)
      prmlbl = [np.round(param, 2) for param in paramstoplot[idx]]
      ax.set_title(f'{sim_idx} {prmlbl}')
      ax.set_xlabel('Trainingloops')
      ax.set_ylabel('Lyapunov Exponent')
      ax.legend(loc="upper right", fontsize='small', ncol=2)

    plt.tight_layout(pad=2.0)
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/LYA_params_{n_params}_nsims_{nsims}_train_{ntimesteps}_regions_{nregions}.png')



def plot_dfas(dfas, best_indices, paramstoplot):

    # Plotting lyapunov

    ntimesteps, nsims, nregions = dfas.shape
    n_params = paramstoplot.shape[1]

    # y_min = np.min(dfas)
    # y_max = np.max(dfas)

    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    axes = axes.ravel()

    for idx, sim_idx in enumerate(best_indices):
        # for sim_idx in range(self.n_work_items):
        ax = axes[idx]
        # ax = axes[sim_idx]

        # Average Lyapunov exponent across regions for the current simulation
        avg_lya = np.mean(dfas[:, sim_idx, :], axis=1)

        # Scatter plot for each region at each timestep
        for region in range(nregions):
            ax.scatter(range(ntimesteps), dfas[:, sim_idx, region], s=10, alpha=0.5, )
            # label=f'Region {region + 1}' if region < 5 else "")

        # Line plot for the average Lyapunov exponent across regions
        ax.plot(range(ntimesteps), avg_lya, color='blue', linewidth=2, label='Average')

        # ax.set_ylim(y_min, y_max)
        prmlbl = [np.round(param, 2) for param in paramstoplot[idx]]
        ax.set_title(f'{sim_idx} {prmlbl}')
        ax.set_xlabel('Trainingloops')
        ax.set_ylabel('Detrended Fluc Anal')
        ax.legend(loc="upper right", fontsize='small', ncol=2)

    plt.tight_layout(pad=2.0)
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/DFA_params_{n_params}_nsims_{nsims}_train_{ntimesteps}_regions_{nregions}.png')



def plot_pcis(pcis, lzcs, best_indices, paramstoplot, n_regions):

    ntimesteps, nsims = pcis.shape
    n_params = paramstoplot.shape[1]
    # print("shapes in pcislzci", ntimesteps, nsims)
    # print(pcis[:, 0])

    # y_min = np.min(pcis)
    # y_max = np.max(pcis)

    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    axes = axes.ravel()

    for idx, sim_idx in enumerate(best_indices):
        ax = axes[idx]
        ax.plot(range(ntimesteps), pcis[:, sim_idx], color='b', linewidth=1.5, label='PCI')
        # ax.plot(range(ntimesteps), np.mean(lzcs[:, sim_idx, :], axis=-1), color='r', linewidth=1.5, label='LZC Avg')

        # ax.set_ylim(y_min, y_max)
        prmlbl = [np.round(param, 2) for param in paramstoplot[idx]]
        ax.set_title(f'{sim_idx} {prmlbl}')
        ax.set_xlabel('Trainingloops')
        ax.set_ylabel('PCI')
        ax.legend(loc="upper right", fontsize='small', ncol=2)

    plt.tight_layout(pad=2.0)
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/PCI_params_{n_params}_nsims_{nsims}_train_{ntimesteps}_regions_{n_regions}.png')


def plot_MSEs(ridge_results, best_indices, paramstoplot, n_regions):
    ntimesteps, nsims = ridge_results.shape
    n_params = paramstoplot.shape[1]

    # y_min = np.min(ridge_results)*100
    # y_max = np.max(ridge_results)*100

    # Create a 4x4 subplot layout
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    axes = axes.ravel()

    for idx, sim_idx in enumerate(best_indices):
        if idx >= len(axes):
            break  # Limit to the available 4x4 grid

        ax = axes[idx]
        # Plot each output series for the current simulation over time
        # for output_idx in range(noutputs):
            # ax.plot(range(ntimesteps), ridge_results[:, sim_idx, output_idx],
            #         label=f'Output {output_idx}', linewidth=1.5)

        ax.plot(range(ntimesteps), ridge_results[:, sim_idx]*100, label=f'MSE', color='b',linewidth=1.5)

        # ax.set_ylim(y_min, y_max)
        # prmlbl = [round(param, 2) for param in paramstoplot[idx]]
        prmlbl = [np.round(param, 2) for param in paramstoplot[idx]]

        ax.set_title(f'{sim_idx} {prmlbl}')
        ax.set_xlabel('Trainingloops')
        ax.set_ylabel('MSE')
        ax.legend(loc="upper right", fontsize='small', ncol=1)

    plt.tight_layout(pad=2.0)
    # os.makedirs('../plots', exist_ok=True) #local
    os.makedirs('./plots', exist_ok=True) #hpc
    # print("curdirp", os.getcwd())
    plt.savefig(f'./plots/MSE_params_{n_params}_nsims_{nsims}_train_{ntimesteps}_regions_{n_regions}.png')


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


if __name__ == '__main__':

    def test_plotmpl():

        # tavg_file = open("../data/tavg_data_async_to_sync", 'rb')
        # tavg_file = open("../data/tavg_b10", 'rb')
        # tavg_data = pickle.load(tavg_file)
        # tavg_file.close()
        # tavg_data(simsteps, states, nodes, paramscombi)

        # tavg = np.load("../data/tavg_b10", allow_pickle=True)
        # tavg_data = np.load("../data/2ndRoundMdpi/b10_FC_noise-4_be0-120.npy", allow_pickle=True)
        # tavg_data = np.load("../data/2ndRoundMdpi/b10_FC_noise-4_be0-120.npy", allow_pickle=True)
        # tavg_data = np.load("../data/2ndRoundMdpi/b10_FC_noise-4_be0-120_g0-1.npy", allow_pickle=True)
        # tavg_data = np.load("../data/2ndRoundMdpi/b10_FC-4_0-120_g3-8.npy", allow_pickle=True)
        tavg_data = np.load("/home/michiel/Documents/Repos/gpu_zerlaut/data/2ndRoundMdpi/b10_FC-4_0-120_g3-8.npy",
                            allow_pickle=True)

        # mooi
        # tavg_data = np.load("../data/2ndRoundMdpi/b10_FC.npy", allow_pickle=True)
        print(tavg_data.shape)


    def testprettyprintparams():
        best_indices = np.array([[5, 12, 23, 47]])  # Indices of best parameters
        best_mse_tloop0 = np.random.rand(1, 64)  # Random MSE values
        best_mse_tloop1 = np.random.randint(1, 100, size=(1, 64))  # Random tloop values
        params = np.random.rand(64, 6) * 10  # Random parameters

        print(best_indices.shape)
        print(best_mse_tloop0.shape)
        print(best_mse_tloop1.shape)
        print(params.shape)

        pretty_print_best_params(best_indices, best_mse_tloop0, best_mse_tloop1, params, 'testing')




    testprettyprintparams()




