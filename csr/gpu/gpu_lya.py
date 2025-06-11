import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from scipy.integrate import solve_ivp

import os
os.environ["PATH"] += ":/usr/local/cuda-10.2/bin"
# print(os.environ["PATH"])

# import matplotlib.pyplot as plt
# import seaborn as sns
# from csr.analysis.conscious_analysis import lyapunov_ex

def computeLYA_gpu(timeseries, loggerobj, mpirank, emb_dim = 3, lag = 1, min_tsep = 2, trajectory_len = 20, tau = 1):

    '''
    Embedding dimension (emb_dim): Captures the complexity of the system's state space.
    Lag (lag): Balances temporal resolution and redundancy in the embedded points.
    Minimum temporal separation (min_tsep): Ensures meaningful distance comparisons by excluding points that are too close in time.
    Trajectory length (trajectory_len): Determines how far ahead the system's divergence is calculated, influencing the Lyapunov exponent's accuracy.
    Time normalization (tau): Scales the calculated Lyapunov exponent to the physical time scale of the system.
    '''


    mod = SourceModule("""
        #include <math.h>
        #include <float.h>

        __device__ float rowwise_euclidean(float* x, float* y, int emb_dim) {
            float sum = 0.0;
            for (int i = 0; i < emb_dim; i++) {
                sum += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return sqrtf(sum);
        }

        __global__ void compute_lyapunov(
            float* timeseries, float* emb_data, float* dists, float* div_traj,
            float* results, int nsims, int nregions, int ntimesteps,
            int emb_dim, int lag, int min_tsep, int trajectory_len, float tau
        ) {
            int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int region_idx = blockIdx.y * blockDim.y + threadIdx.y;

            if (sim_idx >= nsims || region_idx >= nregions) return;

            // Offset into the correct location in global memory
            float* data = timeseries + (sim_idx * nregions + region_idx) * ntimesteps;

            int m = ntimesteps - (emb_dim - 1) * lag;
            // if (m <= 0 || trajectory_len > m) return; 
            
            if (m <= 0 || trajectory_len > m) {
            // Store debug info in the `results` array for the corresponding simulation and region
                //printf("invalid mmmm");
                printf("%d\\n", trajectory_len);
                return;
            }

            // Perform delay embedding
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < emb_dim; j++) {
                    emb_data[(sim_idx * nregions + region_idx) * m * emb_dim + i * emb_dim + j] = data[i + j * lag];
                }
            }

            //if ((i + k) < m && (min_idx + k) < m && min_dist > 0.0f) {
            //    int idx2 = (sim_idx * nregions + region_idx) * m * m + (i + k) * m + (min_idx + k);
            //    sum_log_dist += logf(dists[idx2]);
            //    count++;
            //}

            // Calculate pairwise distances with min_tsep exclusion
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    if (abs(i - j) <= min_tsep) {
                        dists[(sim_idx * nregions + region_idx) * m * m + i * m + j] = FLT_MAX;
                    } else {
                        float dist = rowwise_euclidean(
                            &emb_data[(sim_idx * nregions + region_idx) * m * emb_dim + i * emb_dim],
                            &emb_data[(sim_idx * nregions + region_idx) * m * emb_dim + j * emb_dim],
                            emb_dim
                        );
                        // Clamp distance to avoid logf(0) or NaNs
                        dists[(sim_idx * nregions + region_idx) * m * m + i * m + j] = fmaxf(dist, 1e-10);
                    }
                }
            }

            // Calculate nearest neighbor divergence trajectory
            int ntraj = m - trajectory_len + 1;
            for (int k = 0; k < trajectory_len; k++) {
                float sum_log_dist = 0.0;
                int count = 0;
                for (int i = 0; i < ntraj; i++) {
                    float min_dist = FLT_MAX;
                    int min_idx = -1;
                    for (int j = 0; j < ntraj; j++) {
                        if (dists[(sim_idx * nregions + region_idx) * m * m + i * m + j] < min_dist) {
                            min_dist = dists[(sim_idx * nregions + region_idx) * m * m + i * m + j];
                            min_idx = j;
                        }
                    }
                    if (min_dist > 0.0f) {
                        sum_log_dist += logf(dists[(sim_idx * nregions + region_idx) * m * m + (i + k) * m + (min_idx + k)]);
                        count++;
                    }
                }
                div_traj[(sim_idx * nregions + region_idx) * trajectory_len + k] = count > 0 ? sum_log_dist / count : -FLT_MAX;
            }

            // Perform linear regression to calculate the Lyapunov exponent
            float sum_k = 0.0, sum_div = 0.0, sum_kdiv = 0.0, sum_k2 = 0.0;
            int valid_count = 0;
          for (int k = 0; k < trajectory_len; k++) {
                float div_value = div_traj[(sim_idx * nregions + region_idx) * trajectory_len + k];
                if (isfinite(div_value)) {
                    sum_k += k;
                    sum_div += div_value;
                    sum_kdiv += k * div_value;
                    sum_k2 += k * k;
                    valid_count++;
                }
            }

            float slope = (valid_count * sum_kdiv - sum_k * sum_div) /
                          (valid_count * sum_k2 - sum_k * sum_k);
            results[sim_idx * nregions + region_idx] = slope / tau;
        }
    """)
    nsims, nregions, ntimesteps = timeseries.shape
    results = np.zeros((nsims, nregions), dtype=np.float32)

    # Allocate device memory
    timeseries_gpu = cuda.mem_alloc(timeseries.nbytes)
    emb_data_gpu = cuda.mem_alloc(nsims * nregions * (ntimesteps - (emb_dim - 1) * lag) * emb_dim * 4)
    dists_gpu = cuda.mem_alloc(nsims * nregions * (ntimesteps - (emb_dim - 1) * lag) * (ntimesteps - (emb_dim - 1) * lag) * 4)
    div_traj_gpu = cuda.mem_alloc(nsims * nregions * trajectory_len * 4)
    results_gpu = cuda.mem_alloc(results.nbytes)


    # Copy data to the GPU
    cuda.memcpy_htod(timeseries_gpu, timeseries)

    # Kernel launch configuration
    block_dim = (16, 16, 1)
    grid_dim = (int(np.ceil(nsims / block_dim[0])), int(np.ceil(nregions / block_dim[1])))

    # Get the kernel function
    compute_lyapunov = mod.get_function("compute_lyapunov")

    # Launch the kernel
    compute_lyapunov(
        timeseries_gpu, emb_data_gpu, dists_gpu, div_traj_gpu,
        results_gpu, np.int32(nsims), np.int32(nregions),
        np.int32(ntimesteps), np.int32(emb_dim), np.int32(lag),
        np.int32(min_tsep), np.int32(trajectory_len), np.float32(tau),
        block=block_dim, grid=grid_dim
    )

    if mpirank == 0:
        print("\n")
        loggerobj.info("LYA emb_dim %s", emb_dim)
        loggerobj.info("LYA lag %s", lag)
        loggerobj.info("LYA traj length %s", trajectory_len)
        loggerobj.info("LYA min tstep length %s", min_tsep)
        loggerobj.info("LYA tau %s", tau)
        loggerobj.info("LYA data shape %s", timeseries.shape)
        loggerobj.info("LYA kernel grid %s", grid_dim)
        print("\n")



    # Copy results back to host
    cuda.memcpy_dtoh(results, results_gpu)

    # Free GPU memory
    timeseries_gpu.free()
    emb_data_gpu.free()
    dists_gpu.free()
    div_traj_gpu.free()
    results_gpu.free()

    return results


def gpu_compute_causation_matrix(timeseries):

    import pycuda.autoinit
    import pycuda.driver as cuda
    import numpy as np
    from pycuda.compiler import SourceModule

    kernel_code = """
    __global__ void compute_causation_matrix(float *timeseries, float *causation_matrices, int nsims, int nregions, int ntimesteps) {
        int region_i = blockIdx.x * blockDim.x + threadIdx.x;  // Global index for region i
        int region_j = blockIdx.y * blockDim.y + threadIdx.y;  // Global index for region j
        int sim_idx = blockIdx.z;                             // Simulation index across grid in z dimension

        if (region_i < nregions && region_j < nregions && sim_idx < nsims) {
            float mean_i = 0.0, mean_j = 0.0, var_i = 0.0, var_j = 0.0, cov_ij = 0.0;

            // Calculate means for regions i and j
            for (int t = 0; t < ntimesteps; t++) {
                float value_i = timeseries[(sim_idx * nregions + region_i) * ntimesteps + t];
                float value_j = timeseries[(sim_idx * nregions + region_j) * ntimesteps + t];
                mean_i += value_i;
                mean_j += value_j;
            }
            mean_i /= ntimesteps;
            mean_j /= ntimesteps;

            // Calculate variances and covariance
            for (int t = 0; t < ntimesteps; t++) {
                float diff_i = timeseries[(sim_idx * nregions + region_i) * ntimesteps + t] - mean_i;
                float diff_j = timeseries[(sim_idx * nregions + region_j) * ntimesteps + t] - mean_j;
                var_i += diff_i * diff_i;
                var_j += diff_j * diff_j;
                cov_ij += diff_i * diff_j;
            }

            // Normalize and store the correlation in the causation matrix
            float corr = (sqrtf(var_i) > 0.0f && sqrtf(var_j) > 0.0f) ? cov_ij / (sqrtf(var_i) * sqrtf(var_j)) : 0.0f;
            causation_matrices[(sim_idx * nregions + region_i) * nregions + region_j] = corr;
        }
    }
    """
    # Compile kernel
    mod = SourceModule(kernel_code)
    compute_causation_matrix = mod.get_function("compute_causation_matrix")

    nsims, nregions, ntimesteps = timeseries.shape

    # Allocate memory on the GPU for the timeseries data and causation matrices
    timeseries_flat = timeseries.astype(np.float32).ravel()
    timeseries_gpu = cuda.mem_alloc(timeseries_flat.nbytes)
    cuda.memcpy_htod(timeseries_gpu, timeseries_flat)

    causation_matrices_gpu = cuda.mem_alloc(nsims * nregions * nregions * np.float32().nbytes)

    block = (32, 32, 1)  # Use a 32x32 block for region pairs
    grid_x = (nregions + block[0] - 1) // block[0]  # Grid size in x based on regions
    grid_y = (nregions + block[1] - 1) // block[1]  # Grid size in y based on regions
    grid = (grid_x, grid_y, nsims)

    # Run kernel with a 2D grid
    compute_causation_matrix(
        timeseries_gpu, causation_matrices_gpu,
        np.int32(nsims), np.int32(nregions), np.int32(ntimesteps),
        # block=(block_size, 1, 1), grid=(grid_x, grid_y)
        block=block, grid=grid
    )

    # Copy results from GPU to CPU
    causation_matrices = np.empty((nsims, nregions, nregions), dtype=np.float32)
    cuda.memcpy_dtoh(causation_matrices, causation_matrices_gpu)

    # Free GPU memory
    timeseries_gpu.free()
    causation_matrices_gpu.free()

    return causation_matrices


def score_causation_matrix(causation_matrix, lyapunov_exponents, nsims, threshold=0.1):
    """
    Score the causation matrix based on the number of regions near criticality and
    the average strength of causation relationships.
    """
    scores = []
    for i in range(nsims):
        # Count regions with Lyapunov exponents near zero
        # near_critical_count = np.sum(np.abs(lyapunov_exponents[i]) < threshold)
        near_critical_count = np.sum((lyapunov_exponents[i]) < threshold)

        # Compute average strength of causation relationships
        # Assuming causation_matrix is a square matrix
        avg_causation_strength = np.mean(causation_matrix[i])

        # Score: you can adjust the weighting based on your needs
        score = near_critical_count + avg_causation_strength

        scores.append((i, score))

    # Step 4: Sort scores to find the best causation matrices
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Display the top scores
    print("Top causation matrices based on score:")
    for sim_index, score in sorted_scores[:5]:  # Show top 5
        print(f"Simulation {sim_index}: Score = {score}")

    # Plot in a 2x2 grid
    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # fig.suptitle("Top 4 Causation Matrices by Score")

    top_scores = sorted_scores[:4]
    # print('top_scores', top_scores
    top_matrices = [causation_matrices[topscore[0]] for topscore in top_scores]
    top_lya_mat = [lyapunov_exponents[topscore[0]] for topscore in top_scores]

    top_lya_mat = np.array(top_lya_mat)
    print('top_lya_mat', top_lya_mat.shape)

    # for idx, ax in enumerate(axes.flatten()):
    #     im = ax.imshow(top_matrices[idx], cmap="coolwarm", vmin=-.03, vmax=.03)
    #     ax.set_title(f"Score: {top_scores[idx][1]:.2f}")
    #     ax.set_xlabel("Region")
    #     ax.set_ylabel("Region")

    # for idx, ax in enumerate(axes.flatten()):
    #     im = ax.imshow(top_lya_mat[idx], cmap="coolwarm", vmin=-.03, vmax=.03)
    #     ax.set_title(f"Score: {top_scores[idx][1]:.2f}")
    #     ax.set_xlabel("Region")
    #     ax.set_ylabel("Region")

    # fig, axes = plt.subplots(1, nsims, figsize=(20, 5), sharey=True)
    # for sim_idx in range(nsims):

    # for idx, ax in enumerate(axes.flatten()):
    #     # Reshape to square matrix if each vector represents a 2D structure (e.g., (62,62))
    #     if top_lya_mat.shape[1] == 62 * 62:
    #         matrix = top_lya_mat[idx].reshape(62, 62)
    #     else:
    #         # If each is 1D, we can visualize it as a single-row heatmap or a single column
    #         matrix = top_lya_mat[idx].reshape(1, 62)
    #
    #     im = ax.imshow(matrix, cmap="coolwarm", aspect='auto', vmin=-1, vmax=1)
    #     ax.set_title(f"Matrix {idx + 1}")
    #     ax.set_xlabel("Region Index")
    #     ax.set_ylabel("Region Index" if matrix.shape[0] > 1 else "Single Row")
    #
    # fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal")
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

    return


if __name__ == "__main__":

    def gpu_device_info():
        '''
        Get GPU device information
        TODO use this information to give user GPU setting suggestions
        '''
        dev = cuda.Device(0)
        print('\n')
        print('GPU = %s', dev.name())
        print('TOTAL AVAIL MEMORY: %d MiB', dev.total_memory() / 1024 / 1024)

        # get device information
        att = {'MAX_THREADS_PER_BLOCK': [],
               'MAX_BLOCK_DIM_X': [],
               'MAX_BLOCK_DIM_Y': [],
               'MAX_BLOCK_DIM_Z': [],
               'MAX_GRID_DIM_X': [],
               'MAX_GRID_DIM_Y': [],
               'MAX_GRID_DIM_Z': [],
               'TOTAL_CONSTANT_MEMORY': [],
               'WARP_SIZE': [],
               # 'MAX_PITCH': [],
               'CLOCK_RATE': [],
               'TEXTURE_ALIGNMENT': [],
               # 'GPU_OVERLAP': [],
               'MULTIPROCESSOR_COUNT': [],
               'SHARED_MEMORY_PER_BLOCK': [],
               'MAX_SHARED_MEMORY_PER_BLOCK': [],
               'REGISTERS_PER_BLOCK': [],
               'MAX_REGISTERS_PER_BLOCK': []}

        for key in att:
            getstring = 'cuda.device_attribute.' + key
            # att[key].append(eval(getstring))
            print(key + ': %s', dev.get_attribute(eval(getstring)))

    def random_test_data(nsims, nregions, ntimesteps):

        # Generate random test data
        timeseries = np.random.rand(nsims, nregions, ntimesteps).astype(np.float32)
        return timeseries

    def generate_lorenz_time_series(nsim, nregions, ntimesteps, dt=0.01):
        """Generate a chaotic time series using the Lorenz system."""
        # Lorenz system parameters for chaotic behavior
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        # Define the Lorenz system equations
        def lorenz(t, state):
            x, y, z = state
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - beta * z
            return [dxdt, dydt, dzdt]

        # Initialize time series array
        timeseries = np.zeros((nsim, nregions, ntimesteps), dtype=np.float32)

        # Generate chaotic data for each region and simulation
        for sim in range(nsim):
            for region in range(nregions):
                # Random initial conditions to enhance chaotic variation
                init_state = np.random.uniform(-15, 15, size=3)
                t_span = (0, dt * ntimesteps)
                t_eval = np.linspace(t_span[0], t_span[1], ntimesteps)

                # Integrate Lorenz equations
                sol = solve_ivp(lorenz, t_span, init_state, t_eval=t_eval, method='RK45')

                # Use one component of the Lorenz attractor as the time series
                timeseries[sim, region, :] = sol.y[0]  # Only x component for simplicity

        return timeseries


    def familiar_caus_timeseries(nsims, nregions, ntimesteps):
        """
        Generate test timeseries data with known correlations.
        Each region pair will have a preset correlation coefficient (0.1, 0.2, 0.3, etc.).

        Parameters:
        - nsims: number of simulations
        - nregions: number of regions
        - ntimesteps: number of time steps in each timeseries

        Returns:
        - timeseries: numpy.ndarray with shape (nsims, nregions, ntimesteps)
        """
        timeseries = np.zeros((nsims, nregions, ntimesteps), dtype=np.float32)
        correlation_values = np.linspace(0.1, 1.0, nregions // 2)  # preset correlation values for each pair

        for sim in range(nsims):
            for pair_idx, correlation in enumerate(correlation_values):
                # Create a base signal for each pair of regions
                base_signal = np.random.randn(ntimesteps)
                region1 = 2 * pair_idx
                region2 = region1 + 1

                # Generate correlated signals for the pair of regions
                timeseries[sim, region1] = base_signal
                timeseries[sim, region2] = correlation * base_signal + np.sqrt(1 - correlation ** 2) * np.random.randn(
                    ntimesteps)

        return timeseries


    def generate_non_chaotic_timeseries(nsims, nregions, ntimesteps, frequency=0.05, noise_level=0.01):
        timeseries = np.zeros((nsims, nregions, ntimesteps))
        t = np.arange(ntimesteps)
        sine_wave = np.sin(2 * np.pi * frequency * t)

        for sim in range(nsims):
            for region in range(nregions):
                timeseries[sim, region] = sine_wave + noise_level * np.random.normal(0, 1, ntimesteps)
        return timeseries

    def test_Lyapunov():

        # Parameters for test
        nsims = 2200
        nregions = 96
        ntimesteps = 10
        emb_dim = 6
        lag = 1
        min_tsep = 2
        trajectory_len = 20
        tau = .2

        timeseries = random_test_data(nsims, nregions, ntimesteps)
        # timeseries = generate_lorenz_time_series(nsims, nregions, ntimesteps)
        # timeseries=generate_non_chaotic_timeseries(nsims, nregions, ntimesteps, frequency=0.05)
        print("LYA_ts.shape", timeseries.shape)
        print("LYA_max", timeseries.max())
        print("LYA_min", timeseries.min())

        results = computeLYA_gpu(timeseries, emb_dim , lag, min_tsep, trajectory_len, tau)

        # Print results
        print("Computed Lyapunov Exponents (per simulation and region):")
        print(results)
        print(results.shape)

        return results

    def test_causiation_matrix():

        # Timeseries_data has shape (nsims, nregions, ntimesteps)

        nsims = 36
        nregions = 62
        ntimesteps = 250
        # timeseries = random_test_data(nsims, nregions, ntimesteps)
        timeseries = familiar_caus_timeseries(nsims, nregions, ntimesteps)
        causation_matrices = gpu_compute_causation_matrix(timeseries)
        print(causation_matrices.shape)
        # Extract and print causation matrix for the first simulation to validate results
        causation_matrix = causation_matrices[0]
        print("Causation Matrix (Simulation 0):")
        print(causation_matrix)

        print("\nExpected correlations for pairs (approx):")
        for i, expected_corr in enumerate(np.linspace(0.1, 1.0, nregions // 2)):
            region1 = 2 * i
            region2 = region1 + 1
            print(
                f"Regions {region1} & {region2}: Expected ~{expected_corr:.1f}, Computed {causation_matrix[region1, region2]:.2f}")

        return causation_matrices

    # gpu_device_info()

    lyapunov_exponents = test_Lyapunov()
    # causation_matrices = test_causiation_matrix()
    #
    # score_causation_matrix(causation_matrices, lyapunov_exponents, nsims=36, threshold=0.1)

