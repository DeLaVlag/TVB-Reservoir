import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def computeDFA_gpu(data, loggerobj, mpirank):

    # Define the CUDA kernel for DFA computation
    mod = SourceModule("""
    __global__ void computeDFA(float *data, int num_steps, int num_regions, int *window_sizes, 
                               int num_window_sizes, int nsims, float *alphas) {
        int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;  // parallelize over simulations
        int region_idx = blockIdx.y * blockDim.y + threadIdx.y;  // parallelize over regions
    
        if (sim_idx >= nsims || region_idx >= num_regions) return;
    
        for (int window_idx = 0; window_idx < num_window_sizes; ++window_idx) {
            int window_size = window_sizes[window_idx];
            int num_windows = num_steps / window_size;
    
            float fluctuation = 0.0f;
    
            // Loop over the windows for this region and window size
            for (int i = 0; i < num_windows; ++i) {
                int start_idx = i * window_size;
                int end_idx = start_idx + window_size;
    
                // Linear fit (trend) using the least squares method (order 1 polynomial)
                float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
                for (int j = start_idx; j < end_idx; ++j) {
                    float x = j - start_idx;
                    float y = data[(sim_idx * num_regions * num_steps) + (j * num_regions) + region_idx];  // Access data by simulation and region index
                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_x2 += x * x;
                }
    
                float N = (float)window_size;
                float slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x * sum_x);
                float intercept = (sum_y - slope * sum_x) / N;
    
                // Calculate the fluctuation (root mean square deviation)
                float rms = 0.0f;
                for (int j = start_idx; j < end_idx; ++j) {
                    float x = j - start_idx;
                    float trend = slope * x + intercept;
                    float y = data[(sim_idx * num_regions * num_steps) + (j * num_regions) + region_idx];
                    rms += (y - trend) * (y - trend);
                }
    
                fluctuation += sqrtf(rms / window_size);
            }
    
            // Compute the alpha (slope) of log-log plot of window sizes and fluctuations
            float log_window_size = logf((float)window_size);
            float log_fluctuation = logf(fluctuation);
    
            alphas[(sim_idx * num_regions * num_window_sizes) + (region_idx * num_window_sizes) + window_idx] = log_fluctuation / log_window_size;
        }
    }
    """)

    # Get the computeDFA kernel function
    computeDFA_kernel = mod.get_function("computeDFA")

    # Host function to invoke the PyCUDA kernel

    min_window_size = 10  # Minimum window size
    max_window_size = int(data.shape[2]/2)  # Maximum window size (half of the time series length)
    # min_window_size = 4  # Minimum window size
    # max_window_size = 100 # Maximum window size (half of the time series length)
    num_windows = 10
    window_sizes = np.unique(np.logspace(np.log10(min_window_size), np.log10(max_window_size), num=num_windows, dtype=np.int32))

    # window_sizes = np.array([4, 8, 16, 32, 64], dtype=np.int32)
    # window_sizes = np.array([ 4,  7, 15, 31, 62], dtype=np.int32)

    nsims, num_regions, num_steps = data.shape
    num_window_sizes = len(window_sizes)

    # Allocate memory for the results
    alphas = np.zeros((nsims, num_regions, num_window_sizes), dtype=np.float32)

    # Flatten the data for the GPU
    data_flat = data.ravel().astype(np.float32)

    # Allocate device memory
    data_gpu = cuda.mem_alloc(data_flat.nbytes)
    window_sizes_gpu = cuda.mem_alloc(window_sizes.nbytes)
    alphas_gpu = cuda.mem_alloc(alphas.nbytes)

    # Copy data to the device (GPU)
    cuda.memcpy_htod(data_gpu, data_flat)
    cuda.memcpy_htod(window_sizes_gpu, window_sizes)

    # Define thread and block size
    block_size = (8, 8, 1)  # 8 threads for simulations and 8 for regions
    grid_size = ((nsims + block_size[0] - 1) // block_size[0], (num_regions + block_size[1] - 1) // block_size[1])

    if mpirank == 0:
        print("\n")
        loggerobj.info('DFA Window_sizesdata %s', window_sizes)
        loggerobj.info("DFA data shape %s", data.shape)
        loggerobj.info("DFA kenrlel grid_size %s", grid_size)
        print("\n")

    # Launch the kernel
    computeDFA_kernel(data_gpu, np.int32(num_steps), np.int32(num_regions),
                      window_sizes_gpu, np.int32(num_window_sizes), np.int32(nsims), alphas_gpu,
                      block=block_size, grid=grid_size)

    # Copy the result back to host
    cuda.memcpy_dtoh(alphas, alphas_gpu)

    return np.median(alphas, axis=2)  # Take median across the window sizes

# Example usage
if __name__ == "__main__":

    def generate_test_data(num_sims, num_regions, num_steps):
        # Generate data for each target DFA exponent
        data_close_to_1 = np.cumsum(np.random.randn(num_sims, num_regions, num_steps),
                                    axis=2)  # Random walk (Brownian motion)
        data_close_to_0 = np.random.randn(num_sims, num_regions, num_steps)  # White noise
        data_close_to_minus_1 = np.array([
            np.sin(2 * np.pi * (i + 1) * np.arange(num_steps) / num_steps) for i in range(num_sims * num_regions)
        ]).reshape(num_sims, num_regions, num_steps)  # Sine waves with different frequencies

        return data_close_to_1, data_close_to_0, data_close_to_minus_1


    # Test the computeDFA_gpu function with generated data
    def test_computeDFA():
        num_sims, num_regions, num_steps = 32, 62, 250  # Define the shape of the test data

        # Generate test data
        data_close_to_1, data_close_to_0, data_close_to_minus_1 = generate_test_data(num_sims, num_regions, num_steps)

        # Compute DFA for each test case
        alpha_1 = computeDFA_gpu(data_close_to_1)
        alpha_0 = computeDFA_gpu(data_close_to_0)
        alpha_minus_1 = computeDFA_gpu(data_close_to_minus_1)

        # Print the results for inspection

        # Random Walk (Brownian Motion, H ≈ 0.5)
        # A random walk also has a Hurst exponent near 0.5.
        # However, because each step is based on a cumulative sum of white noise,
        # it has some apparent "momentum."
        print("DFA exponents close to 1 (random walk):", alpha_1)
        # White noise is a purely random process with no memory or persistence;
        # it has a Hurst exponent close to 0.5.
        print("DFA exponents close to 0 (white noise):", alpha_0)
        # Replaced by generate_anti_correlated_signal function
        print("DFA exponents close to -1 (sine wave):", alpha_minus_1)


    def generate_anti_correlated_signal(num_sims, num_regions, num_steps, theta=0.5, mu=0, sigma=0.5):
        """
        Generate anti-correlated signals using an Ornstein-Uhlenbeck process for DFA testing.

        Parameters:
        - num_sims: Number of simulations
        - num_regions: Number of regions
        - num_steps: Length of time series
        - theta: Speed of mean reversion
        - mu: Long-term mean
        - sigma: Volatility (controls amplitude of fluctuations)

        Returns:
        - data: Anti-correlated signal array with shape (num_sims, num_regions, num_steps)
        """
        dt = 1  # Discrete time step
        data = np.zeros((num_sims, num_regions, num_steps), dtype=np.float32)

        for sim in range(num_sims):
            for region in range(num_regions):
                x = np.zeros(num_steps, dtype=np.float32)
                x[0] = np.random.normal(mu, sigma)  # Start with random initial value

                # Generate the OU process
                for t in range(1, num_steps):
                    x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()

                data[sim, region, :] = x

        return data


    # Run the test
    # test_computeDFA()

    num_sims = 64
    num_regions = 68
    num_steps = 250
    anti_correlated_data = generate_anti_correlated_signal(num_sims, num_regions, num_steps, theta=1)
    # print(anti_correlated_data)

    # nsims = 16
    # num_steps = 250
    # num_regions = 68
    # window_sizes = np.array([4, 8, 16, 32, 64], dtype=np.int32)
    #
    # # Randomly generated data for testing (shape: [nsims, num_regions, num_steps])
    # data = np.random.randn(nsims, num_regions, num_steps).astype(np.float32)
    #
    # # Call the PyCUDA DFA computation
    median_alphas = computeDFA_gpu(anti_correlated_data)
    print("Median Alphas:", median_alphas)

