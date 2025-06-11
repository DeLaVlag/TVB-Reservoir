#define MAX_BITS 256

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

//const int stonset = 0; //2500;

//const int nworkitems = 16;
//const int ntrials = 2; // number of simulations/realisations to analyse for one PCI value
//const int nsources = 3;
//const int NBINSARR = 4;

__device__ void binarise_signals(float *signal_m, int t_stim, int nshuffles, int percentile, int nworkitems,
    int ntrials, int nsources, int nbins, int stonset, int *signal_binary, int idx, float *means_prestim, float *signal_centre,
    float *std_prestim, float *signal_centre_norm, float *max_absval_surrogates, float *signal_prestim_shuffle,
    float *shuffle_avg) {

    // Calculate means and signal centre in a single loop
    for (int trial = 0; trial < ntrials; trial++) {
        for (int source = 0; source < nsources; source++) {
            float sum = 0.0;
            for (int t = stonset; t < t_stim; t++) {
                sum += signal_m[(trial * nsources * nbins) + (source * nbins) + t];
//                printf("signm %f ind %d trial %d source %d t %d\n",
//                    signal_m[(trial * nsources * nbins) + (source * nbins) + t],
//                    (trial * nsources * nbins) + (source * nbins) + t, trial, source, t);
            }
            means_prestim[(trial * nsources) + source] = sum / (t_stim - stonset);

            for (int t = 0; t < nbins; t++) {
                float mean_val = means_prestim[trial * nsources + source];
                float signal_val = signal_m[trial * nsources * nbins + source * nbins + t];
                signal_centre[trial * nsources * nbins + source * nbins + t] = signal_val / mean_val - 1;
//                    printf("signm %f ind %d trial %d source %d t %d\n",
//                    signal_m[(trial * nsources * nbins) + (source * nbins) + t],
//                    (trial * nsources * nbins) + (source * nbins) + t, trial, source, t);
            }
//            __syncthreads();
        }
//        __syncthreads();
    }

    // Calculate standard deviation and normalize signal centre in a single loop
    for (int trial = 0; trial < ntrials; trial++) {
        for (int source = 0; source < nsources; source++) {
            float sum_sq = 0.0;
            for (int t = 0; t < t_stim; t++) {
                float centered_val = signal_centre[trial * nsources * nbins + source * nbins + t];
                sum_sq += centered_val * centered_val;
            }
            std_prestim[trial * nsources + source] = sqrt(sum_sq / t_stim);

            float std_val = std_prestim[trial * nsources + source];
            for (int t = 0; t < nbins; t++) {
                float centered_val = signal_centre[(trial * nsources * nbins) + (source * nbins) + t];
                signal_centre_norm[trial * nsources * nbins + source * nbins + t] = centered_val / std_val;
            }
//            __syncthreads();
        }
//        __syncthreads();
    }

    curandState state;
    curand_init(1234, idx, 0, &state);
//    extern __shared__ float shared_mem[];
//    float *signal_prestim_shufflex = shared_mem;

    // Bootstrapping: Shuffle prestim signal in time, intra-trial
    for (int trial = 0; trial < ntrials; trial++) {
        for (int source = 0; source < nsources; source++) {
            for (int t = stonset; t < t_stim; t++) {
                signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + t] =
                    signal_centre_norm[(trial * nsources * nbins) + (source * nbins) + t];
            }
//            __syncthreads();
        }
//        __syncthreads();
    }

    for (int i_shuffle = 0; i_shuffle < nshuffles; i_shuffle++) {
        for (int trial = 0; trial < ntrials; trial++) {
            for (int source = 0; source < nsources; source++) {
                // Shuffle the pre-stimulus signal
                for (int t = stonset; t < t_stim; t++) {
                    int swap_idx = stonset + curand(&state) % (t_stim - stonset);
                    float temp = signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + t];
                    signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + t] =
                        signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + swap_idx];
                    signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + swap_idx] = temp;

//                    atomicAdd(&signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + t],
//                        signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + swap_idx]);
//                    atomicAdd(&signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + swap_idx], temp);
                }
//                __syncthreads();
            }
//            __syncthreads();
        }

        // Calculate average over trials and maximum absolute value in a single loop
        float max_val = 0.0;
        for (int source = 0; source < nsources; source++) {
            for (int t = stonset; t < t_stim; t++) {
                float sum = 0.0;
                for (int trial = 0; trial < ntrials; trial++) {
                    sum += signal_prestim_shuffle[(trial * nsources * t_stim) + (source * t_stim) + t];
                }
                shuffle_avg[source * t_stim + t] = sum / ntrials;
//                atomicAdd(&shuffle_avg[source * t_stim + t], sum / ntrials);

                float val = shuffle_avg[source * t_stim + t];
                if (fabs(val) > max_val) {
                    max_val = fabs(val);
                }
            }
//            __syncthreads();
        }
//        __syncthreads();
        // no raceconditions observed yet but could be
        max_absval_surrogates[i_shuffle] = max_val;
//        atomicAdd(&max_absval_surrogates[i_shuffle], max_val);
    }

    // Sort max_absval_surrogates from low to high
    for (int i = 0; i < nshuffles - 1; i++) {
        for (int j = 0; j < nshuffles - i - 1; j++) {
            if (max_absval_surrogates[j] > max_absval_surrogates[j + 1]) {
                float temp = max_absval_surrogates[j];
                max_absval_surrogates[j] = max_absval_surrogates[j + 1];
//                atomicAdd(&max_absval_surrogates[j], max_absval_surrogates[j + 1]);
                max_absval_surrogates[j + 1] = temp;
//                atomicAdd(&max_absval_surrogates[j + 1], temp);
            }
        }
    }
    // grasping the largest element scaled by percentile (test for 100) and nshuffle size (test case set to 10)
    // max_absval_surrogates is sorted from low to high.
    // correction needed for percentile > nshuffles. if result is 0 then index is 10 which is of other thread mem thus -1
    int thr_idx = 0;
    if (percentile == 1) {
        thr_idx = 0;
    } else {
        thr_idx = (nshuffles - (int)(nshuffles / percentile))-1;
//        printf("maxab[the_idx] %f \n", max_absval_surrogates[thr_idx]);
    }

    float signalThresh = max_absval_surrogates[thr_idx];
//    printf("signalThreshGu %f \n", signalThresh);

//    extern __shared__ float shared_memB[];
//    float *signal_binaryx = shared_memB;
    //Binarize the signal
    // watch out for race conditions! atomic add
    for (int trial = 0; trial < ntrials; trial++) {
        for (int source = 0; source < nsources; source++) {
            for (int t = 0; t < nbins; t++) {
                float val = signal_centre_norm[(trial * nsources * nbins) + (source * nbins) + t];
//                signal_binary[(trial * nsources * nbins) + (source * nbins) + t] = (val > signalThresh) ? 1.0 : 0.0;
                if (val > signalThresh){
//                    atomicAdd(&signal_binary[(trial * nsources * nbins) + (source * nbins) + t], 1.0f);
                    signal_binary[(trial * nsources * nbins) + (source * nbins) + t] = 1;
                }
//                printf("idx %d val %f val > signalThresh %f \n", (trial * nsources * nbins) + (source * nbins) + t,
//                    val, (val > signalThresh) ? 1.0 : 0.0);
            }
//            __syncthreads();
        }
//        __syncthreads();
    }


//    for (int trial = 0; trial < ntrials; trial++) {
//        for (int source = 0; source < nsources; source++) {
//            for (int t = 0; t < nbins; t++) {
//                float val = signal_binary[(trial * nsources * nbins) + (source * nbins) + t];
////                signal_binary[(trial * nsources * nbins) + (source * nbins) + t] =
////                    signal_binaryx[(trial * nsources * nbins) + (source * nbins) + t];
////                printf("tidx %d lidx %d val %f\n", idx, (trial * nsources * nbins) + (source * nbins) + t, val);
//            }
////            __syncthreads();
//        }
//    }
}

__device__ void sort_binj(int *sorted_binJ, int *signal_binary, int* sumCh,
                          int trial, int ntrials, int nsources, int nbins, int t_stim, int idx){

        //* this function writes in the signal binary, comparison with signal binary give crooked report *//
        //* in the first stage the sum of the rows is determined of the binary matrix *//
        //* the 2nd stage sorts all rows based on the largest sum. Here, a diff with the python version occurs, *//
        //* as the sort may reults in duplicate indexes that are not shifted by cuda version but may by python *//
        //* will this pose a problem for the lzc and norm? *//
        //* the sorting algorithm starts on both ends of the loop and swaps the entire row *//
        //* binj keeps nbins size however start iteration from tstim for the sum and sorting *//
        //* copying however needs 2 different indices because of the reduce to tstim size results array there *//

//        t_stim = 0;

        // originally adapted to match mine
        int* binJ = &signal_binary[trial * nsources * nbins];
        // Calculate sum for each row
        for (int i = 0; i < nsources; i++) {
            int sum = 0;
            // start form tstim in order to do only the 2nd half of the binarized matrix
            // should be nbins-tstim if nbins != tstim !! keep symmetric for now
            for (int k = t_stim; k < nbins; k++) {
                sum += binJ[i * nbins + k];
//                if (idx == 1)
//                    printf("idx %d trial %d index %d binj %d \n", idx, trial, i * nbins + k, binJ[i * nbins + k]);
            }
            sumCh[i] = sum;
//            if (idx == 1)
//                printf("idx %d trial %d source %d sum %d \n", idx, trial, i, sum);
        }

        // Sort binJ based on sum
        for (int i = 0; i < nsources - 1; i++) {
            for (int k = 0; k < nsources - i - 1; k++) {
                int sum1 = sumCh[k];
                int sum2 = sumCh[k + 1];
                if (sum1 < sum2) {
                    // Swap rows
                    for (int l = t_stim; l < nbins; l++) {
                        int temp = binJ[k * nbins + l];
                        binJ[k * nbins + l] = binJ[(k + 1) * nbins + l];
                        binJ[(k + 1) * nbins + l] = temp;
                    }
                    int temp = sumCh[k];
                    sumCh[k] = sumCh[k + 1];
                    sumCh[k + 1] = temp;
                }
            }
        }

        // Copy sorted binJ to output array
        // Such a hardship getting the indices right. sortbinj which is (nwi*)ntrials*nsources*tstim large
        // iterates differently the sortedbinj of which only the last part of tstim needs to be computed
        // therefore the k+tstim in the binj index
        for (int i = 0; i < nsources; i++) {
            for (int k = 0; k < t_stim; k++) {
                sorted_binJ[trial * nsources * t_stim + i * t_stim + k] = binJ[i * nbins + (k+t_stim)];
//                if (idx == 1) {
//                    printf("in copy idx %d trial %d index %d binj %d \n", idx, trial, i * nbins + (k+t_stim), binJ[i * nbins + (k+t_stim)]);
//                    printf("in sobi idx %d trial %d index %d binj %d \n", idx, trial,
//                        trial * nsources * t_stim + i * t_stim + k, sorted_binJ[trial * nsources * t_stim + i * t_stim + k]);
//                }
            }
        }
}

__device__ int lz_complexity_2D(int *Dnew, int *ct, int trial, int idx, unsigned long long int *bits, int ntrials, int nsources, int t_stim)
{
    //* fixed this function hopefully, hmm it scales when nsources are increased
    //* the issue was that the lzc did no accumulate over trials. changing c++ to ct[trial]++; fixed it *//
    //* This function in a first loop translates the binjs into bits *//
    //* These bits are then checked for recurring sequences by orring the signal with a shifted version of itself *//
    //* If the shifted parts are unequal (found |=) then complexity accumulated as an int (ct) *//
    //* todo the magic is in the int c = 1, r = 1, q = 1, k = 1, i = 1; vars; unclear what they represent *//

    int c = 1, r = 1, q = 1, k = 1, i = 1;
    ct[trial] = 1;
    bool stop = false;
    int fnd = 0;

    // Convert each column to a sequence of bits
    // checked bitsy part
//    for (int y = 0; y < nbins; y++) {
//        unsigned long long int bit_value = 0;
//        for (int x = 0; x < nsources; x++) {
//            bit_value = (bit_value << 1) | (Dnew[trial * nsources * nbins + x * nbins + y] & 1);
////            printf("Dnew %d \n", Dnew[trial * nsources * nbins + x * nbins + y]);
//        }
//        bits[y] = bit_value;
////        printf("y %d bit_value %d\n", y, bit_value);
//    }
//
//        // Main loop
//    while (!stop) {
////        int a = (q == r) ? i + k - 1 : nsources;
//        int a = 0;
//        if (q==r) a = i + k - 1;
//        else a = nsources;
//
//        int found = 0;
//        for (int shift = 0; shift <= a - k; shift++) {
////            found |= ((bits[q - 1] >> shift) & ((1 << k) - 1)) == (bits[r - 1] & ((1 << k) - 1));
////            if (found) {
////                fnd++;
////                break;
//
//            // Right shift the bits of bits[q - 1] by 'shift' positions and mask the lowest 'k' bits
//            unsigned long long int shifted_bits = (bits[q - 1] >> shift) & ((1 << k) - 1);
//
//            // Mask the lowest 'k' bits of bits[r - 1]
//            unsigned long long int masked_bits = bits[r - 1] & ((1 << k) - 1);
//
//            if (shifted_bits == masked_bits) {
////                printf("idx %d trial %d sht %d\n", idx, trial, shifted_bits);
////                printf("idx %d trial %d mbt %d\n", idx, trial, masked_bits);
//                found = true;
//                fnd++;
//                break;
//
//            }
//        }

    const int bitschunk = 64;
    const int MAX_BITSY = 64;
    // enhanced loop for 128 bit variables for nscources above 64 bit for nsources larger then 68 regions or nsources
    int num_chunks = (nsources +(bitschunk-1))/bitschunk;
//    int num_chunks = 0;
    unsigned long long int bit_chunks[2] = {0};  // Adjust size based on nsources
    for (int y = 0; y < t_stim; y++) {
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            bit_chunks[chunk] = 0;
        }

    // Convert each column to a sequence of bits
//    for (int y = 0; y < t_stim; y++) {
        unsigned long long int bit_value = 0;
        for (int x = 0; x < nsources; x++) {
            int chunk_idx = x / bitschunk;
            int bit_idx = x % bitschunk;
            bit_chunks[chunk_idx] |= ((unsigned long long int)(Dnew[trial * nsources * t_stim + x * t_stim + y] & 1)) << bit_idx;
//            bit_chunks[chunk_idx] = (bit_value << 1) | (Dnew[trial * nsources * t_stim + x * t_stim + y] & 1);
        }
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            bits[y * num_chunks + chunk] = bit_chunks[chunk];
//            bits[y * num_chunks + chunk + trial * t_stim * num_chunks] = bit_chunks[chunk];
        }
    }

    // Main loop
//    while (!stop) {
//        int a = (q == r) ? i + k - 1 : nsources;
//        int found = 0;
//        for (int shift = 0; shift <= a - k; shift++) {
//            for (int chunk = 0; chunk < num_chunks; chunk++) {
//                unsigned long long int shifted_bits = (bits[(q - 1) * num_chunks + chunk] >> shift) & ((1ULL << k) - 1);
//                unsigned long long int masked_bits = bits[(r - 1) * num_chunks + chunk] & ((1ULL << k) - 1);
//                if (shifted_bits == masked_bits) {
//                    found = true;
//                    fnd++;
//                    break;
//                }
//            }
//            if (found) break;
//        }

    while (!stop) {
        int a = (q == r) ? i + k - 1 : nsources;
        bool found = false;

        for (int shift = 0; shift <= a - k; shift++) {
            found = true;
            for (int chunk = 0; chunk < num_chunks; chunk++) {
                unsigned long long int shifted_bits = (bits[(q - 1) * num_chunks + chunk] >> shift) & ((1ULL << k) - 1);
                unsigned long long int masked_bits = bits[(r - 1) * num_chunks + chunk] & ((1ULL << k) - 1);

                // For spans across chunk boundaries
                if (chunk < num_chunks - 1 && shift > 0) {
                    shifted_bits |= bits[(q - 1) * num_chunks + chunk + 1] << (64 - shift);
                    masked_bits |= bits[(r - 1) * num_chunks + chunk + 1] << (64 - shift);
                }

                shifted_bits &= (1ULL << k) - 1;
                masked_bits &= (1ULL << k) - 1;

                if (shifted_bits != masked_bits) {
                    found = false;
                    break;
                }
            }
            if (found) {
                fnd++;
                break;
            }
        }

        if (found==true) {
            k++;
            if (i + k > nsources) {
                r++;
                if (r > t_stim) {
                    c++;
//                    printf("idx %d trial %d cif %d\n", idx, trial, c);
//                    ct[trial]++;
                    stop = true;
                } else {
                    i = 0;
                    q = r - 1;
                    k = 1;
                }
                ct[trial] = c;
//                printf("idx %d trial %d cif %d\n", idx, trial, c);
            }
        } else {
            q--;
            if (q < 1) {
//                ct[trial]++;
                c++;
//                printf("idx %d trial %d cel %d\n", idx, trial, c);
                i = i + k;
                if (i + 1 > nsources) {
                    r++;
                    if (r > t_stim) {
//                        ct[trial]++;
                        c++;
//                        printf("idx %d trial %d cel %d\n", idx, trial, c);
                        stop = true;
                    } else {
                        i = 0;
                        q = r - 1;
                        k = 1;
                    }
                    ct[trial] = c;
                } else {
                    q = r;
                    k = 1;
                }
            }
        }
    }

//    printf("idx %d trial %d found %d c %d\n", idx, trial, fnd, c);
    return c;
}

//__device__ float pci_norm_factor(int *D, int trial, int ntrials, int nsources, int t_stim)
//{
//    float p1 = 0.0f;
////    int p1 = 0;
//    float p0 = 0.0f;
////    int p0 = 0;
//    float H = 0.0f;
////    int L = ntrials*nsources*nbins;
//    int L = nsources*t_stim;
//
//    for (int i = 0; i < L; i++) {
//        p1+= (1.0 * D[i+trial*L] == 1);
////        p1[trial] += (1.0 * D[i] == 1);
////        p1 /= L;
//
//        // when flattened it doesnt follow the structure, as long as the sum is equal. seems so
////        printf("D[i] %d\n", D[i+trial*L]);
//    }
//
////    p1[trial] /= L;
//    p1 /= L;
//    p0 = 1.0f - p1;
////    printf("p1 %f\n", p1);
////    printf("p0 %f\n", p0);
////    p0[trial] = 1.0f - p1[trial];
//
//    if (p1 * p0 != 0) {
////    if (p1[trial] * p0[trial] != 0) {
////        printf("p0=%g\np1=%g\n", p0, p1);
//        H = -p1 * log2f(p1) - p0 * log2f(p0);
////        H = -p1[trial] * log2f(p1[trial]) - p0[trial] * log2f(p0[trial]);
////        printf("H %f\n", (L * H) / log2f(L));
//        return (L * H) / log2f(L);
//    } else {
////        printf("p0=%g\np1=%g\n", p0, p1);
//        return 0.0f;
//    }
//}

// old function causes divide by zero for ntrials = 1 and low complexities.
// this function should fix that or at least no infs with 1 dimension on the statevar position
// in gpudata and ntrial = 2
__device__ float pci_norm_factor(int *D, int trial, int ntrials, int nsources, int t_stim) {
    const float min_threshold = 1e-7f; // Minimum threshold to avoid division by zero
    float p1 = 0.0f;
    float p0 = 0.0f;
    float H = 0.0f;
    int L = nsources * t_stim;

    // Calculate the count of ones in D for the specified trial
    for (int i = 0; i < L; i++) {
        p1 += (1.0f * D[i + trial * L] == 1);
    }

    // Normalize p1 and set minimum thresholds for p1 and p0
    p1 = max(p1 / L, min_threshold);
    p0 = max(1.0f - p1, min_threshold);

    // Calculate entropy H
    H = -p1 * log2f(p1) - p0 * log2f(p0);

    // Normalization factor
    return (L * H) / log2f(L);
}


//extern "C"
__global__  void pci_kernel(
    float *signal_m,
    float *gpu_means_prestim,
    float *gpu_signal_centre,
    float *gpu_std_prestim,
    float *gpu_sigcent_norm,
    float *gpu_maxabsval,
    int *gpu_signal_binary,
    int *gpu_binJ,
    int *gpu_ct,
    unsigned long long int *gpu_bits,
    int *gpu_sumCh,
    float *gpu_pci_lst,
    float *gpu_signal_prestim_shuffle,
    float *gpu_shuffle_avg,
    int nworkitems,
    int ntrials,
    int nsources,
    int nbins,
    int t_stim,
    int nshuffles,
    int percentile,
    int stonset)
{


        // Calculate thread index
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const unsigned int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    const unsigned int size = ntrials * nsources * nbins;
    const unsigned int size_tstim = ntrials * nsources * t_stim;
    const unsigned int size_mean = ntrials * nsources;
    const unsigned int size_shuf = nsources * t_stim;
    const unsigned int bits_size = nbins * ((nsources + 63) / 64);

    // byebye to all the useless threads
    if (idx >= nworkitems) return;

    // Call binarise_signals function
    binarise_signals(
        signal_m + idx * size,
        t_stim,
        nshuffles,
        percentile,
        nworkitems,
        ntrials,
        nsources,
        nbins,
        stonset,
        gpu_signal_binary + idx * size,
//        gpu_signal_binary,
        idx,
        gpu_means_prestim + idx * size_mean,
        gpu_signal_centre + idx * size,
        gpu_std_prestim + idx * size_mean,
        gpu_sigcent_norm + idx * size,
        gpu_maxabsval + idx * nshuffles,
        gpu_signal_prestim_shuffle + idx * size_tstim,
        gpu_shuffle_avg + idx * size_shuf);

    // taking the avg over trials
    float avg_gpu_pci = 0;
    int lzc = 0;
    float norm = 1;
    for (int trial = 0; trial < ntrials; trial++) {

        // switching on sortbinj will malform the binary matrix because it writes in place
        sort_binj(
            gpu_binJ + idx * size_tstim, ////!!!!!!!!!
            gpu_signal_binary + idx * size,
            gpu_sumCh + idx * nsources,
            trial, ntrials, nsources, nbins, t_stim, idx);

        lzc = lz_complexity_2D(
            gpu_binJ + idx * size_tstim,
            gpu_ct + idx * ntrials,
            trial,
            idx,
//            gpu_bits + idx * nbins,
            gpu_bits + idx * bits_size,
            ntrials, nsources, t_stim);
//        printf("idx %d trial %d lzc %d\n", idx, trial, lzc);

        norm = pci_norm_factor(
            // or directly deref array in function call and pass starting address
//            &gpu_binJ[ntrial * nsources * nbins]);
            gpu_binJ + idx * size_tstim,
            trial,
            ntrials, nsources, t_stim);

        avg_gpu_pci += (float(lzc) / norm)/ntrials;

    }
    // Write pci_lsts to gpu_output
    gpu_pci_lst[idx] = avg_gpu_pci;

//    for (int trial = 0; trial < ntrials; trial++) {
//        for (int source = 0; source < nsources; source++) {
//            gpu_results[(idx * ntrials * nsources) + trial * nsources + source] = means_prestim[(trial * nsources) + source];
//        }
//    }
}
