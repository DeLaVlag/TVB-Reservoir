try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
except ImportError:
	# logging.warning('pycuda not available, rateML driver not usable.')
	print('pycuda not available, rateML driver not usable.')

import numpy as np

import os
here = os.path.dirname(os.path.abspath(__file__))

import sys
# add path to PCI dir for pci_bin.py
sys.path.append(os.path.abspath('/'))
# from PCI.pci_bin import *

import tqdm


def cf(array):
    # coerce possibly mixed-stride, double precision array to C-order single precision
    return array.astype(dtype='f', order='C', copy=True)

def ci(array):
    # coerce possibly mixed-stride, double precision array to C-order single precision
    return array.astype(dtype='i', order='C', copy=True)

def make_kernel(source_file, warp_size, nh='nh'):

    try:
        with open(source_file, 'r') as fd:
            source = fd.read()
            source = source.replace('M_PI_F', '%ff' % (np.pi,))
            opts = ['--ptxas-options=-v', '-maxrregcount=32']
            # if lineinfo:
            opts.append('-lineinfo')
            # opts.append('-cudart=legacy')
            # opts.append('-cudart=shared')
            # opts.append('-cuda-device-code-rewrite pretty-print')
            opts.append('-g')
            opts.append('-DWARP_SIZE=%d' % (warp_size,))
            opts.append('-DNH=%s' % (nh,))

            idirs = [here]

            # no_extern_c=True gives the stupid named symbol not found error
            try:
                network_module = SourceModule(
                    source, options=opts, include_dirs=idirs,
                    no_extern_c=True,
                    keep=True, )
            except drv.CompileError as e:
                print('Compilation failure \n %s', e)
                exit(1)

            mod_func = '_Z10pci_kernelPfS_S_S_S_S_PiS0_S0_PyS0_S_S_S_iiiiiiii'

            pci_ker = network_module.get_function(mod_func)

    except FileNotFoundError as e:
        print('%s.\n  PCI.c file not found', e)
        exit(1)

    return pci_ker, 0, 0

def shuffle_prestim_signal(signal, stonset, t_stim, ntrials, nsources, seed=1234):
    rng = np.random.RandomState(seed)
    shuffled_signal = signal.copy()

    for trial in range(ntrials):
        for source in range(nsources):
            prestim_signal = shuffled_signal[trial, source, stonset:t_stim]
            rng.shuffle(prestim_signal)
            shuffled_signal[trial, source, stonset:t_stim] = prestim_signal

    return shuffled_signal

def pytestPCI(GPU_data, gpu_means_prestim, gpu_signal_centre, gpu_std_prestim,
                gpu_sigcent_norm, gpu_signal_prestim_shuffle,
                gpu_maxabsval, gpu_signal_binary, gpu_binJ, gpu_sumch, gpu_ct, gpu_pci_lst_get,
                nworkitems, t_stim, nshuffles, nsources, ntrials, percentile, stonset):

    gpu_means_prestim_get = gpu_means_prestim.get()
    gpu_signal_centre_get = gpu_signal_centre.get()
    gpu_std_prestim_get = gpu_std_prestim.get()
    gpu_signal_centre_norm_get = gpu_sigcent_norm.get()
    gpu_signal_prestim_shuffle_get = gpu_signal_prestim_shuffle.get()
    gpu_maxabsval_get = gpu_maxabsval.get()
    gpu_signal_binary_get = gpu_signal_binary.get()

    gpu_binJ_get = gpu_binJ.get()
    gpu_sumch_get = gpu_sumch.get()
    gpu_ct_get = gpu_ct.get()

    pypcis = []
    for i in tqdm.trange(nworkitems):
        means_prestim = (np.mean(GPU_data[i,:,:,:t_stim], axis=2))
        signal_centre = GPU_data[i,:,:,:] / means_prestim[:, :, np.newaxis] - 1
        std_prestim = np.std(signal_centre[:, :, :t_stim], axis=2)
        signal_centre_norm = signal_centre / std_prestim[:, :, np.newaxis]

        # %% bootstrapping: shuffle prestim signal in time, intra-trial
        signalcn_tuple = tuple(signal_centre_norm)  # not affected by shuffling
        signal_prestim_shuffle = signal_centre_norm[:, :, :t_stim]

        max_absval_surrogates = np.zeros(nshuffles)
        # shuffle_avg = np.zeros((nsources, t_stim))

        # for i_shuffle in range(nshuffles):
        #     # signal_prestim_shuffle = shuffle_prestim_signal(signal_prestim_shuffle, stonset, t_stim, ntrials, nsources,
        #     #                                                 seed=1234)
        #
        #     # Calculate average over trials and maximum absolute value
        #     max_val = 0.0
        #     for source in range(nsources):
        #         for t in range(stonset, t_stim):
        #             shuffle_avg[source, t] = np.mean(signal_prestim_shuffle[:, source, t])
        #             val = shuffle_avg[source, t]
        #             max_val = max(max_val, np.abs(val))
        #     max_absval_surrogates[i_shuffle] = max_val
        #
        # print('signal_prestim_shuffle', signal_prestim_shuffle)
        #
        # # Sort max_absval_surrogates from low to high and calculate the threshold
        # # the smallest threshold is selected
        # max_sorted = np.sort(max_absval_surrogates)
        #
        # # continu with gpu sorted ones to bypass difference in randomness
        max_sorted = gpu_maxabsval_get[i,:]
        #
        if percentile == 1:
            thr_idx = -nshuffles
        else:
            thr_idx = (-int(nshuffles / percentile)) - 1
        signalThresh = max_sorted[thr_idx]
        # print("signalThresh", signalThresh)

        # Binarize the signal
        signalcn = signal_centre_norm
        signal_binary = signalcn > signalThresh
        # print('signalThreshpY', signalThresh)
        # print('signnomrpY', signal_centre_norm)
        # print('signal_binarypy', signal_binary.astype(int))



        # for i_shuffle in range(nshuffles):
        #     for i_source in range(nsources):
        #         for i_trial in range(ntrials):
        #             signal_curr = signal_prestim_shuffle[i_trial, i_source]
        #             # np.random.shuffle(signal_curr)
        #             signal_prestim_shuffle[i_trial, i_source] = signal_curr
        #             # average over trials (removed from the i_trial loop)
        #     shuffle_avg = np.mean(signal_prestim_shuffle, axis=0)
        #     max_absval_surrogates[i_shuffle] = np.max(np.abs(shuffle_avg))
        #
        # # print(max_absval_surrogates)
        #
        # max_sorted = np.sort(max_absval_surrogates)
        #
        # if percentile == 1:
        #     thr_idx = -nshuffles
        # else:
        #     thr_idx = (-int(nshuffles / percentile)) - 1
        # signalThresh = max_sorted[thr_idx]  # correction made for index
        #
        # # %% binarise
        # signalcn = np.array(signalcn_tuple)
        # signal_binary = signalcn > signalThresh


        PCI_trials = []
        lzclist = []
        normlist = []
        for j in range(ntrials):
            # shape should be 2 dimensional
            binJ = signal_binary.astype(int)[j, :, t_stim:]
            # print('binJ)___\n', binJ.astype(int))
            SumCh = np.sum(binJ, axis=1)
            # print('SumPy wi trial', i, j, SumCh)
            Irank = SumCh.argsort()[::-1]
            # print('Irank', j, Irank)
            binJs = binJ[Irank, :]

            # test binjs
            thresholdbinj = 0.02
            maebin = np.mean(np.abs(binJs - gpu_binJ_get[i, j, :, :]))
            equalBIN = maebin <= thresholdbinj
            if not equalBIN:
                print("BINJ absolute error:", maebin)
                print("Arrays not are equal within threshol, workitem:", i, j)
            # print('binJ', binJs)
            # print('gpu_binJ_get', gpu_binJ_get[i, j, :, :])

            # get binj from GPU to test without pyrandoness. binja is already the upper part of the binmatrix
            # binJs = gpu_binJ_get[i,j,:,:]
            # print('binJs', binJs.shape)

            Lempel_Ziv_lst = lz_complexity_2D(binJs, i, j)
            # print('wi trl LZC___', i, j, Lempel_Ziv_lst)
            norm = pci_norm_factor(binJs)
            # print("wi trl pynorm", i, j, norm)

            pci_lst = Lempel_Ziv_lst/norm
            PCI_trials.append(pci_lst)
            lzclist.append(Lempel_Ziv_lst)
            normlist.append(norm)

            # print('pci_lst', pci_lst)

            # threshold = 0.2
            # mae6 = np.mean(np.abs(binJs - gpu_binJ_get[i, j, :, :]))
            # equal6 = mae6 <= threshold

            # if not equal6:
            #     print("Binj absolute error:", mae6)
            #     print("Binj not are equal within threshold, workitem:", i)

            # print('binJs___\n', binJs.astype(int))
            # print('gpubinJs\n', gpu_binJ_get[i, j, :])

            # print('binJs', binJs.astype(int))
        # print('Lempel_Ziv_lst', np.mean(PCI_trials))
        pypcis.append(np.mean(PCI_trials))

        threshold = 0.1

        mae0 = np.mean(np.abs(means_prestim - gpu_means_prestim_get[i,:,:]))
        equal0 = mae0 <= threshold

        if not equal0:
            print("Mean absolute error:", mae0)
            print("Arrays not are equal within threshold:, workitem:", i)

        mae1 = np.mean(np.abs(signal_centre - gpu_signal_centre_get[i,:,:,:]))
        equal1 = mae1 <= threshold

        if not equal1:
            print("SCentre absolute error:", mae1)
            print("Arrays not are equal within threshold:, workitem:", i)

        mae2 = np.mean(np.abs(std_prestim - gpu_std_prestim_get[i, :, :]))
        equal2 = mae2 <= threshold

        if not equal2:
            print("Stddev_prestim absolute error:", mae2)
            print("Arrays not are equal within threshold:, workitem:", i)

        mae3 = np.mean(np.abs(signal_centre_norm - gpu_signal_centre_norm_get[i, :, :, :]))
        equal3 = mae3 <= threshold

        if not equal3:
            print("SCentre_norm absolute error:", mae3)
            print("Arrays not are equal within threshold, workitem:", i)

        # start to be random/ check from max_absval_surrogates
        mae4 = np.mean(np.abs(max_sorted - gpu_maxabsval_get[i, :]))
        equal4 = mae4 <= threshold

        if not equal4:
            print("Maxabsval absolute error:", mae4)
            print("Arrays not are equal within threshol, workitem:", i)

            print('max_sorted', max_sorted)
            print('gpu_maxabsval_get[i, :]', gpu_maxabsval_get[i, :])


        mae5 = np.mean(np.abs(signal_binary - gpu_signal_binary_get[i, :]))
        equal5 = mae5 <= threshold

        sortbinjon = True
        if not equal5 and not sortbinjon:
            print("Binary absolute error:", mae5)
            print("Arrays not are equal within threshold, workitem:", i)


        assert gpu_means_prestim_get[i,:,:].shape == means_prestim.shape
        assert gpu_signal_centre_get[i,:,:].shape == signal_centre.shape
        assert gpu_std_prestim_get[i,:,:].shape == std_prestim.shape
        assert gpu_signal_centre_norm_get[i,:,:].shape == signal_centre_norm.shape
        assert gpu_signal_prestim_shuffle_get[i,:,:].shape == signal_prestim_shuffle.shape
        assert gpu_maxabsval_get[i,:].shape == max_absval_surrogates.shape
        assert gpu_signal_binary_get[i,:].shape == signal_binary.shape


    print('avg over wi pypcis', pypcis)

    thresholdPCI = 0.01
    mae5 = np.mean(np.abs(pypcis - gpu_pci_lst_get))
    equal5 = mae5 <= thresholdPCI
    testPCIvsPY = True
    if not equal5 and testPCIvsPY:
        print("PCIs absolute error:", mae5)
        print("Arrays not are equal within threshold, workitem:", i)

    relerorPCI = np.mean(((np.abs(pypcis - gpu_pci_lst_get))/pypcis) * 100)
    print('PCIs relative error:', relerorPCI)

    print('means', gpu_means_prestim_get.shape)
    print('scent', gpu_signal_centre_get.shape)
    print('stdpr', gpu_std_prestim_get.shape)
    print('scnor', gpu_signal_centre_norm_get.shape)
    print('spshu', gpu_signal_prestim_shuffle_get.shape)
    # print('spshu', gpu_signal_prestim_shuffle_get)
    print('maxab', gpu_maxabsval_get.shape)
    # print('maxab', gpu_maxabsval_get)
    print('binar', gpu_signal_binary_get.shape)
    # print('binar', gpu_signal_binary_get)
    print('gbinJ', gpu_binJ_get.shape)
    print('gsumC', gpu_sumch_get.shape)
    print('gpucs', gpu_ct_get.shape)
    # print('gpuct', gpu_ct_get)


def gpu_mem_info():
    cmd = "nvidia-smi -q -d MEMORY"  # ,UTILIZATION"
    os.system(cmd)  # returns the exit code in unix

def multistim(data):

    # data = np.arange(100)
    # print(data)
    datamultiplestim = []
    stimregions = 1
    bins_averaging = 1
    sim_length = 5000
    stimonset = int(sim_length / 50)  #100
    stimmoment = int(sim_length / 5)  #1000
    regions = 68
    # print('ds', data.shape)
    workitems = data.shape[2]

    # process the data such that it has the shape for the PCI as in pci_bin (ntrials, nsources, nbins, workitems)
    # for single stimulus simple
    prepostall = np.zeros((stimregions, regions, int(data.shape[0]/2), workitems))
    # print('ppsa', prepostall.shape)
    for j in range(workitems):
        for i in range(stimregions):
            # for simple single stiumuls
            prestim = 0
            poststim = int(data.shape[0] / 2)

            # chunk the ts
            prepost = data[prestim:poststim, :, j]
            # print('PPS', prepost.shape)
            prepost = np.mean(prepost.reshape(regions, -1, bins_averaging), axis=2)

            # prepostall.append(prepost)
            prepostall[i, :, :, j] = prepost

            # prepostall = np.zeros((self.regions, self.stimmoment))
    data = np.array(prepostall)


def run_pci_analy(GPU_data, loggerobj, mpirank, nworkitems, ntrials, nsources, nbins, t_stim, nshuffles, percentile, stonset, testtoPY):

    '''

    nbins are the parts of the timeseries to analyse before and after the tstim
    nshuffels the amout of shuffles needed to bypass the variance (or something)
    ntrials are the different attemepts

    '''
    ### data allocation ###
    # it doesnt like cudamalloc on the device side very much
    # only the gpu_pci_lst needs to be accessed. others need to be 'internalised'
    gpu_data = gpuarray.to_gpu(cf(GPU_data))
    gpu_means_prestim = gpuarray.to_gpu(cf(np.zeros((nworkitems, ntrials, nsources),)))
    gpu_signal_centre = gpuarray.to_gpu(cf(np.zeros((nworkitems, ntrials, nsources, nbins),)))
    gpu_std_prestim = gpuarray.to_gpu(cf(np.zeros((nworkitems, ntrials, nsources),)))
    gpu_sigcent_norm = gpuarray.to_gpu(cf(np.zeros((nworkitems, ntrials, nsources, nbins), )))
    gpu_signal_prestim_shuffle = gpuarray.to_gpu(cf(np.zeros((nworkitems, ntrials, nsources, t_stim),)))
    gpu_shuffle_avg = gpuarray.to_gpu(cf(np.zeros((nworkitems, nsources, t_stim), )))
    gpu_maxabsval = gpuarray.to_gpu(cf(np.zeros((nworkitems, nshuffles), )))
    gpu_signal_binary = gpuarray.to_gpu(ci(np.zeros((nworkitems, ntrials, nsources, nbins),)))
    # gpu_shuffle_avg = gpuarray.empty_like(data)

    gpu_binJ = gpuarray.to_gpu(ci(np.zeros((nworkitems, ntrials, nsources, t_stim), )))
    gpu_sumch = gpuarray.to_gpu(ci(np.zeros((nworkitems, nsources), )))

    gpu_ct = gpuarray.to_gpu(ci(np.zeros((nworkitems, ntrials), )))

    bit_chunks = (nsources + 63) // 64
    gpu_bits = gpuarray.to_gpu(np.zeros((nworkitems, nbins, bit_chunks),dtype=np.uint64))

    gpu_pci_lst = gpuarray.to_gpu(cf(np.zeros((nworkitems), )))

    ### make kernel ###
    pci_ker, dfa_ker, lya_ker = make_kernel(source_file=here + '/pci.c', warp_size=32,) # hpc
    # pci_ker, dfa_ker, lya_ker = make_kernel(source_file=here + '/models/pci.c', warp_size=32,) # local
    # todo mend project space hpc to this
    # pci_ker, dfa_ker, lya_ker = make_kernel(source_file=here + '/../PCI/pci.c', warp_size=32,)

    # determine optimal grid recursively
    def dog(fgd):
        maxgd, mingd = max(fgd), min(fgd)
        maxpos = fgd.index(max(fgd))
        if (maxgd - 1) * mingd * bx * by >= nwi:
            fgd[maxpos] = fgd[maxpos] - 1
            dog(fgd)
        else:
            return fgd

    bx, by = 32, 32
    nwi = nworkitems
    rootnwi = int(np.ceil(np.sqrt(nwi)))
    gridx = int(np.ceil(rootnwi / bx))
    gridy = int(np.ceil(rootnwi / by))

    block_dim = bx, by, 1

    fgd = [gridx, gridy]
    dog(fgd)
    grid_dim = fgd[0], fgd[1]

    pci_ker(gpu_data,
            gpu_means_prestim,
            gpu_signal_centre,
            gpu_std_prestim,
            gpu_sigcent_norm,
            gpu_maxabsval,
            gpu_signal_binary,
            gpu_binJ,
            gpu_ct,
            gpu_bits,
            gpu_sumch,
            gpu_pci_lst,
            gpu_signal_prestim_shuffle,
            gpu_shuffle_avg,
            np.intc(nworkitems), np.intc(ntrials),
            np.intc(nsources), np.intc(nbins),
            np.intc(t_stim), np.intc(nshuffles),
            np.intc(percentile), np.intc(stonset),
            block=block_dim, grid=grid_dim)
            # shared=shared_mem_size)
    drv.Context.synchronize()

    # print('gpu_signal_binary', gpu_signal_binary.get())
    gpu_pci_lst_get = gpu_pci_lst.get()
    gpu_ct_get = gpu_ct.get()

    if mpirank == 0:
        loggerobj.info('PCI block_dim %s', block_dim)
        loggerobj.info('PCI grid_dim %s', grid_dim)
        loggerobj.info("PCI result shape %s", gpu_pci_lst_get.shape)
        print("\n")


    testtoPython = testtoPY
    if testtoPython:
        pytestPCI(GPU_data, gpu_means_prestim, gpu_signal_centre, gpu_std_prestim,
                  gpu_sigcent_norm, gpu_signal_prestim_shuffle,
                  gpu_maxabsval, gpu_signal_binary, gpu_binJ, gpu_sumch, gpu_ct, gpu_pci_lst_get,
                  nworkitems, t_stim, nshuffles, nsources, ntrials, percentile, stonset)

    return gpu_pci_lst_get, gpu_ct_get


if __name__ == '__main__':

    def generate_high_complexity_data(nsims, ntrials, nsources, nbins):

        '''
        High Complexity Data:

        Adds more randomness, high-frequency noise, and occasional random jumps across time
        to introduce unpredictable patterns.
        '''

        # Generate random data for high complexity with substantial variability
        high_complexity_data = np.random.normal(loc=0.0, scale=1.0, size=(nsims, ntrials, nsources, nbins))

        # Introduce more high-frequency noise and random jumps across time
        noise = np.random.normal(loc=0.0, scale=0.5, size=(nsims, ntrials, nsources, nbins))
        jumps = np.random.normal(loc=0.0, scale=2.0, size=(nsims, ntrials, nsources, nbins)) * (
                    np.random.rand(nsims, ntrials, nsources, nbins) > 0.95)

        high_complexity_data += noise + jumps
        return high_complexity_data


    def generate_low_complexity_data(nsims, ntrials, nsources, nbins):

        '''
        Low Complexity Data:
        Generates smoother patterns with consistent low-frequency trends and minimal noise across time.
        '''
        # Generate data for low complexity with smoother patterns and low variability
        low_complexity_data = np.random.normal(loc=0.0, scale=0.2, size=(nsims, ntrials, nsources, nbins))

        # Add consistent low-frequency signals across time for each source
        for sim in range(nsims):
            for trial in range(ntrials):
                for source in range(nsources):
                    trend = np.linspace(0, 1, nbins)  # Linear trend
                    low_complexity_data[sim, trial, source, :] += trend + np.sin(np.linspace(0, 2 * np.pi, nbins)) * 0.1

        return low_complexity_data

    # analyse 300ms before and after stimulus moment
    t_analysis = 300  # ms
    # n_inner_steps = int(tavg_period / dt)
    times_l = 1 #ms each iteration on the gpu is 10*.1 ms = 1ms
    nbins_analysis = int(t_analysis / times_l)
    # we need the whole data
    # data = data[:,:,stimonset - nbins_analysis:stimonset + nbins_analysis]

    '''testdata for means'''
    nsims = 10
    # number of simulations/realisations to analyse for one PCI value/ do we need this as we have many trials in gpu threads
    ntrials = 2
    nsources = 62
    nbins, t_stim = 100, 150
    nshuffles = 2
    percentile = 1
    stonset = 0

    # Generate high complexity data
    high_complexity_data = generate_high_complexity_data(nsims, ntrials, nsources, nbins)
    GPU_data = high_complexity_data

    # GPU_data = np.random.rand(nsims, ntrials, nsources, nbins)
    print("GPU_test_data.shape", GPU_data.shape)
    pcis, lzcs = run_pci_analy(GPU_data[:,0], nsims, ntrials, nsources, nbins, t_stim, nshuffles, percentile, stonset, testtoPY=False)

    print("pcis", pcis)