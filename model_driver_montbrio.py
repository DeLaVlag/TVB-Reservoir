from __future__ import print_function

import logging

# necessary set since I broke something
import os

import numpy as np

# os.environ["PATH"] += ":/usr/local/cuda-10.2/bin"
# print(os.environ["PATH"])


from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
# self.my_rank = comm.Get_rank()
# world_size = comm.Get_size()
# if self.my_rank == 0:
#   print("world_size", world_size)

import argparse

from tvb.simulator.lab import *
# from tvb.basic.logger.builder import get_logger

# from tvb.rateML.run.regular_run import regularRun

import os.path
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import time
import tqdm

from csr.gpu.consciousA_GPU import run_pci_analy
# from csr.analysis.conscious_analysis import computeDFA_nolds, lyapunov_ex, computeDFA
from csr.tasks.lorentz import *
# from csr.loss_function import *
# from csr.gpu_lossf import sliding_window_best_fit_loss_pycuda
# from csr.analysis.power10 import get_top_nodes_by_power
from csr.plottools.plotting_multiple import *
# from csr.gpu_fft import compare_fft_sims_inparallel
# from csr.gpu.TikhonovRR import make_prediction, Tikhonov_ridge_reg
from csr.gpu.torch_Tik import *
# from csr.gpu.gpu_TIK_simpler import gpu_TIK_simp
from csr.gpu.gpu_dfa import computeDFA_gpu
from csr.gpu.gpu_lya import computeLYA_gpu
# from csr.gpu.gpu_tikrr import compute_Tik_gpu, testRonGPU
from csr.plottools.heatmaps import *
from csr.gpu.create_task import *
# from csr.gpu.encoding_mechs import *
from csr.gpu.upsamples import *
from csr.plottools.plot_3d_dfapcilya import *

# pshift approximation functions
# from csr.gpu.pshift_approx import *


# for connectome analysis
from collections import Counter

here = os.path.dirname(os.path.abspath(__file__))

class Driver_Setup:

  def get_logger_o(self, loggername, log_file='default.log'):
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
      logger.addHandler(file_handler)

    return logger

  def __init__(self):
    self.args = self.parse_args()

    self.logger = self.get_logger_o('tvb.reservoir')
    self.logger.setLevel(level='INFO' if self.args.verbose else 'WARNING')

    self.my_rank = comm.Get_rank()
    self.world_size = comm.Get_size()
    if self.my_rank == 0:
      self.logger.info("world_size %d", self.world_size)

    self.checkargbounds()

    self.dt = self.args.delta_time
    self.connectivity = self.tvb_connectivity(self.args.n_regions)
    self.weights = self.connectivity.weights / (np.sum(self.connectivity.weights, axis=0) + 1e-12)

    # selected_regions = [10, 17, 30]
    # np.save("data/W_out_original.npy", self.weights[:, selected_regions])  # Shape becomes (96, 3)

    spectral_radius = .95
    self.weights *= spectral_radius / max(abs(np.linalg.eigvals(self.weights)))

    self.lengths = self.connectivity.tract_lengths
    # self.lengths = np.ones((68, 68))
    # self.weights = np.ones((68, 68))

    analyse_lengths = 0
    if analyse_lengths == 1:
      flat_lengths = self.lengths.flatten()
      lengths_counts = Counter(flat_lengths)

      # Print results
      print("Value Counts:")
      for value, count in sorted(lengths_counts.items()):
        print(f"Value {value}: {count} times")

    self.tavg_period = .1
    self.n_inner_steps = int(self.tavg_period / self.dt)

    self.params, self.wi_per_rank, self.params_toplot, self.s0, self.s1 = self.setup_params(
    self.args.n_sweep_arg0,
    self.args.n_sweep_arg1,
    self.args.n_sweep_arg2,
    )

    self.logger.info("params.shape at %s MPI rank %d", self.params.shape, self.my_rank)

    # bufferlength is based on the minimum of the first swept parameter (speed for many tvb models)
    self.n_work_items, self.n_params = self.params.shape
    # copy ntimes along parameter dimension
    new_weights = np.tile(self.weights, (self.n_work_items, 1, 1))
    self.weights = new_weights

    self.buf_len_ = ((self.lengths / self.args.speeds_min / self.dt).astype('i').max() + 1)
    self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of

    self.buf_len = int(32768 / 2)

    self.states = self.args.states
    self.exposures = self.args.exposures

    self.trainingloops = self.args.trainingloops
    self.tlsamples = self.args.tlsamples

    if self.args.gpu_info:
      self.logger.setLevel(level='INFO')
      self.gpu_device_info()
      exit(1)

    if self.my_rank == 0:
      self.logdata()

    self.beta_ridge = self.args.betaridge

  def logdata(self):

    self.logger.info('dt %f', self.dt)
    self.logger.info('n_nodes %d', self.args.n_regions)
    self.logger.info('weights.shape %s', self.weights.shape)
    self.logger.info('lengths.shape %s', self.lengths.shape)
    self.logger.info('tavg period %s', self.tavg_period)
    self.logger.info('n_inner_steps %s', self.n_inner_steps)
    self.logger.info('params shape %s', self.params.shape)

    self.logger.info('nstep %d', self.args.n_time)
    self.logger.info('n_inner_steps %f', self.n_inner_steps)

    # self.logger.info('single connectome, %d x %d parameter space', self.args.n_sweep_arg0, self.args.n_sweep_arg1)
    self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
    self.logger.info('number of states %d', self.states)
    self.logger.info('model %s', self.args.model)
    self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
    self.logger.info('memory for states array on GPU %d MiB',
             (self.buf_len * self.n_work_items * self.states * self.args.n_regions * 4) / 1024 ** 2)

  def checkargbounds(self):

    try:
      assert self.args.n_sweep_arg0 > 0, "Min value for [N_SWEEP_ARG0] is 1"
      assert self.args.n_time > 0, "Minimum number for [-n N_TIME] is 1"
      assert self.args.n_regions > 0, "Min value for  [-tvbn n_regions] for default data set is 68"
      assert self.args.blockszx > 0 and self.args.blockszx <= 32, "Bounds for [-bx BLOCKSZX] are 0 < value <= 32"
      assert self.args.blockszy > 0 and self.args.blockszy <= 32, "Bounds for [-by BLOCKSZY] are 0 < value <= 32"
      assert self.args.delta_time > 0.0, "Min value for [-dt delta_time] is > 0.0, default is 0.1"
      assert self.args.speeds_min > 0.0, "Min value for [-sm speeds_min] is > 0.0, default is 3e-3"
      assert self.args.exposures > 0, "Min value for [-x exposures] is 1"
      assert self.args.states > 0, "Min value for [-s states] is 1"
    except AssertionError as e:
      self.logger.error('%s', e)
      raise

  def tvb_connectivity(self, tvbnodes):
    # white_matter = connectivity.Connectivity.from_file(source_file="connectivity_"+str(tvbnodes)+".zip")
    #sfile = "./data/connectivity_zerlaut_68_newcentres.zip"
    # sfile = "/home/michiel/Documents/Repos/LiquidInterferenceLearning/csr/gpu/data/connectivity_zerlaut_68_newcentres.zip"
    # sfile = "./data/connectivity_68.zip"
    # sfile = "./data/connectivity_96.zip"
    sfile = "connectivity_96.zip"
    # sfile = "./data/connectivity_192.zip"
    # sfile = "./csr/data/connectivity_zerlaut_68_newcentres.zip" #hpc

    # sfile = "/home/michiel/Documents/Repos/LiquidInterferenceLearning/data/connectivity_"+str(tvbnodes)+".zip"
    white_matter = connectivity.Connectivity.from_file(source_file=sfile)
    white_matter.weights = white_matter.weights/np.max(white_matter.weights)
    white_matter.configure()
    return white_matter

  def parse_args(self):  # {{{
    parser = argparse.ArgumentParser(description='Run parameter sweep.')

    # for every parameter that needs to be swept, the size can be set
    parser.add_argument('-s0', '--n_sweep_arg0', default=1, help='num grid points for 1st parameter', type=int)
    parser.add_argument('-s1', '--n_sweep_arg1', default=1, help='num grid points for 2st parameter', type=int)
    parser.add_argument('-s2', '--n_sweep_arg2', default=2, help='num grid points for 3st parameter', type=int)
    parser.add_argument('-s3', '--n_sweep_arg3', default=2, help='num grid points for 3st parameter', type=int)
    parser.add_argument('-s4', '--n_sweep_arg4', default=1, help='num grid points for 3st parameter', type=int)
    parser.add_argument('-s5', '--n_sweep_arg5', default=1, help='num grid points for 3st parameter', type=int)
    parser.add_argument('-s6', '--n_sweep_arg6', default=1, help='num grid points for 3st parameter', type=int)
    parser.add_argument('-n', '--n_time', default=625, help='number of time steps to do', type=int)
    parser.add_argument('-v', '--verbose', default=True, help='increase logging verbosity', action='store_true')
    parser.add_argument('-m', '--model', default='montbrio_heun', help="neural mass model to be used during the simulation")
    parser.add_argument('-s', '--states', default=2, type=int, help="number of states for model")
    parser.add_argument('-x', '--exposures', default=2, type=int, help="number of exposures for model")
    parser.add_argument('-l', '--lineinfo', default=False, help='generate line-number information for device code.', action='store_true')
    parser.add_argument('-bx', '--blockszx', default=32, type=int, help="gpu block size x")
    parser.add_argument('-by', '--blockszy', default=32, type=int, help="gpu block size y")
    parser.add_argument('-val', '--validate', default=False, help="enable validation with refmodels", action='store_true')
    parser.add_argument('-r', '--n_regions', default="96", type=int, help="number of tvb nodes")
    parser.add_argument('-p', '--plot_data', type=int, help="plot res data for selected state")
    parser.add_argument('-w', '--write_data', default=False, help="write output data to file: 'tavg_data", action='store_true')
    parser.add_argument('-g', '--gpu_info', default=False, help="show gpu info", action='store_true')
    parser.add_argument('-dt', '--delta_time', default=.01, type=float, help="dt for simulation")
    parser.add_argument('-cs', '--conduct_speed', default=3, type=float, help="set conduction speed for temporal buffer")
    parser.add_argument('-sm', '--speeds_min', default=1, type=float, help="min speed for temporal buffer")
    parser.add_argument('-tl', '--trainingloops', default=1, type=int, help="number of loops for training")
    parser.add_argument('-tls', '--tlsamples', default=1, type=int, help="number of loops for training")
    parser.add_argument('-br', '--betaridge', default=2.5e-4, type=float, help="number of loops for training") # 3e-1 fpr longer runs to cancel the drift

    args = parser.parse_args()
    return args


  def setup_params(self,
    n0,
    n1,
    n2
    ):
    '''
    This code generates the parameters ranges that need to be set
    '''
    # s0 = np.linspace(.4, .63, n0)
    # s0 = np.linspace(.33, .495, n0)
    # # s0 = np.linspace(2., 4., n0)
    # # sweeparam1 = np.linspace(1, 4, n1) #speed
    # s1 = np.linspace(0.03, 0.05, n1) #speed
    # s2 = np.linspace(1, 9, n2)
    # s2 = np.linspace(1, 9, n3)
    # s2 = np.linspace(1, 9, n4)
    # s2 = np.linspace(1, 9, n5)
    # s2 = np.linspace(1, 9, n6)

    # testing
    # slh = [
    #        # .33, .495,  # global coupling .33, .495,
    #        .6, .6,  # global coupling .33, .495, or fix at 1 for tests
    #        0.00, 0.000,  # weight_noise 0.03, 0.05. or fix to .01 for tests
    #        0, 0,  # External Current -.5, 4.0,
    #        .7, .7,  # Mean heterogeneous noise delta .5, .8,
    #        10, 10,  # Mean Synaptic weight J 12, 16,  #
    #        -4.6, -4.6,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
    #        4, 4.]  # global speed 1., 7.]  #

    # slh = [
    #        1, 30,  # 0.4 - 0.8 prev global coupling .33, .495,
    #        0.01, 0.06,  # weight_noise 0.03, 0.05. or fix to .01 for tests
    #        -5, -4,  # External Current -.5, 4.0,
    #        .07, .08,  # Mean heterogeneous noise delta .5, .8,  #
    #        1, 1,  # Mean Synaptic weight J 12, 16,  #
    #        -5, -5,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
    #        .5, 4.]  # global speed 1., 7.]  #

    # focus on good one
    # slh = [
    #        20.33, 20.33,  # 0.4 - 0.8 prev global coupling .33, .495,
    #        0.06, 0.06,  # weight_noise 0.03, 0.05. or fix to .01 for tests
    #        -5, -4,  # External Current -.5, 4.0,
    #        .07, .08,  # Mean heterogeneous noise delta .5, .8,  #
    #        1, 1,  # Mean Synaptic weight J 12, 16,  #
    #        -5, -5,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
    #        .5, 4.]

    # for massive everything, results in left corner
    # slh = [
    #        20.33, 20.33,  # 0.4 - 0.8 prev global coupling .33, .495,
    #        0.06, 0.06,  # weight_noise 0.03, 0.05. or fix to .01 for tests
    #        -5, 7.0,  # External Current -.5, 4.0,
    #        .1, 2.5,  # Mean heterogeneous noise delta .5, .8,  #
    #        1, 22,  # Mean Synaptic weight J 12, 16,  #
    #        -4, -4,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
    #        4., 4.]

    # zoom in on left corner old where eta and delta where mixed up
    # slh = [
    #        20.33, 20.33,  # 0.4 - 0.8 prev global coupling .33, .495,
    #        0.06, 0.06,  # weight_noise 0.03, 0.05. or fix to .01 for tests
    #        -11, 7.0,  # External Current -.5, 4.0,
    #        -3.2, 2.5,  # Mean heterogeneous noise delta .5, .8,  #
    #        -20, 22,  # Mean Synaptic weight J 12, 16,  #
    #        -4, -4,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
    #        4., 4.]

    # zoom in on left corner
    slh = [
           20, 20,  # 0.4 - 0.8 prev global coupling .33, .495,
           0.01, 0.01,  # weight_noise 0.03, 0.05. or fix to .01 for tests
           -20, 10,  # External Current -.5, 4.0,
           -6, 30,  # Mean Synaptic weight J 12, 16,  #
           -10, 10,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
           1.0, 1.0,  # Mean heterogeneous noise delta .5, .8,  #
           4., 4.]


    # slh = [
    #        20.33, 20.33,  # 0.4 - 0.8 prev global coupling .33, .495,
    #        0.06, 0.06,  # weight_noise 0.03, 0.05. or fix to .01 for tests
    #        -5, 0,  # External Current -.5, 4.0,
    #        .5, .8,  # Mean heterogeneous noise delta .5, .8,  #
    #        12, 16,  # Mean Synaptic weight J 12, 16,  #
    #        -5, -5,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
    #        .5, 4.]

    s0 = np.linspace(slh[0], slh[1], self.args.n_sweep_arg0)  # coupling
    s1 = np.linspace(slh[2], slh[3], self.args.n_sweep_arg1)  # b_e
    s2 = np.linspace(slh[4], slh[5], self.args.n_sweep_arg2)  # weight_noise
    s3 = np.linspace(slh[6], slh[7], self.args.n_sweep_arg3)  # global_speed
    s4 = np.linspace(slh[8], slh[9], self.args.n_sweep_arg4)  # tau_w_e
    s5 = np.linspace(slh[10], slh[11], self.args.n_sweep_arg5)  # a_e
    s6 = np.linspace(slh[12], slh[13], self.args.n_sweep_arg5)  # global speed

    params = itertools.product(s0, s1, s2, s3, s4, s5, s6)
    params = np.array([vals for vals in params], np.float32)

    # if self.my_rank == 0:
    #   print("params_insp.shape", params.shape)
    #   print("params_insp.shape", params)

    # for plotting
    params_toplot = itertools.product(s0, s1)
    params_toplot = np.array([vals for vals in params_toplot], np.float32)

    # mpi stuff
    # print(params.shape)
    wi_total, wi_one_size = params.shape
    wi_per_rank = int(wi_total / self.world_size)
    wi_remaining = wi_total % self.world_size
    # ValueError: cannot reshape array of size 14680064 into shape (24,87381,7)
    params_pr = params.reshape((self.world_size, wi_per_rank, wi_one_size))

    if self.my_rank == 0:
      self.logger.info('remaining wi %f', wi_remaining)
      self.logger.info('params shape per world %s', params.shape)
      self.logger.info('params shape per rank %s', params_pr.shape)

    return params_pr[self.my_rank, :, :].squeeze(), wi_per_rank, params_toplot, s0, s1


  def gpu_device_info(self):
    '''
    Get GPU device information
    TODO use this information to give user GPU setting suggestions
    '''
    dev = drv.Device(0)
    print('\n')
    self.logger.info('GPU = %s', dev.name())
    self.logger.info('TOTAL AVAIL MEMORY: %d MiB', dev.total_memory()/1024/1024)

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
      getstring = 'drv.device_attribute.' + key
      # att[key].append(eval(getstring))
      self.logger.info(key + ': %s', dev.get_attribute(eval(getstring)))

class Driver_Execute(Driver_Setup):

  def __init__(self, ds):
    self.args = ds.args
    self.set_CUDAmodel_dir()
    self.weights, self.lengths, self.params = ds.weights, ds.lengths, ds.params
    self.buf_len, self.states, self.n_work_items = ds.buf_len, ds.states, ds.n_work_items
    self.n_inner_steps, self.n_params, self.dt = ds.n_inner_steps, ds.n_params, ds.dt
    self.exposures, self.logger = ds.exposures, ds.logger
    self.conduct_speed = ds.args.conduct_speed
    self.params_toplot = ds.params_toplot
    self.s0 = ds.s0
    self.trainingloops = ds.trainingloops
    self.tlsamples = ds.tlsamples
    self.beta_ridge = ds.beta_ridge
    self.my_rank = ds.my_rank
    self.world_size = ds.world_size

  def set_CUDAmodel_dir(self):
    # print(os.path.abspath(__file__))
    self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
                 "/models", self.args.model.lower() + '.c')
    self.args.filename = here + self.args.filename

  def set_CUDA_ref_model_dir(self):
    self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
                 "../generatedModels/cuda_refs", self.args.model.lower() + '.c')


  def make_kernel(self, source_file, warp_size, args, lineinfo=False, nh='nh'):

    try:
      with open(source_file, 'r') as fd:
        source = fd.read()
        source = source.replace('pi', '%ff' % (np.pi, ))
        source = source.replace('inf', 'INFINITY')
        source = source.replace('M_PI_F', '%ff' % (np.pi,))
        # opts = ['--ptxas-options=-v', '-maxrregcount=32']
        opts = ['-maxrregcount=32']
        if lineinfo:
          opts.append('-lineinfo')
        opts.append('-DWARP_SIZE=%d' % (warp_size, ))
        opts.append('-DNH=%s' % (nh, ))

        idirs = [here]
        if self.my_rank == 0:
          self.logger.info('nvcc options %r', opts)

        try:
          network_module = SourceModule(
              source, options=opts, include_dirs=idirs,
              no_extern_c=True,
              keep=False,)
        except drv.CompileError as e:
          self.logger.error('Compilation failure \n %s', e)
          exit(1)

        # generic func signature creation
        # mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model, 'jjjjjfPfS_S_S_S_')
        mod_func = '_Z13montbrio_heunjjjjjffiiiiiPfS_S_S_S_S_S_S_S_S_S_S_'

        step_fn = network_module.get_function(mod_func)

    except FileNotFoundError as e:
      self.logger.error('%s.\n  Generated model filename should match model on cmdline', e)
      exit(1)

    return step_fn #}}}

  def cf(self, array):#{{{
    # coerce possibly mixed-stride, double precision array to C-order single precision
    return array.astype(dtype='f', order='C', copy=True)#}}}

  def nbytes(self, data):#{{{
    # count total bytes used in all data arrays
    nbytes = 0
    for name, array in data.items():
      nbytes += array.nbytes
    return nbytes#}}}

  def make_gpu_data(self, data):#{{{
    # put data onto gpu
    gpu_data = {}
    for name, array in data.items():
      try:
        gpu_data[name] = gpuarray.to_gpu(self.cf(array))
      except drv.MemoryError as e:
        self.gpu_mem_info()
        self.logger.error(
          '%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU',
          e, self.params.size)
        exit(1)
    return gpu_data#}}}

  def release_gpumem(self, gpu_data):
    for name, array in gpu_data.items():
      try:
        gpu_data[name].gpudata.free()
      except drv.MemoryError as e:
        self.logger.error('%s.\n\t Freeing mem error', e)
        exit(1)

  def gpu_mem_info(self):

    import subprocess
    import re

    cmd = "nvidia-smi -q -d MEMORY"#,UTILIZATION"
    # os.system(cmd)  # print all mem info

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Extract "FB Memory Usage" section
    fb_mem_section = re.search(r"FB Memory Usage([\s\S]*?)BAR1 Memory Usage", result.stdout)

    if fb_mem_section:
      # Extract memory values using regex
      values = re.findall(r":\s+(\d+)\s+MiB", fb_mem_section.group(1))

      if values:
        print("\n")
        self.logger.info("GPU MEMORY stats")
        self.logger.info("    Total: %d MiB", int(values[0]))
        self.logger.info("    Reserved: %d MiB", int(values[1]))
        self.logger.info("    Used: %d MiB", int(values[2]))
        self.logger.info("    Free: %d MiB", int(values[3]))
        print("\n")
      else:
        self.logger.info("Could not extract memory values.")
    else:
      self.logger.info("Could not find FB Memory Usage section.")


  def run_tvb_rc(self, target_dyn, tearchr_dyn, inject_dyn, inject_dyn_pred, W_out, statesbuf, usebuffer):

    x = target_dyn[0, :]
    y = target_dyn[1, :]
    z = target_dyn[2, :]

    xt = tearchr_dyn[0, :]
    yt = tearchr_dyn[1, :]
    zt = tearchr_dyn[2, :]
    # zt = yt

    # spawn of the chunked prediction to inject datastruct
    learned_thoughts = np.zeros((32, 100, 3))

    # assert (x == y).all() and (y == z).all()

    # for j in range(self.n_work_items):
    #     for i in range(62):
    #       print(self.weights[j, i, 23])

    # print('self.weights', self.weights)
    # setup data#{{{
    # usebuffer is for using the buffer such that the tvb reservoir does not have a cold start
    # however it is now bigger and also seperates training vs prediction phase. todo change usebuffer usage
    if usebuffer == False:
      if self.my_rank == 0:
        self.logger.info("usebuffer %s", usebuffer)
      buffoffzet = 0
      data = { 'weights': self.weights, 'lengths': self.lengths, 'params': self.params.T,
           'x': x, 'y': y, 'z': z, 'xt': xt, 'yt': yt, 'zt': zt,
              'thoughts': learned_thoughts}
      base_shape = self.n_work_items,
      for name, shape in dict(
        tavg0=(self.exposures, self.args.n_regions,),
        tavg1=(self.exposures, self.args.n_regions,),
        state=(self.buf_len, self.states * self.args.n_regions),
        ).items():
        # memory error exception for compute device
        try:
          data[name] = np.zeros(shape + base_shape, 'f')
        except MemoryError as e:
          self.logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large '
                 'for this compute device',
                 e, self.args.n_sweep_arg0, self.args.n_sweep_arg1)
          exit(1)
    else:
      if self.my_rank == 0:
        self.logger.info("usebuffer %s", usebuffer)

      new_timesteps = target_dyn.shape[1] // self.n_inner_steps
      # Reshape the time series for downsampling and compute the mean along the new axis
      YchunkTik = tearchr_dyn.reshape(tearchr_dyn.shape[0], new_timesteps, self.n_inner_steps).mean(axis=2)
      YchunkTik = np.tile(YchunkTik[np.newaxis, :, :], (self.n_work_items, 1, 1))
      YchunkTik = YchunkTik.transpose(0, 2, 1)

      Ypast = target_dyn.reshape(target_dyn.shape[0], new_timesteps, self.n_inner_steps).mean(axis=2)
      Ypast = np.tile(Ypast[np.newaxis, :, :], (self.n_work_items, 1, 1))
      Ypast = Ypast.transpose(0, 2, 1)


      buffoffzet = self.args.n_time * self.n_inner_steps
      data = {'weights': self.weights, 'lengths': self.lengths, 'params': self.params.T,
              'x': x, 'y': y, 'z': z, 'xt': xt, 'yt': yt, 'zt': zt, 'state': statesbuf,
              'thoughts': learned_thoughts}
      base_shape = self.n_work_items,
      for name, shape in dict(
              tavg0=(self.exposures, self.args.n_regions,),
              tavg1=(self.exposures, self.args.n_regions,),
      ).items():
        # memory error exception for compute device
        try:
          data[name] = np.zeros(shape + base_shape, 'f')
        except MemoryError as e:
          self.logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large '
                            'for this compute device',
                            e, self.args.n_sweep_arg0, self.args.n_sweep_arg1)
          exit(1)

    gpu_data = self.make_gpu_data(data)#{{{
    if self.my_rank == 0:
      self.logger.info("gpu_data['xt'].shape %s", gpu_data['xt'].shape)
      self.logger.info("gpu_data['yt'].shape %s", gpu_data['yt'].shape)
      self.logger.info("gpu_data['zt'].shape %s", gpu_data['zt'].shape)

    # setup CUDA stuff#{{{
    step_fn = self.make_kernel(
      source_file=self.args.filename,
      warp_size=32,
      # block_dim_x=self.args.n_sweep_arg0,
      # ext_options=preproccesor_defines,
      # caching=args.caching,
      args=self.args,
      lineinfo=self.args.lineinfo,
      nh=self.buf_len,
      )#}}}

    # setup simulation#{{{
    tic = time.time()

    n_streams = 32
    streams = [drv.Stream() for i in range(n_streams)]
    events = [drv.Event() for i in range(n_streams)]
    tavg_unpinned = []
    stats_unpinned = []

    try:
      tavg = drv.pagelocked_zeros((n_streams,) + data['tavg0'].shape, dtype=np.float32)
      states = drv.pagelocked_zeros((n_streams,) + data['state'].shape, dtype=np.float32)
      # print('states.shape', states.shape)
    except drv.MemoryError as e:
      self.logger.error(
        '%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU',
        e, self.params.size)
      exit(1)

    # determine optimal grid recursively
    def dog(fgd):
      maxgd, mingd = max(fgd), min(fgd)
      maxpos = fgd.index(max(fgd))
      if (maxgd - 1) * mingd * bx * by >= nwi:
        fgd[maxpos] = fgd[maxpos] - 1
        dog(fgd)
      else:
        return fgd

    # n_sweep_arg0 scales griddim.x, n_sweep_arg1 scales griddim.y
    # form an optimal grid recursively
    bx, by = self.args.blockszx, self.args.blockszy
    nwi = self.n_work_items
    rootnwi = int(np.ceil(np.sqrt(nwi)))
    gridx = int(np.ceil(rootnwi / bx))
    gridy = int(np.ceil(rootnwi / by))

    final_block_dim = bx, by, 1

    fgd = [gridx, gridy]
    dog(fgd)
    final_grid_dim = fgd[0], fgd[1]

    assert gridx * gridy * bx * by >= nwi

    if self.my_rank == 0:
      self.logger.info("final_block_dim %s", final_block_dim)
      self.logger.info("final_grid_dim %s", final_grid_dim)

      self.logger.info('history shape %r', gpu_data['state'].shape)
      self.logger.info('gpu_data %s', gpu_data['tavg0'].shape)
      self.logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))
      self.logger.info('final block dim %r', final_block_dim)
      self.logger.info('final grid dim %r', final_grid_dim)

    # run simulation#{{{
    nstep = self.args.n_time

    if self.my_rank == 0:
      self.gpu_mem_info() if self.args.verbose else None

    # thoughtsize = self.delay_timesteps # is delay between teacher and input
    thoughtsize = 20 # is delay between teacher and input
    goforit = 0

    if self.my_rank == 0:
      tqdm_iterator = tqdm.trange(nstep, file=sys.stdout)
    else:
      tqdm_iterator = range(nstep)

    try:
      for i in tqdm_iterator:

        try:
          event = events[i % n_streams]
          stream = streams[i % n_streams]

          if i > 0:
            stream.wait_for_event(events[(i - 1) % n_streams])

          step_fn(np.uintc(i * self.n_inner_steps), np.uintc(self.args.n_regions), np.uintc(self.buf_len),
              np.uintc(self.n_inner_steps), np.uintc(self.n_work_items),
              np.float32(self.dt), np.float32(self.conduct_speed), np.uintc(self.my_rank),
              np.uintc(inject_dyn), np.uintc(buffoffzet), np.uintc(thoughtsize * self.n_inner_steps),
              np.uintc(usebuffer),
              gpu_data['weights'], gpu_data['lengths'], gpu_data['params'],
              gpu_data['state'],
              gpu_data['tavg%d' % (i % 2,)],
              gpu_data['x'],
              gpu_data['y'],
              gpu_data['z'],
              gpu_data['xt'],
              gpu_data['yt'],
              gpu_data['zt'],
              gpu_data['thoughts'],
              block=final_block_dim, grid=final_grid_dim)

          event.record(streams[i % n_streams])
        except drv.LaunchError as e:
          self.logger.error('%s', e)
          exit(1)

        goforit = 0

        tavgk = 'tavg%d' % ((i + 1) % 2,)

        # async wrt. other streams & host, but not this stream.
        if i >= n_streams:
        # if i >= 0:
          stream.synchronize()
          tavg_unpinned.append(tavg[i % n_streams].copy())
          # stats_unpinned.append(states[i % n_streams].copy())

        drv.memcpy_dtoh_async(tavg[i % n_streams], gpu_data[tavgk].ptr, stream=stream)

        # do predcition here for feedbback in the model
        GO = True
        if usebuffer == True and i>int(inject_dyn_pred/self.n_inner_steps) and i % thoughtsize == 0 and GO:
          # print("tavg_unpinned.shape", np.array(tavg_unpinned).shape)
          # print(i)
          tavgpartly = np.array(tavg_unpinned)
          # print(tavgpartly.shape)
          # subtract nstreams of the index as this is copied in later. we dont need to correct as we dont use the
          # first 32 results anyway. the index of tavg is nstreams down on the current
          h = i - thoughtsize - n_streams
          j = h + thoughtsize
          Rpart = tavgpartly[h:j,0,:,:]
          # print('Rpart.shape', Rpart.shape)
          # do partly ridging
          # shapes
          # R (32, 68, 500)
          # Y (32, 500, 3)
          Rchunktik = Rpart.transpose(2,1,0)


          # this one isnt troubled by the streams so correction
          # k = i - thoughtsize # this is fitting for the past
          k = i + thoughtsize # fitting for the future
          # YchunkTikpart = YchunkTik[:,k:i,:]
          if k <=YchunkTik.shape[1]:
            YchunkTikpart = YchunkTik[:,i:k,:]
          # expand for torchs batch matmul
            part_wout = Tik_torch(Rchunktik, YchunkTikpart, 1e-4)

          # do partly predicitions
          Rpartpred = Rpart.transpose(2, 0, 1)
          pred_partly = predi_Torch(Rpartpred, part_wout)
          # pred_partly = predi_Torch(Rpartpred, W_out)

          upsamplemethod = 1
          if upsamplemethod == 0:
            # simple repeating upsample to match GPU time, (8, 200, 3)
            pred_partly_r = np.repeat(pred_partly, self.n_inner_steps, axis=1)
            # print("pred_partly", pred_partly_r.shape)
          elif upsamplemethod == 1:
            pred_partly_r = spline_upsample(pred_partly, self.n_inner_steps)
          elif upsamplemethod == 2:
            pred_partly_r = fft_upsample(pred_partly, self.n_inner_steps)


          # 0 minmax tavg, 1 zscore, 2 normalnorm
          normfunc=2
          if normfunc==0:
            min_tavgpartly = np.min(tavgpartly)
            max_tavgpartly = np.max(tavgpartly)

            normalized_pred_partly = (pred_partly_r - np.min(pred_partly_r)) / (np.max(pred_partly_r) - np.min(pred_partly_r)) * (
                      max_tavgpartly - min_tavgpartly) + min_tavgpartly
          elif normfunc==1:
            mean_vals = np.mean(pred_partly_r, axis=(1, 2), keepdims=True)
            std_vals = np.std(pred_partly_r, axis=(1, 2), keepdims=True)
            normalized_pred_partly = (pred_partly_r - mean_vals) / std_vals
          elif normfunc==2:
            min_vals = np.min(pred_partly_r, axis=(1, 2), keepdims=True)
            max_vals = np.max(pred_partly_r, axis=(1, 2), keepdims=True)

            normalized_pred_partly = (pred_partly_r - min_vals) / (max_vals - min_vals)

          if i == 180 or i == 220 or i == 260 or i == 400:
          #   # plotpred( YchunkTik[:,k:i,:], YchunkTik[:,k:i,:], self.trainingloops, f"Full625{i}")
          #   # plotpred( YchunkTikpart, pred_partly, self.trainingloops, f"ChunkyPred pos{i}")
          #   print("wehere??!!!!!!!!!!!!!!")
            # for ICANN25 paper:
            deltapredchunk = -40
            plotpred_teach_train( Ypast[:,i-40:i+40], pred_partly, YchunkTik[:,i-40:i+40,:], self.trainingloops,
                                  "", deltapredchunk)
            # plot_upsampled(pred_partly, pred_partly_r, f"Spline upsample for pos {i}", n_samples=1)

          # 32,100,3
          # testdata = np.stack([
          #   np.ones((32, 200)),
          #   np.full((32, 200), 2),
          #   np.full((32, 200), 3) ], axis=-1)
          #
          # togpuChunk = testdata.transpose(0,2,1) # 32,3,100
          # togpuChunk = normalized_pred_partly.transpose(0,2,1)
          togpuChunk = pred_partly_r.transpose(0,2,1)
          # print(togpuChunk[4,0,:])
          # togpuChunk = pred_partly_r

          # THE SHAPE THE DATA NEEDS TO BE = (nsims, 3, 200)!!!
          # gpu_data['thoughts'] = gpuarray.to_gpu(self.cf(normalized_pred_partly*1))
          # gpu_data['thoughts'] = gpuarray.to_gpu(self.cf(togpuChunk*1))

          # only amplify the 3rd dim of lorentz by
          scalar = 3.5
          last_result = togpuChunk[:, -1, :]
          scaled_result = last_result * scalar
          togpuChunk[:, -1, :] = scaled_result

          gpu_data['thoughts'] = gpuarray.to_gpu(self.cf(togpuChunk*self.scale_input))
          # gpu_data['thoughts'] = gpuarray.to_gpu(self.cf(testdata))
          # print(pred_partly_r.shape)
          goforit = 1
          # gpu_data['thoughts'] = gpuarray.to_gpu(self.cf(normalized_pred_partly *.5))

          # print("normalized_pred_partly.shape", normalized_pred_partly.transpose(1,0,2).shape)


        if i == nstep - 1:
          states = gpu_data['state'].get()

      # drv.memcpy_dtoh_async(states, gpu_data['state'].ptr, stream=stream)
      # drv.memcpy_dtoh(states, gpu_data['state'].ptr)

      # recover uncopied data from pinned buffer
      if nstep > n_streams:
        for i in range(nstep % n_streams, n_streams):
          stream.synchronize()
          tavg_unpinned.append(tavg[i].copy())
          # print("tavg_unpinned1.shape", np.array(tavg_unpinned).shape)
          # stats_unpinned.append(states[i].copy())

      for i in range(nstep % n_streams):
        stream.synchronize()
        tavg_unpinned.append(tavg[i].copy())
        # print("tavg_unpinned2.shape", np.array(tavg_unpinned).shape)
        # stats_unpinned.append(states[i].copy())

    except drv.LogicError as e:
      self.logger.error('%s. Check the number of states of the model or '
             'GPU block shape settings blockdim.x/y %r, griddim %r.',
             e, final_block_dim, final_grid_dim)
      exit(1)
    except drv.RuntimeError as e:
      self.logger.error('%s', e)
      exit(1)

    # drv.memcpy_dtoh_async(states, gpu_data['state'].ptr, stream=stream)

    # self.logger.info('kernel finish..')
    # release pinned memory
    tavg = np.array(tavg_unpinned)
    # states = np.array(stats_unpinned)
    # states = np.array(states)

    # also release gpu_data
    self.release_gpumem(gpu_data)

    if self.my_rank == 0:

      self.logger.info('kernel finished')
      # self.logger.info('kernel finished')

    # states = 0
    return tavg, states


  def create_task_dyn(self, phase_shift_pis=2, h=0.01, nonzeroffset=0, Sinussus=False):

    np.random.seed(42)

    if Sinussus==False:
      # x0 = np.random.uniform(-20, 40)
      # y0 = np.random.uniform(-25, 50)
      # z0 = np.random.uniform(0, 50)

      # x0 = np.random.uniform(0, 1)
      # y0 = np.random.uniform(0, 1)
      # z0 = np.random.uniform(0, 1)

      # val = np.random.uniform(0, 1)
      #
      # x0 = val
      # y0 = val
      # z0 = val

      x0 = 1
      y0 = 1
      z0 = 1

      # h = 0.005
      x, y, z = forward_euler(x0, y0, z0, h, self.args.n_time * self.n_inner_steps)
      # x, y, z = forward_euler(x0, y0, z0, h, 6250)

      x, y, z = normalize_lorenz_signals(x, y, z)
      fixed_pred_set = np.vstack((x, y, z))
      # teach_pred_set = np.vstack((x, y, z))
      teach_pred_set = np.vstack(shift_lorenz_trajectory(x, y, z, phase_shift_pis=phase_shift_pis, h=h))

    else:

      fixed_pred_set = np.vstack(generate_sinusoidal_signals(time_steps=25000))
      teach_pred_set = np.vstack(generate_sinusoidal_signals(time_steps=25000, pshifts=phase_shift_pis))

    # add the offset for nonzero, non1 and nonegative input
    # fixed_pred_set *= .75
    # teach_pred_set *= .75

    fixed_pred_set += nonzeroffset
    teach_pred_set += nonzeroffset

    return fixed_pred_set, teach_pred_set


  # def Tikhonov_ridge_reg(self, R, Y, beta_ridge):
  #
  #   '''
  #   Tikhonov ridge regression (also known as ridge regression)
  #   Wout=YR^T(RR^T+βI)^−1
  #
  #   Where:
  #
  #     Y is the matrix of target outputs (lorentz)
  #     R is the matrix of reservoir states
  #     β is the regularization parameter
  #     I is the identity matrix
  #
  #   '''
  #
  #   from numpy.linalg import inv
  #
  #   # Compute ridge regression to find W_out
  #   I = np.identity(R.shape[1])  # Identity matrix of size equal to the number of reservoir states
  #   print('R.shape', R.shape)
  #   print('Y.shape', Y.shape)
  #
  #   # Y = Y[:, 50:250] * 1e-6
  #   # Y = Y* 2e-4
  #   W_out = []
  #   for i in range(self.n_work_items):
  #
  #     Ri = R[:, :, i] # shape (n=200x m=68)
  #     R_T = Ri.T      # shape (68x200)
  #
  #     R_T_R = R_T @ Ri # shape (68x68)
  #
  #     # Compute the regularized inverse
  #     # controls the amount of regularization. Adding this regularization term helps to prevent overfitting,
  #     # especially when RTRRTR is close to singular or poorly conditioned.
  #     # The penalty βridge∣∣W∣∣2βridge​∣∣W∣∣2 discourages large coefficients and helps in stabilizing
  #     # the inversion. The larger βridgeβridge​, the stronger the regularization, meaning the weights
  #     # Weights will be smaller.
  #     R_T_R_inv = inv(R_T_R + beta_ridge * I) # Ishape (68x68),
  #
  #     # Compute R^T * Y
  #     R_T_Y = R_T @ Y[i].T  # Shape [68, 3]
  #
  #     # Compute W_out
  #     W_out.append(R_T_R_inv @ R_T_Y)  # Shape [68, 3]
  #
  #   # print(W_out)
  #   W_out = np.array(W_out)
  #   print('W_out.shape', W_out.shape)
  #
  #   return W_out

  def update_output_weights(self, nWout, outregions, nodes_minus_inoutreg):

    '''
    set the weights in the output layer
    or set the weights of particular connections only

    Output regions:
    - Precentral Gyrus; region L: 23, R: 57
    - Paracentral Lobule; region L: 16, R: 50
    - Postcentral Gyrus; region L: 21, R: 55

    nWout[68,3] is the normalized version

    '''

    # recounter the couplingin the weights by predividing by the coupling
    for z in range(self.n_work_items):
    #  nWout[z, :, :] = nWout[z, :, :]/self.s0[z%self.args.n_sweep_arg0]
       nWout[z, :, :] = nWout[z, :, :]/np.max(nWout[z, :, :])

    # copy the normalized Wout to the output regions of TVB
    for j in range(self.n_work_items):
      for x, regio in enumerate(outregions):
        for i, conregion in enumerate(nodes_minus_inoutreg):
          # self.weights[j, conregion, regio] = self.weights[j, conregion, regio] * nWout[j,i,x] *1
          self.weights[j, conregion, regio] = nWout[j,i,x]
            # print(self.weights[j, i, regio])

    # print('nWout',nWout)
    # for j in range(self.n_work_items):
    #   for x, regio in enumerate(outregions):
    #     for i in range(62):
    #       if i not in outregions:
    #         print(self.weights[j, i, regio])

    # copy the normalized Wout to the output regions of TVB
    # interestingly monoweight dont reproduces same as heteroweigths
    # testweights = [.2, .3, .4, .5, .6, .7, .8, .9, 10]
    # for j in range(self.n_work_items):
    #   for x, regio in enumerate([23, 16, 21]):
    #     for i in range(68):
    #       self.weights[j, i, regio] = testweights[j]

    return self.weights

  def compute_pshift(self, new_weights):

    weights = np.maximum(new_weights, 0)
    weights = (weights + weights.T) / 2  # Make symmetric
    np.fill_diagonal(weights, 0)  # No self-loops

    # lengths = connectivity.tract_lengths
    lengths = self.lengths
    lengths = (lengths + lengths.T) / 2  # Make symmetric
    np.fill_diagonal(lengths, 0)  # No self-loops

    spd = shortest_path_delay(weights)
    scd = spectral_delay(weights)
    dfd = diffusion_delay(weights)

    # char_freq_weights = 10.0
    char_freq_weights = compute_characteristic_frequency(weights)
    # print("char_frequency_weights", char_freq_weights)

    char_freq_lenghts = compute_char_frequency_from_delays(lengths)
    # print("char_freq_delay", char_freq_lenghts)

    # 1.17 to set it gleich with the heuristic values for the 96 of 3.8
    char_frequency = blended_characteristic_frequency(char_freq_weights, char_freq_lenghts, alpha=1.17)
    # print("char_freq_blended", char_frequency)

    taus = (spd, scd, dfd)  # Your computed delay estimates

    tau_conn = compute_connection_delay(weights, lengths)  # Compute weighted connection delay
    phase_shift_pis, time_shift = compute_phase_shift_withdelay(taus, tau_conn, char_frequency)

    return phase_shift_pis

  # todo: for the future. every new wout has a new pshift corresponding.
  # todo: however this imposses that teacher and trainer get dimension of nsims
  # def compute_pshift_multi(self):
  #   nsims, nregions, _ = self.weights.shape
  #
  #   pshift_array = np.zeros(nsims)  # Initialize array to store phase shifts per simulation
  #
  #   for i in range(nsims):
  #     weights = np.maximum(self.weights[i], 0)
  #     weights = (weights + weights.T) / 2  # Make symmetric
  #     np.fill_diagonal(weights, 0)  # No self-loops
  #
  #     lengths = self.lengths[i]
  #     lengths = (lengths + lengths.T) / 2  # Make symmetric
  #     np.fill_diagonal(lengths, 0)  # No self-loops
  #
  #     spd = shortest_path_delay(weights)
  #     scd = spectral_delay(weights)
  #     dfd = diffusion_delay(weights)
  #
  #     char_freq_weights = compute_characteristic_frequency(weights)
  #     char_freq_lengths = compute_char_frequency_from_delays(lengths)
  #
  #     char_frequency = blended_characteristic_frequency(char_freq_weights, char_freq_lengths, alpha=1.17)
  #
  #     taus = (spd, scd, dfd)  # Your computed delay estimates
  #     tau_conn = compute_connection_delay(weights, lengths)  # Compute weighted connection delay
  #
  #     phase_shift_pis, _ = compute_phase_shift_withdelay(taus, tau_conn, char_frequency)
  #
  #     pshift_array[i] = phase_shift_pis  # Store result for each simulation
  #
  #   return pshift_array


  def run_pci_analy_gpu(self, tested_ts):

    # # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # # weightstim = np.logspace(-5, 0, 6)
    # # # Number of traces
    # # num_traces = tavg0.shape[2]
    # fig, ax = plt.subplots()
    # # for i in range(6):
    # for i in range(self.n_work_items):
    #   print(i)
    #   # color = colors[i % len(colors)]
    #   # clear_output(wait=True)
    #   plt.plot((tavg0[:, 0, :, i] ), 'k', color='b')
    #   plt.title(f'Plot no {i}')  # Set the title for the plot
    #   plt.xlabel('Time')  # Set the label for the x-axis
    #   plt.ylabel('Firing rate (KHz)')
    #   # plt.savefig(f'plot_weightstim{wght}.png')
    #   plt.show()

    '''testdata for means'''
    nworkitems = self.n_work_items
    # number of simulations/realisations to analyse for one PCI value/ do we need this as we have many trials in gpu threads
    # the trials stuff
    ntrials = 2
    nregions = self.args.n_regions
    nbins, t_stim = 250, 375
    nshuffles = 10
    #sets the position of the threshold. 1 is lowest in array of 10
    percentile = 1
    stonset = 0

    if self.my_rank == 0:
      print("\n")
      self.logger.info('PCI data shape %s', tested_ts.shape)
      self.logger.info('PCI ntrials %s', ntrials)
      self.logger.info('PCI nregions %s', nregions)
      self.logger.info('PCI nbins %s', nbins)
      self.logger.info('PCI t_stim %s', t_stim)
      self.logger.info('PCI nshuffles %s', nshuffles)
      self.logger.info('PCI percentile %s', percentile)
      self.logger.info('PCI stonset %s', stonset)

    pcis, lzcs = run_pci_analy(tested_ts, self.logger, self.my_rank, nworkitems, ntrials, nregions, nbins, t_stim, nshuffles, percentile, stonset, False)
    # print('lzcs.shape',np.average(lzcs, axis=1).shape)

    # min_window_size = 10  # Minimum window size
    # max_window_size = int(100 / 2)  # Maximum window size (half of the time series length)
    # window_sizes = np.unique(np.logspace(np.log10(min_window_size), np.log10(max_window_size), num=10, dtype=int))
    # window_sizes = [10, 20, 50, 100, 200]
    # print(window_sizes)
    # not on GPU yet
    # dfa = []
    # lya = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # lya = np.ones(self.params_toplot.shape[0])
    # dfa = np.ones(nworkitems)
    # lya = np.ones(nworkitems)
    # for i in range(nworkitems):
    #   dfa.append(computeDFA_nolds((tavg0[:,0,:,i]*1e-3)))
      # dfa.append(computeDFA((tested_ts[:250, 0, :, i] * 1e3), window_sizes))
      # lya.append(lyapunov_ex((tested_ts[:500, 0, :, i] * 1e3)))

    return pcis, lzcs


  def downsample(self, target_dyn_test):

    window_size = self.n_inner_steps
    # original window size is too small for range 200:2500 as it does not include all
    # window_size = 12

    if target_dyn_test.shape[1] % window_size != 0:
      raise ValueError("The number of timesteps must be divisible by the window size.")

    # Calculate the new number of timesteps after downsampling
    new_timesteps = target_dyn_test.shape[1] // window_size

    # Reshape the time series for downsampling and compute the mean along the new axis
    target_dyn_test = target_dyn_test.reshape(target_dyn_test.shape[0], new_timesteps, window_size).mean(axis=2)

    # print("target_dyn_test_shape", target_dyn_test.shape)

    plotdowndym = 0
    if plotdowndym:
      import matplotlib.pyplot as plt
      plt.figure(figsize=(15, 8))

      for i in range(3):
        plt.subplot(3, 1, i + 1)
        if i == 0:
          plt.plot(np.arange(0, 2000, 10), target_dyn_test[i]+10, label=f'Downsampled Series {i + 1}', linestyle='--',
            marker='o')
        if i == 1:
          plt.plot(np.arange(0, 2000, 10), target_dyn_test[i]+20, label=f'Downsampled Series {i + 1}', linestyle='--',
             marker='o')
        if i == 2:
          plt.plot(np.arange(0, 2000, 10), target_dyn_test[i], label=f'Downsampled Series {i + 1}', linestyle='--',
             marker='o')
        plt.title(f'Time Series {i + 1}')
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

      plt.show()

    return target_dyn_test

  def rescale(self, targetdyn, tavg, outregions):
    """Rescale targetdyn according to 0 < r < tavg.max as r is always positive."""

    scaled_target_all = []  # Store all scaled targets for each work item
    normalized_tsss = []  # Store all scaled targets for each work item

    for sim_idx in range(self.n_work_items):
      # scaled_target = []  # Temporary storage for each work item

      # for i in range(tavg.shape[1]):
      # for regnr, region_idx in enumerate(outregions):
      #   # Step 1: Get the minimum and maximum of targetdyn
      #   target_min = targetdyn[:, regnr].min()
      #   target_max = targetdyn[:, regnr].max()
      #
      #   tavg_min = tavg[:, region_idx, sim_idx].min()
      #   tavg_max = tavg[:, region_idx, sim_idx].max()
      #
      #   scaled_dyn = targetdyn[:, regnr]
      #
      #   scaled_dyn = (scaled_dyn - target_min) / (target_max - target_min) * (tavg_max - tavg_min) + tavg_min
      #
      #   # Normalizing it is too much?
      #   if target_max != target_min:
      #     scaled_dyn = (scaled_dyn - target_min) / (target_max - target_min)
      #   else:
      #     scaled_dyn = 0
      #
      #   scaled_target.append(scaled_dyn)

      # min_vals = targetdyn.min(axis=-1, keepdims=True)
      # max_vals = targetdyn.max(axis=-1, keepdims=True)
      #
      # # Avoid division by zero
      # range_vals = max_vals - min_vals
      # range_vals[range_vals == 0] = 1.0  # Set to 1 if no range (constant values)
      #
      # # Normalize Y
      # targetdyn = (targetdyn - min_vals) / range_vals

      # Normalize the whole TAVG
      # for allregidx in range(self.args.n_regions):
      tavg_min = tavg[:, :, sim_idx].min()
      tavg_max = tavg[:, :, sim_idx].max()
      ts_slice = tavg[:, :, sim_idx]

      if tavg_max != tavg_min:
        normalized_tss = (ts_slice - tavg_min) / (tavg_max - tavg_min)
        # print("normalized_tss.shape", np.array(normalized_tss).shape)
      else:
        normalized_tss = 0

      # scaled_target_all.append(np.array(targetdyn))
      # scaled_target_all.append(np.array(scaled_target))
      normalized_tsss.append(np.array(normalized_tss))  # From shape (16, 68, 200)

    # print("Type of normalized_tsss:", type(normalized_tsss))
    # print("Length of normalized_tsss:", len(normalized_tsss))
    # for i, item in enumerate(normalized_tsss):
    #   print(f"Shape of normalized_tsss[{i}]:", np.shape(item))

    global_min = targetdyn.min()
    global_max = targetdyn.max()
    targetdyn_norm = (targetdyn - global_min) / (global_max - global_min + 1e-6)

    plotdowndym = 0
    if plotdowndym:

      plt.figure(figsize=(16, 8))  # Define the figure size
      # Loop over each time series and plot it in a subplot
      for i in range(tavg.shape[1]):
        plt.subplot(tavg.shape[1], 1, i+1)  # Create a subplot with 3 rows and 1 column (adjust as per data)

        # Plot each scaled target time series
        plt.plot(scaled_target[i], label=f'Rescaled Series {i + 1}', linestyle='--')

        # Customize the plot for each time series
        plt.title(f'Time Series {i + 1}')
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.legend()  # Display legend
        plt.grid()  # Display grid for better readability

      # Ensure the subplots don't overlap
      plt.tight_layout()

      # Show the plot
      # plt.show()array.astype(dtype='f', order='C', copy=True)

    # print("tavg.shape", tavg.shape)

    targetdyn_expanded = np.expand_dims(targetdyn_norm, axis=0)  # Shape: [1, nts, noutpus]
    targetdyn_tiled = np.tile(targetdyn_expanded, (self.n_work_items, 1, 1))
    # print("ttargetdyn_tiled_rescale.shape", targetdyn_tiled.shape)
    # return np.array(scaled_target_all), np.array(normalized_tsss)
    return np.array(targetdyn_tiled), np.array(normalized_tsss)
    # return np.array(scaled_target_all), tavg.transpose(2, 0, 1)

  def mse_with_decay(self, y, y_pred, decay_rate=0.1, T=250):
    """
    Compute MSE for each simulation with exponential decay correction.

    Args:
        y (numpy.ndarray): True outputs of shape (nsims, ntimesteps, noutputs).
        y_pred (numpy.ndarray): Predicted outputs of shape (nsims, ntimesteps, noutputs).
        decay_rate (float): Exponential decay rate.

    Returns:
        sorted_mse (numpy.ndarray): Sorted MSE percentages with decay correction.
        sorted_indices (numpy.ndarray): Indices of sorted MSE percentages.
    """
    # Calculate raw MSE per simulation
    mse_per_sim = np.mean(np.mean((y - y_pred) ** 2, axis=2), axis=1)
    if self.my_rank == 0:

      print("mse_per_sim", mse_per_sim[np.argsort(mse_per_sim)])

    # Apply exponential decay correction
    simulation_indices = np.arange(len(mse_per_sim))  # Assuming time or simulation index
    # print("simulation_indices", simulation_indices)
    # decay_factors = np.exp(-decay_rate * simulation_indices)
    # decay_factors = np.exp(-np.arange(y.shape[1]) * decay_rate)  # Decay factors along the time axis
    n_samples = y.shape[1]
    decay_factors = np.exp(-np.arange(n_samples, dtype=np.float32) /T)  # Decay factors along the time axis
    decay_factors = decay_factors / decay_factors.sum()  # Normalize it?
    # corrected_mse = mse_per_sim * decay_factors
    # print("corrected_mse", corrected_mse)

    # print("decay_factors", decay_factors)

    weighted_mse_per_sim = np.sum((y - y_pred) ** 2 * decay_factors[:, np.newaxis], axis=1)
    mse_per_sim = np.mean(weighted_mse_per_sim, axis=1)

    mean_y = np.mean(y, axis=(1, 2), keepdims=True)  # Mean over time and features
    var_y = np.mean((y - mean_y) ** 2, axis=(1, 2))  # Variance over time and features

    # Normalize MSE by variance of y
    normalized_mse = mse_per_sim / var_y

    # Convert to percentages // without variance
    corrected_mse_percentage = (normalized_mse / np.sum(normalized_mse)) * 100
    # print("corrected_msecorrected_mse_percentage", corrected_mse_percentage)

    # Sort by smallest corrected MSE percentage
    sorted_indices = np.argsort(corrected_mse_percentage)
    sorted_mse = corrected_mse_percentage[sorted_indices]

    return sorted_mse, sorted_indices


  def run_all(self):

    np.random.seed(88)

    tic = time.time()


    # todo
    # noise to initial weights

    inregions = [10, 17, 30]
    outregions = [16, 21, 23] # orig

    # weights are now nsims, region, region size. which is not necessary
    # only necessary for output nodes which are reset thus shape 32, 3, 68
    # input_weight_matrix = get_input_weight_matrix(self.weights[0], inregions)
    # print(input_weight_matrix.shape)

    # outregions = [16, 17, 18, 19, 20, 21]
    # inregions = [30]
    # outregions = [23]
    scale_sims = 2.5
    traindelay = 0
    preddelay = 100 # for pci to have 300 before and after stimulus
    trainstart = int(scale_sims * (50  + traindelay))
    trainend =   int(scale_sims * (250 + traindelay))
    predstart =  int(scale_sims * (50  + preddelay) )# in ridge regression
    predend =    int(scale_sims * (150 + preddelay) ) # in ridge regression
    analstart = 50
    analend = 250
    # run training for 2500 step of which 2000 are used for rigde reg. these are on GPU *10 scale
    # as the GPU does 10 steps and averages it to 1 step. inject dynamics till inject_dyn
    inject_dyn_ridge = int(250 * self.n_inner_steps * scale_sims)
    inject_dyn_pred = int((50 + preddelay) * self.n_inner_steps * scale_sims)

    dynrun = int(2000 * scale_sims)

    MSEs = []
    pcis = []
    lzcs = []
    dfas = []
    lyas = []
    Ys = []
    Ypreds = []

    avgMSE = np.zeros(self.params.shape[0])

    # for MSE
    # T is the lergth of the forecast
    # D is the number of env variables. 3 for lorentz
    T = int(100*scale_sims)
    D = 1

    # for FFT
    window = 100
    step = 1

    # set which lorentz dimension should be x,y or z. equal values equal equal input
    set_dyn = [0,1,2]

    nodes_minus_inoutreg_anal = [node for node in list(range(self.args.n_regions)) if
                node not in (inregions + outregions)]

    # nodes_minus_inoutreg = [node for node in list(range(self.args.n_regions)) if
    #                              node not in (inregions + outregions)]
    nodes_minus_inoutreg = range(self.args.n_regions)

    # assert window == (predend-predstart)

    best_mse_tloop = np.full((self.n_work_items, 2), [float('inf'), -1])
    # best_mse_tloop = np.full((self.n_work_items, 4), None, dtype=object)
    # best_mse_tloop = np.full((self.n_work_items, 4), float('inf'))

    # pshift = np.pi / 5
    # pshift = np.pi * .75 # wel aardig
    # pshift = np.pi * 1 # for sinusoidals
    # pshift = 4.6 # for lorentz
    # pshift = 1.75 # sofar the best for standard 68 nodes
    # pshift = 3.8 # for h.004 for lorentz for 96 nodes!
    self.pshift = 3.8 # for h.004 for lorentz for 96 nodes!
    # pshift = 3.26# for zerlaut68belimconn
    dominant_frq = 1.4 #Hz
    sampling_rate = 100 #Hz
    time_shift_seconds = self.pshift / (2 * np.pi * dominant_frq)
    self.delay_timesteps = int(round(time_shift_seconds * sampling_rate))

    self.scale_input = .5

    # pshift = -1.5
    #pshift = 0 # for lorentz
    nonzeroffset = 0
    #todo: for sinussen the br and nonzeroffset were changed

    # Set the task to sinus instead of lorentz
    SinusTask=False
    stepsizeforLorentz = 0.004

    # create fixed set
    notfixpredictiontask = False
    if notfixpredictiontask == False:
      # fixed_pred_set = np.vstack(generate_sinusoidal_signals())
      # teach_pred_set = np.vstack(generate_sinusoidal_signals(pshifts=pshift))

      fixed_pred_set, teach_pred_set = self.create_task_dyn(phase_shift_pis=self.pshift, h=stepsizeforLorentz,
                                                            nonzeroffset=nonzeroffset, Sinussus=SinusTask)

      # lorentz_params = {
      #   "x_range": (-20, 20),
      #   "y_range": (-30, 30),
      #   "z_range": (0, 50)
      # }
      # # lorentz_params = {
      # #       "frequencies": (0.1, 0.2, 0.3)  # Frequencies for x, y, and z respectively
      # # }
      # encoding_type = "normalized"
      # dynamics = generate_enc_lorenz_dynamics(sigma=10.0, rho=28.0, beta=8 / 3, dt=0.01, steps=2500)
      # fixed_pred_set = enc_at_iwm(input_weight_matrix, dynamics, lorentz_params, encoding_type=encoding_type).T
      # plot_target_dyn(fixed_pred_set, teach_pred_set, "Prediction Dyanmics")
      # plt.show()

    # self.trainingloops = 2
    Ys_Ypreds = np.full((2, self.params.shape[0], (predend-predstart), len(outregions)), float('inf'))
    # print("Ys_Ypredsshape", Ys_Ypreds.shape)

    # train on which state
    # 0 for r
    # 1 for V
    r_or_V = 0

    if self.my_rank == 0:
      self.logger.info('trainstart:trainend %d %d', trainstart, trainend)
      self.logger.info('predstart:predend %d %d', predstart, predend)
      self.logger.info('inject_dyn_ridge, inject_dyn_pred %d %d', inject_dyn_ridge, inject_dyn_pred)
      # print('dynrun', dynrun)
      self.logger.info("delay_timesteps %d", self.delay_timesteps)
      self.logger.info("T %d", T)

    best_indices = np.zeros((self.n_work_items), dtype=int)
    # Wout = np.zeros((self.n_work_items, self.args.n_regions, len(set_dyn)))

    for tloop in range(self.trainingloops):

      # print('self.weights', self.weights)

      # create 3d dynamics to drive/train/feed the model
      # target_dyn_train = np.vstack(generate_sinusoidal_signals())
      # teachr_dyn_train = np.vstack(generate_sinusoidal_signals(pshifts=pshift))

      # todo for now focus on the best one and give that shift to all
      if tloop > 0:
        # print("self.weights[best_indices][0].shape", self.weights[best_indices][0].shape)
        # self.pshift = self.compute_pshift(self.weights[best_indices][0])
        # print("self.pshift", self.pshift)
        self.pshift = 3.8

      if notfixpredictiontask == True:

        target_dyn_train, teachr_dyn_train = self.create_task_dyn(phase_shift_pis=self.pshift, h=stepsizeforLorentz,
                                                                  nonzeroffset=nonzeroffset, Sinussus=SinusTask)

      else:
        # use a fixed prediction set
        target_dyn_train = fixed_pred_set
        teachr_dyn_train = teach_pred_set


      # dynamics = generate_enc_lorenz_dynamics(sigma=10.0, rho=28.0, beta=8 / 3, dt=0.01, steps=2500)
      # target_dyn_train = enc_at_iwm(input_weight_matrix, dynamics, lorentz_params, encoding_type=encoding_type).T
      # target_dyn_train = np.vstack(self.create_task_dyn()) # lorentz dyn

      # RUN SIMULATION for RIDGE REGRESSION
      trainer = target_dyn_train[set_dyn, :inject_dyn_ridge]
      tearchr = teachr_dyn_train[set_dyn, :inject_dyn_ridge]
      if self.my_rank == 0:
        self.logger.info("training loop commencing")
        self.logger.info('target_dyn.shape %s', target_dyn_train[set_dyn, :inject_dyn_ridge].shape)
        self.logger.info('trainer0.shape %s', trainer.shape)
        self.logger.info('tearchr0.shape %s', tearchr.shape)
        self.logger.info('phase shift %f', self.pshift)

      if self.trainingloops == 0:
        # inverted_dyn = -target_dyn_train
        plot_target_dyn(trainer, tearchr, "")

      statesbuf = 0
      dummyWout = 0

      # scale trainer / teacher
      trainer = trainer*self.scale_input
      tearchr = tearchr*self.scale_input

      tavg0, statesbuf = self.run_tvb_rc(trainer, tearchr, inject_dyn_ridge, inject_dyn_pred, dummyWout, statesbuf, usebuffer=False)
      if self.my_rank == 0:
        self.logger.info("tavg0.shape %s", tavg0.shape)
        self.logger.info("states.shape %s", statesbuf.shape)

      # if self.trainingloops == 1:
      #   # inverted_dyn = -target_dyn_train
      #   # plot_tavgs(tavg0[:, 0, :, 0], "after train injection r")
      #   # plot_tavgs(tavg0[:, 1, :, 0], "after train injection V")
      #   plot_tavgs_notstacked(tavg0[:, 0, :, 0], "after train injection r")
      #   plot_tavgs_notstacked(tavg0[:, 1, :, 0], "after train injection V")

      # compute regression for weights
      # the same input is used to train the network
      # Regularization parameter for ridge regression
      beta_ridge = self.beta_ridge
      # beta_ridge = 1
      if self.my_rank == 0:
        self.logger.info("NaNs in tavg0: %s", np.isnan(tavg0).any())
        self.logger.info("Infinite values in tavg0: %s", np.isinf(tavg0).any())
        self.logger.info('betaridge %f', beta_ridge)

      # SCALE TARGET DYNAMICS for rigdereg and MSE

      if self.n_inner_steps != 1:
      # downsample to match gpu frequency as 10 steps are averaged to 1. 10 to match gpu time
      # :(trainend-trainstart) is sufficient for the length of the training dynamics
        downsampled_training_dyn = self.downsample(target_dyn_train[set_dyn, 0:dynrun])
        downsampled_teacher_dyn = self.downsample(teachr_dyn_train[set_dyn, 0:dynrun])
        if self.my_rank == 0:
          self.logger.info("downsampled.shape %s", downsampled_training_dyn.shape)
      else:
        downsampled_training_dyn = trainer
        downsampled_teacher_dyn = tearchr

      if self.trainingloops == 0:
        # plot the dynamics
        inverted_dyn = downsampled_training_dyn
        plot_target_dyn(inverted_dyn, downsampled_teacher_dyn, "Downsampled training dynamics")

      # correct the offset in generating the sinussus
      downsampled_training_dyn = downsampled_training_dyn - nonzeroffset
      tavg0 = tavg0 - nonzeroffset

      # not normalizing the regression seems to make it worse
      notnormalizeregression = 0
      if notnormalizeregression == 1:
        # should be shapes:
        # sdt shape 16 200 3 -> Y
        # tavgcopy 16, 200, 68 -> R
        scaled_dyn_train, tavg_copy_train = downsampled_training_dyn, tavg0[trainstart:trainend, r_or_V, :, :]
        tavg_copy_train = tavg_copy_train.transpose(2, 0, 1)

        # copy the tavgcopy into the shape for Y
        scaled_dyn_train = np.expand_dims(scaled_dyn_train.T, axis=0)  # Add a new axis at the beginning
        scaled_dyn_train = np.repeat(scaled_dyn_train, self.n_work_items, axis=0)


      else:
        if self.my_rank == 0:
          self.logger.info("downsampled_training_dyn.shape %s", downsampled_training_dyn.shape)
          self.logger.info("downsampled_teacher_dyn.shape %s", downsampled_teacher_dyn.shape)

        # rescale only of the target data
        # normalizing seems to have a negative effect on the prediction (bacomes too uniform)
        scaled_dyn_train, tavg_copy_train = self.rescale(downsampled_training_dyn[set_dyn, :].T,
                                        tavg0[trainstart:trainend, r_or_V, :, :],
                                        outregions)

        scaled_dyn_treachr, _ = self.rescale(downsampled_teacher_dyn[set_dyn, :].T,
                                        tavg0[trainstart:trainend, r_or_V, :, :],
                                        outregions)

      # print(scaled_dyn_train[0,:,2])

      # RIDGE REGRESSION
      if self.my_rank == 0:
        self.logger.info("scaled_dyn_train.shape %s", scaled_dyn_train.shape)
        self.logger.info("normalize_tavg.shape %s", tavg_copy_train.shape)
        self.logger.info("scaled_dyn_treachr.shape %s", scaled_dyn_treachr.shape)

      # tavg_copy_train = np.transpose(tavg_copy_train, axes=(0, 2, 1)) \
      #   if tavg_copy_train.ndim == 3 else np.ascontiguousarray(tavg_copy_train)
      # print(tavg_copy_train.transpose(1, 2, 0)[:, nodes_minus_inoutreg, :].shape)

      # Wout = Tikhonov_ridge_reg(tavg_copy_train.transpose(1, 2, 0)[:, nodes_minus_inoutreg, :],
      #                           scaled_dyn_train.transpose(0, 2, 1), beta_ridge,
      #                           self.n_work_items)

      # Wout = gpu_TIK_simp(tavg_copy_train.transpose(1, 2, 0)[:, nodes_minus_inoutreg, :],
      #                           scaled_dyn_train.transpose(0, 2, 1), beta_ridge)

      # Define the rolling shift
      # shift = -5  # Rolling to predict the next timestep
      # # Roll each simulation's timeseries
      # rolled_dyn_train = np.roll(scaled_dyn_train, shift, axis=1)
      rolled_dyn_train = scaled_dyn_treachr


      Rtrain = tavg_copy_train.transpose(0, 2, 1)[:, nodes_minus_inoutreg, :]

      if self.trainingloops == 1:
        plot_target_dyn(scaled_dyn_train[0].T, rolled_dyn_train[0].T, "Input and Teacher Dynamics")
        # plot_target_dyn(Rtrain, "Rtrains")
        # plot_tavgs_notstacked(Rtrain[0].T, "Rtrains")

      # shapes
      # Rtain (32, 68, 500)
      # rolled_dyn_train (32, 500, 3)
      # todo use cobrawap to heuristcally estimate the phase shift of temporal relation input and output
      # todo check out sampling freq of tvb sim and adapt the lorentz
      Wout = Tik_torch(Rtrain, rolled_dyn_train, beta_ridge)
      if self.my_rank == 0:
        self.logger.info("Wout.shape %s", Wout.shape)

      earlytest = False
      if earlytest == True:
        # (4, 250, 96) @ (4, 96, 3)
        # early fits look good. probably some tweaking and would be good
        pred_early = predi_Torch(tavg0[predstart:predend,0, ].transpose(2, 0, 1), Wout)
        plotpred( rolled_dyn_train, pred_early, self.trainingloops, f"ChunkyPred pos")

      # print(scaled_dyn_train.transpose(0, 2, 1)[20, 0, 114])
      # print(scaled_dyn_train.transpose(0, 2, 1)[20, 0, 116])
      # print(scaled_dyn_train.transpose(0, 2, 1)[20, 0, 114])
      # testRonGPU(tavg_copy_train.transpose(1, 2, 0)[:, nodes_minus_inoutreg, :], scaled_dyn_train.transpose(0, 2, 1))

      # Wout = compute_Tik_gpu(tavg_copy_train[:, nodes_minus_inoutreg, :],
      #                           scaled_dyn_train, beta_ridge)
      # print("Wout", Wout)

      # # Test Tikhonov
      # R=tavg0[trainstart:trainend, 0, nodes_minus_inoutreg, :]
      # Y=scaled_dyn_train
      # test_ridge_regression_visual(R, Y, beta_ridge, self.n_work_items)

      # update weights after the prediction to include the bias substraction
      # the bias can only be applied after 1 iteration minimum as the ypred is from the previous iteration
      # the single bias value for each simulation is subtracted from all the weights connected to the output layers
      bias_correction = 0
      if tloop > 0 and bias_correction == 1:
      #   bias = np.mean(Ypred - Y, axis=(1, 2))  # Shape: (nsims,)
      #   Wout = Wout - bias[:, np.newaxis, np.newaxis]
      #   print("Bias magnitude:", bias)
      #   print("Weight range:", np.min(Wout), np.max(Wout))

        Ypred = torch.tensor(Ypred, dtype=torch.float32, device="cuda:0")
        Y = torch.tensor(Y, dtype=torch.float32, device="cuda:0")
        WoutTorch = torch.tensor(Wout, dtype=torch.float32, device="cuda:0")

        bias = torch.mean(Ypred - Y, dim=(1, 2))  # [nsims, 1]

        # Check the magnitude of the bias
        print(f"Calculated Bias: {bias}")

        # Normalize or scale the bias if necessary
        bias = bias / torch.std(Ypred - Y, dim=(1, 2))  # Normalize by error

      # Apply the bias correction to the weights
        for i in range(self.n_work_items):
          # WoutTorch[i] -= bias[i].unsqueeze(1).unsqueeze(2)  # Subtract the bias from the output weights
          print(f"Original bias for simulation {i}: {bias[i]}")

          # Check the weight before bias subtraction
          print(f"Original W_out[{i}] before bias subtraction: {WoutTorch[i]}")

          WoutTorch[i] -= bias[i].view(1, 1)  # Subtract the bias from the output weights
          print(f"Updated Weight Range: {WoutTorch.min()}, {WoutTorch.max()}")

        Wout = WoutTorch.cpu().numpy()

      self.update_output_weights(Wout, outregions, nodes_minus_inoutreg)

      go_ahead = True
      if tloop % self.tlsamples == 0 and go_ahead:
        #### PREDICTION ####

        if self.my_rank == 0:
          self.logger.info("Prediction Phase")
          # self.logger.info("tl%tls %f", tloop % self.tlsamples)

        # create test data to do predictions
        if notfixpredictiontask == True:
          # target_dyn_pred = generate_sinusoidal_signals()
          # target_dyn_pred = np.vstack(self.create_task_dyn()) # lorentz dynamics
          # dynamics = generate_enc_lorenz_dynamics(sigma=10.0, rho=28.0, beta=8 / 3, dt=0.01, steps=2500)
          # target_dyn_pred = enc_at_iwm(input_weight_matrix, dynamics, lorentz_params, encoding_type=encoding_type).T
          # teachr_dyn_pred = generate_sinusoidal_signals(pshifts=pshift)

          target_dyn_pred, teachr_dyn_pred = self.create_task_dyn(phase_shift_pis=self.pshift, h=stepsizeforLorentz,
                                                                  nonzeroffset=nonzeroffset, Sinussus=SinusTask)

        else:
          # use a fixed prediction set
          target_dyn_pred = fixed_pred_set
          teachr_dyn_pred = teach_pred_set

        if self.trainingloops == 0:
          plot_target_dyn(fixed_pred_set, teach_pred_set, "fixed one")

        # alter for injection from 250
        trainer = target_dyn_pred[set_dyn, :] * self.scale_input
        tearchr = teachr_dyn_pred[set_dyn, :] * self.scale_input

        tavg1, _ = self.run_tvb_rc(trainer, tearchr, inject_dyn_pred,
                                   inject_dyn_pred, Wout,
                                   statesbuf, usebuffer=True)

        if self.trainingloops == 1:
          plot_tavgs_notstacked(tavg1[:, 0, :, 0], "pred r")
          plot_tavgs_notstacked(tavg1[:, 1, :, 0], "pred V")

        if self.my_rank == 0:
          self.logger.info('target_dyn_pred %s', target_dyn_pred[set_dyn, :inject_dyn_pred].T.shape)
          self.logger.info('teachr_dyn_pred %s', teachr_dyn_pred[set_dyn, :inject_dyn_pred].T.shape)
          self.logger.info("tavg1.shape %s", tavg1.shape)
          self.logger.info("NaNs in tavg1: %s", np.isnan(tavg1).any())
          self.logger.info("Infinite values in tavg1: %s", np.isinf(tavg1).any())

        # downsample AND rescale for loss / factor 10 to convert to gpu time
        # this sets also the prediction window size of the data for the fft.
        notdownsample = 0
        if notdownsample == 1:
          # downsampled_predict_dyn = target_dyn_pred[set_dyn, 150:250]
          downsampled_predict_dyn = teachr_dyn_pred[set_dyn, predstart:predend]
        else:
          # downsampled_predict_dyn = self.downsample(target_dyn_pred[set_dyn, predstart*self.n_inner_steps:predend*self.n_inner_steps])
          # downsampled_predict_dyn = self.downsample(target_dyn_pred[set_dyn, 500:1700])
          # downsampled_predict_dyn = self.downsample(teachr_dyn_pred[set_dyn, 500:1700]) # signal will be fitted with this
          # downsampled_predict_dyn = self.downsample(teachr_dyn_pred[set_dyn, predstart*self.n_inner_steps:predend*self.n_inner_steps])   # signal will be fitted with this
          downsampled_predict_dyn = self.downsample(teachr_dyn_pred[set_dyn])   # signal will be fitted with this
          downsampled_target_dyn_4predplot = self.downsample(target_dyn_pred[set_dyn])   # signal will be fitted with this
          if self.my_rank == 0:
            self.logger.info('target_dyn_predafterdowns %s', downsampled_predict_dyn.shape)

        # offset correction
        downsampled_predict_dyn = downsampled_predict_dyn - nonzeroffset
        tavg1 = tavg1 - nonzeroffset

        # normalization for prediction seem to work averechts
        notnormalizeprediction = 0
        if notnormalizeprediction == 1:
          # spd shape 16 100 3 -> Y
          # tavgcopy 16, 250, 68 -> R
          # should be R(16, 62, 100), Y(16, 100, 3)
          scaled_predict_dyn, tavgcopy = downsampled_predict_dyn, tavg1[:, r_or_V, :, :]
          tavgcopy = tavgcopy.transpose(2, 0, 1)

          # copy the tavgcopy into the shape for Y
          scaled_predict_dyn = np.expand_dims(scaled_predict_dyn.T, axis=0)  # Add a new axis at the beginning
          scaled_predict_dyn = np.repeat(scaled_predict_dyn, self.n_work_items, axis=0)


        else:
          # rescale only of the target data
          # normalizing seems to have a negative effect on the prediction (bacomes too uniform)
          scaled_predict_dyn, tavgcopy = self.rescale(downsampled_predict_dyn[set_dyn, :].T,
                                          tavg1[:, r_or_V, :, :],
                                          outregions)

          scaled_target_dyn_4predplot, _ = self.rescale(downsampled_target_dyn_4predplot[set_dyn, :].T,
                                          tavg1[:, r_or_V, :, :],
                                          outregions)


        # scaled_dyn_pred = np.array(scaled_dyn_pred)
        if self.my_rank == 0:
          self.logger.info('scaled_predict_dyn.shape %s', scaled_predict_dyn.shape)
          self.logger.info('tavgcopy.shape %s', tavgcopy.shape)

        if self.trainingloops == 1:
          plot_target_dyn(scaled_predict_dyn[0, predstart: predend].T, scaled_predict_dyn[0, predstart: predend].T, "scaled_predict_dyn Dyanmics")
          plot_target_dyn(downsampled_predict_dyn, downsampled_predict_dyn, "downsampled_predict_dyn Dyanmics")

        # plot comparison part to behold the magnitude and signal
        # plot_y_true_in_sim_windows(scaled_dyn_tests.T)


        # Test Tikhonov and predict 100 timesteps
        # R = tavgcopy[:, 50:170, nodes_minus_inoutreg]
        R = tavgcopy[:, predstart:predend, nodes_minus_inoutreg]

        if self.trainingloops == 0:
          # inverted_dyn = -target_dyn_train
          plot_tavgs(R[0], "tavg[0] ")

        poffset = 0
        yoffset = 0 # scale up + 3 when creating the task
        R = R.transpose(0, 2, 1)+poffset
        # R = tavg1[50:150, 0, nodes_minus_inoutreg, :]

        # Y = scaled_dyn_pred.T

        Y = scaled_predict_dyn[:, predstart: predend] - yoffset #.transpose(0, 2, 1)
        # Y2plot = scaled_target_dyn_4predplot[:, predstart: predend] - yoffset #.transpose(0, 2, 1)

        # ridgeres, pureMSE = make_prediction(R, Y, Wout, self.trainingloops)

        # pureMSE = predi_Torch(R, Y, Wout)
        # shapes for the prediction should be
        # R(16, 62, 100), Y(16, 100, 3)
        # pureMSE, Ypred, normalized_MSE = predi_Torch_mse_exdecay_plus(R, Y, Wout, T)
        # pureMSE, Ypred, normalized_MSE = predi_Torch_msexpdecaycorr(R, Y, Wout, T)
        pureMSE, Ypred = predi_Torch_norm(R, Y, Wout)
        normalized_MSE = pureMSE

        pureMSE = np.clip(pureMSE, 0, 2.5)


        if self.my_rank == 0:
          self.logger.info("Y.shapeb4preditorc %s", Y.shape)
          self.logger.info("R.shapeb4preditorc %s", R.shape)
          self.logger.info("R.min %s", R.min())
          self.logger.info("R.max %s", R.max())
          self.logger.info("Ypred.max %s", Ypred.shape)


        # print("Bla Sorted MSE Percentages:", normalized_MSE*100)
        # sorted_indices_bla = np.argsort(normalized_MSE)
        # sorted_mse_bla = normalized_MSE[sorted_indices_bla]
        # print("Bla Sorted Simulation Indices:", sorted_mse_bla)

        # collect all Y and Ypreds
        # Ys=Y[:, 20:120]
        # Ypreds=Ypred[:, 20:120]
        Ys=Y[:]
        Ypreds=Ypred[:]

        # numpy equivalent to pytorch test
        # sorted_mse, sorted_indices_simp = self.mse_with_decay(Y, Ypred, decay_rate=0.1)
        # # # Print Results
        # print("Sorted MSE Percentages:", sorted_mse)
        # print("Sorted Simulation Indices:", sorted_indices_simp)

        power = 0
        if power == 1 and self.trainingloops > 1:
          signalYpred = Ypred[0, :, 0]
          signalY = Y[0, :, 0]
          N = len(signalYpred)  # Number of samples
          signal_fftYpred = fft(signalYpred)
          signal_fftY = fft(signalY)
          signal_powerYpred = np.abs(signal_fftYpred) ** 2  # Power spectrum
          signal_powerY = np.abs(signal_fftY) ** 2  # Power spectrum

          # Compute corresponding frequencies
          sampling_rate = 1/self.dt
          freqs = fftfreq(N, d=1 / sampling_rate)

          # Focus on positive frequencies
          positive_freqs = freqs[1:N // 2]
          positive_powerYpred = signal_powerYpred[1:N // 2]
          positive_powerY = signal_powerY[1:N // 2]

          # Find the dominant frequency
          dominant_frequencyYpred = positive_freqs[np.argmax(positive_powerYpred)]
          dominant_frequencyY = positive_freqs[np.argmax(positive_powerY)]

          plt.figure(figsize=(10, 6))
          plt.plot(positive_freqs, positive_powerYpred, label="Prediction")
          plt.plot(positive_freqs, positive_powerY, label="Teacher")
          plt.title("Frequency Spectrum")
          plt.xlabel("Frequency (Hz)")
          plt.ylabel("Power")
          plt.grid(True)
          plt.legend()
          print("dominant_frequencYpred", dominant_frequencyYpred)
          print("dominant_frequencY", dominant_frequencyY)
          print(np.mean(positive_powerYpred), np.std(positive_powerYpred))

        # ridgeres = np.array(ridgeres, dtype=object)
        # print('ridgeresshape', ridgeres.shape)

        MSEs.append(normalized_MSE)

        # for i in range(self.n_work_items):
        #   avgMSE[i] = (normalized_MSE[i] + avgMSE[i])/2

        # compare and store the best MSE
        # for i in range(self.n_work_items):
        #   if normalized_MSE[i] < best_mse_tloop[i, 0]:
        #   # if result[i, 0] < best_mse_tloop[i, 0] or best_mse_tloop[i, 0] is None:
        #     best_mse_tloop[i, 0] = normalized_MSE[i]  # Update best MSE
        #     best_mse_tloop[i, 1] = tloop
        #     # best_mse_tloop[i, 2] = result[i, 1]
        #     # best_mse_tloop[i, 3] = result[i, 2]
        #     # Ys_Ypreds[0, i] = Y[i, 20:120]
        #     # Ys_Ypreds[1, i] = Ypred[i, 20:120]
        #     Ys_Ypreds[0, i] = Y[i]
        #     Ys_Ypreds[1, i] = Ypred[i]

        # do the conscious stuff
        # todo investigate effect duration pertubation/reaction of network
        pci, lzc = self.run_pci_analy_gpu(tavg1[:,0,:,:])
        # clip solution for the extreme value
        pci = np.clip(pci, -1, 3)

        # pcis.append(pci)
        # lzcs.append(lzc)
        # lzcs.append(lzc)

        print('tavg1shape.T', tavg1[:,:,:,0].shape)
        np.save("data/tavg1_pred0", tavg1[:,:,:,0])
        np.save("data/tavg1_pred1", tavg1[:,:,:,1])
        np.save("data/tavg1_pred2", tavg1[:,:,:,2])
        np.save("data/tavg1_pred3", tavg1[:,:,:,3])

        # dfas.append(computeDFA_gpu(np.ascontiguousarray(tavg1.T[:,:,0,:])))
        dfa = computeDFA_gpu(np.ascontiguousarray(tavg1.T[:,nodes_minus_inoutreg_anal,0,:]), self.logger, self.my_rank)
        # print("dfas.shape", np.array(dfas).shape)
        dfa = np.mean(dfa, axis=1)

        # lyas.append(computeLYA_gpu(np.ascontiguousarray(tavg1.T[:,:,0,:]), emb_dim=6, lag=1, min_tsep=2, trajectory_len=20, tau=.2))
        lya = computeLYA_gpu(np.ascontiguousarray(tavg1.T[:,nodes_minus_inoutreg_anal,0,predstart:predend]),
                             self.logger, self.my_rank, emb_dim=4, lag=5, min_tsep=10, trajectory_len=50, tau=self.dt)
        # lyas.append(computeLYA_gpu(np.ascontiguousarray(tavgcopy.transpose(0,2,1)), emb_dim=6, lag=1, min_tsep=2, trajectory_len=20, tau=.2))
        # print("Computed Lyapunov Exponents (per simulation and region):", lyas)
        lya = np.mean(lya, axis=1)

        if self.my_rank == 0:
          self.logger.info('tavgforanal %s', tavg1.T[:, nodes_minus_inoutreg_anal, 0, :].shape)  # 16,68,625
          self.logger.info("Anal res pci %s", pci.shape)
          self.logger.info("Anal res lzc %s", lzc.shape)
          self.logger.info("Anal res dfa %s", dfa.shape)
          self.logger.info("Anal res lya %s", lya.shape)

        # stack the parameters with the analysis results for easy parameters selection during plotting
        pci4plot = np.hstack((self.params, pci.reshape(-1, 1))) # (4,7) @ (4,1)
        dfa4plot = np.hstack((self.params, dfa.reshape(-1, 1)))
        lya4plot = np.hstack((self.params, lya.reshape(-1, 1)))
        mse4plot = np.hstack((self.params, pureMSE.reshape(-1, 1)))

        # plot_combined_3d_combined(pci4plot[:, [2, 3, 4, 7]], dfa4plot[:, [2, 3, 4, 7]], lya4plot[:, [2, 3, 4, 7]], whattosave=self.n_params,
        #                           whattosave2=self.args.n_regions)

    comm.Barrier()

    if self.my_rank == 0:
      self.logger.info("pci4plot %s", pci4plot.shape)
      self.logger.info("dfa4plot %s", dfa4plot.shape)
      self.logger.info("lya4plot %s", lya4plot.shape)
      self.logger.info("mse4plot %s", mse4plot.shape)

    # gather all the resuls and reshape them to be stacked on the nsims dimension
    # (worldsize, nsims, results) to (worldsize * nsims, results)
    resultshape = self.params.shape[1]+1
    nsims = self.n_work_items

    # get all the results
    all_pci4plot = np.array(comm.gather(pci4plot, root=0))
    all_dfa4plot = np.array(comm.gather(dfa4plot, root=0))
    all_lya4plot = np.array(comm.gather(lya4plot, root=0))
    all_mse4plot = np.array(comm.gather(mse4plot, root=0))

    # collect the mses
    all_mses = np.array(comm.gather(pureMSE, root=0))

    # collect the mses
    all_params_g = np.array(comm.gather(self.params, root=0))

    all_Ys = np.array(comm.gather(Y, root=0))
    all_Y_preds = np.array(comm.gather(Ypred, root=0))

    all_Wouts = np.array(comm.gather(Wout, root=0))

    MPI.Finalize()

    if self.my_rank == 0:

      # stack all together AFTER the mpi has finalized as only root node has these results
      all_pci4plot = all_pci4plot.reshape(self.world_size * nsims, resultshape)
      all_dfa4plot = all_dfa4plot.reshape(self.world_size * nsims, resultshape)
      all_lya4plot = all_lya4plot.reshape(self.world_size * nsims, resultshape)
      all_mse4plot = all_mse4plot.reshape(self.world_size * nsims, resultshape)

      all_mses = all_mses.reshape(self.world_size * nsims)
      all_params = all_params_g.reshape(self.world_size * nsims, self.params.shape[1])

      _, _, n_p_timesteps, lorentzdymensions = all_Ys.shape
      # print("lorentzdymensions, n_p_timesteps", lorentzdymensions, n_p_timesteps)
      all_Ys = all_Ys.reshape(self.world_size * nsims, n_p_timesteps, lorentzdymensions)
      all_Y_preds = all_Y_preds.reshape(self.world_size * nsims, n_p_timesteps, lorentzdymensions)

      all_Wouts = all_Wouts.reshape(self.world_size * nsims, len(nodes_minus_inoutreg), lorentzdymensions)

      # the last param is the result
      set_params = [2, 3, 4, 7]
      all_p = all_pci4plot[:, set_params]
      all_d = all_dfa4plot[:, set_params]
      all_l = all_lya4plot[:, set_params]
      all_m = all_mse4plot[:, set_params]

      # stack all results
      all_result = np.hstack((all_p[:, :3], all_p[:, 3:], all_d[:, 3:], all_l[:, 3:], all_m[:, 3:]))

      # set the mask to sort the results only for the theorized values
      # mask_cons_hihgdfa_lowlya = (0.44 < all_result[:, 3]) & (all_result[:, 3] < 0.67) & (all_result[:, 4] >= 0.5) & (
      #           all_result[:, 5] <= 0.3) #& (all_result[:, 5] <= 0.2)
      # mask_cons_lowdfa_highlya = (0.44 < all_result[:, 3]) & (all_result[:, 3] < 0.67) & (all_result[:, 4] < 0.5) & (
      #           all_result[:, 5] > 0.3) #& (all_result[:, 5] <= 0.2)
      # mask_uncon_highdfa_lowlya = (0.12 < all_result[:, 3]) & (all_result[:, 3] < 0.31) & (all_result[:, 4] >= 0.5) & (
      #           all_result[:, 5] <= 0.3) #& (all_result[:, 5] <= 0.2)
      # mask_uncon_lowdfa_highlya = (0.12 < all_result[:, 3]) & (all_result[:, 3] < 0.31) & (all_result[:, 4] < 0.5) & (
      #           all_result[:, 5] > 0.3) #& (all_result[:, 5] <= 0.2)

      # based on relative ranges what is found
      mask_cons_hihgdfa_lowlya = (all_result[:, 3] >= 1.5)  & (
                all_result[:, 4] >= 0.5) & (all_result[:, 5] <= 0.3)
      mask_cons_lowdfa_highlya = (all_result[:, 3] >= 1.5)  & (
                all_result[:, 4] < 0.5) & (all_result[:, 5] > 0.3)
      mask_uncon_highdfa_lowlya = (all_result[:, 3] < 1.5)  & (
                all_result[:, 4] >= 0.5) & (all_result[:, 5] <= 0.3)
      mask_uncon_lowdfa_highlya = (all_result[:, 3] < 1.5)  & (
                all_result[:, 4] < 0.5) & (all_result[:, 5] > 0.3)

      # apply the mask
      cons_hihgdfa_lowlya  = all_result[mask_cons_hihgdfa_lowlya]
      cons_lowdfa_highlya  = all_result[mask_cons_lowdfa_highlya]
      uncon_highdfa_lowlya = all_result[mask_uncon_highdfa_lowlya]
      uncon_lowdfa_highlya = all_result[mask_uncon_lowdfa_highlya]

      # Sort results in descending order based on the last column (all_m)
      cons_hihgdfa_lowlya_res = cons_hihgdfa_lowlya[np.argsort(cons_hihgdfa_lowlya[:, 6])]
      cons_lowdfa_highlya_res = cons_lowdfa_highlya[np.argsort(cons_lowdfa_highlya[:, 6])]
      uncon_highdfa_lowlya_res = uncon_highdfa_lowlya[np.argsort(uncon_highdfa_lowlya[:, 6])]
      uncon_lowdfa_highlya_res = uncon_lowdfa_highlya[np.argsort(uncon_lowdfa_highlya[:, 6])]

      print("Nominal Metric Values Filter")
      print("I   J   eta   PCI   DFA   LYA   MSE")
      print(cons_hihgdfa_lowlya_res)
      print(cons_lowdfa_highlya_res)
      print(uncon_highdfa_lowlya_res)
      print(uncon_lowdfa_highlya_res)

      self.logger.info("all_pci4plot %s", all_p.shape)
      self.logger.info("all_dfa4plot %s", all_d.shape)
      self.logger.info("all_lya4plot %s", all_l.shape)
      self.logger.info("all_mse4plot %s", all_m.shape)

      # make the 3d plots
      plot_3d_simulation_par(all_p,
                             all_d,
                             all_l,
                             all_m,
                             whattosave=self.n_params, whattosave2=self.args.n_regions)

          # print("pci4plot[:, [2, 3, 4, 7]]\n", pci4plot[:, [2, 3, 4, 7]])
          # print("dfa4plot[:, [2, 3, 4, 7]]\n", dfa4plot[:, [2, 3, 4, 7]])
          # print("lya4plot[:, [2, 3, 4, 7]]\n", lya4plot[:, [2, 3, 4, 7]])

      params_histogram(all_p,
                       all_d,
                       all_l,
                       all_m,
                       'Montbrió-Pazo-Rosin')

      #@@@ POST PROCESSING @@@#
      if self.trainingloops == 0:
        plot_frequency_spectra(Ys_Ypreds[0], Ys_Ypreds[1], self.dt)

      best_indices = np.argsort(all_mses)
      paramstoplot = all_params[best_indices, :]
      if self.my_rank == 0:
        self.logger.info("all_params_g.shape %s", all_params_g.shape)
        self.logger.info("all_params.shape %s", all_params.shape)
        self.logger.info("best_indices.shape %s", best_indices.shape)
        self.logger.info("all_Y.shape %s", all_Ys.shape)
        self.logger.info("all_Y_preds.shape %s", all_Y_preds.shape)

      np.save("data/W_out_MBR_inc_inout_MB.npy", all_Wouts[best_indices][0])

      np.save("data/all_p.npy_MB", all_p)
      np.save("data/all_d.npy_MB", all_d)
      np.save("data/all_l.npy_MB", all_l)
      np.save("data/all_m.npy_MB", all_m)

      plotreaction =0
      if plotreaction == 1:
        # inverted_dyn = -target_dyn_train
        # plot_tavgs(tavg1[:,0,:,0], "tavg1[:,0,:,0] ")
        plot_tavgs_notstacked(tavg1[:, 0, :, best_indices[0]], f"prediction res R idx {best_indices[0]} params {paramstoplot[0]}")
        plot_tavgs_notstacked(tavg1[:, 1, :, best_indices[0]], f"prediction res V idx {best_indices[0]} params {paramstoplot[0]}")

      # the running avg results sorted
      # sorted_avg_indices = np.argsort(avgMSE)
      # sorted_avgMSE = avgMSE[sorted_avg_indices]

      MSEs = np.array(MSEs, dtype=np.float32)
      median_MSEs = np.median(MSEs, axis=0)
      sorted_medi_indices = np.argsort(median_MSEs)[:32]

      MSEstoPlot = all_mses[best_indices]

      # plot the best and the worst results
      # best_indices[:128] is just used as a range, arrays already are sorted
      # MSEstoPlot = np.concatenate((MSEstoPlot[:64], MSEstoPlot[-64:]))
      # paramstoplot = np.concatenate((paramstoplot[:64], paramstoplot[-64:]))
      pretty_print_best_params(best_indices, MSEstoPlot, best_mse_tloop[:, 1], paramstoplot, "Sorted best achievable MSE")

      print("MSEs.shape", MSEs.shape)
      plot_mse = False
      if plot_mse == True:
        plot_MSEs(MSEs, best_indices, paramstoplot, self.args.n_regions)

      # Todo broken by multi version
      predY = all_Y_preds[best_indices]
      # predY = predY.transpose(0,2,1)
      trueY = all_Ys[best_indices]
      # trueY = trueY.transpose(0,2,1)

      # predY = Ypred[best_indices]
      # predY = predY.transpose(0, 2, 1)
      # trueY = Y[best_indices]
      # trueY = trueY.transpose(0, 2, 1)

      plotpred(trueY, predY, self.trainingloops, "Best ever MSE")
      # plotpred_teach_train(true, pred, trainer, self.trainingloops, "Best ever MSE")

    plotcstuff = False
    if plotcstuff == True:
      plot_pcis(pcis, lzcs, sorted_medi_indices, paramstoplot, self.args.n_regions)
      plot_dfas(dfas, sorted_medi_indices, paramstoplot)
      # plot_lyas(lyas, sorted_medi_indices, paramstoplot)


    plotheatmaps = True
    if plotheatmaps == True:
      # generic_heatmap(median_MSEs, "Mean Squared Error", self.trainingloops)
      generic_heatmap(pci, "Complexity Index", self.trainingloops)
      # generic_heatmap(dfa, "Detrended Fluctuation Analysis", self.trainingloops)
      # generic_heatmap(lya, "Lyapunv Exponent", self.trainingloops)

    # plot all 16 first preds
    # plot_predictions_vs_true_fullsims(true, pred, "besteverpost")
    plotallpred = False
    if plotallpred == True:
      rounded_params = np.round(paramstoplot, 2)
      # plot_predictions_vs_true_fullsims(Ys_Ypreds[0, best_indices], Ys_Ypreds[1, best_indices], rounded_params)
      plot_predictions_vs_true_fullsims(Ys[best_indices], Ypreds[best_indices], rounded_params)

    # print("tavg1plotshape", tavg1[50:150, 0, inregions, sorted_medi_indices].shape)
    # plotpred(tavg1[50:150, 0, inregions, sorted_medi_indices], Ypreds[sorted_medi_indices, :, :], self.trainingloops)

    extra_test = 0
    if extra_test == 1:
    #   loss, bestwindows = sliding_window_best_fit_loss_pycuda(scaled_dyn_pred.T, tavg_fft,
    #                                T, D, window_size=window, step_size=step)
    #
    #   fftprec, fftbest = compare_fft_sims_inparallel(scaled_dyn_pred, tavg_fft,
    #                               window_size=window, step_size=step, debug=True)
    #   # fftprec, fftbest = compare_fft_sims_inparallel(scaled_dyn_pred, tavg1[analstart:analend, 0, outregions, :],
    #   #                                                window_size=window, step_size=step, debug=True)
    #
    #   print("fftprec.shape", fftprec.shape)
    #   print("fftbest.shape", fftbest.shape)
    #
    #   # outregions = [16, 21, 23]
    #   mask = np.ones(self.args.n_regions, dtype=bool)
    #   mask[inregions] = False
    #   tavg_min_inregions = tavg1[analstart:analend, 0, :, :][:, mask, :]
    #   top10_power_regions = get_top_nodes_by_power(tavg_min_inregions, outregions)
    #
    #   for param, top_regions in top10_power_regions.items():
    #     print(f"Top nodes for parameter {param}: {top_regions}")
    #
    #   # print(loss)
    #   # for ls in loss:
    #   #   print("loss, {:.2f}".format(ls))
    #
      median_lyas = np.random.rand(self.n_params)
      bestwindows = np.random.rand(3, self.n_params)
      fftprec = np.random.rand(self.n_params)
      fftbest = np.random.rand(self.n_params)
      plot_multple(tavg0[:250, :, :, sorted_medi_indices], self.params, self.args.n_time, pcis, np.average(lzcs, axis=1), median_dfas, median_lyas,
                   median_MSEs, bestwindows, fftprec, fftbest, window, step,'g', 'sig', True, True,
                  self.args.n_regions, self.trainingloops)

    # show plots eventually
    notrunonhpc = True
    if notrunonhpc == True:
      plt.show()

    toc = time.time()
    elapsed = toc - tic

    if self.my_rank == 0:
      # print('Output shape (simsteps, states, bnodes, n_params) %s', tavg1.shape)
      # print('Finished TVB training successfully in: {0:.3f}'.format(elapsed))
      # print('Training rounds:', self.trainingloops)

      self.logger.info('Output shape (simsteps, states, bnodes, n_params) %s', tavg1.shape)
      self.logger.info('Finished TVB training successfully in: {0:.3f}'.format(elapsed))
      self.logger.info('Training rounds:'.format(self.trainingloops))
      self.logger.info('and in {0:.3f} M step/s'.format(
        1e-6 * self.args.n_time * self.n_inner_steps * self.n_work_items / elapsed))

    return tavg1


if __name__ == '__main__':

  driver_setup = Driver_Setup()

  sig_tavgs = []
  for sig_i in range(1):
    tavgGPU = Driver_Execute(driver_setup).run_all()
    sig_tavgs.append(tavgGPU)

  # print("sig_tavgs.shape", sig_tavgs.shape)


  # simtime = driver_setup.args.n_time
  # # simtime = 10
  # regions = driver_setup.args.n_regions
  # g = 1.0
  # # g = 0.0042
  # s = 1.0
  # dt = driver_setup.dt
  # period = 1
  #
  # # generic model definition
  # model = driver_setup.args.model.capitalize()+'T'
  #
  # # non generic model names
  # # model = 'MontbrioT'
  # # model = 'RwongwangT'
  # # model = 'OscillatorT'
  # # model = 'DumontGutkin'
  # # model = 'MontbrioPazoRoxin'
  # # model='Generic2dOscillator'
  # (time, tavgCPU) = regularRun(simtime, g, s, dt, period).simulate_python(model)
  #
  # print('CPUshape', tavgCPU.shape)
  # print('GPUshape', tavgGPU.shape)
  #
  # # check for deviation tolerance between GPU and CPU
  # # for basic coupling and period = 1
  # # using euler deterministic solver
  # max_err = []
  # x = 0
  # for t in range(0, simtime):
  #   # print(t, 'tol:', np.max(np.abs(actual[t] - expected[t, :, :, 0])))
  #   # print(t, 'tol:', np.max(np.abs(tavgCPU[t,:,:,0], tavgGPU[t,:,:,0])))
  #   print(t)
  #   # print('C', tavgCPU[t,:,:,0])
  #   # print('G', tavgGPU[t,:,:,0])
  #   # print(t, 'tol:', np.max(np.abs(tavgCPU[t,:,:,0] - tavgGPU[t,:,:,0])))
  #   np.testing.assert_allclose(tavgCPU[t, :, :, 0], tavgGPU[t, :, :, 0], 2e-5 * t * 2, 1e-5 * t * 2)
