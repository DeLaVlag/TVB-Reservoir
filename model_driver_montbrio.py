from __future__ import print_function

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD

import argparse

from tvb.simulator.lab import *

import os
os.environ["PATH"] += ":/usr/local/cuda-10.2/bin"

import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import time
import tqdm

from metrics.pci_driver import run_pci_analy
from utils.lorentz import *
import itertools
from utils.torch_Tik import *
from metrics.gpu_dfa import computeDFA_gpu
from metrics.gpu_lya import computeLYA_gpu
from utils.create_task import *
from utils.upsamples import *
from plottools.plot_print_all import *
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

    spectral_radius = .95
    self.weights *= spectral_radius / max(abs(np.linalg.eigvals(self.weights)))

    self.lengths = self.connectivity.tract_lengths

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

    self.params, self.wi_per_rank, self.params_toplot, self.s0, self.s1 = self.setup_params()

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
    sfile = "connectivity_96.zip"
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


  def setup_params(self):
    '''
    This code generates the parameters ranges that need to be set
    '''

    # for paper simulations
    slh = [
           20, 20,  # 0.4 - 0.8 prev global coupling .33, .495,
           0.01, 0.01,  # weight_noise 0.03, 0.05. or fix to .01 for tests
           -20, 10,  # External Current -.5, 4.0,
           -6, 30,  # Mean Synaptic weight J 12, 16,  #
           -10, 10,  # Constant parameter to scale the rate of feedback from the firing rate variable to itself -5., -3.,
           1.0, 1.0,  # Mean heterogeneous noise delta .5, .8,  #
           4., 4.]

    s0 = np.linspace(slh[0], slh[1], self.args.n_sweep_arg0)  # coupling
    s1 = np.linspace(slh[2], slh[3], self.args.n_sweep_arg1)  # b_e
    s2 = np.linspace(slh[4], slh[5], self.args.n_sweep_arg2)  # weight_noise
    s3 = np.linspace(slh[6], slh[7], self.args.n_sweep_arg3)  # global_speed
    s4 = np.linspace(slh[8], slh[9], self.args.n_sweep_arg4)  # tau_w_e
    s5 = np.linspace(slh[10], slh[11], self.args.n_sweep_arg5)  # a_e
    s6 = np.linspace(slh[12], slh[13], self.args.n_sweep_arg5)  # global speed

    params = itertools.product(s0, s1, s2, s3, s4, s5, s6)
    params = np.array([vals for vals in params], np.float32)

    # for plotting
    params_toplot = itertools.product(s0, s1)
    params_toplot = np.array([vals for vals in params_toplot], np.float32)

    # mpi stuff
    wi_total, wi_one_size = params.shape
    wi_per_rank = int(wi_total / self.world_size)
    wi_remaining = wi_total % self.world_size
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

        # func signature creation
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

    # spawn of the chunked prediction to inject datastruct
    learned_thoughts = np.zeros((32, 100, 3))

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
          tavgpartly = np.array(tavg_unpinned)
          # subtract nstreams of the index as this is copied in later. we dont need to correct as we dont use the
          # first 32 results anyway. the index of tavg is nstreams down on the current
          h = i - thoughtsize - n_streams
          j = h + thoughtsize
          Rpart = tavgpartly[h:j,0,:,:]
          # do partly ridging
          Rchunktik = Rpart.transpose(2,1,0)

          k = i + thoughtsize # fitting for the future
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
            # for ICANN25 paper:
            deltapredchunk = -40
            plotpred_teach_train( Ypast[:,i-40:i+40], pred_partly, YchunkTik[:,i-40:i+40,:], self.trainingloops,
                                  "", deltapredchunk)

          togpuChunk = pred_partly_r.transpose(0,2,1)

          # only amplify the 3rd dim of lorentz by
          scalar = 3.5
          last_result = togpuChunk[:, -1, :]
          scaled_result = last_result * scalar
          togpuChunk[:, -1, :] = scaled_result

          gpu_data['thoughts'] = gpuarray.to_gpu(self.cf(togpuChunk*self.scale_input))

        if i == nstep - 1:
          states = gpu_data['state'].get()

      # recover uncopied data from pinned buffer
      if nstep > n_streams:
        for i in range(nstep % n_streams, n_streams):
          stream.synchronize()
          tavg_unpinned.append(tavg[i].copy())

      for i in range(nstep % n_streams):
        stream.synchronize()
        tavg_unpinned.append(tavg[i].copy())

    except drv.LogicError as e:
      self.logger.error('%s. Check the number of states of the model or '
             'GPU block shape settings blockdim.x/y %r, griddim %r.',
             e, final_block_dim, final_grid_dim)
      exit(1)
    except drv.RuntimeError as e:
      self.logger.error('%s', e)
      exit(1)


    tavg = np.array(tavg_unpinned)
    self.release_gpumem(gpu_data)

    if self.my_rank == 0:

      self.logger.info('kernel finished')

    return tavg, states


  def create_task_dyn(self, phase_shift_pis=2, h=0.01, nonzeroffset=0, Sinussus=False):

    np.random.seed(42)

    if Sinussus==False:

      # for significance we use the following conditions
      # initial_conditions = [
      #   [1.0, 1.0, 1.0],
      #   [0.1, 0.0, 0.0],
      #   [5.0, 5.0, 5.0],
      #   [-1.0, -1.0, 0.5],
      #   [2.0, 3.0, 4.0],
      # ]

      x0 = 1
      y0 = 1
      z0 = 1

      # h = 0.005
      x, y, z = forward_euler(x0, y0, z0, h, self.args.n_time * self.n_inner_steps)

      x, y, z = normalize_lorenz_signals(x, y, z)
      fixed_pred_set = np.vstack((x, y, z))
      teach_pred_set = np.vstack(shift_lorenz_trajectory(x, y, z, phase_shift_pis=phase_shift_pis, h=h))

    else:

      fixed_pred_set = np.vstack(generate_sinusoidal_signals(time_steps=25000))
      teach_pred_set = np.vstack(generate_sinusoidal_signals(time_steps=25000, pshifts=phase_shift_pis))

    fixed_pred_set += nonzeroffset
    teach_pred_set += nonzeroffset

    return fixed_pred_set, teach_pred_set


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
       nWout[z, :, :] = nWout[z, :, :]/np.max(nWout[z, :, :])

    # copy the normalized Wout to the output regions of TVB
    for j in range(self.n_work_items):
      for x, regio in enumerate(outregions):
        for i, conregion in enumerate(nodes_minus_inoutreg):
          self.weights[j, conregion, regio] = nWout[j,i,x]

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

    char_freq_lenghts = compute_char_frequency_from_delays(lengths)

    # 1.17 to set it gleich with the heuristic values for the 96 of 3.8
    char_frequency = blended_characteristic_frequency(char_freq_weights, char_freq_lenghts, alpha=1.17)
    # print("char_freq_blended", char_frequency)

    taus = (spd, scd, dfd)  # Your computed delay estimates

    tau_conn = compute_connection_delay(weights, lengths)  # Compute weighted connection delay
    phase_shift_pis, time_shift = compute_phase_shift_withdelay(taus, tau_conn, char_frequency)

    return phase_shift_pis


  def run_pci_analy_gpu(self, tested_ts):

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

    normalized_tsss = []  # Store all scaled targets for each work item

    for sim_idx in range(self.n_work_items):

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

      normalized_tsss.append(np.array(normalized_tss))  # From shape (16, 68, 200)

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


    targetdyn_expanded = np.expand_dims(targetdyn_norm, axis=0)  # Shape: [1, nts, noutpus]
    targetdyn_tiled = np.tile(targetdyn_expanded, (self.n_work_items, 1, 1))

    return np.array(targetdyn_tiled), np.array(normalized_tsss)


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

    nodes_minus_inoutreg = range(self.args.n_regions)

    best_mse_tloop = np.full((self.n_work_items, 2), [float('inf'), -1])

    self.pshift = 3.8 # for h.004 for lorentz for 96 nodes!
    dominant_frq = 1.4 #Hz
    sampling_rate = 100 #Hz
    time_shift_seconds = self.pshift / (2 * np.pi * dominant_frq)
    self.delay_timesteps = int(round(time_shift_seconds * sampling_rate))

    self.scale_input = .5

    nonzeroffset = 0
    # Set the task to sinus instead of lorentz
    SinusTask=False
    stepsizeforLorentz = 0.004

    # create fixed set
    notfixpredictiontask = False
    if notfixpredictiontask == False:

      fixed_pred_set, teach_pred_set = self.create_task_dyn(phase_shift_pis=self.pshift, h=stepsizeforLorentz,
                                                            nonzeroffset=nonzeroffset, Sinussus=SinusTask)

    # self.trainingloops = 2
    Ys_Ypreds = np.full((2, self.params.shape[0], (predend-predstart), len(outregions)), float('inf'))

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


      # RUN SIMULATION for RIDGE REGRESSION
      trainer = target_dyn_train[set_dyn, :inject_dyn_ridge]
      tearchr = teachr_dyn_train[set_dyn, :inject_dyn_ridge]
      if self.my_rank == 0:
        self.logger.info("training loop commencing")
        self.logger.info('target_dyn.shape %s', target_dyn_train[set_dyn, :inject_dyn_ridge].shape)
        self.logger.info('trainer0.shape %s', trainer.shape)
        self.logger.info('tearchr0.shape %s', tearchr.shape)
        self.logger.info('phase shift %f', self.pshift)

      statesbuf = 0
      dummyWout = 0

      # scale trainer / teacher
      trainer = trainer*self.scale_input
      tearchr = tearchr*self.scale_input

      tavg0, statesbuf = self.run_tvb_rc(trainer, tearchr, inject_dyn_ridge, inject_dyn_pred, dummyWout, statesbuf, usebuffer=False)
      if self.my_rank == 0:
        self.logger.info("tavg0.shape %s", tavg0.shape)
        self.logger.info("states.shape %s", statesbuf.shape)

      # compute regression for weights
      # the same input is used to train the network
      # Regularization parameter for ridge regression
      beta_ridge = self.beta_ridge
      # beta_ridge = 1
      if self.my_rank == 0:
        self.logger.info("NaNs in tavg0: %s", np.isnan(tavg0).any())
        self.logger.info("Infinite values in tavg0: %s", np.isinf(tavg0).any())
        self.logger.info('betaridge %f', beta_ridge)

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

      # RIDGE REGRESSION
      if self.my_rank == 0:
        self.logger.info("scaled_dyn_train.shape %s", scaled_dyn_train.shape)
        self.logger.info("normalize_tavg.shape %s", tavg_copy_train.shape)
        self.logger.info("scaled_dyn_treachr.shape %s", scaled_dyn_treachr.shape)

      rolled_dyn_train = scaled_dyn_treachr


      Rtrain = tavg_copy_train.transpose(0, 2, 1)[:, nodes_minus_inoutreg, :]

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

      self.update_output_weights(Wout, outregions, nodes_minus_inoutreg)

      go_ahead = True
      if tloop % self.tlsamples == 0 and go_ahead:
        #### PREDICTION ####

        if self.my_rank == 0:
          self.logger.info("Prediction Phase")

        # create test data to do predictions
        if notfixpredictiontask == True:

          target_dyn_pred, teachr_dyn_pred = self.create_task_dyn(phase_shift_pis=self.pshift, h=stepsizeforLorentz,
                                                                  nonzeroffset=nonzeroffset, Sinussus=SinusTask)

        else:
          # use a fixed prediction set
          target_dyn_pred = fixed_pred_set
          teachr_dyn_pred = teach_pred_set

        # alter for injection from 250
        trainer = target_dyn_pred[set_dyn, :] * self.scale_input
        tearchr = teachr_dyn_pred[set_dyn, :] * self.scale_input

        tavg1, _ = self.run_tvb_rc(trainer, tearchr, inject_dyn_pred,
                                   inject_dyn_pred, Wout,
                                   statesbuf, usebuffer=True)

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

        # Test Tikhonov and predict 100 timesteps
        R = tavgcopy[:, predstart:predend, nodes_minus_inoutreg]

        poffset = 0
        yoffset = 0 # scale up + 3 when creating the task
        R = R.transpose(0, 2, 1)+poffset
        Y = scaled_predict_dyn[:, predstart: predend] - yoffset #.transpose(0, 2, 1)
        pureMSE, Ypred = predi_Torch_norm(R, Y, Wout)
        normalized_MSE = pureMSE

        pureMSE = np.clip(pureMSE, 0, 2.5)


        if self.my_rank == 0:
          self.logger.info("Y.shapeb4preditorc %s", Y.shape)
          self.logger.info("R.shapeb4preditorc %s", R.shape)
          self.logger.info("R.min %s", R.min())
          self.logger.info("R.max %s", R.max())
          self.logger.info("Ypred.max %s", Ypred.shape)

        Ys=Y[:]
        Ypreds=Ypred[:]

        MSEs.append(normalized_MSE)

        pci, lzc = self.run_pci_analy_gpu(tavg1[:,0,:,:])
        # clip solution for the extreme value
        pci = np.clip(pci, -1, 3)

        print('tavg1shape.T', tavg1[:,:,:,0].shape)
        np.save("data/tavg1_pred0", tavg1[:,:,:,0])
        np.save("data/tavg1_pred1", tavg1[:,:,:,1])
        np.save("data/tavg1_pred2", tavg1[:,:,:,2])
        np.save("data/tavg1_pred3", tavg1[:,:,:,3])

        dfa = computeDFA_gpu(np.ascontiguousarray(tavg1.T[:,nodes_minus_inoutreg_anal,0,:]), self.logger, self.my_rank)
        dfa = np.mean(dfa, axis=1)

        lya = computeLYA_gpu(np.ascontiguousarray(tavg1.T[:,nodes_minus_inoutreg_anal,0,predstart:predend]),
                             self.logger, self.my_rank, emb_dim=4, lag=5, min_tsep=10, trajectory_len=50, tau=self.dt)
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

      #@@@ POST PROCESSING @@@#
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

      MSEs = np.array(MSEs, dtype=np.float32)
      MSEstoPlot = all_mses[best_indices]

      pretty_print_best_params(best_indices, MSEstoPlot, best_mse_tloop[:, 1], paramstoplot, "Sorted best achievable MSE")

    predY = all_Y_preds[best_indices]
    trueY = all_Ys[best_indices]
    plotpred(trueY, predY, self.trainingloops, "Best ever MSE")

    # show plots eventually
    notrunonhpc = True
    if notrunonhpc == True:
      plt.show()

    toc = time.time()
    elapsed = toc - tic

    if self.my_rank == 0:
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
