#!/bin/bash/env bashasd

ml Stages/2024
ml GCC/12.3.0  OpenMPI/4.1.5 ParaStationMPI/5.9.2-1
ml CUDA Python SciPy-Stack numba
ml PyCUDA mpi4py/3.1.4 scikit-learn matplotlib
ml PyTorch/2.1.2


export PYTHONPATH=$PYTHONPATH:/p/project1/cslns/$USER/LiquidInterferenceLearning/TVB/tvb_library

