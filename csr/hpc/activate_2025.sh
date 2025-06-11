#!/bin/bash/env bashasd

ml Stages/2025
ml GCC/13.3.0  
#ml OpenMPI/4.1.5 
ml ParaStationMPI//5.11.0-1
ml CUDA Python SciPy-Stack numba
#ml PyCUDA 
ml mpi4py/4.0.1 scikit-learn matplotlib
ml PyTorch/2.5.1


export PYTHONPATH=$PYTHONPATH:/p/project1/cslns/$USER/LiquidInterferenceLearning/TVB/tvb_library

