
# ###################################### CONFIGURATION ####################################

# print('STARTING PREPROCESSING \n \n')

# import os 
# from configparser import ConfigParser
# import numpy as np
# import psana as ps
# import sys
# import os
# import os.path


# os.environ['PS_SRV_NODES'] = '1' #added for SLURM

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
 

# file = 'config_test.ini'
# config = ConfigParser()
# config.read(file)

# t1 = float(config['waveform']['MBESslice_inner_t1']);
# t2 = float(config['waveform']['MBESslice_inner_t2']) 

# t1_t2 = t1+t2;

# print('\n t1 = %s  t2 = %s \n' % (t1,t2))
# print('\n  t1+t2 = %s' % t1_t2);

# fft_deadtime = float(config['Peak Finding']['FFT_deadtime']);

# print(' \n fft_deadtime = %s' % fft_deadtime);

print('submission completed')