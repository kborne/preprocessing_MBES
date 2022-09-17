import sys
import numpy as np
import scipy as sp
import time
import h5py


expt='tmoly0220' 

runs = np.arange(205,210+1)

tstart = time.time();

for run_n,run in enumerate(runs):
    
    trun = time.time();

    start = time.time()
    
    raw = h5py.File("/cds/data/psdm/tmo/tmoly0220/scratch/ffb/preproc/v6/run%i_v6.h5" % (run), 'r') 
    
    # define the order to sort all subsequent arrays to experimental time
    order  = raw['timestamp'][:].argsort(); 
    shots = len(order)
    
    
    # Get things related to the photon energy for photon-electron correlation and subsequent FEL spectral binning
    photEn_dummy = np.array(raw['zoneplatespec'])[order,:]; 
    
#     # Get things related to the delay value
    
    
    atm_trace = np.array(raw['atm_trace'])[order,:];
    



    # FEL Pulse Energy for GMD
    pulsEn = np.array(raw['xgmd_energy'])[order];

    # Event codes
    goose =  np.array(raw['evrs'][:,68].astype(bool))[order];   # ~ laser mistimed by several nanoseconds
    xrayq =  np.array(~raw['evrs'][:,161].astype(bool))[order]; # ~ BY-kick no FEL present
    ph60hz = np.array(raw['evrs'][:,41].astype(bool))[order];   # ~ sort by even/odd shots to account for drifting in the undulator


    hf_dummy = np.array(raw['MBESpk_inner_t'])[order,:];
    
    
        
    t_extract = time.time()-trun
    
    print('\t run %s load time: %s' % (run,t_extract))

# do stuff
t_end = time.time() - tstart

print('Total Extraction TIme: %s' % t_end)