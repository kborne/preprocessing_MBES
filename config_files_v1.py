
###################################### CONFIGURATION ####################################

print('STARTING PREPROCESSING \n \n')

import os 
from configparser import ConfigParser
import numpy as np

import psana as ps

# from psana.hexanode.PyCFD import PyCFD


import sys
import os
import os.path


os.environ['PS_SRV_NODES'] = '1' #added for SLURM

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
 

file = 'config_v1.ini'
config = ConfigParser()
config.read(file)


###################################### PREPROCESS FOLDER ###################################

preprocessed_folder = config['DataSource']['preprocessed_folder'];


# print("current dir is: %s" % (os.getcwd()))

if os.path.isdir(preprocessed_folder):
    print('\n' + preprocessed_folder + "   :::  already exists \n")
else:
    print('\n' + preprocessed_folder + " ::: does not exists \n making it ... \n")
    os.mkdir(preprocessed_folder)
    

###################################### EXPERIMENT AND RUN NUMBER ###################################

exp = config['DataSource']['experiment']
#run_number = config['DataSource']['run_number']
run_number = int(sys.argv[1])

print('\n Calling data:   experiment = %s   run number = %s \n' % (exp,run_number))

suffix = config['DataSource']['h5_suffix']

# filename = preprocessed_folder+'run%d_%s.h5' % (run_number,suffix) #changeme
# print('{0}, {1}, {2}'.format('a', 'b', 'c'))
filename = '{0}/run{1}_{2}.h5'.format(preprocessed_folder,run_number,suffix)

if (os.path.isfile(filename) or os.path.isfile(filename.split('.')[0]+'_part0.h5')):
    raise ValueError('h5 files for run {0} already exist! check folder: {1}'.format(run_number, preprocessed_folder))
else:
    print('\n generating file: {0} \n ' .format(filename))
    
    
    

# ds = ps.DataSource(exp=exp, run=run_number,dir='/cds/data/drpsrcf/tmo/tmoly7820/xtc')
ds = ps.DataSource(exp=exp, run=int(run_number))
smd = ds.smalldata(filename=filename,batch_size=100) #IMP: batch_size=20 for 120 Hz. 
#Batch size is the number of events that are held in the memory before being to transferred to disk

update = 100 # Print update (per core)
default_val = -9999.0

######### MBES preproc settings ##########
MBESslice_inner_t1, MBESslice_inner_t2= float(config['waveform']['MBESslice_inner_t1']),float(config['waveform']['MBESslice_inner_t2']) #upper and lower bounds in us for inner channel
MBESslice_outer_t1, MBESslice_outer_t2= float(config['waveform']['MBESslice_outer_t1']),float(config['waveform']['MBESslice_outer_t2'])  #upper and lower bounds in us for outer channel
chan_inner = 3
chan_outer = 5
chan_pd = 6
chan_i0pd = 2

########### Hit Finding settings #######

# FFT ~ independent variables
FFT_thresh,FFT_deadtime,FFT_nskip = int(config['Peak Finding']['FFT_Threshold']),int(config['Peak Finding']['FFT_deadtime']),int(config['Peak Finding']['FFT_nskip']);



#xiangli@slac.stanford.edu 02/20/2018


from scipy.optimize import bisect


class PyCFD:

    def __init__(self, params):
        self.sample_interval = params['sample_interval']
        self.delay = int(params['delay']/self.sample_interval)
        self.fraction = params['fraction']
        self.threshold = params['threshold']
        self.walk = params['walk']
        self.polarity = 1 if params['polarity']=='Positive' else -1
        self.timerange_low = params['timerange_low']
        self.timerange_high = params['timerange_high']
        self.offset = params['offset']
        self.xtol = 0.1*self.sample_interval

        
    def NewtonPolynomial3(self,x,x_arr,y_arr):
    
        d_0_1 = (y_arr[1] - y_arr[0])/(x_arr[1] - x_arr[0])
        d_1_2 = (y_arr[2] - y_arr[1])/(x_arr[2] - x_arr[1])
        d_2_3 = (y_arr[3] - y_arr[2])/(x_arr[3] - x_arr[2])
        
        d_0_1_2 = (d_1_2 - d_0_1)/(x_arr[2] - x_arr[0])
        d_1_2_3 = (d_2_3 - d_1_2)/(x_arr[3] - x_arr[1])        
        d_0_1_2_3 = (d_1_2_3 - d_0_1_2)/(x_arr[3] - x_arr[0])
        
        c0 = y_arr[0]
        c1 = d_0_1
        c2 = d_0_1_2
        c3 = d_0_1_2_3
        
        return c0 + c1*(x-x_arr[0]) + c2*(x-x_arr[0])*(x-x_arr[1]) + c3*(x-x_arr[0])*(x-x_arr[1])*(x-x_arr[2])
    
            
    def CFD(self,wf, wt):        
        
        wf = wf[(wt>self.timerange_low)&(wt<self.timerange_high)] 
        wt = wt[(wt>self.timerange_low)&(wt<self.timerange_high)] #choose the time window of interest        
        
        wf_1 = wf[:-self.delay] #original waveform
        wf_2 = wf[self.delay:] #delayed waveform
       
        wf_cal = wf_1 - self.fraction*wf_2 #bipolar waveform
        wf_cal_m_walk = self.polarity*wf_cal-self.walk+self.polarity*(self.fraction*self.offset-self.offset) #bipolar signal minus the walk level
        wf_cal_m_walk_sign = np.sign(wf_cal_m_walk) 

        wf_cal_ind = np.where((wf_cal_m_walk_sign[:-1] < wf_cal_m_walk_sign[1:]) & 
        (wf_cal_m_walk_sign[1:] != 0) & ((wf_cal_m_walk[1:] - wf_cal_m_walk[:-1]) >= 1e-8))[0] #find the sign change locations of wf_cal_m_walk

        #check if the orignal signal is above the threhold at sign change locations of wf_cal_m_walk
        wf_cal_ind_ind = np.where(self.polarity*wf_1[wf_cal_ind] > (self.threshold+self.polarity*self.offset))[0]  

        
        t_cfd_arr = np.empty([0,])
        t_cfd_fdbk = [];
        
        #The arrival time t_cfd is obtained from the Newton Polynomial fitted to the 4 data points around the location found from above.
        try:
            for ind in wf_cal_ind_ind:

                t_arr = wt[(wf_cal_ind[ind]-1):(wf_cal_ind[ind]+3)]

                wf_cal_m_walk_arr = wf_cal_m_walk[(wf_cal_ind[ind]-1):(wf_cal_ind[ind]+3)]
            
                if len(t_arr) != 4 or len(wf_cal_m_walk_arr) != 4:
                    t_cfd_fdbk.append('array !=4')
                    continue
            
                if (t_arr[1] - t_arr[0])==0 or (t_arr[2] - t_arr[1])==0 or (t_arr[3] - t_arr[2])==0:
                    t_cfd_fdbk.append('i - (i-1) = 0')
                    continue
                
                if (t_arr[2] - t_arr[0])==0 or (t_arr[3] - t_arr[1])==0 or (t_arr[3] - t_arr[0])==0:
                    t_cfd_fdbk.append('i - (i-2) = 0')
                    continue
            
                t_cfd = bisect(self.NewtonPolynomial3,t_arr[1],t_arr[2],args=(t_arr, wf_cal_m_walk_arr),xtol=self.xtol)
                t_cfd_fdbk.append('good')

                t_cfd_arr = np.append(t_cfd_arr,t_cfd)
        except:
            t_cfd_fdbk.append('bad')
            t_cfd_arr = np.append(t_cfd_arr,wt[wf_cal_ind[ind]])

        return t_cfd_arr,t_cfd_fdbk,wf_cal_m_walk,wt[wf_cal_ind[wf_cal_ind_ind]]

# CFD Parameters ~ written to dictionary
CFD_params = {'sample_interval':0.00016,
     'fraction': float(config['Peak Finding']['CFD_fraction']),
     'delay':0.00016*float(config['Peak Finding']['CFD_delay']),
     'polarity':'Positive',
     'threshold':float(config['Peak Finding']['CFD_threshold']),
     'walk':int(config['Peak Finding']['CFD_walk']),
     'timerange_low':MBESslice_inner_t1,
     'timerange_high':MBESslice_inner_t2,
     'offset':float(config['Peak Finding']['CFD_offset'])};

CFD = PyCFD(CFD_params)



# SciPy Parameters 

from scipy.signal import find_peaks, peak_prominences

SciPy_height = float(config['Peak Finding']['SciPy_height'])
SciPy_threshold = float(config['Peak Finding']['SciPy_threshold'])
SciPy_distance = float(config['Peak Finding']['SciPy_distance'])
SciPy_prominence = float(config['Peak Finding']['SciPy_prominence'])
SciPy_width = float(config['Peak Finding']['SciPy_width'])
SciPy_plateau_size = float(config['Peak Finding']['SciPy_plateau_size'])


##########################################

##### deconvolution peak finder ##########
deconv_folder='/reg/neh/home/tdd14/TMO/MBES/'
sys.path.append(deconv_folder)
from FFT_peakfinder_v2 import peakfindFFT, fix_wf_baseline
maxhits=1000 # for number of electron hits
raw_resp_inner=np.load('/reg/neh/home/tdd14/TMO/MBES/raw_resp.npy') #changed 20210521
raw_resp_inner=np.load('/reg/neh/home/tdd14/TMO/MBES/raw_resp_inner.npy') #changed 20220420
# the above has some parameters hard coded in
##########################################

##########################################
Nfound = np.array([0])
Nbad = np.array([0])
times = None
# #####################################################
for run in ds.runs():
    
    # detectors - epics defined below
    timing = run.Detector("timing")
    hsd = run.Detector("hsd")
#     tmo_atmopal = run.Detector("tmo_atmopal") #defined below with different variable name
    tmo_fzpopal = run.Detector('tmo_fzpopal')
    gmd = run.Detector("gmd")
    xgmd = run.Detector("xgmd")
    ebeam = run.Detector("ebeam")
    pcav = run.Detector("pcav") # not used for the moment
    det_las_t = run.Detector('las_fs14_target_time')
    det_las_d = run.Detector('las_fs14_target_time_dial')
#     # det_lxt = run.Detector('lxt_ttc')
    
    #ATM motor and ATM opal
    det_dly = run.Detector('las_atm_dly')
    det_atmopal = run.Detector('tmo_atmopal')
    
    atm_roi = np.s_[600:780,:]
    atm_save_roi = np.s_[400:800,:]
    atm_background = None

    kernel = np.ones(100)/100 #smoothing function
    offset = 50 #offset for CFD
    
    #atm edge position and height
    step_pos = 0
    step_height = 0
    
    if hasattr(run, 'epicsinfo'):
        epics_strs = [item[0] for item in run.epicsinfo.keys()][1:] # first one is weird
        epics_detectors = [run.Detector(item) for item in epics_strs]

    max_nevent = int(config['DataSource']['max_nevent']);
    for nevent, event in enumerate(run.events()):

        if nevent > max_nevent:
            break
        
        if nevent%update==0: print("Event number: %d, Valid shots: %d" % (nevent, Nfound))
#         data ={}            
        data = {'epics_'+epic_str: epic_det(event) for epic_str, epic_det in zip(epics_strs, epics_detectors)}
        
        if any(type(val) not in [int, float] for val in data.values()):
            print("Bad EPICS: %d" % nevent)
            Nbad += 1
            continue

    
        hsd_data = hsd.raw.waveforms(event)
        if hsd_data is None:
            print("Bad HSD: %d" % nevent)
            Nbad += 1
            continue
        zps = tmo_fzpopal.raw.image(event)
        if zps is None:
            print("Bad zp-spectrometer: %d" % nevent)
            Nbad += 1
            continue
        evrs = timing.raw.eventcodes(event)
        if evrs is None:
            print("Bad EVRs: %d" % nevent)
            Nbad += 1
            continue
        '''  Delay Stage Motors '''
        laser_delay = det_las_t(event)
        if laser_delay is None: 
            print('Laser Delay is None, continuing: %d' % nevent)
            Nbad+=1
            continue
        else:
            data['laser_delay'] = laser_delay;
        
        las_atm_dly = det_dly(event)
        if las_atm_dly is None: 
            print('las_atm_dly is None, continuing: %d' % nevent)
            Nbad+=1
            data['las_atm_dly'] = np.nan
        else:
            data['las_atm_dly'] = las_atm_dly
        
        # lxt_delay = det_lxt(event)
        # if laser_delay is None: 
        #     print('LXT is None, continuing: %d' % nevent)
        #     Nbad+=1
        #     continue
        # else:
        #     data['lxt_delay'] = lxt_delay
            
        laser_dial = det_las_d(event)
        if laser_dial is None:
            print("Bad Laser Dial: %d" %nevent)
        else:
            data['laser_dial'] = laser_dial
            
        evrs = np.array(evrs)
        if evrs.dtype == int:
            data['evrs'] = evrs.copy()
        else:
            print("Bad EVRs: %d" % nevent)
        
        bad = False
        for (detname, method), attribs in run.detinfo.items():
            if bad: break
            if (detname not in ['timing', 'hsd', 'tmo_fzpopal', 'epicsinfo', 'tmo_opal1', 'xtcav']) and not ((detname=='tmo_atmopal') and (method=='raw')): #xtcav added 20210510
                for attrib in attribs:
#                     print(attrib)
                    if detname!='tmo_atmopal' or attrib not in ['proj_ref', 'proj_sig', 'calib', 'image', 'raw']:
                        val = getattr(getattr(locals()[detname], method), attrib)(event)
                        if val is None:
                            if detname in ['ebeam', 'gmd', 'xgmd'] and evrs[161]: # BYKIK gives None for these, but we still want to process the shots
                                val = default_val
                            else:
                                bad = True
                                print("Bad %s: %d" % (detname, nevent))
                                Nbad += 1
                                break
                        data[detname+'_'+attrib] = val
        if bad:
            continue
        
        
        #VLS:
        
        #Zone Plate pixel position 
        zps_data = tmo_fzpopal.raw.image(event)
        
        ## pixel peak position is identified by summing along the axis and max position of the summation
        zps_pixel = 100+np.argmax(np.sum(zps_data[100:900],axis=1))
        px_2d_width = 25 #spacing to cut the 2d image slice to store. Typically 50 pixel width. 25 on either side
        px_edge_width = 5 #spacing to integrate for the line out. 
        
        zps_2d = zps_data[zps_pixel-px_2d_width:zps_pixel+px_2d_width][:]
        zps = zps_data[zps_pixel-px_edge_width:zps_pixel+px_edge_width][:].mean(0)#
        zps_bg = zps_data[zps_pixel-px_2d_width:zps_pixel-px_2d_width+px_edge_width][:].mean(0)#
        
        #Saving Zone plate 2D cut, lineout and background lineout
        data['zoneplate_photon_pixel']=np.argmax(np.subtract(zps,zps_bg))
        data['zoneplate2d'] = zps_2d.copy()
        data['zoneplatespec'] = zps.copy()
        data['zoneplateBG'] = zps_bg.copy()
        
        ################# ATM edge finding ###################
        atmimg = det_atmopal.raw.image(event)
                
        if ( ( data['evrs'][68] > 0.5 ) or (data['evrs'][161] > 0.5) ):
            bkgd = atmimg[atm_roi]
            step_pos = 0
            step_height = 0

            if atm_background is None:
                atm_background = bkgd
            else:
                atm_background = 0.5*atm_background+0.5*bkgd

            atm_out = np.sum( bkgd.astype(float)/atm_background.astype(float) ,0)

        if ( (data['evrs'][67] > 0.5 ) and (data['evrs'][161] < 0.5) ):
            
            if atm_background is None:
                atm_out = np.sum( atmimg[atm_roi].astype(float),0)
            else:
                #Divide ATM signal then sum
                atm_out = np.sum( atmimg[atm_roi].astype(float)/atm_background.astype(float) ,0)

            #Analyze with CFD
            ana = atm_out[offset:]-atm_out[:-offset]

            convolved = np.convolve(ana, kernel)
            convolved /= np.max(convolved)
            convolved = convolved[len(kernel):-len(kernel)]

            q = np.argmax(convolved)
            o = int(offset/2) + int(len(kernel)/2)
            t_s = np.max([q+o-offset*2,0])
            t_f = np.min([q+o+offset*2,len(atm_out)])
            step_trace = atm_out[t_s:t_f]
            step_pos = q
            step_height = np.max(step_trace) - np.min(step_trace)

#                 evt_dict['atm_img_slice'] = atmimg[atm_save_roi]
        data['atm_trace'] = atm_out
        data['atm_pos'] = step_pos
        data['atm_height'] = step_height
        
        ############### MBES Traces: Waveform and Hit finder#################
        
        # get MBES waveform data
        if times is None:
            times = hsd_data[chan_inner]['times'] * 1e6 # times will be same for all  channels
            MBESslice_inner_i1, MBESslice_inner_i2 = np.argmin(abs(MBESslice_inner_t1-times)), \
                                                     np.argmin(abs(MBESslice_inner_t2-times))
            MBESslice_outer_i1, MBESslice_outer_i2 = np.argmin(abs(MBESslice_outer_t1-times)), \
                                                     np.argmin(abs(MBESslice_outer_t2-times))
#             times=times[MBESslice_i1:MBESslice_i2+1] # removed 20210510
        
        wf_MBES_inner = hsd_data[chan_inner][0].astype('float')
        wf_MBES_inner = fix_wf_baseline(wf_MBES_inner)
            
        wf_MBES_outer = hsd_data[chan_outer][0].astype('float')
        wf_MBES_outer = fix_wf_baseline(wf_MBES_outer)
            
        data['MBES_wf_inner'] = -wf_MBES_inner[MBESslice_inner_i1: MBESslice_inner_i2+1].copy()
        data['MBES_wf_outer'] = -wf_MBES_outer[MBESslice_outer_i1: MBESslice_outer_i2+1].copy()
        
        
        # Original threshold was "5". Increasing it based on Sergey's plot for run 74.
        # Attempted with '6', '7' '8' seemed reasonable, '10' was too high
        # Settled on '7'
        
        
############################################################ now run hit finder on MBES waveform - inner anode ##########################################################################################
    ###### FFT routine    
    
    # parameters from initilization file
    
        tpk_inner, Ipk_inner = peakfindFFT(-wf_MBES_inner, threshold=FFT_thresh, raw_resp=raw_resp_inner, deadtime=FFT_deadtime, nskip=FFT_nskip)[:,1].astype(int), np.zeros(maxhits)
        tpk_inner = tpk_inner[tpk_inner<len(times)]
        if len(tpk_inner)>maxhits:
            print('Warning! Number of peaks on inner anode (%i) exceeds \'maxhits\' (%i)' % (len(tpk_inner), maxhits))
            data['MBESpk_inner_t'] = times[tpk_inner[:maxhits]].copy() # in us
            data['MBESpk_inner_I'] = -wf_MBES_inner[tpk_inner[:maxhits]].copy()
        else:
            tpk_inner_times = np.ones(maxhits) * default_val
            if len(tpk_inner)>0:
                tpk_inner_times[:len(tpk_inner)] = times[tpk_inner]
                data['MBESpk_inner_t'] = tpk_inner_times.copy() #in us
                Ipk_inner[:len(tpk_inner)] = -wf_MBES_inner[tpk_inner]
                data['MBESpk_inner_I'] = Ipk_inner.copy()

        ###### CFD routine
        
        # parameters for CFD are loaded above
        
        pd_inner,pd_fdbk,binary_sig,ind = CFD.CFD(-wf_MBES_inner,times)
        data['binary_sig'] = binary_sig;
        if len(pd_inner) > maxhits:
            print('Warning! CFD HIT FINDER Number of peaks on inner anode (%i) exceeds \'maxhits\' (%i)' % (len(tpk_inner), maxhits))
            data['inner_t_CFD'] = pd_inner[:maxhits]
        else:
            pd_inner_times = np.ones(maxhits) * default_val
            if len(pd_inner)>0:
                pd_inner_times[:len(pd_inner)] = pd_inner
                data['inner_t_CFD'] = pd_inner_times.copy() #in us
                
                
                
        ###### SciPy routine
            
        tpk_sp_inner= find_peaks(-wf_MBES_inner,height=SciPy_height, threshold=SciPy_threshold, distance=SciPy_distance, prominence=SciPy_prominence, width=SciPy_width, plateau_size=SciPy_plateau_size)[0]
        tpk_sp_inner = tpk_sp_inner[tpk_sp_inner<len(times)]
        if len(tpk_sp_inner) > maxhits:
            print('Warning! CFD HIT FINDER Number of peaks on inner anode (%i) exceeds \'maxhits\' (%i)' % (len(tpk_inner), maxhits))
            data['inner_t_SciPy'] = times[tpk_sp_inner[:maxhits]].copy()
        else:
            tpk_sp_inner_times = np.ones(maxhits) * default_val
            if len(tpk_sp_inner)>0:
                tpk_sp_inner_times[:len(tpk_sp_inner)] = times[tpk_sp_inner]
                data['inner_t_SciPy'] = tpk_sp_inner_times.copy() #in us
        
            
        #data['MBESpk_inner_I'] = Ipk_inner.copy()
        
        # now run hit finder on MBES waveform - outer anode
        # tpk_outer, Ipk_outer = peakfindFFT(-wf_MBES _outer, threshold=5, raw_resp=raw_resp_inner, deadtime=10, nskip=20)[:,1].astype(int),  np.zeros(maxhits)
        tpk_outer, Ipk_outer = peakfindFFT(-wf_MBES_outer, threshold=7, raw_resp=raw_resp_inner, deadtime=10, nskip=20)[:,1].astype(int), np.zeros(maxhits)
        tpk_outer = tpk_outer[tpk_outer<len(times)]
        if len(tpk_outer)>maxhits:
            print('Warning! Number of peaks on outer anode (%i) exceeds \'maxhits\' (%i)' % (len(tpk_outer), maxhits))
            data['MBESpk_outer_t'] = times[tpk_outer[:maxhits]].copy() # in us
            data['MBESpk_outer_I'] = -wf_MBES_outer[tpk_outer[:maxhits]].copy()
        else:
            tpk_outer_times = np.ones(maxhits) * default_val
            if len(tpk_outer)>0:
                tpk_outer_times[:len(tpk_outer)] = times[tpk_outer.astype(int)]
                data['MBESpk_outer_t'] = tpk_outer_times.copy() #in us
                Ipk_outer[:len(tpk_inner)] = -wf_MBES_outer[tpk_inner]
                data['MBESpk_outer_I'] = Ipk_outer.copy()
        
#        #data['MBESpk_outer_I'] = Ipk_outer.copy()

        # take the photodioide waveform and save its sum to h5
        wf_pd = hsd_data[chan_pd][0].astype('float')
        wf_pd = fix_wf_baseline(wf_pd)
        data['wf_pd_sum'] = wf_pd[4500:6000].sum()
        # take a slice of the first 10,000 elements of the waveform
        data['wf_pd_slice'] = wf_pd[4500:7000]
        
        wf_i0pd = hsd_data[chan_i0pd][0].astype('float')
        data['wf_i0pd_sum'] = wf_i0pd[4500:6000].sum()
        # take a slice of the first 10,000 elements of the waveform
        data['wf_i0pd_slice'] = wf_i0pd[4500:7000]
        
        valid_data = True
        for key, val in data.items():
            if (type(val) not in [int, float]) and (not hasattr(val, 'dtype')):
                print("Bad data:", key)
                valid_data = False
                break
        
        if valid_data:
            smd.event(event, **data)
            Nfound += 1
        else:
            Nbad += 1
            continue
        
if smd.summary:
    try:
        Nbad = smd.sum(Nbad)
    except:
        Nbad = -9999
        print('Problem with Nbad!')
    try:
        Nfound = smd.sum(Nfound)
    except:
        Nfound = -9999
        print('Problem with Nfound!')
    smd.save_summary(Nfound=Nfound, Nbad=Nbad, MBESslice_inner_t1=MBESslice_inner_t1, MBESslice_inner_t2=MBESslice_inner_t2, \
                     MBESslice_outer_t1=MBESslice_outer_t1, MBESslice_outer_t2=MBESslice_outer_t2)
    
smd.done()
    
print('----------------- DONE -----------------')
#if rank == (size - 1):
#    perms = '444' # fo-fo-fo
#    for f in [filename.split('.')[0]+'_part0.h5', filename]:
#        os.chmod(f, int(perms, base=8))



