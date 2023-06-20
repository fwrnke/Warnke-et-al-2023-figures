"""
Generating seismic wiggle traces for workflow schematic figure (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-12-02

"""
import os
import sys

import segyio
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import gain, rescale, _print_info_TOPAS_moratorium

_print_info_TOPAS_moratorium()

#%% FUNCTIONS

def smooth(y, box_pts):
    """Smooth seismic trace."""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#%% MAIN
if __name__ == '__main__':

    dir_fig = './auxiliary'
    
    dir_file = 'C:/PhD/processing/TOPAS/TAN2006/pockmarks_3D'
    file = '20200703015945_UTM60S.sgy'
    path = os.path.join(dir_file, file)
        
    basepath, filename = os.path.split(path)
    basename, suffix = os.path.splitext(filename)
    
    with segyio.open(path, 'r', strict=False, ignore_geometry=True) as src:
        n_traces = src.tracecount           # total number of traces
        dt = segyio.tools.dt(src) / 1000    # sample rate [ms]
        n_samples = src.samples.size        # total number of samples
        twt = src.samples                   # two way travel time (TWTT) [ms]
        
        print(f'n_traces:  {n_traces}')
        print(f'n_samples: {n_samples}')
        print(f'dt:        {dt} ms')
        
        tracl = src.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]  # Trace sequence number within line – numbers continue to increase if additional reels are required on same line.
        tracr = src.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]  # Trace sequence number within reel – each reel starts at trace number one
        fldr = src.attributes(segyio.TraceField.FieldRecord)[:]  # field record number
        
        delrt = src.attributes(segyio.TraceField.DelayRecordingTime)[:]  # Delay recording time (ms)
        
        # get seismic data [amplitude]; transpose to fit numpy data structure
        data = src.trace.raw[:].T  # eager version (completely read into memory)
    
    #%% Plot wiggle
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    idx_tr = 323
    idx_twt = np.where((twt >= 755) & (twt <= 775))[0]
    _data = data[idx_twt, idx_tr]
    
    filt_size_start = 100
    filt_size_end = 400
    filt = np.ones_like(_data)
    filt[:filt_size_start] = np.exp(np.log(1e-1 / 1) * (np.arange(filt_size_start) - 1) / (filt_size_start - 1))[::-1]
    # filt[-filt_size:,] = np.linspace(1, 0, filt_size)  # linear
    filt[-filt_size_end:,] = np.exp(np.log(1e-1 / 1) * (np.arange(filt_size_end) - 1) / (filt_size_end - 1))
    data_smooth = smooth(_data * filt, 5)
    
    data_smooth_gain = gain(data_smooth, twt=twt[idx_twt] / 1000, tpow=2, gpow=0.5)
    data_smooth_gain = rescale(abs(data_smooth_gain), 0, 0.2) * np.sign(data_smooth)
    
    ax.plot(twt[idx_twt], _data, c='grey', marker='o', markersize=2, ls='-', label='original')
    ax.plot(twt[idx_twt], data_smooth, c='b', ls='-', label='smooth')
    ax.plot(twt[idx_twt], data_smooth_gain, c='g', ls='-', label='smooth (gain)')
    
    ax.legend()
    fig.tight_layout()
    
    plt.savefig(os.path.join(dir_fig, f'{basename}_wiggle_trace.svg'))
    
    #%% Plot spike
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    idx_tr_spike = tracr[fldr == 103443][0]
    idx_twt_spike = np.where((twt >= 680) & (twt <= 710))[0]
    _spike = data[idx_twt_spike, idx_tr_spike]
    
    ax.plot(twt[idx_twt_spike], _spike, c='k', ls='-', label='original')
    fig.tight_layout()
    
    plt.savefig(os.path.join(dir_fig, f'{basename}_wiggle_trace_spike.svg'))
    
    #%%
    # from scipy.signal import chirp
    # signal = chirp(twt[idx_twt_spike], f0=1000, t1=twt[idx_twt_spike][-1], f1=1000)
    
    pts = 100
    pts_end = int(pts * 0.1)
    signal = np.sin(np.arange(pts))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
    factor = np.linspace(0.2, 1, pts)
    factor[-pts_end:] = np.linspace(0.9, 0.4, pts_end)
    ax.plot(signal * factor, 'k')
    
    plt.savefig(os.path.join(dir_fig, f'{basename}_wiggle_trace_spike_synthetic.svg'))
