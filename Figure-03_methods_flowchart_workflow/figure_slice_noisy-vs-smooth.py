"""
Short description.


"""
import os

import xarray as xr
import matplotlib.pyplot as plt

#%% MAIN

if __name__ == '__main__':
    
    dir_fig = './auxiliary'
    dir_work = r'E:\PhD\processing\TOPAS\TAN2006\paper\rev1'
    file = 'cube_center_IDW_env_5x5m_0+05ms_freq-il-xl_preproc_IL-725-825_XL-800-900-trunc_FFT_hard_niter-50.nc'
    file_smooth = 'cube_center_IDW_env_5x5m_0+05ms_freq-il-xl_preproc_IL-725-825_XL-800-900-trunc_FFT_hard_niter-50_gaussian-1.nc'
    
    chunks = dict(freq_twt=1, iline=-1, xline=-1)
    cube = xr.open_dataset(os.path.join(dir_work, file), engine='h5netcdf', chunks=chunks)
    cube_smooth = xr.open_dataset(os.path.join(dir_work, file_smooth), engine='h5netcdf', chunks=chunks)
    
    selection = dict(freq_twt=0.075, method='nearest')
    var = 'freq_env_interp.real'
    kwargs = dict(cmap='Greys', vmin=-0.2, vmax=0.2, add_colorbar=False, add_labels=False)
    dpi = 100
    
    freq_slice = cube[var].sel(**selection)
    freq_slice_smooth = cube_smooth[var].sel(**selection)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(aspect='equal'))
    freq_slice.T.plot(ax=ax, **kwargs)
    ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(dir_fig, f'slice_freq_{selection["freq_twt"]}kHz.png'), dpi=dpi)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(aspect='equal'))
    freq_slice_smooth.T.plot(ax=ax, **kwargs)
    ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(dir_fig, f'slice_freq_{selection["freq_twt"]}kHz_smooth.png'), dpi=dpi)

