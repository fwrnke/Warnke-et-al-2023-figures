"""
Create Figure showing interpolation results using different bin sizes (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2023-01-31

"""
import os
import sys
import glob

import numpy as np
import xarray as xr
import matplotlib as mpl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import plot_cube_slices

xr.set_options(keep_attrs=True)

def sort_slices(s):
    """Sort slices based on user-defined filename extracts."""
    _, _, bins, domain, _ = os.path.basename(s).split('_')
    return bin_sizes.index(bins) * 10 + domains.index(domain)

#%% MAIN
if __name__ == '__main__':

    dir_fig = os.path.dirname(os.path.abspath(__file__))
    dir_work = dir_fig
    fig_name = os.path.basename(dir_fig)

    bin_sizes = ['7+5x7+5', '10x10', '15x15', '25x25']
    domains = ['iline', 'xline', 'twt']
    var = 'env'
    
    dpi = 300
    
    # load cube subsets (netCDF)
    slices = glob.glob(os.path.join(dir_work, '*.nc'))
    slices = np.split(np.asarray(sorted(slices, key=sort_slices)), 4)
    slices = [tuple(a) for a in slices]
    
    slices_interp_bin_sizes = []
    folds = []
    folds_da = []
    for i, tuple_slices in enumerate(slices):
        _list = []
        for j, path_slice in enumerate(tuple_slices):
            ds = xr.open_dataset(path_slice, engine="h5netcdf")
            da = ds[var].T if 'twt' in path_slice else ds[var]
            _list.append(da)
            if j == 2:  # use fold from time slice
                folds.append(ds['fold'].attrs['coverage_perc'])
                folds_da.append(ds['fold'].load())
        slices_interp_bin_sizes.append(_list)
        
    # fill 25x25 m cube
    slices_interp_bin_sizes[-1][0] = (
        slices_interp_bin_sizes[-1][0].where(slices_interp_bin_sizes[-1][0] > 0, np.nan)
        .interpolate_na(dim='xline', keep_attrs=True)
        .fillna(0.0)
    )
    slices_interp_bin_sizes[-1][1] = (
        slices_interp_bin_sizes[-1][1].where(slices_interp_bin_sizes[-1][1] > 0, np.nan)
        .interpolate_na(dim='iline', keep_attrs=True)
        .fillna(0.0)
    )
    slices_interp_bin_sizes[-1][2] = (
        slices_interp_bin_sizes[-1][2].where(slices_interp_bin_sizes[-1][2] > 0, np.nan)
        .interpolate_na(dim='iline', keep_attrs=True)
        .fillna(0.0)
    )
    
    #%% [PLOTTING] combined sparse + interp

    with mpl.rc_context({'font.family': 'Arial', 'mathtext.fontset': 'stix'}):

        subfigures = (2, 2)
        clim_env = (0, 2)

        titles = [
            f'{bs.replace("+",".")} m bins\n({f:.2f}% coverage)' for bs, f in zip(bin_sizes, folds)
        ]

        plot_dict = plot_cube_slices(
            slices=slices_interp_bin_sizes,
            var=var,
            il_sel=None,
            xl_sel=None,
            twt_sel=None,
            subfigures=subfigures,
            subfigure_index='C',
            label_slices=[False] * len(titles),
            title=titles,
            label_subplots=('a)', 'b)', 'c)', 'd)'),
            clim=clim_env,
            kw_figure=dict(figsize=(8, 8)),
            kw_gridspec=dict(wspace=0.02, hspace=0.02, bottom=0.12, top=0.90),
            plot_spectrum=True,
        )

        #%% export figure as PNG
        figure_number = 14
        plot_dict['fig'].savefig(
            os.path.join(dir_fig, f"Figure-{figure_number:02d}_different_bin_sizes_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches='tight',
        )
        # plot_dict['fig'].savefig(
        #     os.path.join(dir_fig, f"Figure-{figure_number:02d}_different_bin_sizes.tiff"),
        #     dpi=dpi,
        #     bbox_inches='tight',
        # )
