"""
Create Figure showing interpolation results in time and frequency domain (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-11-02

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
    _, domain, kind, _ = os.path.basename(s).split('_')
    return [d.lower() for d in domains].index(domain) * 10 + [o.lower() for o in order].index(kind)

#%% MAIN
if __name__ == "__main__":

    dir_fig = os.path.dirname(os.path.abspath(__file__))
    dir_work = dir_fig
    fig_name = os.path.basename(dir_fig)

    il_sel = 750
    xl_sel = 600
    twt_sel = 775
    
    var = "env"
    domains = ["Sparse", "Time", "Frequency"]
    order = ["il", "xl", "twt"]
    indices = [il_sel, xl_sel, twt_sel]
    
    dpi = 300
    
    slices = glob.glob(os.path.join(dir_work, '*.nc'))
    slices = np.split(np.asarray(sorted(slices, key=sort_slices)), 3)
    slices = [tuple(a) for a in slices]
    
    slices_interp_bin_sizes_ = []
    for i, tuple_slices in enumerate(slices):
        _list = []
        for path_slice in tuple_slices:
            _list.append(xr.open_dataset(path_slice, engine="h5netcdf")[var])
        slices_interp_bin_sizes_.append(_list)

    #%% [PLOTTING] combined sparse + time + freq

    with mpl.rc_context({"font.family": "Arial", "mathtext.fontset": "stix"}):

        subfigures = (3, 1)  # (1,2)
        clim_env = (0, 2)

        titles = ["Sparse cube"] + [f"{d} domain" for d in domains[1:]]

        plot_dict = plot_cube_slices(
            slices=slices_interp_bin_sizes_,
            var=var,
            il_sel=None,
            xl_sel=None,
            twt_sel=None,
            subfigures=subfigures,
            subfigure_index="C",
            label_slices=(True, False, False),
            title=titles,
            label_subplots=("a)", "b)", "c)"),
            clim=clim_env,
            kw_figure=dict(figsize=(4, 12)),
            kw_gridspec=dict(wspace=0.02, hspace=0.02, bottom=0.15),
            plot_spectrum=True,
        )

        #%% export figure as PNG
        figure_number = 12
        plot_dict["fig"].savefig(
            os.path.join(dir_fig, f"Figure-{figure_number:02d}_time_vs_frequency_interp_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
        # plot_dict["fig"].savefig(
        #     os.path.join(dir_fig, f"Figure-{figure_number:02d}_time_vs_frequency_interp.tiff"),
        #     dpi=dpi,
        #     # bbox_inches="tight",
        # )
