"""
Create Figure showing interpolation results using only ilines/xlines (Warnke et al., 2023).

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
import matplotlib.ticker as mticker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import plot_cube_slices

def sort_slices(s):
    """Sort slices based on user-defined filename extracts."""
    _, _, bins, subset, domain, _ = os.path.basename(s).split('_')
    if subset == 'full':
        subset = ''
    else:
        subset = f'_{subset}'
    return bin_sizes.index(bins) * 100 + subsets.index(subset) * 10 + domains.index(domain)

#%% MAIN

if __name__ == '__main__':

    dir_fig = os.path.dirname(os.path.abspath(__file__))
    dir_work = dir_fig

    bin_sizes = ['5x5', '5x5', '5x5', '5x15', '15x5']
    subsets = ['', '_ILINES', '_XLINES', '_ILINES', '_XLINES']
    domains = ['iline', 'xline', 'twt']
    var = 'env'
    
    dpi = 300
    
    slices_interp_bin_sizes = []
    folds = []
    folds_da = []
    
    # load cube subsets (netCDF)
    slices = glob.glob(os.path.join(dir_work, '*.nc'))
    slices = np.split(np.asarray(sorted(slices, key=sort_slices)), 5)
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

    slices_interp_bin_sizes = slices_interp_bin_sizes[:-2] + [None] + slices_interp_bin_sizes[-2:]
    
    #%% [PLOTTING] combined sparse + interp

    with mpl.rc_context({'font.family': 'Arial', 'mathtext.fontset': 'stix'}):

        subfigures = (2, 3)

        clim_env = (0, 2)

        idx_nan = slices_interp_bin_sizes.index(None)
        titles = [f'{bs} m' for bs in bin_sizes]
        names = ['Full', 'ilines', 'xlines', 'ilines', 'xlines']
        titles = [f'{n} ({t})' for n, t in zip(names, titles)]
        titles.insert(idx_nan, '')

        plot_dict = plot_cube_slices(
            slices=slices_interp_bin_sizes,
            var=var,
            il_sel=None,  # il_sel_list,
            xl_sel=None,  # xl_sel_list,
            twt_sel=None,  # twt_sel_list,
            subfigures=subfigures,
            subfigure_index='C',
            label_slices=[False] * len(titles),
            title=titles,
            label_subplots=('a)', 'b)', 'c)', 'd)', 'e)', 'f)'),
            clim=clim_env,  # (clim_env, clim_env, clim_env, clim_freq, clim_freq, clim_freq),
            # kw_subfigure=dict(wspace=0.05, hspace=0.05),
            kw_figure=dict(figsize=(12, 8.2)),
            kw_gridspec=dict(wspace=0.02, hspace=0.02, bottom=0.12, top=0.92),
            plot_spectrum=True,
        )

        # =====================================================================================
        #                                   BIN COVERAGE
        # =====================================================================================
        fontsize = 10

        fig = plot_dict['subfigs'][3]
        sharex = sharey = False
        ax_fold = fig.subplots(
            2,
            2,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=dict(
                wspace=0.25,
                hspace=0.35,
                left=0.15,
                right=0.95,
                top=0.9,
                bottom=0.1,
            ),
        )
        ax_fold = ax_fold.ravel()

        vmax = 3
        subtitles = ['ilines', 'xlines'] * 2
        for i, fold in enumerate(folds_da[1:]):
            im = fold.T.plot(
                ax=ax_fold[i],
                cmap='magma',
                vmin=0,
                vmax=vmax,
                add_colorbar=False,
            )
            fold_perc = fold.attrs["coverage_perc"]
            dil = fold['iline'].attrs['dil']
            dxl = fold['xline'].attrs['dxl']

            ax_fold[i].set_aspect(aspect=dil / dxl)

            ax_fold[i].set_title(
                f'{subtitles[i]} ({dil:.0f}x{dxl:.0f} m)',  # | {fold_perc:.2f}%',
                fontsize=fontsize + 1,
                fontweight='semibold',
            )
            ax_fold[i].text(
                0.95,
                0.1,
                f'{fold_perc:.2f}%',
                va='center',
                ha='right',
                color='white',
                fontsize=fontsize + 2,
                fontweight='bold',
                transform=ax_fold[i].transAxes,
            )

        cbar = fig.colorbar(
            im,
            ax=ax_fold.tolist(),
            shrink=0.9,
            orientation='vertical',
            pad=0.05,
            extend='max',
            label='fold',
        )
        cbar.ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        cbar.ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

        high_res_x = (200, 100)
        high_res_y = (100, 50)
        low_res = (50, 10)
        for i, ax in enumerate(ax_fold):
            ax.tick_params(axis='both', labelsize=fontsize)

            if i < 2:
                ax.set_xlabel('')
            else:
                ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            if i % 2 == 1:
                ax.set_ylabel('')
            else:
                ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

            if i < 3:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(high_res_y[0]))
                ax.yaxis.set_minor_locator(mticker.MultipleLocator(high_res_y[1]))
            else:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(low_res[0]))
                ax.yaxis.set_minor_locator(mticker.MultipleLocator(low_res[1]))
            if i != 2:
                ax.xaxis.set_major_locator(mticker.MultipleLocator(high_res_x[0]))
                ax.xaxis.set_minor_locator(mticker.MultipleLocator(high_res_x[1]))
            else:
                ax.xaxis.set_major_locator(mticker.MultipleLocator(low_res[0]))
                ax.xaxis.set_minor_locator(mticker.MultipleLocator(low_res[1]))

        fig.suptitle('Bin coverages', y=0.99, fontsize=12, fontweight='semibold')

        fig.text(
            0.075,
            0.975,
            'd)',
            transform=fig.transSubfigure,
            va="center",
            ha="center",
            fontsize=16,
            fontweight='semibold',
            family='Times New Roman'
        )

        #%% export figure as PNG
        figure_number = 15
        plot_dict['fig'].savefig(
            os.path.join(dir_fig, f"Figure-{figure_number:02d}_only_ilines_xlines_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches='tight',
        )
        # plot_dict['fig'].savefig(
        #     os.path.join(dir_fig, f"Figure-{figure_number:02d}_only_ilines_xlines.tiff"),
        #     dpi=dpi,
        #     bbox_inches='tight',
        # )
