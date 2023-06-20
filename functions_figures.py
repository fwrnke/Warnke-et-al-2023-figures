"""
Helper functions required for creating figures in Warnke et al. (2023).

@author: fwrnke
@email:  fwrnke@mailbox.org, fwar378@aucklanduni.ac.nz
@date:   2022-12-03

"""
import warnings
from collections.abc import Iterable

import numpy as np
import xarray as xr
import xrft
import scipy.interpolate as interp

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredOffsetbox


def _print_info_TOPAS_moratorium():
    print("[INFO]  **********************************************************************************")
    print()
    print("[INFO]    Please note that the source data for this script will be made available through ")
    print("[INFO]    a public data repository after an initial 12 month moratorium.  ")
    print()
    print("[INFO]  **********************************************************************************")
    print()

class AnchoredScaleBar(AnchoredOffsetbox):
    """Create anchored scale bar for figure."""

    def __init__(
        self,
        transform,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        loc=4,
        pad=0.2,
        borderpad=0.5,
        sep=5,
        prop=None,
        barcolor="black",
        barwidth=None,
        fontsize=12,
        **kwargs,
    ):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor

        Source: https://gist.github.com/dmeliza/3251476

        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea

        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0, sizey), 0, -sizey, ec=barcolor, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx, textprops=dict(fontsize=fontsize))
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely, textprops=dict(fontsize=fontsize, rotation=90))
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(
            self, loc, pad=pad, borderpad=borderpad, child=bars, prop=prop, frameon=False, **kwargs
        )


def plot_cube_slices(
    cube=None,
    slices: tuple = None,
    var: str = None,
    il_sel: int = None,
    xl_sel: int = None,
    twt_sel: float = None,
    # dim: str = 'twt',
    clim: tuple = None,
    cmap: str = "Greys",
    subfigures: tuple = None,
    subfigure_index: str = "C",
    label_slices: bool = False,
    title: str = None,
    label_subplots: str = None,
    plot_spectrum: bool = None,
    kw_figure: dict = None,
    kw_subfigure: dict = None,
    kw_slice_indicator: dict = None,
    kw_gridspec: dict = None,
):
    """Plot selected slices (time, iline, xline) of input 3D cube."""
    args = locals()
    dict_keys = ["fig", "axes", "cbar"]

    if not cube and not slices:
        raise ValueError("Either `cube` or `slices` must be provided!")

    if subfigures is not None and isinstance(subfigures, tuple):
        nrows, ncols = subfigures
        nfigs = np.prod(subfigures)

        dict_keys[0] = "subfigs"

        if not isinstance(label_slices, bool) and nfigs != len(label_slices):
            warnings.warn("Mismatch between number of `subfigures` and `label_slices`!")
            print(label_slices)
            label_slices = list(label_slices) + [False] * (nfigs - len(label_slices))
            print(label_slices)
        if not isinstance(label_subplots, bool) and nfigs != len(label_subplots):
            warnings.warn("Mismatch between number of `subfigures` and `label_subplots`!")
            print(label_subplots)
            label_subplots = list(label_subplots) + [""] * (nfigs - len(label_subplots))
            print(label_subplots)

        kwargs_figure = dict(figsize=(ncols * 6, nrows * 6))
        if isinstance(kw_figure, dict):
            kwargs_figure.update(kw_figure)

        kwargs_subfigure = dict(wspace=0.1, hspace=0.1)
        if isinstance(kw_subfigure, dict):
            kwargs_subfigure.update(kw_subfigure)

        fig = plt.figure(layout=None, **kwargs_figure)  # layout=None
        # print(fig)
        subfigs = fig.subfigures(nrows, ncols, **kwargs_subfigure)
        subfigs = subfigs.ravel(order=subfigure_index)
        # print(subfigs)

        # remove keys to be passed to plotting function
        for key in ["kw_figure", "kw_subfigure", "subfigures", "subfigure_index"]:
            del args[key]

        plot_params = []
        if cube is not None and isinstance(cube, Iterable):
            key = "cube"
            data = cube
        elif (
            slices is not None and isinstance(slices, Iterable) and isinstance(slices[0], Iterable)
        ):
            key = "slices"
            data = slices
        else:
            raise ValueError("Incorrect input format")

        for i, _data in enumerate(data):
            if _data is None:
                continue
            args[key] = _data
            args["fig"] = subfigs[i]
            args["clim"] = (
                clim[i] if isinstance(clim, Iterable) and isinstance(clim[0], Iterable) else clim
            )
            args["label_slices"] = (
                label_slices[i] if isinstance(label_slices, Iterable) else label_slices
            )
            args["title"] = title[i] if isinstance(title, (tuple, list)) else title
            args["label_subplots"] = (
                label_subplots[i] if isinstance(label_subplots, (tuple, list)) else label_subplots
            )
            args["il_sel"] = il_sel[i] if isinstance(il_sel, Iterable) else il_sel
            args["xl_sel"] = xl_sel[i] if isinstance(xl_sel, Iterable) else xl_sel
            args["twt_sel"] = twt_sel[i] if isinstance(twt_sel, Iterable) else twt_sel
            plot_params.append(_plot_cube_slices(**args))
        noutputs = len([d for d in data if d is not None])

        # results = plot_params
        results = dict(fig=fig)
        results.update(
            dict(
                zip(
                    dict_keys,
                    [
                        a.tolist()
                        for a in np.split(
                            np.transpose(plot_params).ravel(),
                            # np.arange(nfigs, nfigs * 2 + 1, nfigs))
                            np.arange(noutputs, noutputs * 2 + 1, noutputs),
                        )
                    ],
                )
            )
        )

        if None in data:
            idx_none = [i for i, x in enumerate(data) if x is None]
            for i in idx_none:
                results["subfigs"].insert(i, subfigs[i])

    elif not subfigures and (
        isinstance(cube, Iterable)
        or (isinstance(slices, Iterable) and isinstance(slices[0], (list, tuple)))
    ):
        raise ValueError("`subfigures` must be provided when multiple datasources are given!")

    else:
        del args["kw_subfigure"]
        del args["subfigures"]

        results = dict(zip(dict_keys, _plot_cube_slices(**args)))

    return results


def _plot_cube_slices(
    cube=None,
    slices: tuple = None,
    var: str = None,
    il_sel: int = None,
    xl_sel: int = None,
    twt_sel: float = None,
    # dim: str = 'twt',
    clim: tuple = None,
    cmap: str = "Greys",
    fig=None,
    label_slices: bool = False,
    title: str = None,
    label_subplots: str = None,
    plot_spectrum: bool = None,
    kw_figure: dict = None,
    kw_slice_indicator: dict = None,
    kw_gridspec: dict = None,
):
    if isinstance(cube, xr.DataArray):
        data_il = cube.sel(iline=il_sel)
        data_xl = cube.sel(xline=xl_sel)
        data_twt = cube.sel(twt=twt_sel).T
    elif isinstance(cube, xr.Dataset):
        data_il = cube[var].sel(iline=il_sel)
        data_xl = cube[var].sel(xline=xl_sel)
        data_twt = cube[var].sel(twt=twt_sel).T

    if isinstance(slices, Iterable):
        data_il, data_xl, data_twt = slices
        il_sel = int(data_il["iline"].data)
        xl_sel = int(data_xl["xline"].data)
        twt_sel = float(data_twt["twt"].data)

    if clim is None:
        vmin = vmin_slice = 0
        vmax = vmax_slice = np.mean(
            [np.percentile(data_il, 99), np.percentile(data_xl, 99), np.percentile(data_twt, 99)]
        )
        show_both_colobars = False
    elif isinstance(clim, dict):
        vmin, vmax = clim.get(
            "profiles", (0, np.mean([np.percentile(data_il, 99), np.percentile(data_xl, 99)]))
        )
        vmin_slice, vmax_slice = clim.get("slice", (0, np.percentile(data_twt, 99)))
        show_both_colobars = True
    elif isinstance(clim, str) and clim == "freq":
        vmin, vmax = 0, np.mean([np.percentile(data_il, 99), np.percentile(data_xl, 99)])
        vmin_slice, vmax_slice = 0, np.percentile(data_twt, 99)
        show_both_colobars = True
    else:
        vmin = vmin_slice = min(clim)
        vmax = vmax_slice = max(clim)
        show_both_colobars = False

    kwargs_ilxl = dict(
        cmap=cmap, vmin=vmin, vmax=vmax, yincrease=False, add_colorbar=False, add_labels=False
    )
    kwargs_slice = dict(
        cmap=cmap, vmin=vmin_slice, vmax=vmax_slice, add_colorbar=False, add_labels=False
    )

    kwargs_labels_axes = dict(fontsize=12, fontweight="normal")
    kwargs_ticklabels = dict(labelsize=10)

    kwargs_slice_indicator = dict(c="blue", lw=1)
    if isinstance(kw_slice_indicator, dict):
        kwargs_slice_indicator.update(kw_slice_indicator)

    # cbar_shrink = 0.95  # 0.75
    dim = "twt" if not any("freq" in d for d in data_il.dims) else "freq_twt"
    dim_twt = dim
    dim_iline = "iline"
    dim_xline = "xline"
    hscale = 1.0
    shape = dict(zip(data_twt.dims, data_twt.shape))
    if dim == "twt":
        shape.update({dim_twt: data_il[dim_twt].size * data_il[dim_twt].attrs.get("dt", 0.05)})
        # print('twt', shape)
    elif "freq" in dim:
        # print('dim:', dim)
        shape.update(
            {
                dim_twt: data_il[dim_twt].size
                * float(f"{data_il[dim_twt].attrs.get('spacing', 0.005):g}")
            }
        )
        dim_iline = "freq_iline"
        dim_xline = "freq_xline"

        kwargs_ilxl.update(
            dict(
                cmap="inferno",
                norm=mcolors.LogNorm(vmin=kwargs_ilxl["vmin"] + 1e-10, vmax=kwargs_ilxl["vmax"]),
                vmin=None,
                vmax=None,
            )
        )
        kwargs_slice.update(
            dict(
                cmap="inferno",
                norm=mcolors.LogNorm(vmin=kwargs_slice["vmin"] + 1e-10, vmax=kwargs_slice["vmax"]),
                vmin=None,
                vmax=None,
            )
        )
        # kwargs_slice.update({'cmap': 'inferno'})
        # print('freq', shape)

    dil = data_twt[dim_iline].attrs.get("dil")
    dxl = data_twt[dim_xline].attrs.get("dxl")
    if dil == dxl:
        width_ratios = [shape[dim_iline], shape[dim_xline]]
        height_ratios = [shape[dim_xline], shape[dim_xline] * hscale]
    elif dil > dxl:  # only ILINES
        width_ratios = [shape[dim_iline] / (dil / dxl), shape[dim_xline]]
        height_ratios = [shape[dim_xline], shape[dim_xline] * hscale]
    elif dil < dxl:  # only XLINES
        width_ratios = [shape[dim_iline], shape[dim_xline] / (dxl / dil)]
        height_ratios = [shape[dim_xline], shape[dim_xline] * hscale]

    wpad_subfig = 0.02
    hpad_subfig = 0.02
    gridspec_kw = dict(
        wspace=0,
        hspace=0,
        left=0.15 + wpad_subfig,
        right=0.9 - wpad_subfig,
        bottom=0.075 + hpad_subfig,
        top=0.95 - hpad_subfig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(kw_gridspec, dict):
        gridspec_kw.update(kw_gridspec)

    kwarg_figure = dict(figsize=(6, 6))
    if isinstance(kw_figure, dict):
        kwarg_figure.update(kw_figure)

    if fig is not None:
        axes = fig.subplots(2, 2, gridspec_kw=gridspec_kw)
        transform = fig.transSubfigure
    else:
        fig, axes = plt.subplots(2, 2, gridspec_kw=gridspec_kw, **kwarg_figure)
        transform = fig.transFigure
    axes = dict(zip([dim_twt, "empty", "xl", "il"], axes.ravel()))

    # plot data
    im_twt = data_twt.plot(ax=axes[dim_twt], **kwargs_slice)
    # axes[dim_twt].set_aspect("equal")
    im_xl = data_xl.plot(ax=axes["xl"], **kwargs_ilxl)
    _ = data_il.plot(ax=axes["il"], **kwargs_ilxl)
    
    if plot_spectrum:
        bbox = axes["empty"].get_position()
        bbox_edit = bbox.from_bounds(bbox.x0 * 1.4, bbox.y0 * 1.2, 0.2, 0.2)
        # axes['empty'].set_position([0.65, 0.6, 0.25, 0.25])
        axes["empty"].set_position(bbox_edit)

        axcbar_spectrum = axes["empty"].inset_axes([1 + 0.02, 0.025, 0.04, 0.95])

        _freq = xrft.power_spectrum(
            data_twt.drop(["x", "y"]),
            dim=["iline", "xline"],
            window="hamming",
            true_phase=True,
            true_amplitude=True,
            chunks_to_segments=False,
            scaling="density",
            keep_attrs=True,
        )
        xr.apply_ufunc(np.abs, _freq).plot(
            ax=axes["empty"],
            cmap="inferno",
            robust=True,
            norm=mcolors.LogNorm(vmin=0.0001, vmax=1.0),
            # vmin=0, vmax=0.2,
            add_colorbar=True,
            cbar_ax=axcbar_spectrum,
        )
        axcbar_spectrum.tick_params(pad=1)
        axes["empty"].set_title("Power spectrum", fontsize=kwargs_ticklabels.get("labelsize"))
        axes["empty"].set_xlabel("")
        axes["empty"].set_ylabel("")
        # axes['empty'].set_xlabel('spatial frequency')
        # axes['empty'].set_ylabel('spatial frequency')
        axes["empty"].set_xlim(-0.5, 0.5)
        axes["empty"].set_ylim(-0.5, 0.5)
        axes["empty"].set_aspect("equal")

        axes["empty"].xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        axes["empty"].xaxis.set_minor_locator(mticker.MultipleLocator(0.1))

        axes["empty"].yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        axes["empty"].yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    else:
        axes["empty"].axis("off")

    il_minmax = (data_twt[dim_iline].values.min(), data_twt[dim_iline].values.max())
    xl_minmax = (data_twt[dim_xline].values.min(), data_twt[dim_xline].values.max())
    twt_minmax = (data_il[dim_twt].values.min(), data_il[dim_twt].values.max())

    # plot slice indicators
    axes[dim_twt].axvline(il_sel, **kwargs_slice_indicator)
    axes[dim_twt].axhline(xl_sel, **kwargs_slice_indicator)

    axes["xl"].axvline(il_sel, **kwargs_slice_indicator)
    axes["xl"].axhline(twt_sel, **kwargs_slice_indicator)

    axes["il"].axvline(xl_sel, **kwargs_slice_indicator)
    axes["il"].axhline(twt_sel, **kwargs_slice_indicator)

    if label_slices:
        if isinstance(label_slices, str) and label_slices == "outside":
            # cbar_shrink = 0.65  # reduce size to avoid overlap

            factor = -0.05
            kw_label_slices_left = dict(
                c=kwargs_slice_indicator.get("c", "k"),
                backgroundcolor="white",
                va="center",
                ha="right",
            )
            kw_label_slices_right = kw_label_slices_left.copy()
            kw_label_slices_right.update(ha="center")
            # dict(c=kwargs_slice_indicator.get('c', 'k'), va='center', ha='right')
        else:
            factor = 0.03
            kw_label_slices_left = dict(
                c=kwargs_slice_indicator.get("c", "k"), va="bottom", ha="left"
            )
            kw_label_slices_right = kw_label_slices_left.copy()
            kw_label_slices_right.update(rotation=90, rotation_mode="anchor")

        axes[dim_twt].text(
            il_minmax[0] + shape[dim_iline] * factor, xl_sel, f"{xl_sel:g}", **kw_label_slices_left
        )
        ypos = (
            xl_minmax[1] - shape[dim_xline] * factor
            if label_slices == "outside"
            else xl_minmax[0] + shape[dim_xline] * factor
        )
        axes[dim_twt].text(il_sel, ypos, f"{il_sel:g}", **kw_label_slices_right)

        axes["xl"].text(
            il_minmax[0] + shape[dim_iline] * factor,
            twt_sel,
            f"{twt_sel:g}",
            **kw_label_slices_left,
        )
        axes["xl"].text(
            il_sel, twt_minmax[1] - shape[dim_twt] * factor, f"{il_sel:g}", **kw_label_slices_right
        )

        axes["il"].text(
            xl_minmax[0] + shape[dim_xline] * factor,
            twt_sel,
            f"{twt_sel:g}",
            **kw_label_slices_left,
        )
        axes["il"].text(
            xl_sel, twt_minmax[1] - shape[dim_twt] * factor, f"{xl_sel:g}", **kw_label_slices_right
        )

    # add colobar
    pad_cbar = gridspec_kw["wspace"] if gridspec_kw["wspace"] > 0 else 0.02
    axcbar = axes[dim_twt].inset_axes([1 + pad_cbar, 0.04, 0.04, 0.92])
    cbar = fig.colorbar(im_twt, cax=axcbar, extend="max" if var == "env" else "both")
    # cbar.ax.yaxis.set_tick_params(pad=-15)
    cbar.ax.tick_params(pad=1)

    if show_both_colobars:
        axcbar2 = axes["il"].inset_axes([1 + pad_cbar, 0.04, 0.03, 0.92])
        fig.colorbar(im_xl, cax=axcbar2, extend="max" if var == "env" else "both")
        # cbar2.ax.tick_params(labelsize=kwargs_ticklabels.get('labelsize', 10) - 2, pad=1)

    # cbar = fig.colorbar(im_twt, ax=axes['empty'], shrink=cbar_shrink, location="bottom", fraction=0.2,
    #                     extend="max" if var == "env" else "both")
    # cbar.set_label(data_twt.attrs.get('seismic_attribute',''), labelpad=-40, **kwargs_labels_axes)

    # add title
    if title is not None:
        xtxt = 0.3 if plot_spectrum else 0.6
        ytxt = 1.45 if plot_spectrum else 0.5
        add_fontsize = 2 if "\n" not in title else 0
        axes["empty"].text(
            xtxt,
            ytxt,
            title,
            transform=axes["empty"].transAxes,
            va="bottom",
            ha="center",
            fontsize=kwargs_labels_axes.get("fontsize", 12) + add_fontsize,
            fontweight="semibold",
        )

    if label_subplots is not None:
        fig.text(
            0.075,
            0.975,
            label_subplots,
            transform=transform,
            va="center",
            ha="center",
            fontsize=kwargs_labels_axes.get("fontsize", 12) + 4,
            fontweight="semibold",
            fontfamily="Times New Roman",
        )

    # set SLICE parameter
    _label = data_twt[dim_xline].attrs.get("long_name", "")
    _unit = data_twt[dim_xline].attrs.get("units", None)
    axes[dim_twt].set_ylabel(
        f"{_label} ({_unit})" if _unit is not None else f"{_label}", **kwargs_labels_axes
    )
    axes[dim_twt].tick_params(
        axis="x", which="both", top=True, labeltop=True, pad=1, bottom=False, labelbottom=False
    )
    # axes[dim_twt].tick_params(axis='y', which='both', right=True, labelright=False)

    # set XLINE parameter
    _label = data_xl[dim_twt].attrs.get("standard_name", "")
    _unit = data_xl[dim_twt].attrs.get("units", None)
    axes["xl"].set_ylabel(
        f"{_label} ({_unit})" if _unit is not None else f"{_label}", **kwargs_labels_axes
    )
    _label = data_xl[dim_iline].attrs.get("long_name", "")
    _unit = data_xl[dim_iline].attrs.get("units", None)
    axes["xl"].set_xlabel(
        f"{_label} ({_unit})" if _unit is not None else f"{_label}", **kwargs_labels_axes
    )

    # set ILINE parameter
    _label = data_il[dim_xline].attrs.get("long_name", "")
    _unit = data_il[dim_xline].attrs.get("units", None)
    axes["il"].yaxis.set_label_position("right")
    axes["il"].set_xlabel(
        f"{_label} ({_unit})" if _unit is not None else f"{_label}", **kwargs_labels_axes
    )
    axes["il"].tick_params(
        which="both",
        axis="both",
        top=False,
        labeltop=False,
        left=False,
        labelleft=False,
        right=True if not show_both_colobars else False,
        labelright=True if not show_both_colobars else False,
    )

    # set general parameter
    for ax in list(axes.values()) + [cbar.ax]:
        ax.tick_params(axis="both", **kwargs_ticklabels)

    if dim == "twt":
        for name, ax in axes.items():
            if name != "empty":
                ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
                ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))

            if name == dim_twt:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
                ax.yaxis.set_minor_locator(mticker.MultipleLocator(25))
            elif name != "empty":
                ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
                ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))

        # remove first iline x-axis label --> avoid overlap
        xlims = axes["il"].get_xlim()
        xticks = axes["il"].get_xticks().tolist()
        if len(xticks) > 4:
            xticklabels = [f"{x:g}" if i > 1 else "" for i, x in enumerate(xticks)]
            # print('xticks:', xticks)
            axes["il"].set_xticks(xticks)
            axes["il"].set_xticklabels(xticklabels)
            axes["il"].set_xlim(xlims)
    else:
        # frequency axis
        for ax in [axes["xl"], axes["il"]]:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

        # remove first freq y-axis label --> avoid overlap
        ylims = axes["xl"].get_ylim()
        yticks = axes["xl"].get_yticks().tolist()
        # print('yticks', yticks)
        yticklabels = [f"{y:g}" if y > 0 else "" for y in yticks]
        # print('xticks:', xticks)
        axes["xl"].set_yticks(yticks)
        axes["xl"].set_yticklabels(yticklabels)
        axes["xl"].set_ylim(ylims)

        # remove first iline x-axis label --> avoid overlap
        xlims = axes["il"].get_xlim()
        xticks = axes["il"].get_xticks().tolist()
        xticklabels = [f"{x:g}" if i > 0 else "" for i, x in enumerate(xticks)]
        # print('xticks:', xticks)
        axes["il"].set_xticks(xticks)
        axes["il"].set_xticklabels(xticklabels)
        axes["il"].set_xlim(xlims)

    return (fig, axes, cbar)


# =================================================================================================
#                                           UTILS
# =================================================================================================

def xprint(*args, kind: str = 'info', verbosity: int = 0, **kwargs) -> None:
    """Thin wrapper function for build-in print() to add informative prefix and color-coded print statements."""
    verbosity = 1 if verbosity is True else verbosity
    prefixes = {
        'info': ('\033[39m', '[INFO]  ', 1),
        'warning': ('\033[33m\033[1m', '[WARNING]  ', 0),
        'error': ('\033[31m\033[1m', '[ERROR]  ', 0),
        'success': ('\033[32m', '[SUCCESS]  ', 1),
        'debug': ('\033[36m', '[DEBUG]  ', 2),
    }
    prefix = prefixes.get(kind, None)
    if prefix is None:
        args = args
        verbosity_lvl = 1
    else:
        color, prefix_, verbosity_lvl = prefix
        args = [f'{color}{prefix_}'] + ['{arg}'.format(arg=i) for i in args] + ['\033[0m']

    if verbosity_lvl <= verbosity:
        print(*args, **kwargs)


def rescale(a, vmin=0, vmax=1):
    """
    Rescale array to given range (default: [0, 1]).

    Parameters
    ----------
    a : np.ndarray
        Input array to rescale/normalize.
    vmin : float, optional
        New minimum value (default: `0`).
    vmax : float, optional
        New maximum value (default: `1`).

    Returns
    -------
    np.ndarray
        Rescaled input array.

    """
    a = np.asarray(a)
    _vmin = np.nanmin(a)
    _vmax = np.nanmax(a)

    vmin = _vmin if vmin is None else vmin
    vmax = _vmax if vmax is None else vmax

    if _vmin == _vmax:
        return a
    return vmin + (a - _vmin) * ((vmax - vmin) / (_vmax - _vmin))


def slice_valid_data(data, nso):
    """
    Account for zero padded input data and return only valid data samples (!= 0).

    Parameters
    ----------
    data : np.ndarray
        Seismic section (samples x traces).
    nso : int
        Original number samples per trace (from binary header).

    Returns
    -------
    np.ndarray
        "Unpadded" seismic section with only valid (non-zero) samples (2D).
    idx_start_slice : np.ndarray
        Array of starting indices for valid data slice per trace.

    """
    # get indices of first valid sample
    # = 0: trace was padded at bottom
    # > 0: trace was padded at top
    idx_start_slice = (data != 0).argmax(axis=0)
    # create index array
    indexer = np.transpose(np.arange(nso) + idx_start_slice[:, None])
    # return sliced traces and indices
    return np.take_along_axis(data, indexer, axis=0), idx_start_slice


def depth2twt(depth, v=1500, units='s'):
    """Convert depth (m) to two-way travel time (TWT in sec)."""
    if units == 's':
        return depth / (v / 2)
    elif units == 'ms':
        return (depth / (v / 2)) * 1000
    elif units == 'ns':
        return (depth / (v / 2)) * 1e-06


def twt2depth(twt, v=1500, units='s'):
    """Convert two-way travel time (TWT in sec) to depth (m)."""
    if units == 's':
        return (v / 2) * twt
    elif units == 'ms':
        return (v / 2) * (twt / 1000)
    elif units == 'ns':
        return (v / 2) * (twt / 1e-06)


def twt2samples(twt, dt: float, units='s'):
    """Convert TWT (sec) to samples (#) based on sampling interval `dt` (sec)."""
    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6

    return twt / dt


def samples2twt(samples, dt: float):
    """Convert samples (#) to TWT (in dt units!) based on sampling interval `dt`."""
    return samples * dt


def depth2samples(depth, dt: float, v=1500, units='s'):
    """Convert depth (m) to samples (#) given a sampling interval `dt` and acoustic velocity."""
    _twt = depth2twt(depth, v=v)

    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6

    return twt2samples(_twt, dt=dt)


def samples2depth(samples, dt: float, v=1500, units='s'):
    """Convert samples (#) to depth (m) given a sampling interval `dt` and acoustic velocity."""
    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6
    else:
        raise ValueError(f'Unit "{units}" is not supported for `dt`.')

    _twt = samples2twt(samples, dt=dt)

    return twt2depth(_twt, v=v)


def pad_array(a, n: int, zeros=False):
    """
    Pad 1D input array with `n` elements at start and end (mirror of array).

    Parameters
    ----------
    a : np.ndarray
        1D input array.
    n : int
        Number of elements to add at start and end.
    zeros : bool, optional
        Add zeros instead of mirrored values (default: `False`).

    Returns
    -------
    np.ndarray
        Padded input array.

    """
    if zeros:
        return np.concatenate((np.zeros(n), a, np.zeros(n)))
    else:
        # # mirror array
        # pad_start = a[0] + np.abs(a[1:n+1][::-1] - a[0])
        # pad_end = a[-1] + np.abs(a[-n-1:-1][::-1] - a[-1])

        # mirror array AND flip upside down
        pad_start = a[0] - np.abs(a[1 : n + 1][::-1] - a[0])
        pad_end = a[-1] - np.abs(a[-n - 1 : -1][::-1] - a[-1])
        return np.concatenate((pad_start, a, pad_end))


# =================================================================================================
#                                           SIGNAL
# =================================================================================================

def gain(
    data,
    twt,
    tpow=0.0,  # multiply data by t^tpow
    epow=0.0,  # multiply data by exp(epow*t)
    etpow=1.0,  # multiply data by exp(epow*t^etpow)
    ebase=None,  # use as base for exp function (default: e)
    gpow=0.0,  # take signed gpowth power of scaled data
    clip=None,  # clip any value whose magnitude exceeds clipval
    pclip=None,  # clip any value greater than clipval
    nclip=None,  # clip any value less than clipval
    qclip=None,  # clip by quantile on absolute values on trace
    bias=None,  # bias data by adding an overall bias value
    scale=1.0,  # multiply data by overall scale factor
    norm: bool = False,  # divide data by overall scale factor
    norm_rms: bool = False,  # normalize using RMS amplitude
    copy: bool = True,
    axis=0,
):
    """
    Apply various different types of gain for either single trace (1D array) or seismic section (2D array).
    
    !!! warning "Copyright"
        
        This function is a Python implementation of the Seismic Unix `sugain` module.
        Please refer to the license file `LICENSE_SeismicUnix`!
    
    
    Parameters
    ----------
    data : np.ndarray
        Seismic trace (nsamples,) or section (nsamples, ntraces).
    twt : np.ndarray
        Array of samples (in seconds TWT) appropriate spacing of sample rate (`dt`).
    tpow : float, optional
        Multiply data by t^tpow (default: `0.0`).
    epow : float, optional
        Multiply data by exp(epow*t) (default: `0.0`).
    etpow : float, optional
        Multiply data by exp(epow*t^etpow) (default: `1.0`).
    ebase : float, optional
        Base of exponential function (default: `e`).
    gpow : float, optional
        Take signed gpowth power of scaled data (default: `0.0`).
    clip : float, optional
        Clip any value whose magnitude exceeds clipval (default: `None`).
    pclip : float, optional
        Clip any value greater than clipval (default: `None`).
    nclip : float, optional
        Clip any value less than clipval (default: `None`).
    qclip : float, optional
        Clip by quantile on absolute values on trace (default: `None`).
    bias : float, optional
        Bias data by adding an overall bias value (default: `None`).
    scale : float, optional
        Multiply data by overall scale factor (default: `1.0`).
    norm : bool, optional
        Divide data by overall scale factor (default: `False`).
    norm_rms : bool, optional
        Normalize using RMS amplitude (default: `False`).
    copy : bool, optional
        Copy input data (no change of input data) (default: `True`).
    axis : int, optional
        Axis along which to gain (default: `0`).

    Returns
    -------
    data : np.ndarray
        Input data with applied gain function(s) along `axis`.
    
    Notes
    -----
    By default, the input array will be copied (`copy=True`) to avoid updating of the input data in place.
    
    References
    ----------
    [^1]: `sugain` module help, [http://sepwww.stanford.edu/oldsep/cliner/files/suhelp/sugain.txt](http://sepwww.stanford.edu/oldsep/cliner/files/suhelp/sugain.txt)

    """
    if copy:
        data = data.copy()

    if data.ndim == 1:
        nsamples, ntraces, ndim = data.size, None, 1
    elif data.ndim == 2:
        if axis == 0:
            nsamples, ntraces = data.shape
        else:
            ntraces, nsamples = data.shape
        ndim = 2
    else:
        if axis == 0:
            nsamples, nil, nxl = data.shape
        elif axis == 2 or axis == -1:
            nil, nxl, nsamples = data.shape
        else:
            raise ValueError('For 3D datasets the time axis must be either first or last.')
        ndim = 2

    for param, name in zip(
        [tpow, epow, etpow, gpow, clip, pclip, nclip, qclip, bias, scale],
        ['tpow', 'epow', 'etpow', 'gpow', 'clip', 'pclip', 'nclip', 'qclip', 'bias', 'scale'],
    ):
        if (param is not None) and not isinstance(param, (int, float)):
            raise ValueError(f'`{name}` must be either int or float')

    # bias
    if (bias is not None) and (bias != 0.0):
        data += bias

    # tpow
    if (tpow is not None) and (tpow != 0.0):
        tpow_fact = np.power(twt, tpow)
        tpow_fact[0] = 0.0 if twt[0] == 0.0 else np.power(twt[0], tpow)
        if ndim == 1:
            data *= tpow_fact
        else:
            data *= tpow_fact[:, None]

    # epow & etpow (& ebase)
    if epow is not None and epow != 0.0:
        # etpow
        etpow_fact = np.power(twt, etpow)
        # epow
        if ebase is None:
            epow_fact = np.exp(epow * etpow_fact)
        else:
            epow_fact = np.power(ebase, epow * etpow_fact)
        if ndim == 1:
            data *= epow_fact
        else:
            data *= epow_fact[:, None]

    # gpow (take signed gpowth power of scaled data)
    if (gpow is not None) and (gpow != 0.0):
        # workaround to prevent numpy from complaining about negative numbers
        data = np.sign(data) * np.abs(data) ** gpow

    # clip
    if clip is not None:
        data = np.where(data > clip, clip, data)
        data = np.where(data < -clip, -clip, data)

    # pclip
    if pclip is not None:
        data = np.where(data > pclip, pclip, data)

    # nclip
    if nclip is not None:
        data = np.where(data < nclip, nclip, data)

    # qclip
    if qclip is not None:
        qclip_per_trace = np.quantile(np.abs(data), q=qclip, axis=axis)
        data = np.where(data > qclip_per_trace, qclip_per_trace, data)

    # norm_rms
    if norm_rms:
        data = rms_normalization(data, axis=axis)

    # scale
    if (scale is not None) and (scale != 1.0):
        data = data * scale if not norm else data * 1 / scale

    return data


def rms(array, axis=None):
    r"""
    Calculate the RMS amplitude(s) of a given array.

    Parameters
    ----------
    array : np.ndarray
        Amplitude array.
    axis : int, tuple, list (optional)
        Axis for RMS amplitude calculation (default: `None`, i.e. single value for whole array).

    Returns
    -------
    rms : np.ndarray
        Root mean square (RMS) amplitude(s).

    $$
    rms = \sqrt{\frac{\sum{a^2}}{N}}
    $$

    """
    if axis is None:
        N = array.size
    elif isinstance(axis, int):
        N = array.shape[axis]
    elif isinstance(axis, (tuple, list)):
        N = np.prod([array.shape[ax] for ax in axis])

    return np.sqrt(np.sum(array**2, axis=axis) / N)


def rms_normalization(signal, axis=None):
    """
    Normalize signal using RMS amplitude of input array.

    Parameters
    ----------
    signal : np.ndarray
        Input trace(s).
    axis : int, optional
        Axis used for RMS amplitude calculation (default: `None`, i.e. whole array).

    Returns
    -------
    np.ndarray
        Normalized signal using RMS amplitude.

    References
    ----------
    [^1]: [https://superkogito.github.io/blog/2020/04/30/rms_normalization.html](https://superkogito.github.io/blog/2020/04/30/rms_normalization.html)

    """
    signal = np.asarray(signal)
    _rms = rms(signal, axis=axis)
    _rms[_rms == 0.0] = 1.0

    return signal / _rms


# =================================================================================================
#                                           FILTER
# =================================================================================================

def filter_interp_1d(
    data, method='IQR', kind='cubic', win=11, threshold=3.0, filter_boundaries=True
):  # noqa
    """
    Remove outliers using the IQR (inter-quartile range) method and
    interpolate using user-specified `kind` (default: 'cubic').
    Return outlier-removed and interpolated input array.

    Parameters
    ----------
    data : np.ndarray
        Input data (1D).
    method : str, optional
        Filter method to use (default: `IQR`).
    kind : str, optional
        Interpolation method for scipy.interpolate.interp1d (default: `cubic`).
    win : int, optional
        Size of moving window if required by chosen method (default: `11`).
    threshold : float, optional
        Threshold used for median absolute deviation (MAD) (default: `3.0`).
    filter_boundaries : bool, optional
        Filter flagged outlier indices at start and end of input array to avoid
        edge effects (if present despite padding) (default: `True`).

    Returns
    -------
    data_interp : np.ndarray
        Filtered and interpolated data.

    """
    METHODS = ['IQR', 'z-score', 'r_z-score', 'MAD', 'doubleMAD', 'r_doubleMAD', 'r_singleMAD']
    KIND_LIST = [
        'linear',
        'nearest',
        'nearest-up',
        'zero',
        'slinear',
        'quadratic',
        'cubic',
        'previous',
        'next',
    ]

    if data.ndim != 1:
        raise ValueError('data must be 1D array!')
    if kind not in KIND_LIST:
        raise ValueError(f'Parameter `kind` must be one of {KIND_LIST}')

    # get outlier indices
    if method == 'IQR':
        idx = iqr_filter(data)
    elif method == 'z-score':
        idx = zscore_filter(data)
    elif method == 'r_z-score':
        idx = moving_zscore_filter(data, win=win)
    elif method == 'MAD':
        idx = mad_filter(data, threshold=threshold, mad_mode='single')
    elif method == 'doubleMAD':
        idx = mad_filter(data, threshold=threshold, mad_mode='double')
    elif method == 'r_doubleMAD':
        idx = moving_mad_filter(data, win=win, threshold=threshold, mad_mode='double')
    elif method == 'r_singleMAD':
        idx = moving_mad_filter(data, win=win, threshold=threshold, mad_mode='single')
    else:
        raise ValueError(f'Given method ist not valid. Choose from {METHODS}')

    # filter flagged outlier indices at start and end of input array
    if filter_boundaries:
        # find consecutive flagged values
        ## get differences
        diff_idx = np.diff(idx)
        ## split into arrays holding consecutive flagged values
        diff_idx_split = np.split(diff_idx, np.nonzero(diff_idx > 1)[0])

        # check if first index is in input
        if np.isin(0, idx):
            # number of consecutive indices at start (add one due to split location)
            n_exclude_start = diff_idx_split[0].size + 1
            # exclude indices from flagged ones
            idx = idx[n_exclude_start:]

        # check last index is in input
        if np.isin(data.size - 1, idx):
            # number of consecutive indices at end
            n_exclude_end = diff_idx_split[-1].size
            # exclude indices from flagged ones
            idx = idx[:-n_exclude_end]

    # compute sampling indices
    x = np.arange(data.size)  # updated/altered input data

    # mask outliers for interpolation
    mask = np.ones(data.size, dtype='bool')
    mask[idx] = 0
    _data = data[mask]
    _x = x[mask]

    # create interpolation function
    _interp = interp.interp1d(_x, _data, kind=kind)
    # interpolate masked values
    data_interp = _interp(x)

    return data_interp


def median_abs_deviation(x, axis=-1):
    """
    Return the median absolute deviation (MAD) from given input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    mad : np.ndarray
        Median absolute deviation (MAD) of input array.

    """
    if x.ndim == 1:
        mad = np.median(np.abs(x - np.median(x, axis=axis)))
    elif x.ndim == 2:
        mad = np.median(np.abs(x.T - np.median(x, axis=axis)).T, axis=axis)
    else:
        raise ValueError(f'Input arrays with < {x.ndim} > dimensions are not supported!')
    return mad


def median_abs_deviation_double(x, axis=-1):
    """
    Return the median absolute deviation (MAD) for unsymmetric distributions.
    Computes the deviation from median for both sides (left & right).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : TYPE, optional
        Axis to compute median on (default: -1).

    Returns
    -------
    mad : np.ndarray
        Median absolute deviation (MAD) of input array.
    
    References
    ----------
    [^1]: [https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/](https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/)

    """
    if x.ndim == 1:
        med = np.median(x, axis=axis)
        diff = np.abs(x - med)
        mad_left = np.median(diff[x <= med])
        mad_right = np.median(diff[x >= med])
        if mad_left == 0 or mad_right == 0:
            raise ValueError('one side of median absolute deviation is zero')
        mad = np.repeat(mad_left, len(x))
        mad[x > med] = mad_right
    elif x.ndim == 2:
        # compute median for each window
        med = np.median(x, axis=axis)
        # difference from median (per window)
        diff = np.abs(x - med[:, None])

        # define column of reference value (in window)
        idx_col = x.shape[-1] // 2
        # left side MAD
        mad_left = np.median(diff[(x <= med[:, None])[:, idx_col]], axis=axis)
        mad_left[mad_left == 0] = 1
        # right side MAD
        mad_right = np.median(diff[(x >= med[:, None])[:, idx_col]], axis=axis)
        mad_right[mad_right == 0] = 1

        # create and fill output array
        mad = np.ones((x.shape[0],), dtype=x.dtype)
        mad[(x <= med[:, None])[:, idx_col]] = mad_left
        mad[(x >= med[:, None])[:, idx_col]] = mad_right
    else:
        raise ValueError(f'Input arrays with < {x.ndim} > dimensions are not supported!')

    return mad.astype(x.dtype)


def smooth(data, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters
    ----------
    data : np.ndarray
        1D input data array.
    window_len : int, optional
        Input window length, should be odd integer (default: 11).
    window : str, optional
        Tpye of smoothing window function (default: 'hanning').

    Returns
    -------
    out :
        smoothed input data
    
    References
    ----------
    [^1]: [https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html](https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)

    """
    if data.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if data.size < window_len:
        raise ValueError(
            f'Input data should be longer ({data.size}) than the window length ({window_len}).'
        )

    if window_len < 3:
        return data

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if window_len % 2 == 0:  # even
        left, right = window_len // 2, window_len // 2 + 1
    else:  # odd
        left, right = window_len // 2 + 1, window_len // 2 + 1
    # print(f'left: {left}, right {right} ')

    s = np.r_[data[left - 1 : 0 : -1], data, data[-2 : -right - 1 : -1]]
    # print('padded signal: ', len(s))

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    out = np.convolve(s, w / w.sum(), mode='valid')

    return out


def zscore_filter(data, axis=-1):
    """Z-score filter for outlier detection. Return array of outlier indices."""
    z_score = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
    return np.nonzero(np.logical_or(z_score < -1, z_score > 1))[0]


def moving_zscore_filter(data, win, axis=-1):  # noqa
    """
    Return array of outlier indices using moving z-score filter for outlier detection of length `win`.
    
    """
    mean = smooth(data, window_len=win, window='hanning')
    z_score = (data - mean) / np.std(data, axis=axis)
    return np.nonzero(np.logical_or(z_score < -1, z_score > 1))[0]


def iqr_filter(a, axis=-1):
    """Inter-quartile range (IQR) filter for outlier detection. Return array of outlier indices."""
    quantiles = np.quantile(a, [0.25, 0.75], axis=axis, keepdims=True)
    q1 = quantiles[0]
    q3 = quantiles[1]
    iqr = q3 - q1
    iqr_upper = q3 + 1.5 * iqr
    iqr_lower = q1 - 1.5 * iqr

    return np.nonzero(np.logical_or(a < iqr_lower, a > iqr_upper))[0]


def mad_filter(a, threshold=3, axis=-1, mad_mode='single'):
    """Median Absolute Deviation (MAD) filter. Return array of outlier indices."""
    med = np.median(a, axis=axis)
    if mad_mode == 'single':
        mad = median_abs_deviation(a)
    elif mad_mode == 'double':
        mad = median_abs_deviation_double(a)
    return np.nonzero((np.abs(a - med) / mad) > threshold)[0]


def moving_mad_filter(a, win, threshold=3, axis=-1, mad_mode='single'):  # noqa
    """Moving Median Absolute Deviation (MAD) filter of length `win`. Return array of outlier indices."""
    if (type(win) != int) or (win % 2 != 1):
        raise ValueError('window length must be odd integer')

    win_half = (win - 1) // 2

    # pad start and end of input array
    a_pad = pad_array(a, win_half)

    # create moving windows (as views)
    windows = moving_window(a_pad, window_length=win)

    # compute moving median
    moving_med = np.median(windows, axis=-1)

    # compute moving MAD
    if mad_mode == 'single':
        moving_mad = median_abs_deviation(windows)
    elif mad_mode == 'double':
        moving_mad = median_abs_deviation_double(windows)

    # account for case MAD == 0 (prone to false outlier detection)
    moving_mad[moving_mad == 0] = 1

    return np.nonzero((np.abs(a - moving_med) / moving_mad) > threshold)[0]


def moving_window(a, window_length: int, step_size: int = 1):
    """
    Create moving windows of given window length over input array (as view).

    Parameters
    ----------
    a : np.ndarray
        1D input array.
    window_length : int
        Length of moving window.
    step_size : int
        Step size of moving window (default: 1).

    Returns
    -------
    view : np.ndarray
        View of array according to `window_length` and `step_size`.

    References
    ----------
    [^1]: [https://stackoverflow.com/a/6811241](https://stackoverflow.com/a/6811241)
    [^2]: [https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html](https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html)
    [^3]: [https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788](https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788)

    """
    shape = a.shape[:-1] + (a.shape[-1] - window_length + 1 - step_size + 1, window_length)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def detect_seafloor_reflection(
    data,
    idx_slice_start=None,
    nsta=None,
    nlta=None,
    win=30,
    win_mad=None,
    win_mad_post=None,
    win_median=11,
    n: int = 5,
    post_detection_filter: bool = True,
):
    """
    Detect seafloor reflection using the STA/LTA algorithm.
    Its commonly applied in seismology that evaluates the ratio of short- and long-term energy density.
    The initially sample indices found by the STA/LTA algorithm are used to
    create individual search windows per trace (idx - win <= x <= idx + win).
    Return indices of maximum amplitude(s) within individual search windows (shape: (ntraces,)).

    Parameters
    ----------
    data : np.ndarray
        Input seismic section (samples x traces).
    idx_slice_start : np.ndarray, optional
        Index of first non-padded sample in original data.
    nsta : TYPE, optional
        Length of short time average window (in samples). If `None`: 0.1% of total samples.
    nlta : TYPE, optional
        Length of long time average window (in samples). If `None`: 5% of total samples.
    win : int, optional
        Number of samples to pad search window with (default: `20`).
        Set search window to `win` samples deeper and `win` x 2 samples shallower than baseline.
    win_mad : int, optional
        Number of traces used for Median Absolute Deviation (MAD) filtering.
        If None (default), this window is set to 5% of total traces.
    win_mad_post : int, optional
        Number of traces used for Median Absolute Deviation (MAD) filtering (after detection).
        If None (default), this window is set to 1% of total traces.
    win_median : int, optional
        Number of traces for rolling median filter, should be odd integer (default: `11`).
    post_detection_filter : bool, optional
        Apply optional Median Absolute Deviation (MAD) filtering
        after actual seafloor detection (default: `True`).

    Returns
    -------
    np.ndarray
        Indices of samples at maximum amplitude (per trace).

    """
    nsamples, ntraces = data.shape

    # check for zero traces (e.g. from merging)
    cnt_zero_traces = np.count_nonzero(data, axis=0)
    n_zero_traces = ntraces - np.count_nonzero(cnt_zero_traces, axis=0)

    # mask zero traces if found
    if n_zero_traces > 0:
        mask_nonzero_traces = cnt_zero_traces.astype('bool')
        data = data[:, mask_nonzero_traces]

    if nsta is None:
        nsta = int(np.around(nsamples * 0.001))
    if nlta is None:
        nlta = int(np.around(nsamples * 0.05))

    if nsta < 3:
        nsta = 3
        nlta = 50
        print(f'[WARNING]    Changed nsta={nsta} and nlta={nlta}!')

    # (1) calc standard STA/LTA from data array
    sta_lta = sta_lta_filter(data, nsta, nlta, axis=0)

    # (2) detect first significant amplitude peak (sample indices)
    # CAUTION: could be outlier (e.g. noise bursts in water column)
    #          but that misdetection will be filtered in the subsequent step!
    idx_sta_lta = np.argmax(sta_lta, axis=0)

    if idx_slice_start is not None:
        idx_sta_lta += idx_slice_start
        
    if idx_slice_start is not None:
        # (3) replace seafloor detections outside sample range with median value
        idx_sta_lta = np.where(
            np.logical_or(idx_sta_lta > nsamples - idx_slice_start, idx_sta_lta < idx_slice_start),
            np.median(idx_sta_lta),
            idx_sta_lta,
        )

    # # (3) outlier detection & removal #TODO: unnecessary?
    if win_mad is None:
        win_mad = int(idx_sta_lta.size * 0.02)
        win_mad = win_mad + 1 if win_mad % 2 == 0 else win_mad  # must be odd
        win_mad = 7 if win_mad < 7 else win_mad                 # at least 7 traces

    idx_sta_lta = filter_interp_1d(
        idx_sta_lta, method='r_doubleMAD', kind='cubic', threshold=3, win=win_mad
    ).astype('int')

    # (4) apply moving median filter to remove large outliers
    idx_sta_lta = moving_median(idx_sta_lta, win_median, padded=True).astype('int')
    
    # (5) detect `actual` first break amplitude
    # init index array
    idx_arr = np.arange(nsamples)[:, None]
    # create mask from slices (upper index <= slice <= lower index)
    idx_upper, idx_lower = (idx_sta_lta - win), (idx_sta_lta + win)  # *2
    mask = (idx_arr >= idx_upper) & (idx_arr <= idx_lower)
    # get indices from mask
    indices = np.apply_along_axis(np.nonzero, 0, mask).squeeze()

    # subset input array using indices of search window
    sta_lta_win = np.take_along_axis(data, indices, axis=0)

    # get `n` largest values for each trace subset
    # n = 5
    idx_nlargest = np.argpartition(-sta_lta_win, n, axis=0)[:n]
    # sort the indices for each trace (ascending order)
    idx_nlargest = np.take_along_axis(
        idx_nlargest, axis=0, indices=np.argsort(idx_nlargest, axis=0)
    )

    # get indices to split `idx_nlargest` into groups of different peak amplitudes
    idx_nlargest_sel = [
        np.nonzero(tr > 1)[0][0] if np.nonzero(tr > 1)[0].size > 0 else n
        for tr in np.diff(idx_nlargest, 1, axis=0).T
    ]

    # split the index array of `n` largest values and select first significant (positive) amplitude
    idx_nlargest_sel = [
        np.split(tr, [i])[0] if i != 0 else np.array([tr[i]])
        for tr, i in zip(idx_nlargest.T, idx_nlargest_sel)
    ]

    # get index of max. amplitude within selected maxima of first significant (positive) amplitude (NOTE: subset index!)
    idx_peak_amp = np.asarray(
        [
            nlarge[np.argmax(tr[i])]
            for nlarge, tr, i in zip(idx_nlargest.T, sta_lta_win.T, idx_nlargest_sel)
        ]
    )

    # convert subset indices to indices of seismic section
    idx_peak_amp += idx_upper
    
    if n_zero_traces > 0:
        x = np.arange(0, ntraces)  # create trace idx WITH zero traces
        x_masked = x[mask_nonzero_traces]  # masked zero traces
        # create interpolation function
        _interp = interp.interp1d(x_masked, idx_peak_amp, kind='linear')
        # interpolate masked indices of zero traces
        idx_peak_amp = _interp(x).astype('int')

    # (6) additional outlier detection & removal
    if post_detection_filter:
        if win_mad_post is None:
            win_mad_post = int(idx_sta_lta.size * 0.01)
            win_mad_post = (
                win_mad_post + 1 if win_mad_post % 2 == 0 else win_mad_post
            )  # must be odd
            win_mad_post = 7 if win_mad_post < 7 else win_mad_post  # at least 7 traces
        idx_peak_amp = filter_interp_1d(
            idx_peak_amp, method='r_doubleMAD', kind='cubic', threshold=3, win=win_mad_post
        ).astype('int')

    return idx_peak_amp.astype('int')


def sta_lta_filter(a, nsta: int, nlta: int, axis=-1):  # noqa
    """
    Compute the STA/LTA ratio (short-time-average / longe-time-average)
    by continuously calculating the average values of the absolute amplitude
    of a seismic trace in two consecutive moving-time windows.

    Parameters
    ----------
    a : np.ndarray
        Seismic trace (1D) or section (2D).
    nsta : int
        Length of short time average window (samples).
    nlta : int
        Length of long time average window (samples).
    axis : int, optional
        Axis for which to compute STA/LTA ratio (default: -1).

    Returns
    -------
    np.ndarray
        Either 1D or 2D array of STA/LTA ratio (per trace).
    
    References
    ----------
    [^1]: Withers et al. (1998) A comparison of select trigger algorithms for automated global seismic phase and event detection,
          [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.245&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.245&rep=rep1&type=pdf)
    [^2]: Trnkoczy, A. (2012) Understanding and parameter setting of STA/LTA trigger algorithm,
          [https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content](https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content)
    [^3]: ObsPy, [https://docs.obspy.org/_modules/obspy/signal/trigger.html#classic_sta_lta_py](https://docs.obspy.org/_modules/obspy/signal/trigger.html#classic_sta_lta_py)

    """
    if any(s == 1 for s in a.shape):
        a = np.squeeze(a, axis=-1).copy()

    # calculate moving average
    sta = np.cumsum(a**2, axis=axis).astype('float')

    # copy for LTA
    lta = sta.copy()

    # compute the STA and the LTA
    if a.ndim == 1:
        sta[nsta:] = sta[nsta:] - sta[:-nsta]
        sta /= nsta
        lta[nlta:] = lta[nlta:] - lta[:-nlta]
        lta /= nlta

        # pad zeros
        sta[: nlta - 1] = 0
    elif a.ndim == 2:
        sta[nsta:, :] = sta[nsta:, :] - sta[:-nsta, :]
        sta /= nsta
        lta[nlta:, :] = lta[nlta:, :] - lta[:-nlta, :]
        lta /= nlta

        # pad zeros
        sta[: nlta - 1, :] = 0

    # avoid division by zero!
    return np.divide(sta, lta, out=np.zeros_like(sta, dtype=sta.dtype), where=(lta != 0))


def moving_median(a, win: int = 3, padded=False):
    """
    Apply moving median of given window size.
    Optional padding of input array using half the window size to avoid edge effects.

    Parameters
    ----------
    a : np.ndarray
        Input data (1D).
    win : int, optional
        Number of data points within moving window (default: `3`).
    padded : bool, optional
        Pad start and end of array (default: `False`).

    Returns
    -------
    np.ndarray
        Moving median of input data.

    """
    if padded:
        half_win = (win - 1) // 2
        a = pad_array(a, half_win)

    windows = moving_window(a, window_length=win)

    return np.median(windows, axis=-1)
