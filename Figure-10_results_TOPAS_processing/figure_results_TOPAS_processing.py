"""
Create Figure illustrating results from 2D TOPAS (pre)processing steps (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-10-28

"""
import os
import sys

import numpy as np
import segyio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import hilbert

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import AnchoredScaleBar, rms_normalization, _print_info_TOPAS_moratorium

_print_info_TOPAS_moratorium()

#%% FUCTIONS

def envelope(signal, axis=-1):
    """
    Compute envelope of a seismic trace (1D), section (2D) or cube (3D) using the Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Seismic trace (1D) or section (2D).
    axis : int, optional
        Axis along which to do the transformation (default: -1).

    Returns
    -------
    np.ndarray
        Amplitude envelope of input array along `axis`.

    """
    signal_analytic = hilbert(signal, axis=axis)
    return np.abs(signal_analytic).astype(signal.dtype)


def open_segy(path, extended_attrs: bool = False, **kwargs_segyio):
    """Open SEG-Y file and return trace header information."""
    if kwargs_segyio == {}:
        kwargs_segyio = dict(strict=False, ignore_geometry=True)

    params = dict()

    with segyio.open(path, "r", **kwargs_segyio) as src:
        params["n_traces"] = src.tracecount  # total number of traces
        params["dt"] = segyio.tools.dt(src) / 1000  # sample rate [ms]
        params["n_samples"] = src.samples.size  # total number of samples
        params["twt"] = src.samples  # two way travel time (TWTT) [ms]

        if extended_attrs:
            params["tracl"] = src.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[
                :
            ]  # Trace sequence number within line – numbers continue to increase if additional reels are required on same line.
            params["tracr"] = src.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[
                :
            ]  # Trace sequence number within reel – each reel starts at trace number one
            params["fldr"] = src.attributes(segyio.TraceField.FieldRecord)[
                :
            ]  # field record number
            params["swdep"] = src.attributes(segyio.TraceField.SourceWaterDepth)[:]

            params["delrt"] = src.attributes(segyio.TraceField.DelayRecordingTime)[
                :
            ]  # Delay recording time (ms)

        # get seismic data [amplitude]; transpose to fit numpy data structure
        params["data"] = src.trace.raw[:].T  # eager version (completely read into memory)

    return params


def plot_seismic_image(
    data,
    dt=None,
    twt=None,
    traces=None,
    cmap="Greys",
    show_colormap=True,
    show_xaxis_labels=True,
    gain=1,
    norm=False,
    title=None,
    env=False,
    reverse=False,
    units="ms",
    ax=None,
    interpolation="antialiased",
    label_kwargs=None,
    plot_kwargs=None,
):
    """
    Plot seismic traces of SEG-Y file as image using specified colormap and gain.

    Parameters
    ----------
    data : numpy.array
        2D array of SEG-Y trace data..
    dt : float, optional
        Sampling interval in specified units (default: seconds).
        The default is None.
    twt : np.array, optional
        1D array of two-way traveltimes (TWT, default: seconds).
        The default is None.
    traces : np.array, optional
        1D array of trace indices. The default is None.
    cmap : str, optional
        Matplotlib-compatible string of colormap. The default is 'Greys'.
    gain : int, optional
        Custom gain parameter (for visualization only). The default is 1.
    norm :
        Normalize amplitude of trace(s) using `rms` or `peak` amplitude.
    title : str
        Figure title (e.g. filename). The default is None.
    env : bool, optional
        Envelope as input data type. The default is False (amplitude).
    reverse : bool, optional
        Reverse profile orientation for plotting. The default is False.
    units : str, optional
        Time units (y-axis). The default is 'ms'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Axes handle.
    colormap : matplotlib.colorbar.Colorbar
        Colormap handle.

    """
    # get samples and traces from data
    nsamples, ntraces = data.shape

    # create time axis (convert dt fro ms to s)
    if dt is None and twt is None:
        raise ValueError("Either dt or twt required")
    elif dt is not None:
        twt = np.linspace(0, dt * nsamples, nsamples)
    elif twt is not None:
        dt = np.mean(np.diff(twt))

    # normalize
    if norm is True or isinstance(norm, str) and norm.lower() == "rms":
        data = rms_normalization(data, axis=0)
    elif isinstance(norm, str) and norm.lower() in ["max", "peak"]:
        data /= np.max(np.abs(data))

    # set plotting extent [xmin, xmax, ymin, ymax]
    _offset = 0.5 * dt
    extent = [-_offset, ntraces + _offset, twt[-1] + _offset, twt[0] - _offset]

    # clip amplitude data for plotting
    if gain is not None:
        clip_percentile = ((1 - gain) * 2) + 97.5  # empirically tested
        vm = np.percentile(data, clip_percentile)  # clipping
    else:
        vm = np.max(np.abs(data))
    # adjust parameter for colormap
    vmax = vm
    if env:
        vmin = 0
        data_label = "envelope"
    else:
        vmin = -vm
        data_label = "amplitude"

    x_label = "Trace #" if traces is None else "Field record number"
    y_label = f"Time ({units})"

    if label_kwargs is None:
        label_kwargs = dict(labels_size=12, ticklabels_size=10, title_size=12)

    # create figure and axes
    if plot_kwargs is None:
        plot_kwargs = dict(figsize=(16, 8))

    if ax is None:
        fig, ax = plt.subplots(1, 1, **plot_kwargs)
    else:
        fig = ax.get_figure()

    # plot data
    profile = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=extent,
        interpolation=interpolation,
    )
    # create colormap
    if show_colormap:
        cbar = fig.colorbar(
            profile,
            ax=ax,
            pad=0.025,
            fraction=0.05,  # pad=0.01
            location="right",
            orientation="vertical",
            format="%.3f",
        )
        cbar.ax.set_ylabel(
            data_label, labelpad=25, rotation=270, fontsize=label_kwargs["labels_size"]
        )
        cbar.ax.tick_params(axis="y", labelsize=label_kwargs["ticklabels_size"])

    # set x-axis
    ## ticks
    if traces is not None:
        if ntraces < 25:
            xticks = np.arange(0, ntraces, 1)
            xticklabels = [str(t) for t in traces]
        else:  # too many labels to plot for every trace
            xticks = np.arange(
                0, ntraces + 1, np.around(ntraces // 10, 1 - len(str(ntraces // 10)))
            )
            xticks = np.append(xticks, np.atleast_1d(ntraces - 1), axis=0)
            xticks = xticks[xticks < ntraces]
            xticklabels = [str(t) for t in traces[xticks]]
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticklabels, rotation=45, ha="left", fontsize=label_kwargs["ticklabels_size"]
        )
    else:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
    ax.xaxis.tick_top()

    ## labels
    if show_xaxis_labels:
        ax.set_xlabel(x_label, fontweight="semibold", fontsize=label_kwargs["labels_size"])
        ax.xaxis.set_label_position("top")
    else:
        ax.set_xticklabels([])

    # set y-axis
    ## ticks
    # ax.set_ylim([twt.max(), twt.min()])
    ax.tick_params(
        axis="y", which="minor", direction="out", bottom=False, top=False, left=True, right=False
    )
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(11))
    ## labels
    ax.set_ylabel(y_label, fontweight="semibold", fontsize=label_kwargs["labels_size"])

    ax.tick_params(axis="both", labelsize=label_kwargs["ticklabels_size"])

    # set subplot title
    if title is not None:
        ax.set_title(title, fontweight="semibold", fontsize=label_kwargs["title_size"])

    # reverse profile plot if needed
    if reverse:
        ax.invert_xaxis()

    if ax and show_colormap:
        return fig, ax, profile, cbar
    return fig, ax, profile


def plot_inset_wiggle(
    ax,
    loc_inset: list,
    data: np.ndarray,
    twt_slice: slice,
    trace_slice: slice,
    scaler: float = 1.0,
    add_connectors: bool = True,
    add_color: bool = False,
    kwargs_plot: dict = None,
    kwargs_con: dict = None,
):
    """Plot inset figure showing noise spike."""
    if kwargs_plot is None or kwargs_plot == {}:
        kwargs_plot = dict(c="black", alpha=1, lw=1)
    if kwargs_con is None or kwargs_con == {}:
        kwargs_con = dict(edgecolor="black", alpha=1, lw=0.5)

    traces = data[twt_slice, trace_slice]
    if not traces.min() >= 0.0:
        kwargs_plot["lw"] = 0.5

    ax_wiggle = ax.inset_axes(loc_inset)
    y = np.arange(traces.shape[0])
    for i in range(traces.shape[-1]):
        ax_wiggle.plot(traces[:, i] + (i + 1) / scaler, y, **kwargs_plot)
        if traces.min() >= 0.0:
            ax_wiggle.fill_betweenx(
                y,
                (i + 1) / scaler,
                traces[:, i] + (i + 1) / scaler,
                where=(traces[:, i] + (i + 1) / scaler >= (i + 1) / scaler),
                color=kwargs_plot.get("c", "k"),
            )
        else:
            ax_wiggle.axvline((i + 1) / scaler, c="black", lw=0.5, zorder=5)
            if add_color:
                ax_wiggle.fill_betweenx(  # positive
                    y,
                    (i + 1) / scaler,
                    traces[:, i] + (i + 1) / scaler,
                    where=(traces[:, i] + (i + 1) / scaler >= (i + 1) / scaler),
                    color="blue",
                )
                ax_wiggle.fill_betweenx(  # negative
                    y,
                    (i + 1) / scaler,
                    traces[:, i] + (i + 1) / scaler,
                    where=(traces[:, i] + (i + 1) / scaler <= (i + 1) / scaler),
                    color="red",
                )

    ax_wiggle.set_ylim(0, traces.shape[0])
    ax_wiggle.invert_yaxis()
    ax_wiggle.tick_params(
        axis="both", which="both", left=False, labelleft=False, bottom=False, labelbottom=False
    )

    # add connectors
    if add_connectors:
        patch, connectors = ax.indicate_inset(
            bounds=spike_bounds, inset_ax=ax_wiggle, transform=ax.transData, **kwargs_con
        )
        for c, v in zip(
            connectors, [True, True, False, False]
        ):  # lower_left, upper_left, lower_right, upper_righ
            c.set_visible(v)

        return ax_wiggle, (patch, connectors)
    
    return ax_wiggle


def plot_inset_static(
    ax,
    loc_inset: list,
    data: np.ndarray,
    profile,
    xlims: list = None,
    ylims: list = None,
    scaler: float = 10.0,
    kwargs_con: dict = None,
):
    """Plot inset figure showing close-up of static effect."""
    if kwargs_con is None or kwargs_con == {}:
        kwargs_con = dict(ec="black", alpha=1, lw=0.5)

    ax_static = ax.inset_axes(loc_inset)
    ax_static.imshow(
        data,
        cmap=profile.get_cmap(),
        vmin=profile.get_clim()[0] * scaler,
        vmax=profile.get_clim()[1] * scaler,
        aspect="auto",
        extent=profile.get_extent(),
        interpolation=profile.get_interpolation(),
    )

    ax_static.set_xlim(xlim_static)
    ax_static.set_ylim(ylim_static)
    ax_static.invert_yaxis()
    ax_static.tick_params(
        axis="both", which="both", left=False, labelleft=False, bottom=False, labelbottom=False
    )
    # add connectors
    patch, connectors = ax.indicate_inset_zoom(inset_ax=ax_static, **kwargs_con)
    patch.set(edgecolor=kwargs_con.get("edgecolor", "k"), linewidth=kwargs_con.get("lw", 1))
    for c, v in zip(
        connectors, [False, True, False, True]
    ):  # lower_left, upper_left, lower_right, upper_righ
        c.set_visible(v)

    return ax_static, (patch, connectors)


#%% MAIN

if __name__ == "__main__":

    dir_work = "C:/PhD/processing/TOPAS/TAN2006/pockmarks_3D"
    dir_fig = "."

    file = "20200704001310"
    dist_traces = 2.25  # meter

    # paths_segy = glob.glob(os.path.join(dir_work, file + "*.sgy"))
    path_raw = "../20200704001310_UTM60S.sgy"
    path_proc = "../20200704001310_UTM60S_static_tide_mistie_despk.sgy"

    segy_raw = open_segy(path_raw, extended_attrs=True)
    segy_proc = open_segy(path_proc, extended_attrs=True)

    #%% [PLOT]
    
    dpi = 600
    
    cmap_amp = "RdBu"
    cmap_env = "Greys"
    gain_amp = 3
    gain_env = 1.5

    norm = "rms"

    twt_min = 725
    twt_max = 925

    scalebar_dist = 1000

    labelsize = 12
    ticksize = 10

    color = "black"
    alpha = 1

    kwargs_labels = {"labels_size": labelsize, "ticklabels_size": ticksize}
    kwargs_colorbar = dict(
        pad=0.04, fraction=0.1, shrink=0.9, extend="both", orientation="horizontal", format="%.3g"
    )
    kwargs_colorbar_label = dict(
        labelpad=-42, rotation=None, fontsize=labelsize, fontweight="semibold"
    )

    kwargs_con = dict(edgecolor="white", lw=0.5, alpha=1)

    with mpl.rc_context(
        {"font.family": "Arial", "mathtext.default": "default", "mathtext.fontset": "stixsans"}
    ):

        # ========== CREATE FIGURE ==========
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(8, 7.5),
            constrained_layout=True,
            gridspec_kw={"wspace": 0.075, "hspace": 0.05},
        )

        # ========== RAW DATA ==========
        _, _, profile_raw_amp = plot_seismic_image(
            segy_raw["data"],
            twt=segy_raw["twt"],
            traces=None,
            cmap=cmap_amp,
            show_colormap=False,
            show_xaxis_labels=True,
            gain=gain_amp,
            norm=norm,
            ax=axes[0, 0],
            label_kwargs=kwargs_labels,
        )

        # define spike inset
        spike_xy = (138, 858)
        spike_offset = (5, 13)
        spike_offset_bounds = (spike_offset[0] * 2, spike_offset[1])
        spike_offset_data = (spike_offset[0] // 2, spike_offset[1])
        # x0, y0, width, height
        spike_bounds = [
            spike_xy[0] - spike_offset_bounds[0],
            spike_xy[1] - spike_offset_bounds[1],
            spike_offset_bounds[0] * 2 + 1,
            spike_offset_bounds[1] * 2 + 1,
        ]

        twt_slice = slice(
            int((spike_xy[1] - spike_offset_data[1] - segy_raw["twt"][0]) / segy_raw["dt"]),
            int((spike_xy[1] + spike_offset_data[1] - segy_raw["twt"][0] + 1) / segy_raw["dt"]),
            1,
        )
        trace_slice = slice(
            spike_xy[0] - spike_offset_data[0], spike_xy[0] + spike_offset_data[0] + 1, 1
        )
        loc_spike = [0.1, 0.02, 0.2, 0.35]

        # plot spike inset
        ax_raw_amp_static, _ = plot_inset_wiggle(
            axes[0, 0],
            loc_inset=loc_spike,
            scaler=3,
            data=segy_raw["data"],
            twt_slice=twt_slice,
            trace_slice=trace_slice,
        )

        # annotate spike
        ax_raw_amp_static.annotate(
            "",
            xy=(0.35, 0.1),
            xycoords="axes fraction",
            xytext=(0.35, 0.9),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="|-|", shrinkA=0, shrinkB=0, mutation_scale=4),
            annotation_clip=False,
        )
        ax_raw_amp_static.text(
            0.175,
            0.5,
            "20 ms",
            transform=ax_raw_amp_static.transAxes,
            rotation=90,
            rotation_mode="anchor",
            va="center",
            ha="center",
        )

        # define static inset
        loc_static = [0.5, 0.15, 0.2, 0.4]
        xlim_static = [2275, 2295]  # [2275, 2325]
        ylim_static = [751.5, 754.5]

        # plot static inset
        ax_raw_amp_static, _ = plot_inset_static(
            axes[0, 0],
            loc_inset=loc_static,
            data=segy_raw["data"],
            profile=profile_raw_amp,
            xlims=xlim_static,
            ylims=ylim_static,
            scaler=1.0,
            kwargs_con=kwargs_con,
        )

        # add scale bar
        width = abs(np.subtract(*xlim_static)) * 0.5
        sizex = width / dist_traces
        kwargs_scalebar_inset = dict(
            sizex=sizex,
            sizey=None,
            loc="lower right",
            pad=0.5,
            borderpad=0.25,
            sep=3,
            labelx=f"{width:.0f} m",
            labely=None,
            barwidth=2,
        )
        sb = AnchoredScaleBar(ax_raw_amp_static.transData, **kwargs_scalebar_inset)
        ax_raw_amp_static.add_artist(sb)
        
        # define static inset (WIGGLE)
        loc_static_wiggle = [0.7, 0.15, 0.28, 0.4]
        trace_slice_wiggle = slice(2275, 2295, 2)  # [2275, 2325]
        twt_slice_wiggle = slice(
            int((ylim_static[0] - segy_raw["twt"][0]) / segy_raw["dt"]),
            int((ylim_static[1] - segy_raw["twt"][0]) / segy_raw["dt"])
        )

        # plot static inset (WIGGLE)
        plot_inset_wiggle(
            axes[0, 0],
            loc_inset=loc_static_wiggle,
            scaler=2,
            data=segy_raw["data"],
            twt_slice=twt_slice_wiggle,
            trace_slice=trace_slice_wiggle,
            add_connectors=False,
            add_color=True,
        )

        # ---------- create colormap (AMPLITUDE) ----------
        cbar = fig.colorbar(profile_raw_amp, ax=axes[0, 0], **kwargs_colorbar)
        cbar.ax.set_xlabel("Amplitude", **kwargs_colorbar_label)
        cbar.ax.tick_params(axis="x", labelsize=kwargs_labels["ticklabels_size"])

        # ========== RAW DATA (ENVELOPE) ==========
        segy_raw_env = envelope(segy_raw["data"], axis=0)
        _, _, profile_raw_env = plot_seismic_image(
            segy_raw_env,
            twt=segy_raw["twt"],
            traces=None,
            cmap=cmap_env,
            show_colormap=False,
            show_xaxis_labels=True,
            gain=gain_env,
            norm=False,
            env=True,
            ax=axes[0, 1],
            label_kwargs=kwargs_labels,
        )
        axes[0, 1].set_ylabel(None)

        # plot spike inset
        ax_raw_env_spike, _ = plot_inset_wiggle(
            axes[0, 1],
            loc_inset=loc_spike,
            scaler=3,
            data=segy_raw_env,
            twt_slice=twt_slice,
            trace_slice=trace_slice,
        )
        # annotate spike
        ax_raw_env_spike.annotate(
            "",
            xy=(0.35, 0.1),
            xycoords="axes fraction",
            xytext=(0.35, 0.9),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="|-|", shrinkA=0, shrinkB=0, mutation_scale=4),
            annotation_clip=False,
        )
        ax_raw_env_spike.text(
            0.185,
            0.5,
            "20 ms",
            transform=ax_raw_env_spike.transAxes,
            rotation=90,
            rotation_mode="anchor",
            va="center",
            ha="center",
        )

        # plot static inset
        ax_raw_env_static, _ = plot_inset_static(
            axes[0, 1],
            loc_inset=loc_static,
            data=segy_raw_env,
            profile=profile_raw_env,
            xlims=xlim_static,
            ylims=ylim_static,
            scaler=10.0,
            kwargs_con=kwargs_con,
        )
        # add scale bar
        sb = AnchoredScaleBar(ax_raw_env_static.transData, **kwargs_scalebar_inset)
        ax_raw_env_static.add_artist(sb)
        
        # plot static inset (WIGGLE)
        plot_inset_wiggle(
            axes[0, 1],
            loc_inset=loc_static_wiggle,
            scaler=2,
            data=segy_raw_env,
            twt_slice=twt_slice_wiggle,
            trace_slice=trace_slice_wiggle,
            add_connectors=False,
            add_color=True,
        )

        # ---------- create colormap (ENVELOPE) ----------
        kwargs_colorbar.update({"extend": "max"})
        cbar = fig.colorbar(profile_raw_env, ax=axes[0, 1], **kwargs_colorbar)
        cbar.ax.set_xlabel("Envelope", **kwargs_colorbar_label)
        cbar.ax.tick_params(axis="x", labelsize=kwargs_labels["ticklabels_size"])

        # ========== PROCESSED DATA ==========
        _, _, profile_proc_amp = plot_seismic_image(
            segy_proc["data"],
            twt=segy_proc["twt"],
            traces=None,
            cmap=cmap_amp,
            show_colormap=False,
            show_xaxis_labels=True,
            gain=gain_amp,
            norm=norm,
            ax=axes[1, 0],
            label_kwargs=kwargs_labels,
        )
        axes[1, 0].set_xlabel(None)

        # plot spike inset
        plot_inset_wiggle(
            axes[1, 0],
            loc_inset=loc_spike,
            scaler=20,
            data=segy_proc["data"],
            twt_slice=twt_slice,
            trace_slice=trace_slice,
        )

        # plot static inset
        ax_proc_amp_static, _ = plot_inset_static(
            axes[1, 0],
            loc_inset=loc_static,
            data=segy_proc["data"],
            profile=profile_proc_amp,
            xlims=xlim_static,
            ylims=ylim_static,
            scaler=1.0,
            kwargs_con=kwargs_con,
        )

        # add scale bar
        sb = AnchoredScaleBar(ax_proc_amp_static.transData, **kwargs_scalebar_inset)
        ax_proc_amp_static.add_artist(sb)
        
        # plot static inset (WIGGLE)
        plot_inset_wiggle(
            axes[1, 0],
            loc_inset=loc_static_wiggle,
            scaler=2,
            data=segy_proc["data"],
            twt_slice=twt_slice_wiggle,
            trace_slice=trace_slice_wiggle,
            add_connectors=False,
            add_color=True,
        )

        # ========== PROCESSED DATA (ENVELOPE) ==========
        segy_proc_env = envelope(segy_proc["data"], axis=0)
        _, _, profile_proc_env = plot_seismic_image(
            segy_proc_env,
            twt=segy_proc["twt"],
            traces=None,
            cmap=cmap_env,
            show_colormap=False,
            show_xaxis_labels=True,
            gain=gain_env,
            norm=False,
            env=True,
            ax=axes[1, 1],
            label_kwargs=kwargs_labels,
        )
        axes[1, 1].set_xlabel(None)
        axes[1, 1].set_ylabel(None)

        # plot spike inset
        plot_inset_wiggle(
            axes[1, 1],
            loc_inset=loc_spike,
            scaler=20,
            data=segy_proc_env,
            twt_slice=twt_slice,
            trace_slice=trace_slice,
        )

        # plot static inset
        ax_proc_env_static, _ = plot_inset_static(
            axes[1, 1],
            loc_inset=loc_static,
            data=segy_proc_env,
            profile=profile_proc_env,
            xlims=xlim_static,
            ylims=ylim_static,
            scaler=5.0,
            kwargs_con=kwargs_con,
        )
        # add scale bar
        sb = AnchoredScaleBar(ax_proc_env_static.transData, **kwargs_scalebar_inset)
        ax_proc_env_static.add_artist(sb)
        
        # plot static inset (WIGGLE)
        plot_inset_wiggle(
            axes[1, 1],
            loc_inset=loc_static_wiggle,
            scaler=2,
            data=segy_proc_env,
            twt_slice=twt_slice_wiggle,
            trace_slice=trace_slice_wiggle,
            add_connectors=False,
            add_color=True,
        )

        # ========== AXIS PARAMS ==========
        kwargs_axis_labels = dict(
            fontsize=labelsize,
            fontweight="semibold",
            backgroundcolor="white",
        )

        labels_subplots = ["a", "b", "c", "d"]
        labels_subplots = [f"{s})" for s in labels_subplots]
        pos_subplot_labels = (-0.025, 1.075)  # 0.935
        # pos_subplot_labels = (-0.075, 1.075)
        # pos_subplot_labels_left = (-0.15, 1.05)
        # pos_subplot_labels_right = (-0.075, 1.05)
        kwargs_subplot_labels = dict(
            ha="center", va="center", color="black", fontsize=18, fontweight="bold", family="Times New Roman",
            backgroundcolor='white'
        )

        for i, ax in enumerate(axes.flatten()):
            ax.set_ylim((twt_max, twt_min))

            ax.tick_params(axis="y", which="both", left=True, labelleft=True, right=True)

            if i % 2 == 1:
                ax.tick_params(axis="y", which="both", labelleft=False)

            if i > 1:
                ax.tick_params(axis="x", which="both", labeltop=False)

            ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(25))

            sizex = scalebar_dist / dist_traces
            sb = AnchoredScaleBar(
                ax.transData,
                sizex=sizex,
                sizey=None,
                loc="lower right",
                pad=0.5,
                borderpad=0.25,
                sep=3,
                labelx=f"{scalebar_dist} m",
                labely=None,
                barwidth=2,
            )
            ax.add_artist(sb)

            # adjust xlabel/ylabel positions
            label_offset = 0.03
            if i <= 1:
                label_xaxis = ax.get_xlabel()
                label_xaxis_txt = ax.set_xlabel(label_xaxis, ha="left", **kwargs_axis_labels)
                ax.xaxis.set_label_coords(0.05, 1.04)

            if i % 2 == 0:
                label_yaxis = ax.get_ylabel()
                label_yaxis_txt = ax.set_ylabel(label_yaxis, ha="right", **kwargs_axis_labels)
                ax.yaxis.set_label_coords(-0.04, 0.98)

            # profile orientation
            kwargs_orientation = kwargs_subplot_labels.copy()
            del kwargs_orientation['family']
            del kwargs_orientation['backgroundcolor']
            kwargs_orientation.update(fontsize=14, fontweight="normal")
            ax.text(0.05, 0.935, "N", transform=ax.transAxes, **kwargs_orientation)
            ax.text(0.95, 0.935, "S", transform=ax.transAxes, **kwargs_orientation)

            # ----- SUBPLOT LABELS -----
            ax.text(
                *pos_subplot_labels,
                labels_subplots[i],
                transform=ax.transAxes,
                **kwargs_subplot_labels,
            )

        # ---------- ROW / COLUMN LABELS ----------
        kwargs_headers = dict(
            fontsize=labelsize + 3, fontweight="semibold", color=color, va="center", ha="center"
        )
        # rows
        xy_row_headers = (-0.2, 0.5)
        axes[0, 0].text(
            *xy_row_headers,
            "Unprocessed",
            rotation=90,
            rotation_mode="anchor",
            transform=axes[0, 0].transAxes,
            **kwargs_headers,
        )
        axes[1, 0].text(
            *xy_row_headers,
            "Processed",
            rotation=90,
            rotation_mode="anchor",
            transform=axes[1, 0].transAxes,
            **kwargs_headers,
        )
        # columns
        xy_col_headers = (0.5, 1.15)
        axes[0, 0].text(
            *xy_col_headers, "Full waveform", transform=axes[0, 0].transAxes, **kwargs_headers
        )
        axes[0, 1].text(
            *xy_col_headers, "Envelope", transform=axes[0, 1].transAxes, **kwargs_headers
        )

        #%% save figure
        figure_number = 10
        plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_TOPAS_processing_{dpi}dpi.png"), dpi=dpi)
        # plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_TOPAS_processing.pdf"), dpi=dpi)
