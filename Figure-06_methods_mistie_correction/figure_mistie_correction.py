"""
Create Figure illustrating mistie correction application (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-10-25

"""
import os
import sys
import glob

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import segyio
from scipy.signal import hilbert, correlate, correlation_lags

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import AnchoredScaleBar, _print_info_TOPAS_moratorium

_print_info_TOPAS_moratorium()

#%% FUNCTIONS

def find_nearest(array, value, return_index=False):
    """Find index of nearest value to input `value` in `array`)."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_index:
        return idx
    return array[idx]


def open_segy(path, return_params=True):
    """Open SEG_Y file and return trace data as numpy.ndarray."""
    params = dict()

    with segyio.open(path, "r", strict=False, ignore_geometry=True) as src:
        params["n_traces"] = src.tracecount  # total number of traces
        params["dt"] = segyio.tools.dt(src) / 1000  # sample rate [ms]
        params["n_samples"] = src.samples.size  # total number of samples
        params["twt"] = src.samples  # two way travel time (TWTT) [ms]
        print(f"[INFO]    {path_iline}")
        print(f"n_traces:  {params['n_traces']}")
        print(f"n_samples: {params['n_samples']}")
        print(f"dt:        {params['dt']} ms")

        # eager version (completely read into memory)
        data = src.trace.raw[:].T
        print("# of samples:      ", data.size)
        print("# of zero samples: ", data.size - np.count_nonzero(data))

    try:
        src.close()
    except IOError:
        print("[WARNING]    SEG-Y file was already closed.")

    if return_params:
        return data, params
    return data


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


def cross_correlation_shift(cc):
    """
    Return cross-correlation shift (in samples) between correlated signals.

    Parameters
    ----------
    cc : np.ndarray
        Cross-correlation between two signals (1D).

    Returns
    -------
    shift : int
        Lag shift between both signals.
            shift < 0  --->  signal A later than signal B
            shift > 0  --->  signal A earlier than signal B

    """
    zero_idx = int(np.floor(len(cc) / 2))
    idx = np.argmax(cc) if np.abs(np.max(cc)) >= np.abs(np.min(cc)) else np.argmin(cc)
    return zero_idx - idx


def plot_traces(
    ax,
    data_traces,
    twt,
    twt_limits: tuple,
    df,
    idx,
    color="blue",
    marker: tuple = ("o", "s"),
    offset_labels: float = 0.25,
    offset_start: float = 0.0,
    offset_factor: float = 1.5,
    index_letter: str = "i",
    title: str = None,
    plot_yaxis=True,
    kw_text: dict = None,
    kw_marker: dict = None,
    return_data: bool = False,
):
    """Plot traces at intersection."""
    #
    kwargs_text = dict(fontsize=12)
    if kw_text is not None:
        kwargs_text.update(kw_text)

    kwargs_marker = dict(s=24, edgecolor=None, facecolor=color, clip_on=False)
    if kw_marker is not None:
        kwargs_marker.update(kw_marker)

    mask = (twt >= twt_limits[0]) & (twt <= twt_limits[1])
    twt = twt[mask]

    tracr = df.loc[idx, "tracr"].values - 1  # `tracr` starts with 1!
    traces = data_traces[:, tracr][mask]
    # compute reference trace (from trace mixing)
    trace_ref = traces.mean(axis=-1)

    offsets = np.arange(
        offset_start, offset_start + traces.shape[1] * offset_factor, offset_factor
    )
    markers = list(marker[-1]) * len(offsets)
    markers[len(markers) // 2] = marker[0]

    index_letter_rm = "\mathrm{" + index_letter + "}"  # noqa
    labels = (
        [f"{index_letter_rm}-{x}" for x in np.arange(1, len(markers[: len(markers) // 2]) + 1)]
        + [index_letter_rm]
        + [f"{index_letter_rm}+{x}" for x in np.arange(1, len(markers[len(markers) // 2 :]))]
    )
    labels = ["$\mathrm{tr}_{" + label + "}$" for label in labels]  # noqa

    # plot original traces
    for i, offset in enumerate(offsets):
        ax.plot(traces[:, i] + offset, twt, c=color, lw=0.5, clip_on=False)
        ax.fill_betweenx(
            twt,
            offset,
            traces[:, i] + offset,
            where=(traces[:, i] + offset >= offset),
            color=color,
            clip_on=False,
        )
        # add marker
        ax.scatter(offset, twt_limits[0] - offset_labels, marker=markers[i], **kwargs_marker)
        # add label
        ax.text(
            offset,
            twt_limits[1] + offset_labels,
            labels[i],
            va="center",
            ha="center",
            **kwargs_text,
        )
        # plot '+'
        if i > 0:
            ax.text(
                offset - offset_factor / 2,
                twt_limits[0] + (twt_limits[1] - twt_limits[0]) / 2,
                "+",
                va="center",
                ha="center",
                color="k",
                fontsize=20,
                fontweight="semibold",
            )

    # plot reference trace
    offset_factor_ref = offset_factor * 1.2
    offset_ref = offset + offset_factor_ref
    ax.plot(trace_ref + offset_ref, twt, c=color, lw=1, clip_on=False)
    ax.fill_betweenx(
        twt,
        offset_ref,
        trace_ref + offset_ref,
        where=(trace_ref + offset_ref >= offset_ref),
        color=color,
        clip_on=False,
    )

    # plot '='
    ax.text(
        offset_ref - offset_factor_ref / 2,
        twt_limits[0] + (twt_limits[1] - twt_limits[0]) / 2,
        "=",
        va="center",
        ha="center",
        color="k",
        fontsize=20,
        fontweight="semibold",
    )

    # add label
    ax.text(
        offset_ref,
        twt_limits[1] + offset_labels + offset_labels / 2,
        "\n".join(
            [
                "$\mathrm{tr_{" + f"{index_letter},avg" + "}}$",  # noqa
                "$\mathrm{tr_{" + f"{index_letter},env" + "}}$",  # noqa
            ]
        ),
        va="center",
        ha="center",
        linespacing=0.8,
        **kwargs_text,
    )

    # add title
    if title is not None:
        ax.set_title(title, color=color, pad=25, **kwargs_text)

    # plot trace envelope
    offset_ref_env = offset_ref + offset_factor / 20
    trace_ref_env = envelope(trace_ref)
    ax.plot(trace_ref_env + offset_ref_env, twt, c=color, lw=2, clip_on=False)

    ax.set_ylim(twt_limits)
    ax.set_xlim([-0.75, offset_ref_env])
    ax.invert_yaxis()

    # remove ticks
    ax.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
    for loc in ["top", "right", "bottom"]:
        ax.spines[loc].set_visible(False)

    # plot TWT axis
    if plot_yaxis:
        # ax.spines['left'].set_position(('outward', 5))
        ax.set_ylabel("TWT (ms)", **kwargs_text)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    if return_data:
        return traces, trace_ref, trace_ref_env, twt

    return None


#%% MAIN

if __name__ == "__main__":

    dir_work = ".."
    dir_fig = os.path.dirname(os.path.abspath(__file__))

    file = "../TOPAS_metadata.aux"
    iline = "20200704001310"
    xline = "20200709015919_merge"
    
    dpi = 600

    # set map extent
    xy = (297350, 5125100)
    width = height = 9500
    # set inset extent
    xy_inset = (297930.35, 5124945.93)
    winset = hinset = 15

    index_letters = ["i", "j"]

    # load auxiliary CSV
    path_aux = os.path.join(dir_fig, file)
    df_aux = pd.read_csv(path_aux, sep=",", parse_dates=["time"])
    df_aux[["dx", "dy"]] = df_aux[["x", "y"]] - xy_inset

    # calc X and Y offsets
    df_iline = df_aux[df_aux["line"] == iline]
    df_xline = df_aux[df_aux["line"] == xline]

    #%% load data

    # inline
    path_iline = glob.glob(os.path.join(dir_work, f"{iline}*_mistie_*.sgy"))[0]
    data_iline, params_iline = open_segy(path_iline)

    # xline
    path_xline = glob.glob(os.path.join(dir_work, f"{xline}*_mistie_*.sgy"))[0]
    data_xline, params_xline = open_segy(path_xline)

    #%%
    with mpl.rc_context(
        {"font.family": "Arial", "mathtext.default": "default", "mathtext.fontset": "stixsans"}
    ):

        ticklabelsize = 11
        textsize = 12
        colors = {"il": "blue", "xl": "green"}
        
        name_profile_NS = 'Profile N-S'  # iline
        name_profile_WE = 'Profile W-E'  # xline.split("_")[0],

        # ========== CREATE FIGURE ==========
        fig, ax = plt.subplot_mosaic(
            [["map", "map"], ["il", "xl"], ["ref", "cc"]],
            subplot_kw=dict(aspect="auto"),
            gridspec_kw=dict(height_ratios=[1, 0.5, 0.5]),
            figsize=(6, 12),
        )
        fig.subplots_adjust(top=0.975, bottom=0.08, left=0.15, right=0.88, wspace=0.5, hspace=0.25)

        # ========== a) MAP ==========
        ax["map"].set_aspect("equal")

        for name, line in df_aux.groupby(by="line"):
            c = (
                colors.get("il", "grey")
                if iline in name
                else colors.get("xl", "grey")
                if xline in name
                else "grey"
            )
            lw = 2 if any(i in name for i in [iline, xline]) else 0.5
            zorder = 3 if any(i in name for i in [iline, xline]) else 2
            ax["map"].plot(line["x"], line["y"], c=c, lw=lw, zorder=zorder)

        ax["map"].tick_params(
            axis="both",
            which="both",
            direction="in",
            labelsize=ticklabelsize,
            pad=3,
            top=True,
            labeltop=True,
            right=True,
            labelright=True,
        )
        for label in ax["map"].get_yticklabels():
            pos = label.get_unitless_position()
            angle = 90 if pos[0] == 0.0 else -90
            label.set_rotation(angle)
            label.set_va("center")

        # xticks
        ax["map"].xaxis.set_major_locator(mticker.MultipleLocator(2000))
        ax["map"].xaxis.set_minor_locator(mticker.MultipleLocator(1000))
        ax["map"].xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        # yticks
        ax["map"].yaxis.set_major_locator(mticker.MultipleLocator(2000))
        ax["map"].yaxis.set_minor_locator(mticker.MultipleLocator(1000))
        ax["map"].yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        # set map extent
        ax["map"].set_xlim([xy[0] - width // 2, xy[0] + width // 2])
        ax["map"].set_ylim([xy[1] - height // 2, xy[1] + height // 2])

        # add scale bar
        sb = AnchoredScaleBar(
            ax["map"].transData,
            sizex=1000,
            sizey=None,
            loc="lower right",
            pad=1,
            borderpad=0.5,
            sep=5,
            labelx="1 km",
            labely=None,
            barwidth=3,
        )
        ax["map"].add_artist(sb)

        # annotate
        kwargs_txt = dict(fontsize=textsize, fontweight="semibold", va="center", ha="center")
        ax["map"].text(
            297900, 5123000, name_profile_NS, color=colors.get("il", "k"), rotation=-77, **kwargs_txt
        )
        ax["map"].text(
            300300,
            5125000,
            name_profile_WE,
            color=colors.get("xl", "k"),
            rotation=11,
            **kwargs_txt,
        )

        bbox_ax = ax["map"].get_position()
        # print(bbox_ax)
        bbox_ax.update_from_data_x(np.asarray([0.10, bbox_ax.xmax]))
        ax["map"].set_position(bbox_ax)
        # print(ax["map"].get_position())

        # ----- b) CREATE INSET MAP -----
        # compute closet trace to profile intersection
        idx_il = df_iline["dx"].abs().nsmallest(3).index
        idx_xl = df_xline["dx"].abs().nsmallest(3).index
        indices = [sorted(idx_il), sorted(idx_xl)]
        offsets_inset = [
            [(0.5, 0.75), (-1.75, -0.75), (0.5, -1.0)],
            [(-0.5, -1.5), (0.25, -1.5), (-2.5, 1.5)],
        ]

        loc_inset = [0.65, 0.7, 0.42, 0.42]
        axinset = ax["map"].inset_axes(loc_inset)
        for i, (name, line) in enumerate(
            df_aux[df_aux["line"].isin([iline, xline])].groupby(by="line")
        ):
            c = (
                colors.get("il", "grey")
                if iline in name
                else colors.get("xl", "grey")
                if xline in name
                else "grey"
            )
            lw = 1.5 if any(i in name for i in [iline, xline]) else 0.5
            axinset.plot(
                line["x"],
                line["y"],
                c=c,
                lw=lw,
                marker="o",
                markersize=3,
                markeredgecolor=None,
                markerfacecolor=c,
            )

            index_letter_rm = "\mathrm{" + index_letters[i] + "}"  # noqa
            labels = (
                [
                    f"{index_letter_rm}-{x}"
                    for x in np.arange(1, len(indices[i][: len(indices[i]) // 2]) + 1)
                ]
                + [index_letter_rm]
                + [
                    f"{index_letter_rm}+{x}"
                    for x in np.arange(1, len(indices[i][len(indices[i]) // 2 :]))
                ]
            )
            labels = ["$\mathrm{tr}_{" + label + "}$" for label in labels]  # noqa

            for k, (idx, xy) in enumerate(line.loc[indices[i], ["x", "y"]].iterrows()):
                axinset.annotate(
                    labels[k],
                    xy=(xy["x"], xy["y"]),
                    xycoords="data",
                    xytext=(xy["x"] + offsets_inset[i][k][0], xy["y"] + offsets_inset[i][k][1]),
                    textcoords="data",
                    va="center",
                    ha="left",
                    fontsize=textsize + 2,
                )

        # set inset extent
        axinset.set_xlim([xy_inset[0] - winset // 2, xy_inset[0] + winset // 2])
        axinset.set_ylim([xy_inset[1] - hinset // 2, xy_inset[1] + hinset // 2])

        ms = 9
        mc = "D"
        mm = "s"
        kwargs_pts_center = dict(
            lw=0, marker=mc, markersize=ms, markeredgewidth=0, markeredgecolor=None
        )
        kwargs_pts_mix = dict(
            lw=0, marker=mm, markersize=ms, markeredgewidth=0, markeredgecolor=None
        )

        xy_il = df_aux.loc[idx_il, ["x", "y"]]
        xy_xl = df_aux.loc[idx_xl, ["x", "y"]]
        # closest
        axinset.plot(
            xy_il["x"].iloc[0],
            xy_il["y"].iloc[0],
            markerfacecolor=colors.get("il", "k"),
            **kwargs_pts_center,
        )
        axinset.plot(
            xy_xl["x"].iloc[0],
            xy_xl["y"].iloc[0],
            markerfacecolor=colors.get("xl", "k"),
            **kwargs_pts_center,
        )
        # mixed
        axinset.plot(
            xy_il["x"].iloc[1:],
            xy_il["y"].iloc[1:],
            markerfacecolor=colors.get("il", "k"),
            **kwargs_pts_mix,
        )
        axinset.plot(
            xy_xl["x"].iloc[1:],
            xy_xl["y"].iloc[1:],
            markerfacecolor=colors.get("xl", "k"),
            **kwargs_pts_mix,
        )

        # add scale bar
        sizex = 2
        sb_inset = AnchoredScaleBar(
            axinset.transData,
            sizex=sizex,
            sizey=None,
            loc="lower left",
            pad=1,
            borderpad=0.2,
            sep=3,
            labelx=f"{sizex} m",
            labely=None,
            barwidth=3,
        )
        axinset.add_artist(sb_inset)

        # remove ticks and ticklabels
        axinset.tick_params(
            axis="both", which="both", left=False, labelleft=False, bottom=False, labelbottom=False
        )

        # highlight inset
        patch, connectors = ax["map"].indicate_inset_zoom(axinset, edgecolor="black", alpha=0.8)
        for c, v in zip(connectors, [False, True, True, False]):
            c.set_visible(v)

        # ========== c) ILINE & d) XLINE TRACES ==========
        twt_limits = (751, 754)

        traces_il, trace_ref_il, trace_ref_env_il, twt = plot_traces(
            ax["il"],
            data_iline,
            params_iline["twt"],
            twt_limits=twt_limits,
            df=df_iline,
            idx=idx_il,
            color=colors.get("il", "k"),
            marker=(mc, mm),
            index_letter=index_letters[0],
            title=name_profile_NS,
            plot_yaxis=True,
            return_data=True,
            kw_text=dict(fontsize=textsize + 2),
        )  # 'N-S profile'

        shift = 7  # samples
        data_xline_shift = np.row_stack(
            (data_xline[shift:], np.zeros((shift, data_xline.shape[-1])))
        )
        traces_xl, trace_ref_xl, trace_ref_env_xl, _ = plot_traces(
            ax["xl"],
            data_xline_shift,
            params_xline["twt"],
            twt_limits=twt_limits,
            df=df_xline,
            idx=idx_xl,
            color=colors.get("xl", "k"),
            marker=(mc, mm),
            index_letter=index_letters[1],
            title=name_profile_WE,
            plot_yaxis=False,
            return_data=True,
            kw_text=dict(fontsize=textsize + 2),
        )  # 'W-E profile'

        # ========== e) REFERENCE TRACES ==========
        traces_ref = [trace_ref_il, trace_ref_xl]
        traces_ref_env = [trace_ref_env_il, trace_ref_env_xl]
        offsets = [0.5, 2]
        titles = [name_profile_NS, name_profile_WE]
        offsets_title = [0.35, 2.15]

        alpha = 0.3
        for tr, tr_env, color, offset, idx_s, title, offset_title in zip(
            traces_ref, traces_ref_env, colors.values(), offsets, index_letters, titles, offsets_title
        ):
            ax["ref"].axhline(
                twt[tr_env.argmax()], xmin=0.15, xmax=1.05, c=color, ls="--", lw=1, clip_on=False
            )
            ax["ref"].plot(tr + offset, twt, c=color, lw=1, alpha=alpha, clip_on=False)
            ax["ref"].fill_betweenx(
                twt,
                offset,
                tr + offset,
                where=(tr + offset >= offset),
                alpha=alpha,
                color=color,
                ec=None,
                clip_on=False,
            )
            ax["ref"].plot(tr_env + offset, twt, c=color, lw=1.5, clip_on=False)

            ax["ref"].text(
                offset,
                twt_limits[1] + 0.25,
                "$\mathrm{tr_{" + f"{idx_s},env" + "}}$",  # noqa
                va="center",
                ha="center",
                fontsize=textsize + 2,
                linespacing=0.8,
            )
            
            # title
            ax["ref"].text(
                offset_title,
                twt_limits[0] - 0.15,
                title,
                va="center",
                ha="center",
                fontsize=textsize,
                linespacing=0.8,
                color=color,
            )

        # add shift arrow
        x_arrow = -0.3
        ax["ref"].annotate(
            "",
            xy=(x_arrow, twt[trace_ref_env_il.argmax()]),
            xytext=(x_arrow, twt[trace_ref_env_xl.argmax()]),
            arrowprops=dict(arrowstyle="|-|", shrinkA=0, shrinkB=0, mutation_scale=4),
            annotation_clip=False,
        )

        ax["ref"].set_xlim([-0.5, 2.5])
        ax["ref"].set_ylim(twt_limits)
        ax["ref"].invert_yaxis()

        # ax['ref'].spines['left'].set_position(('outward', 5))
        ax["ref"].set_ylabel("TWT (ms)", fontsize=textsize)
        ax["ref"].yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax["ref"].yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

        ax["ref"].tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        for loc in ["top", "bottom", "right"]:
            ax["ref"].spines[loc].set_visible(False)

        # ========== f) CROSS-CORRELATION ==========
        cc = correlate(trace_ref_env_il, trace_ref_env_xl, mode="same", method="fft")
        lags = correlation_lags(len(trace_ref_env_il), len(trace_ref_env_xl), mode="same") * -1
        cc_shift = cross_correlation_shift(cc)

        x_offset = 2
        ax["cc"].plot(cc + x_offset, lags, c="k", lw=1)
        ax["cc"].axhline(0, xmin=0.35, xmax=0.9, c="k", ls="--", lw=1)
        ax["cc"].axhline(cc_shift, xmin=0.35, xmax=0.9, c="red", ls="--", lw=2)

        # add shift arrow
        x_arrow = 7.9
        ax["cc"].annotate(
            "",
            xy=(x_arrow, 0),
            xytext=(x_arrow, cc_shift),
            arrowprops=dict(arrowstyle="|-|", shrinkA=0, shrinkB=0, mutation_scale=4),
        )

        # add annotation shift
        ax["cc"].text(
            0.82,
            0.63,
            f"Shift = {cc_shift}",
            color="red",
            fontsize=textsize + 2,
            fontweight="semibold",
            va="center",
            ha="center",
            transform=ax["cc"].transAxes,
        )

        # add 'corr(x, y)'
        ax["cc"].text(
            0.65,
            -0.1,
            (
                "$\mathrm{xcorr(tr_{"  # noqa
                + ", tr_{".join([f"{idx_s},env" + "}" for idx_s in index_letters])[:-1]
                + "}})$"
            ),
            va="center",
            ha="center",
            fontsize=textsize + 2,
            linespacing=0.8,
            transform=ax["cc"].transAxes,
        )

        ax["cc"].set_xlim([0, 8])
        ax["cc"].set_ylim([-50, 50])

        ax["cc"].spines["left"].set_position(("outward", -25))
        ax["cc"].set_ylabel("lag (samples)", fontsize=textsize)
        ax["cc"].yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax["cc"].yaxis.set_minor_locator(mticker.MultipleLocator(5))

        ax["cc"].tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        for loc in ["top", "bottom", "right"]:
            ax["cc"].spines[loc].set_visible(False)

        # ----- SUBPLOT LABELS -----
        pos_subplot_labels_left = (-0.38, 1.05)
        pos_subplot_labels_right = (-0.075, 1.05)
        kwargs_subplot_labels = dict(
            ha="center", va="center", color="black", fontsize=20, fontweight="bold", family='Times New Roman'
        )
        ax["map"].text(
            *pos_subplot_labels_right, "a)", transform=ax["map"].transAxes, **kwargs_subplot_labels
        )
        axinset.text(*(0.125, 0.875), "b)", transform=axinset.transAxes, **kwargs_subplot_labels)
        ax["il"].text(
            *pos_subplot_labels_left, "c)", transform=ax["il"].transAxes, **kwargs_subplot_labels
        )
        ax["xl"].text(
            *pos_subplot_labels_right, "d)", transform=ax["xl"].transAxes, **kwargs_subplot_labels
        )
        ax["ref"].text(
            *pos_subplot_labels_left, "e)", transform=ax["ref"].transAxes, **kwargs_subplot_labels
        )
        ax["cc"].text(
            *pos_subplot_labels_right, "f)", transform=ax["cc"].transAxes, **kwargs_subplot_labels
        )

        #%% save figure
        figure_number = 6
        plt.savefig(
            os.path.join(dir_fig, f"Figure-{figure_number:02d}_mistie_correction_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
        # plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_mistie_correction.pdf"), bbox_inches="tight")
