"""
Create Figure illustrating custom static correction method (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-10-25

"""
import os
import sys
from functools import partial

import numpy as np
import segyio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.signal import savgol_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import (
    samples2twt,
    twt2samples,
    xprint,
    slice_valid_data,
    filter_interp_1d,
    moving_mad_filter,
    detect_seafloor_reflection,
    rms_normalization,
    AnchoredScaleBar,
    _print_info_TOPAS_moratorium,
)

_print_info_TOPAS_moratorium()


#%% LOAD DATA
if __name__ == "__main__":

    dir_fig = "."

    path = "../20200704001310_UTM60S.sgy"
    basepath, filename = os.path.split(path)
    basename, suffix = os.path.splitext(filename)

    with segyio.open(path, "r", strict=False, ignore_geometry=True) as src:
        n_traces = src.tracecount  # total number of traces
        dt = segyio.tools.dt(src) / 1000  # sample rate [ms]
        n_samples = src.samples.size  # total number of samples
        twt = src.samples  # two way travel time (TWTT) [ms]

        print(f"n_traces:  {n_traces}")
        print(f"n_samples: {n_samples}")
        print(f"dt:        {dt} ms")
        print(f"n_traces: {n_traces}")

        tracl = src.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[
            :
        ]  # Trace sequence number within line – numbers continue to increase if additional reels are required on same line.
        tracr = src.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[
            :
        ]  # Trace sequence number within reel – each reel starts at trace number one
        fldr = src.attributes(segyio.TraceField.FieldRecord)[:]  # field record number

        delrt = src.attributes(segyio.TraceField.DelayRecordingTime)[
            :
        ]  # segyio.TraceField.DelayRecordingTime

        swdep = src.attributes(segyio.TraceField.SourceWaterDepth)[:]
        scalel = src.attributes(segyio.TraceField.ElevationScalar)[:]
        if all(s > 0 for s in scalel):
            swdep = swdep * np.abs(scalel)
        elif all(s < 0 for s in scalel):
            swdep = swdep / np.abs(scalel)

        # sampling frequency [Hz]
        Fs = 1 / (dt / 1000)

        # eager version (completely read into memory)
        data_src = src.trace.raw[:].T
        print("# of samples:      ", data_src.size)
        print("# of zero samples: ", data_src.size - np.count_nonzero(data_src))

        # extract infos from binary header
        hns = src.bin[segyio.BinField.Samples]  # samples per trace
        nso = src.bin[segyio.BinField.SamplesOriginal]  # original number samples per trace

    try:
        src.close()
    except IOError:
        print("[WARNING]    SEG-Y file was already closed.")

    if n_samples < 200:
        raise ValueError(
            f"Input SEG-Y with {n_samples} samples is too short for static correction."
        )

    verbosity = 1

    if "pad" in basename:
        xprint(
            "Account for variable DelayRecordingTimes (`delrt`)", kind="info", verbosity=verbosity
        )

        data_src_sliced, idx_start_slice = slice_valid_data(data_src, nso)
        idx_amp = detect_seafloor_reflection(data_src_sliced)
        idx_amp += idx_start_slice
        twt_seafloor = twt[idx_amp]
    else:
        data_amp_smart = detect_seafloor_reflection(data_src)

    #%% PLOT: seafloor detection method

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    gain = 0
    norm = True
    clip_percentile = ((1 - gain) * 2) + 97.5  # empirically tested
    vm = np.percentile(data_src, clip_percentile)  # clipping
    _offset = 0.5 * dt
    extent = [-_offset, n_traces + _offset, twt[-1] + _offset, twt[0] - _offset]

    # plot data
    ## normalize
    if norm is True or isinstance(norm, str) and norm.lower() == "rms":
        data_show = rms_normalization(data_src, axis=0)
    elif isinstance(norm, str) and norm.lower() in ["max", "peak"]:
        data_show = data_src * np.max(np.abs(data_src))
    else:
        data_show = data_src

    ax.imshow(data_show, cmap="seismic", vmin=-vm, vmax=vm, aspect="auto", extent=extent)

    # plot seafloor detection
    if "pad" in basename:
        ax.plot(twt_seafloor, "r-", lw=3, label="detected seafloor")
    else:
        ax.plot(twt[data_amp_smart], "r-", lw=3, label="detected seafloor")

    # ax.plot(tracl, data_amp_smart, 'b-', label='detected seafloor')
    ax.set_xlabel("Traces [#]", fontsize=18)
    ax.set_ylabel("TWT [ms]", fontsize=18)
    ax.legend(fontsize=15)
    fig.tight_layout()

    #%% PLOT: seafloor static + residuals
    with mpl.rc_context({"font.family": "Arial", "mathtext.fontset": "stix"}):
        win_MAD = 21
        threshold_MAD = 3
        win_sg = 11
        limit_perc = False
        limit_samples = 10
        limit_by_MAD = False

        twt_min = 751
        twt_max = 760

        dpi = 600

        if "pad" in basename:
            data_input = idx_amp
        else:
            data_input = data_amp_smart

        # ========== FIGURE ==========
        # one-column:               3.33 inches
        # one-and-one-third-column: 4.33 inches
        # two-column:               6.66 inches
        scale = 2.5
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(3.33 * scale, 3 * scale),  # (10, 8)
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.subplots_adjust(hspace=0.1)

        # ========== (a) SEAFLOOR ==========
        ## rolling MAD filter+
        axes[0].plot(tracl, twt[data_input], c="lightgrey", ls="-", label="seafloor")
        idx_MAD_r = moving_mad_filter(
            data_input, win=win_MAD, threshold=threshold_MAD, mad_mode="double"
        )
        axes[0].plot(
            tracl[idx_MAD_r],
            twt[data_input[idx_MAD_r]],
            c="red",
            ls="",
            marker="o",
            label="outlier",
        )
        data_amp_smart_mad_r = filter_interp_1d(
            data_input, method="r_doubleMAD", kind="cubic", threshold=threshold_MAD, win=win_MAD
        )
        axes[0].plot(
            tracl,
            twt[np.around(data_amp_smart_mad_r, 0).astype("int")],
            c="grey",
            ls="-",
            label="outlier removed",
        )

        ## savgol filter
        data_amp_smart_mad_r_lowpass = savgol_filter(
            data_amp_smart_mad_r, window_length=win_sg, polyorder=1, deriv=0
        )
        axes[0].plot(
            tracl,
            twt[np.around(data_amp_smart_mad_r_lowpass, 0).astype("int")],
            c="blue",
            ls="-",
            label="smoothed",
        )

        # ---------- INSET AXIS ----------
        inset = [0.025, 0.1, 0.45, 0.45]
        inset_xlim = [164310, 164400]
        inset_ylim = [751.25, 751.85]

        shadow = mpl.patches.Rectangle(
            np.asarray(inset[:2]) * [1.3, 0.85],
            width=inset[2],
            height=inset[3],
            fc="grey",
            ec=None,
            alpha=0.5,
            zorder=3,
            transform=axes[0].transAxes,
        )
        axes[0].add_patch(shadow)

        axinset = axes[0].inset_axes(inset)
        axinset.plot(tracl, twt[data_input], c="lightgrey", ls="-", label="seafloor")
        axinset.plot(
            tracl[idx_MAD_r],
            twt[data_input[idx_MAD_r]],
            c="red",
            ls="",
            marker="o",
            label="outlier",
        )
        axinset.plot(
            tracl,
            twt[np.around(data_amp_smart_mad_r, 0).astype("int")],
            c="grey",
            ls="-",
            label="outlier removed",
        )
        axinset.plot(
            tracl,
            twt[np.around(data_amp_smart_mad_r_lowpass, 0).astype("int")],
            c="blue",
            ls="-",
            label="smoothed",
        )
        axinset.set_xlim(inset_xlim)
        axinset.set_ylim(inset_ylim)
        axinset.invert_yaxis()
        axinset.tick_params(
            which="both",
            top=False,
            labeltop=False,
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=False,
            labelright=False,
        )

        axinset.text(
            0.2,
            0.87,
            "Seafloor (no outliers)",
            ha="left",
            va="center",
            color="grey",
            fontsize=15,
            fontweight="semibold",
            transform=axinset.transAxes,
        )
        axinset.annotate(
            "Baseline\n(smoothed)",
            xy=(0.55, 0.65),
            xycoords="axes fraction",
            xytext=(0.35, 0.3),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=3,
                edgecolor="blue",
                facecolor="blue",
                lw=1.5,
            ),
            ha="left",
            va="center",
            color="blue",
            fontsize=15,
            fontweight="semibold",
        )
        axinset.annotate(
            "Outlier",
            xy=(164391, 751.675),
            xycoords="data",
            xytext=(0.9, 0.1),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=5,
                edgecolor="red",
                facecolor="red",
                lw=1.5,
            ),
            ha="right",
            va="center",
            color="red",
            fontsize=15,
            fontweight="semibold",
        )

        # add scalebars
        ax_ = axes[0].inset_axes(inset)
        ax_.set_xlim(inset_xlim)
        ax_.set_ylim(inset_ylim)
        ax_.remove()
        sb = AnchoredScaleBar(
            ax_.transData,
            sizex=20,
            sizey=0.1,
            loc="lower left",
            labelx="100 m",
            labely="0.1 ms",
            barwidth=2,
        )
        axinset.add_artist(sb)

        # highlight inset
        patch, connectors = axes[0].indicate_inset_zoom(axinset, edgecolor="black")
        for c, v in zip(connectors, [False, True, False, True]):
            c.set_visible(v)

        kw_subplot_labels = dict(va="center", ha="center", fontsize=20, fontweight="semibold", fontfamily='Times New Roman')
        axes[0].text(-0.1, 1.0, "a)", transform=axes[0].transAxes, **kw_subplot_labels)
        axinset.text(0.05, 0.90, "b)", transform=axinset.transAxes, **kw_subplot_labels)

        # ========== (b) RESIDUALS ==========
        color_residuals = "green"
        
        # calc residuals
        residual_static_amp_smart_samples = data_amp_smart_mad_r_lowpass - data_amp_smart_mad_r
        static = residual_static_amp_smart_samples.copy()

        # if set: clip static values using given percentile
        if limit_perc is not None and limit_perc is not False:
            clip = np.percentile(np.abs(static), limit_perc)
            static = np.where(np.abs(static) > clip, clip * np.sign(static), static)

        # if set: clip static values using user-specified number of samples
        if isinstance(limit_samples, (float, int)):
            static = np.where(
                np.abs(static) > limit_samples, limit_samples * np.sign(static), static
            )

        # if set: clip static values using median absolute deviation (multiplied by factor)
        if limit_by_MAD is not False:
            limit_by_MAD = limit_by_MAD if isinstance(limit_by_MAD, (int, float)) else 3
            threshold = int(np.ceil(np.median(np.abs(static)) * limit_by_MAD))
            static = np.where(np.abs(static) > threshold, threshold * np.sign(static), static)

        axes[1].plot(tracl, static * dt, c=color_residuals, ls="-", lw=1, label="residual statics")

        # labels
        fontsize_labels = 16
        axes[0].invert_yaxis()
        axes[0].set_ylabel("TWT (ms)", fontsize=fontsize_labels)

        axes[0].text(
            164665,
            751.5,
            "Pockmark",
            ha="center",
            va="center",
            color="black",
            fontsize=18,
            fontweight="semibold",
            transform=axes[0].transData,
        )

        axes[1].set_xlabel("Trace (#)", fontsize=fontsize_labels)
        axes[1].set_ylabel("$\Delta$t (ms)", fontsize=fontsize_labels)  # noqa

        axes[1].text(
            0.2,
            0.8,
            "Residuals",
            ha="center",
            va="center",
            color=color_residuals,
            fontsize=18,
            fontweight="semibold",
            transform=axes[1].transAxes,
        )
        axes[1].annotate(
            "",
            xy=(164720, 0.25),
            xycoords="data",
            xytext=(0.75, 0.9),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
                shrinkA=6,
                shrinkB=0,
                edgecolor="k",
                facecolor="k",
                lw=1,
            ),
        )
        axes[1].annotate(
            "",
            xy=(164740, -0.25),
            xycoords="data",
            xytext=(0.75, 0.9),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=-0.1",
                shrinkA=8,
                shrinkB=0,
                edgecolor="k",
                facecolor="k",
                lw=1,
            ),
        )
        axes[1].annotate(
            "Clipped",
            xy=(164720, 0.25),
            xycoords="data",
            xytext=(0.75, 0.9),
            textcoords="axes fraction",
            ha="left",
            va="center",
            color="k",
            fontsize=18,
            fontweight="normal",
        )

        # secondary axis (upper subplot)
        secax_upper = axes[0].secondary_yaxis("right", functions=(lambda x: x, lambda y: y))

        # secondary axis (lower subplot)
        secax_lower = axes[1].secondary_yaxis(
            "right", functions=(partial(twt2samples, dt=dt), partial(samples2twt, dt=dt))
        )
        secax_lower.set_ylabel(
            "$\Delta$ Samples", va="center", fontsize=fontsize_labels  # noqa
        )  # , rotation=270

        # spines
        spine_offset = 5
        axes[0].spines["bottom"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        # axes[0].spines['left'].set_position(('outward', spine_offset))
        # axes[0].spines['right'].set_position(('outward', spine_offset))
        # secax_upper.spines['right'].set_position(('outward', spine_offset))
        axes[0].tick_params(bottom=False)

        axes[1].spines["top"].set_visible(False)
        # axes[1].spines['bottom'].set_position(('outward', spine_offset))
        # axes[1].spines['left'].set_position(('outward', spine_offset))
        # axes[1].spines['right'].set_position(('outward', spine_offset))
        # secax_lower.spines['right'].set_position(('outward', spine_offset))

        # # legend
        # handles, labels = axes[0].get_legend_handles_labels()
        # order = [0,2,3,1]
        # legend = axes[0].legend([handles[idx] for idx in order],
        #                         [labels[idx] for idx in order],
        #                         fontsize=fontsize_labels,
        #                         markerscale=2,
        #                         )
        # for line in legend.get_lines():
        #     line.set_linewidth(3)

        # add TWT axis ticks
        for ax in [axes[0], secax_upper]:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
            ax.tick_params(which="major", length=6)
            ax.tick_params(which="minor", length=3)
            # yticklabels = ax.get_yticklabels()
            # print(yticklabels)
            # yticklabels = [txt if i % 2 == 1 else '' for i,txt in enumerate(yticklabels)]
            # print(yticklabels)
            # ax.set_yticklabels(yticklabels)

        fontsize_ticks = fontsize_labels - 2
        for ax in axes:
            ax.set_xlim(tracl[0], tracl[-1])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(fontsize_ticks)

        for label in secax_upper.get_xticklabels() + secax_upper.get_yticklabels():
            label.set_fontsize(fontsize_ticks)

        for label in secax_lower.get_xticklabels() + secax_lower.get_yticklabels():
            label.set_fontsize(fontsize_ticks)

        for ax in axes:
            ax.set_xlim([164200, 165000])
        axes[0].tick_params(which="minor", bottom=False)
        axes[0].set_ylim(twt_min, twt_max)
        axes[0].invert_yaxis()

        axes[1].xaxis.set_major_locator(mticker.MultipleLocator(100))
        axes[1].xaxis.set_minor_locator(mticker.MultipleLocator(10))
        axes[1].tick_params(which="major", length=6)
        axes[1].tick_params(which="minor", length=3)

        # ----- highlight patch -----
        alpha = 0.3
        rect_xmin = 164560
        rect_width = 200
        color = "pink"
        axes[0].text(
            0.71,
            0.5,
            "\n".join(("Pockmark", "interval")),
            ha="left",
            va="center",
            color=color,
            fontsize=20,
            fontweight="semibold",
            transform=axes[0].transAxes,
        )
        rect_0 = mpl.patches.Rectangle(
            (rect_xmin, 751),
            rect_width,
            10,
            fc=color,
            ec=None,
            alpha=alpha,
            zorder=1,
            clip_on=False,
            transform=axes[0].transData,
        )
        axes[0].add_artist(rect_0)
        y_lim = axes[1].get_ylim()
        rect_1 = mpl.patches.Rectangle(
            (rect_xmin, y_lim[0]),
            rect_width,
            y_lim[1] * 2,
            fc=color,
            ec=None,
            alpha=alpha,
            zorder=1,
            clip_on=True,
            transform=axes[1].transData,
        )
        axes[1].add_artist(rect_1)

        axes[1].text(-0.1, 0.99, "c)", transform=axes[1].transAxes, **kw_subplot_labels)

        #%% savefig
        figure_number = 4
        plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_static_correction_{dpi}dpi.png"), dpi=dpi, bbox_inches="tight")
        # plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_static_correction.pdf"), bbox_inches="tight")
