"""
Create Figure illustrating trace resampling and envelope (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-10-20

"""
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from scipy.signal import resample, hilbert

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#%% FUNCTIONS

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


def freq_spectrum(signal, Fs, n=None, taper=True, return_minmax=False):
    """
    Compute frequency spectrum of input signal given a sampling rate (Fs).

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.
    Fs : int
        Sampling rate/frequency (Hz).
    n : int, optional
        Length of FFT, i.e. number of points (default: len(signal)).
    taper : TYPE, optional
        Window function applied to time signal to improve frequency domain properties.
        The default is True.

    Returns
    -------
    f : np.ndarray
        Array of signal frequencies.
    a_norm : np.ndarray
        Magnitude of amplitudes per frequency.
    f_min : float
        Minimum frequency with actual signal content.
    f_max : float
        Maximum frequency with actual signal content.

    """
    # signal length (samples)
    N = len(signal)
    # select window function
    if taper:
        win = np.blackman(N)
    else:
        win = np.ones((N))
    # apply tapering
    s = signal * win

    # number of points to use for FFT
    if n is None:
        n = N

    # calc real part of FFT
    a = np.abs(np.fft.rfft(s, n))
    # calc frequency array
    f = np.fft.rfftfreq(n, 1 / Fs)

    # scale magnitude of FFT by used window and factor of 2 (only half-spectrum)
    a_norm = a * 2 / np.sum(win)

    if return_minmax:
        # get frequency limits using calculated amplitude threshold
        slope = np.abs(np.diff(a_norm) / np.diff(f))  # calculate slope
        threshold = (slope.max() - slope.min()) * 0.001  # threshold amplitude
        f_limits = np.where(a_norm > threshold)[0]  # get frequency limits
        f_min, f_max = f[f_limits[0]], f[f_limits[-1]]  # select min/max frequencies
        f_min, f_max = np.min(f_limits), np.max(f_limits)  # select min/max frequencies
        return f, a_norm, f_min, f_max
    else:
        return f, a_norm


def find_nearest(array, value):  # noqa
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def moving_average(a, n=3):  # noqa
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


if __name__ == "__main__":
    #% LOAD DATA
    
    dir_data = os.path.dirname(os.path.abspath(__file__))
    file_data = "20200702231201_UTM60S_trace-50_raw.csv"
    
    dpi = 600
    
    data = np.loadtxt(os.path.join(dir_data, file_data), dtype="float", delimiter=",")
    twt = data[:, 0]
    trace = data[:, 1]
    dt = round((twt[1] - twt[0]) * 10000) / 10000
    
    #%% RESAMPLING
    
    # (OPTIONAL) upsampling trace for smoother infill of positive amplitudes
    factor_upsampling = 2
    window = None
    trace_upsampled = resample(trace, num=trace.size * factor_upsampling, window=window)
    twt_upsampled = resample(twt, num=twt.size * factor_upsampling, window=window)
    
    # downsample trace
    factor_downsampling = 2
    
    ## scipy.signal.resample
    tr_resample, twt_resample = resample(trace, num=trace.size // 2, t=twt, window=window)
    tr_resample_up, twt_resample_up = resample(
        tr_resample, num=tr_resample.size * 4, t=twt_resample, window=window
    )
    
    #%% EVELOPE
    trace_env = envelope(trace)
    trace_resampled_env = envelope(tr_resample)
    
    #%% FREQUENCIES
    nsmooth = 31
    trim_start = nsmooth // 2 - 1 if nsmooth % 2 == 0 else nsmooth // 2
    trim_end = nsmooth // 2
    
    # frequency spectrum: trace
    f_trace, amp_trace = freq_spectrum(trace, Fs=1 / (dt / 1000))
    amp_trace /= amp_trace.max()
    amp_trace_avg = moving_average(amp_trace, n=nsmooth)
    
    # frequency spectrum: resampled
    f_trace_r, amp_trace_r = freq_spectrum(tr_resample, Fs=1 / (dt * 2 / 1000))
    amp_trace_r /= amp_trace_r.max()
    amp_trace_r_avg = moving_average(amp_trace_r, n=nsmooth)
    
    # frequency spectrum: envelope
    f_trace_env, amp_trace_env = freq_spectrum(trace_resampled_env, Fs=1 / (dt * 2 / 1000))
    amp_trace_env /= amp_trace_env.max()
    # amp_trace_env_avg = moving_average(amp_trace_env, n=nsmooth)
    
    #%% PLOTTING
    with mpl.rc_context({"font.family": "Arial"}):
    
        # set time window
        t_min = 749 - dt / 2
        t_max = 751 + dt / 2
    
        # set defaults
        kwargs_labels = dict(fontsize=16, fontweight="normal")
        kwargs_subplot_labels = dict(fontsize=16, fontweight="semibold", pad=15)
        kwargs_marker_trace = dict(
            marker="o", markerfacecolor="grey", markeredgecolor="grey", markersize=5
        )
        kwargs_marker_resmpl = dict(
            marker="o", markerfacecolor="navy", markeredgecolor="navy", markersize=5
        )
    
        # ==================== CREATE FIGURE ====================
        fig, axes = plt.subplots(
            2,
            3,
            sharey=False,
            sharex=False,
            figsize=(7, 8),  # (8,10)
            gridspec_kw={"height_ratios": [0.8, 0.2]},
        )
        fig.subplots_adjust(wspace=0.2, hspace=0.05)
    
        for ax in axes.flat:
            ax.patch.set_alpha(0)
    
        # ========== (A) plot original trace (with samples) ==========
        # axes[0,0].text(0.5, -0.075, '(a) input trace',
        #              transform=axes[0,0].transAxes,
        #              va='center', ha='center',
        #              **kwargs_labels)
        axes[0, 0].set_title("Input trace", **kwargs_subplot_labels)
        axes[0, 0].plot(trace, twt, c="k", label="original trace", **kwargs_marker_trace)
        axes[0, 0].fill_betweenx(
            twt_upsampled, 0, trace_upsampled, where=(trace_upsampled >= 0), color="k"
        )
    
        # annotate trace
        idx_twt = find_nearest(twt, 749.75)
        xytrace = (float(trace[twt == idx_twt] - 0.1), idx_twt)
        axes[0, 0].annotate(
            "Trace",
            xy=xytrace,
            xycoords="data",
            xytext=(0.85, 0.70),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=5,
                edgecolor="k",
                facecolor="k",
                lw=1.5,
            ),
            horizontalalignment="center",
            verticalalignment="center",
            color="k",
            **kwargs_labels,
        )
        # annotate samples
        idx_twt = find_nearest(twt, 750.125)
        xytrace = (float(trace[twt == idx_twt]), idx_twt)
        axes[0, 0].annotate(
            "Samples",
            xy=xytrace,
            xycoords="data",
            xytext=(0.60, 0.38),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=3,
                edgecolor="grey",
                facecolor="grey",
                lw=1.5,
            ),
            horizontalalignment="left",
            verticalalignment="center",
            color="grey",
            **kwargs_labels,
        )
    
        # ========== (B) plot downsampled trace (with samples) ==========
        # axes[0,1].text(0.5, -0.075, '(b) resampled',
        #              transform=axes[0,1].transAxes,
        #              va='center', ha='center',
        #              **kwargs_labels)
        axes[0, 1].set_title("Resampled", **kwargs_subplot_labels)
        axes[0, 1].plot(trace, twt, c="grey", alpha=0.5, label="original trace", **kwargs_marker_trace)
        axes[0, 1].plot(
            tr_resample, twt_resample, c="navy", label="downsampled trace", **kwargs_marker_resmpl
        )
    
        # annotate reduced samples
        color = "navy"
        idx_twt = find_nearest(twt, 750.15)
        xytrace = (float(trace[twt == idx_twt]), idx_twt)
        axes[0, 1].annotate(
            "Reduced\nsamples",
            xy=xytrace,
            xycoords="data",
            xytext=(0.7, 0.35),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=5,
                edgecolor=color,
                facecolor=color,
                lw=1.5,
            ),
            horizontalalignment="left",
            verticalalignment="center",
            color=color,
            **kwargs_labels,
        )
        # create arrow (2nd subplot) --> check subplot (C)!
        idx_twt = find_nearest(twt, 749.7)
        xy = np.array([float(trace[twt == idx_twt]), idx_twt])
        axes[0, 1].add_patch(
            mpatches.FancyArrowPatch(
                xy + np.array([0.35, -0.15]),
                xy + np.array([0.1, 0.0]),
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=0,
                edgecolor=color,
                facecolor=color,
                lw=1.5,
                mutation_scale=20,
            )
        )
    
        # ---------- create inset figure ----------
        loc_inset = [-0.25, 0.75, 0.55, 0.2]
        axinset = axes[0, 1].inset_axes(loc_inset)
        axinset.tick_params(
            top=False,
            labeltop=False,
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=False,
            labelright=False,
            which="both",
        )
        # plot traces
        axinset.plot(trace, twt, c="grey", alpha=0.5, label="original trace", **kwargs_marker_trace)
        axinset.plot(
            tr_resample, twt_resample, c="navy", label="downsampled trace", **kwargs_marker_resmpl
        )
        axinset.set_xlim(-0.05, 0.05)
        axinset.set_ylim(749.01, 749.12)
        axinset.invert_yaxis()
        # highlight inset
        patch, connectors = axes[0, 1].indicate_inset_zoom(axinset, edgecolor="black")
        for c, v in zip(connectors, [False, True, True, False]):
            c.set_visible(v)
    
        # annotate "dt_original"
        xy = np.array([-0.02, 749.025])
        axinset.add_patch(
            mpatches.FancyArrowPatch(
                xy,
                xy + np.array([0, dt]),
                arrowstyle="|-|",
                # connectionstyle='arc3',
                shrinkA=0,
                shrinkB=0,
                edgecolor="grey",
                facecolor="grey",
                mutation_scale=3,
                lw=1.5,
            )
        )
        xytext = xy + np.array([-0.005, dt / 2])
        axinset.text(*xytext, "$\Delta$t", va="center", ha="right", color="grey", **kwargs_labels)  # noqa
    
        # annotate "dt_resampled"
        xy = np.array([0.012, 749.05])
        arrow_dt = mpatches.FancyArrowPatch(
            xy,
            xy + np.array([0, dt * 2]),
            arrowstyle="|-|",
            # connectionstyle='arc3',
            shrinkA=0,
            shrinkB=0,
            edgecolor="navy",
            facecolor="navy",
            mutation_scale=3,
            lw=1.5,
        )
        axinset.add_patch(arrow_dt)
        xytext = xy + np.array([0.005, dt])
        axinset.text(
            *xytext, "$\Delta$t$_{\mathsf{r}}$", va="center", ha="left", color="navy", **kwargs_labels  # noqa
        )
    
        # add inset shadow
        shadow = mpl.patches.Rectangle(
            np.asarray(loc_inset[:2]) * [0.9, 0.99],
            width=loc_inset[2],
            height=loc_inset[3],
            fc="grey",
            ec=None,
            alpha=0.3,
            zorder=5,
            clip_on=False,
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].add_patch(shadow)
    
        # ========== (C) plot trace envelope (with samples) ==========
        axes[0, 2].set_title("Envelope", **kwargs_subplot_labels)
        axes[0, 2].plot(trace, twt, c="grey", alpha=0.5, label="original trace")
        axes[0, 2].plot(tr_resample, twt_resample, c="navy", label="downsampled trace")
        axes[0, 2].plot(trace_env, twt, c="grey", lw=3, alpha=0.5, label="env(original trace)")
        axes[0, 2].plot(
            trace_resampled_env, twt_resample, c="navy", lw=3, label="env(downsampled trace)"
        )
        axes[0, 2].axvline(x=0, c="k", lw=1, alpha=0.5, zorder=0)
    
        # annotate downsampled trace
        idx_twt = find_nearest(twt, 749.58)
        xytrace = (float(trace[twt == idx_twt]), idx_twt)
        axes[0, 2].annotate(
            "Downsampled\ntrace",
            xy=xytrace,
            xycoords="data",
            xytext=(-0.1, 0.75),
            textcoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="center",
            color=color,
            **kwargs_labels,
        )
        # create arrow (3rd subplot)
        idx_twt = find_nearest(twt, 749.58)
        xy = np.array([float(trace[twt == idx_twt]), idx_twt])
        axes[0, 2].add_patch(
            mpatches.FancyArrowPatch(
                xy + np.array([-0.35, -0.05]),
                xy,
                arrowstyle="-|>",
                connectionstyle="arc3",
                shrinkA=0,
                shrinkB=5,
                edgecolor=color,
                facecolor=color,
                lw=1.5,
                mutation_scale=20,
            )
        )
        # add envelope label
        axes[0, 2].text(
            0.160,
            749.45,
            "Envelope",
            rotation=-50,
            ha="left",
            va="top",
            color="navy",
            fontsize=kwargs_labels["fontsize"],
            fontweight="bold",
        )
    
        # add TWT axis ticks
        axes[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axes[0, 0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
        # adjust plot appearance
        for i, ax in enumerate(axes[0, :]):
            ax.set_ylim(t_min, t_max)
            ax.invert_yaxis()
    
            for spine in ["left", "right", "bottom"]:
                ax.spines[spine].set_visible(False)
            ax.spines["top"].set_position(("outward", 2))
    
            ax.set_xlabel("Amplitude", **kwargs_labels)
            ax.xaxis.set_label_position("top")
    
            ax.tick_params(
                top=True,
                labeltop=True,
                bottom=False,
                labelbottom=False,
                direction="in",
                which="both",
                labelsize=12,
            )
            ax.tick_params(axis="x", pad=0)
            if i > 0:
                ax.tick_params(left=False, labelleft=False)
    
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
        axes[0, 0].spines["left"].set_visible(True)
        axes[0, 0].spines["left"].set_position(("outward", 2))
    
        ylabel = axes[0, 0].set_ylabel(
            "TWT (ms)", backgroundcolor="white", ha="right", **kwargs_labels
        )
        axes[0, 0].yaxis.set_label_coords(-0.117, 1.01)
    
        # ========== FREQUENCY SPECTRA ==========
        freq_fs_labels = 12
        kwargs_text = dict(
            ha="left",
            va="center",
            fontsize=11,
            fontweight="semibold",
        )
    
        axes[1, 0].plot(f_trace / 1000, amp_trace, c="lightgrey", label="trace")
        axes[1, 0].plot(
            f_trace[trim_start:-trim_end] / 1000,
            amp_trace_avg,
            c="black",
            label=f"{nsmooth}-sample\naverage",
        )
        axes[1, 0].set_ylabel("Amplitude\n(normalized)", fontsize=freq_fs_labels)
        axes[1, 0].text(
            0.22, 0.85, "Trace", c="lightgrey", transform=axes[1, 0].transAxes, **kwargs_text
        )
        axes[1, 0].text(
            0.57,
            0.55,
            f"Running\naverage\n({nsmooth}-samples)",
            c="black",
            transform=axes[1, 0].transAxes,
            **kwargs_text,
        )
    
        axes[1, 1].plot(f_trace_r / 1000, amp_trace_r, c="navy", alpha=0.3, label="resampled")
        axes[1, 1].plot(
            f_trace_r[trim_start:-trim_end] / 1000,
            amp_trace_r_avg,
            c="navy",
            label=f"{nsmooth}-sample\naverage",
        )
        axes[1, 1].text(
            0.22, 0.85, "Resampled", c="navy", alpha=0.3, transform=axes[1, 1].transAxes, **kwargs_text
        )
        axes[1, 1].text(
            0.57,
            0.55,
            f"Running\naverage\n({nsmooth}-samples)",
            c="navy",
            transform=axes[1, 1].transAxes,
            **kwargs_text,
        )
    
        axes[1, 2].plot(f_trace_r / 1000, amp_trace_env, c="navy", lw=2)
        # axes[1,2].plot(f_trace_r[trim_start:-trim_end]/1000, amp_trace_env_avg, c='navy')
        axes[1, 2].text(
            0.3,
            0.2,
            "Envelope",
            c="navy",
            transform=axes[1, 2].transAxes,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    
        for i, ax in enumerate(axes[1, :]):
            ax.set_ylim(-0.02, 1)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
            if i > 0:
                ax.set_yticklabels([])
    
            ax.set_xlabel("Frequency (kHz)", fontsize=freq_fs_labels)
    
            if i < 2:
                ax.set_xlim(1.55, 6.45)
            else:
                ax.set_xlim(-0.1, 3.1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.tick_params(which="both", direction="in", labelsize=12)
    
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_position(("outward", 2))
    
        for i, (ax, label) in enumerate(zip(axes.ravel(), ["a)", "b)", "c)", "", "", ""])):
            if i < 3:
                ax.text(
                    -0.05,
                    1.14,
                    label,
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    fontsize=20,
                    fontweight="semibold",
                    fontfamily='Times New Roman',
                    color="black",
                )
    
        #%% save figure
        figure_number = 8
        plt.savefig(
            os.path.join(dir_data, f"Figure-{figure_number:02d}_resampling_envelope_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
        # plt.savefig(os.path.join(dir_data, f"Figure-{figure_number:02d}_resampling_envelope.pdf"), bbox_inches="tight")
