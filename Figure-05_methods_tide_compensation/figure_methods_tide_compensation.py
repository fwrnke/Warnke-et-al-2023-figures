"""
Create Figure illustrating the compensation of varying tidal elevation (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-10-25

"""
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch

from tpxo_tide_prediction import tide_predict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import depth2twt, twt2depth, AnchoredScaleBar

# set units
depth2twt_ = partial(depth2twt, units='ms')

#%% MAIN

if __name__ == "__main__":

    dir_fig = "."
    dir_work = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        dir_tpxo = sys.argv[1]
    else:
        # sys.exit('\n[ERROR]    Path of "TPXO9_atlas_v5" files is required (needs to be downloaded separately)!')
        dir_tpxo = r'D:\scripts\packages\tpxo_tide_prediction\data\v5'  # r"D:\tides\TPXO9_atlas_v5"
    
    path_tid = os.path.join(dir_work, "../TOPAS_metadata.aux")
    path_tid = path_tid if os.path.isfile(path_tid) else os.path.join(dir_work, "./TOPAS_metadata.aux")
    path_tid = path_tid if os.path.isfile(path_tid) else None
        
    df_tide_all = pd.read_csv(path_tid, sep=",", parse_dates=["time"])
    tmin = np.datetime64(df_tide_all["time"].min(), "m")
    tmax = np.datetime64(df_tide_all["time"].max(), "m")

    profile = "20200704001310"
    df_tide = df_tide_all[df_tide_all["line"] == profile]

    # sort
    df_tide = df_tide.sort_values(by="time")

    tmin_p = np.datetime64(df_tide["time"].min(), "m")
    tmax_p = np.datetime64(df_tide["time"].max(), "m")
    
    dpi = 600
    
    #%% PLOT: single profile

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 4))

    ax[0].plot(df_tide["fldr"].values, df_tide["tide_m"].values, "k-", label="tidal elevation (m)")
    ax[1].plot(
        df_tide["fldr"].values, df_tide["tide_ms"].values, "b-", label="tidal elevation (ms)"
    )
    ax[2].plot(
        df_tide["fldr"].values,
        df_tide["tide_samples"].values,
        "g-",
        label="tidal elevation (samples)",
    )

    ax[0].set_ylabel("tide (m)")
    ax[1].set_ylabel("tide (ms)")
    ax[2].set_ylabel("tide (samples)")
    ax[2].set_xlabel("field recording number (#)")

    for axis in ax:
        axis.legend()

    fig.tight_layout()
    plt.subplots_adjust(hspace=0)

    #%% TIDAL ELEVATION DURING SURVEY

    lat = np.array([-44])
    lon = np.array([174.47])
    times = np.arange(tmin, tmax, np.timedelta64(1, "m"))

    tides = tide_predict(dir_tpxo, lat, lon, times)

    #%% PLOT FIGURE

    with mpl.rc_context({"font.family": "Arial"}):
        font_labels = dict(fontsize=16, fontweight="normal")
        labelsize = 11
        spine_offset = 5
        dt = 0.025

        # ----- FIGURE -----
        # one-column:               3.33 inches
        # one-and-one-third-column: 4.33 inches
        # two-column:               6.66 inches
        scale = 2.2
        fig, axes = plt.subplot_mosaic(
            [["tide_top"], ["empty"], ["tide"], ["profile"]],
            figsize=(3.33 * scale, 3.8 * scale),
            gridspec_kw={"height_ratios": [3, 0.05, 2, 2]},
        )
        fig.subplots_adjust(hspace=0.2)
        axes["empty"].remove()

        # ----- OVERALL TIDAL ELEVATION -----
        axes["tide_top"].plot(times, tides, c="black")
        axes["tide_top"].set_xlim(times[0], times[-1])
        axes["tide_top"].tick_params(axis="x", direction="in", rotation=15, labelsize=labelsize)
        axes["tide_top"].tick_params(
            axis="y",
            which="both",
            labelsize=labelsize,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )
        for tick in axes["tide_top"].xaxis.get_majorticklabels():
            tick.set_horizontalalignment("center")
        axes["tide_top"].set_ylabel("Tidal elevation (m)", **font_labels)
        axes["tide_top"].yaxis.set_label_position("right")
        axes["tide_top"].set_ylim(-1, 1)
        axes["tide_top"].yaxis.set_major_locator(mticker.MultipleLocator(1))
        axes["tide_top"].yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

        # secondary y-axis (time)
        ax_tide_top_ms = axes["tide_top"].secondary_yaxis("left", functions=(depth2twt_, twt2depth))
        ax_tide_top_ms.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_tide_top_ms.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
        ax_tide_top_ms.set_ylabel("Tidal elevation $\Delta$t (ms)", **font_labels)  # noqa
        ax_tide_top_ms.tick_params(axis="both", labelsize=labelsize)

        # plot tide subset (along profile)
        times_mask = (times >= tmin_p) & (times <= tmax_p)
        axes["tide_top"].plot(times[times_mask], tides[times_mask], c="blue", lw=4)

        # create LEFT side of Connection patch
        con = ConnectionPatch(
            xyA=(0.388, 0.52),
            coordsA=axes["tide_top"].transAxes,
            xyB=(0, 1),
            coordsB=axes["tide"].transAxes,
            color="blue",
        )
        fig.add_artist(con)
        # create RIGHT side of Connection patch
        con = ConnectionPatch(
            xyA=(0.393, 0.69),
            coordsA=axes["tide_top"].transAxes,
            xyB=(1, 1),
            coordsB=axes["tide"].transAxes,
            color="blue",
        )
        fig.add_artist(con)

        axes["tide_top"].text(
            0.98,
            0.92,
            "Tidal elevation during survey",
            ha="right",
            va="center",
            color="black",
            # rotation=13, rotation_mode='anchor',
            transform=axes["tide_top"].transAxes,
            **font_labels
        )

        # ----- TIDE ALONG PROFILE -----
        axes["tide"].plot(
            df_tide["fldr"].values, df_tide["tide_m"].values, "b-", label="tidal elevation (m)"
        )
        axes["tide"].set_ylabel("Tidal elevation (m)", **font_labels)
        axes["tide"].yaxis.set_label_position("right")
        axes["tide"].tick_params(axis="x", direction="in", labelbottom=False)
        axes["tide"].tick_params(
            axis="y",
            which="both",
            labelsize=labelsize,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )
        axes["tide"].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        axes["tide"].yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        axes["tide"].set_xlim(df_tide["fldr"].values.min(), df_tide["fldr"].values.max())

        # secondary axis (time)
        ax_tide_ms = axes["tide"].secondary_yaxis("left", functions=(depth2twt_, twt2depth))
        ax_tide_ms.set_ylabel("Tidal elevation\n $\Delta$t (ms)", **font_labels)  # noqa
        ax_tide_ms.tick_params(axis="both", labelsize=labelsize)
        ax_tide_ms.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_tide_ms.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        axes["tide"].text(
            0.5,
            0.6,
            "Tidal elevation along profile",
            ha="center",
            va="center",
            color="blue",
            rotation=14.5,
            rotation_mode="anchor",
            transform=axes["tide"].transAxes,
            **font_labels
        )

        # ----- SEAFLOOR ALONG PROFILE -----
        axes["profile"].plot(
            df_tide["time"], df_tide["seafloor_static"], c="grey", label="seafloor_static"
        )
        axes["profile"].plot(
            df_tide["time"],
            df_tide["seafloor_static_tide"],
            c="blue",
            label="seafloor_static_tide",
        )

        axes["profile"].set_xlim(df_tide["time"].min(), df_tide["time"].max())
        loc_dt = mdates.MinuteLocator(byminute=np.arange(0, 65, 5))
        axes["profile"].xaxis.set_major_locator(loc_dt)
        fmt_datetime = mdates.DateFormatter("%H:%M")  # %Y-%m-%d %H:%M:%S
        axes["profile"].xaxis.set_major_formatter(fmt_datetime)
        axes["profile"].tick_params(axis="x", direction="in", labelsize=labelsize)
        axes["profile"].tick_params(axis="y", labelsize=labelsize)

        axes["profile"].set_xlabel(
            df_tide["time"].dt.date.min().strftime("%Y-%m-%d"), **font_labels
        )

        axes["profile"].yaxis.set_major_locator(mticker.MultipleLocator(2))
        axes["profile"].yaxis.set_minor_locator(mticker.MultipleLocator(1))
        axes["profile"].set_ylabel("TWT (ms)", **font_labels)

        axes["profile"].invert_yaxis()

        # secondary axis (depth)
        ax_profile_m = axes["profile"].secondary_yaxis(
            "right",
            functions=(partial(twt2depth, units='ms'), depth2twt)
        )
        ax_profile_m.set_ylabel("Depth (m)", **font_labels)
        ax_profile_m.tick_params(axis="both", labelsize=labelsize)
        ax_profile_m.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax_profile_m.yaxis.set_minor_locator(mticker.MultipleLocator(1))

        axes["profile"].text(
            0.2,
            0.15,
            "Tide corrected",
            ha="center",
            color="blue",
            transform=axes["profile"].transAxes,
            **font_labels
        )

        axes["profile"].text(
            0.67,
            0.62,
            "Seafloor",
            ha="left",
            color="grey",
            transform=axes["profile"].transAxes,
            **font_labels
        )

        # ----- SCALE BARS -----
        sb = AnchoredScaleBar(
            axes["tide"].transData,
            sizex=200,
            sizey=None,
            loc="lower right",
            pad=1,
            borderpad=0.5,
            sep=5,
            labelx="1 km",
            labely=None,
            barwidth=3,
        )
        axes["tide"].add_artist(sb)

        sb = AnchoredScaleBar(
            axes["tide"].transData,
            sizex=200,
            sizey=None,
            loc="lower right",
            pad=1,
            borderpad=0.5,
            sep=5,
            labelx="1 km",
            labely=None,
            barwidth=3,
        )
        axes["profile"].add_artist(sb)

        # ----- SUBPLOT LABELS -----
        # pos_subplot_labels = (-0.1, 1.0)
        kwargs_subplot_labels = dict(
            ha="center", va="center", color="black", fontsize=20, fontweight="bold", family='Times New Roman'
        )
        axes["tide_top"].text(
            *(-0.1, 1.075), "a)", transform=axes["tide_top"].transAxes, **kwargs_subplot_labels
        )
        axes["tide"].text(
            *(-0.1, 1.075), "b)", transform=axes["tide"].transAxes, **kwargs_subplot_labels
        )
        axes["profile"].text(
            *(-0.1, 1.0), "c)", transform=axes["profile"].transAxes, **kwargs_subplot_labels
        )

        #%% save figure
        figure_number = 5
        plt.savefig(
            os.path.join(dir_fig, f"Figure-{figure_number:02d}_tide_compensation_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
        # plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_tide_compensation.pdf"), bbox_inches="tight")
