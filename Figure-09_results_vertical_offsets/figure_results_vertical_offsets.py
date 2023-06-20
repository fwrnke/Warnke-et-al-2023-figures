"""
Create Figure illustrating the results from vertical offset corrections (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-10-27

"""
import os

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cmocean

#%% FUNCTIONS

def round_to_closest(value: float, step: float) -> float:
    """
    Round input `value` to closest `step`.

    Parameters
    ----------
    value : float
        Input value.
    step : float
        Step size (e.g., 0.5).

    Returns
    -------
    float
        Rounded input value.

    Examples
    --------
    >>> round_to_closest(1.4, 0.5)
    1.5

    >>> round_to_closest(1.78, 0.5)
    2.0

    """
    return round(value * (1 / step)) / (1 / step)


#%% MAIN

if __name__ == "__main__":

    dir_work = os.path.dirname(os.path.abspath(__file__))
    dir_fig = os.path.dirname(os.path.abspath(__file__))

    file = "../TOPAS_metadata.aux"

    path_aux = os.path.join(dir_work, file)
    df_aux = pd.read_csv(path_aux, sep=",", parse_dates=["time"])
    
    dpi = 600

    #%% [PLOT]
    with mpl.rc_context(
        {"font.family": "Arial", "mathtext.default": "default", "mathtext.fontset": "stixsans"}
    ):

        ticklabelsize = 10
        textsize = 14

        # set map extent
        xy = (297200, 5125100)
        width = height = 8500

        units = "ms"  # 'ms', 'm', 'samples'
        s = 0.2  # markersize
        alpha = 0.75
        cmap_offset = "RdBu"
        cmap_seafloor = cmocean.cm.deep
        vmin_sf = 735
        vmax_sf = 775

        subset_factor = 3
        x = df_aux["x"].values[::subset_factor]
        y = df_aux["y"].values[::subset_factor]
        corrections = ["static", "tide", "mistie"]

        # ========== CREATE FIGURE ==========
        fig, axes = plt.subplots(
            nrows=2, ncols=3, figsize=(11, 6.3), subplot_kw={"aspect": "equal"}
        )
        fig.subplots_adjust(top=0.92, bottom=0.05, left=0.075, right=0.95, wspace=0.2, hspace=0.1)

        kwargs_row_label = dict(
            x=-0.2,
            y=0.5,
            fontsize=textsize,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
        )

        kwargs_cbar = dict(
            fraction=0.075,
            shrink=0.95,
            pad=0.02,
            orientation="vertical",
            extend="both",
        )

        # ========== VERTICAL OFFSETS ==========
        for i in range(axes.shape[-1]):
            c = df_aux[f"{corrections[i]}_{units}"][::subset_factor]
            vminmax = round_to_closest(max(c.abs().min(), c.abs().max()), 0.25)
            ax_offset = axes[0, i].scatter(
                x,
                y,
                s=s,
                c=c,
                marker="o",
                alpha=alpha,
                cmap=cmap_offset,
                vmin=-vminmax,
                vmax=vminmax,
                rasterized=True
            )

            cbar_ticks = np.arange(-vminmax, vminmax + 0.5, 0.5)
            cbar = fig.colorbar(
                ax_offset,
                ax=axes[0, i],
                format=lambda x, _: f"{x:.1g}",  # + f" {units}",
                ticks=cbar_ticks,
                **kwargs_cbar,
            )
            cbar.ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            cbar.ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

        # label row
        axes[0, 0].text(
            s=f"Vertical offset ({units})", transform=axes[0, 0].transAxes, **kwargs_row_label
        )

        # label cols
        kwargs_col_label = kwargs_row_label.copy()
        kwargs_col_label.update(x=0.5, y=1.15, rotation=0)
        for ax, label in zip(
            axes[0, :], ["Static correction", "Tide compensation", "Mistie correction"]
        ):
            ax.text(s=label, transform=ax.transAxes, **kwargs_col_label)

        # ========== SEAFLOOR ==========
        for i in range(axes.shape[-1]):
            c = df_aux[f'seafloor_{"_".join([c for c in corrections[:i+1]])}'][::subset_factor]
            ax_offset = axes[1, i].scatter(
                x,
                y,
                s=s,
                c=c,
                marker="o",
                alpha=alpha,
                cmap=cmap_seafloor,
                vmin=vmin_sf,
                vmax=vmax_sf,
                rasterized=True
            )

            cbar = fig.colorbar(
                ax_offset, ax=axes[1, i], format=lambda x, _: f"{x:.0f}", **kwargs_cbar  # ms
            )
            cbar.ax.invert_yaxis()
            cbar.ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
            cbar.ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

        # label row
        axes[1, 0].text(
            s="Detected seafloor (ms)", transform=axes[1, 0].transAxes, **kwargs_row_label
        )

        # ---------- LABELS & ANNOTATION ----------
        subplot_labels = ["a", "c", "e", "b", "d", "f"]
        subplot_labels = [f"{s})" for s in subplot_labels]
        # pos_subplot_labels = (0.075, 0.925)  # inside
        pos_subplot_labels = (-0.075, 1.025)  # outside

        kwargs_subplot_labels = dict(
            ha="center", va="center", color="black", fontsize=20, fontweight="bold", family="Times New Roman"
        )

        for i, ax in enumerate(axes.ravel()):
            # xticks
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2500))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(500))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
            # yticks
            ax.yaxis.set_major_locator(mticker.MultipleLocator(2500))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(500))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
            # set map extent
            ax.set_xlim([xy[0] - width // 2, xy[0] + width // 2])
            ax.set_ylim([xy[1] - height // 2, xy[1] + height // 2])

            ax.tick_params(axis="both", which="major", length=6)
            ax.tick_params(axis="both", which="both", direction="in")

            if i == 0:
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelsize=ticklabelsize,  # pad=2,
                    top=True,
                    labeltop=True,
                    right=True,
                    labelright=False,
                )
                for ylabel in ax.get_yticklabels():
                    pos = ylabel.get_unitless_position()
                    angle = 90 if pos[0] == 0.0 else -90
                    ylabel.set_rotation(angle)
                    ylabel.set_va("center")
                    ylabel.set_ha("right")

                for xlabel in ax.get_xmajorticklabels():
                    pos = xlabel.get_unitless_position()
                    if pos[1] == 0.0:
                        xlabel.set_position((pos[0], pos[1] - 0.02))
            elif i in [1, 2]:
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelsize=ticklabelsize,
                    top=True,
                    labeltop=True,
                    right=True,
                    bottom=True,
                    labelbottom=True,
                    left=True,
                    labelleft=False,
                )
                for xlabel in ax.get_xmajorticklabels():
                    pos = xlabel.get_unitless_position()
                    if pos[1] == 0.0:
                        xlabel.set_position((pos[0], pos[1] - 0.02))
            elif i == 3:
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelsize=ticklabelsize,  # pad=2,
                    top=True,
                    labeltop=False,
                    right=True,
                    labelright=False,
                )
                for ylabel in ax.get_yticklabels():
                    pos = ylabel.get_unitless_position()
                    angle = 90 if pos[0] == 0.0 else -90
                    ylabel.set_rotation(angle)
                    ylabel.set_va("center")
                    ylabel.set_ha("right")
            elif i in [4, 5]:
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelsize=ticklabelsize,
                    top=True,
                    labeltop=False,
                    right=True,
                    bottom=True,
                    labelbottom=True,
                    left=True,
                    labelleft=False,
                )

            if i >= 2:
                xy_arrow_offset = [750, 750]
                kwargs_arrows = dict(
                    xycoords="data",
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-|>", shrinkA=0, shrinkB=0, ec="black", fc="black"
                    ),
                )

                xy_arrow = np.array([299000, 5126150])
                ax.annotate("", xy=xy_arrow, xytext=xy_arrow + xy_arrow_offset, **kwargs_arrows)

                xy_arrow = np.array([297150, 5124750])
                ax.annotate("", xy=xy_arrow, xytext=xy_arrow + xy_arrow_offset, **kwargs_arrows)

                xy_arrow = np.array([298100, 5122900])
                ax.annotate("", xy=xy_arrow, xytext=xy_arrow + xy_arrow_offset, **kwargs_arrows)

            # subplot labels
            ax.text(
                *pos_subplot_labels,
                subplot_labels[i],
                transform=ax.transAxes,
                **kwargs_subplot_labels,
            )

        #%% save figure
        figure_number = 9
        plt.savefig(os.path.join(dir_fig, f'Figure-{figure_number:02d}_vertical_offsets_{dpi}dpi.png'), dpi=dpi)
        # plt.savefig(os.path.join(dir_fig, f'Figure-{figure_number:02d}_vertical_offsets.pdf'), dpi=dpi)
