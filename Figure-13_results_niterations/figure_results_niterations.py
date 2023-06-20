"""
Create Figure showing interpolation results using different maximum iterations (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-11-02

"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#%% FUNCTIONS

def annotate_frequency(
    ax, dataframe, idx: int, xpos: int, xytext: tuple, color: str = 'k', **kwargs_labels
):
    """Annotate selected frequency."""
    freqs = np.linspace(0, 10, 2001, endpoint=True)

    ax.annotate(
        f'{freqs[idx]*1000:g} Hz',
        xy=(xpos, dataframe.loc[xpos, idx]),
        xycoords='data',
        xytext=xytext,
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='-|>',
            connectionstyle='arc3',
            shrinkA=0,
            shrinkB=3,
            edgecolor=color,
            facecolor=color,
            lw=1,
        ),
        horizontalalignment='center',
        verticalalignment='center',
        color=color,
        **kwargs_labels,
    )


#%% MAIN

if __name__ == '__main__':

    dir_fig = os.path.dirname(os.path.abspath(__file__))
    dir_work = dir_fig

    niterations = [10, 30, 50, 100]
    domains = ['iline', 'xline', 'twt']
    var = 'env'
    
    dpi = 600
    
    #%% [AUX] cost vs iteration

    columns = ['niterations', 'runtime_s']

    df_dict = {}
    for niter in niterations:
        folder = f'cube_center_IDW_env_5x5m_0+05ms_freq-il-xl_preproc_FFT_hard_niter-{niter}'
        df = pd.read_csv(os.path.join(dir_fig, f'runtimes_{folder}.txt'), sep=';', header=None)
        ncols = df.shape[1]
        df.rename(
            columns=dict(zip(list(df.columns), columns + [i for i in range(1, ncols - 2 + 1)])),
            inplace=True,
        )
        df_dict[niter] = df

    #%% [AUX] runtimes
    nworkers = 8
    nthreads = 16

    # runtimes from report
    runtime_col = 'runtime (parallel)'
    computation_col = 'computation time'
    runtimes = {
        10: {
            runtime_col: 149.01,
            # 'ntasks': 16123,
            computation_col: 2271,
            # 'transfer_time': 9.87
        },
        30: {
            runtime_col: 326.96,
            # 'ntasks': 16123,
            computation_col: 5028,
            # 'transfer_time': 19.87
        },
        50: {
            runtime_col: 494.42,
            # 'ntasks': 16123,
            computation_col: 7620,
            # 'transfer_time': 25.38
        },
        100: {
            runtime_col: 935,
            # 'ntasks': 16123,
            computation_col: 14580,
            # 'transfer_time': 29.77
        },
    }

    #%% [PLOTTING] combined sparse + interp
    with mpl.rc_context({'font.family': 'Arial', 'mathtext.fontset': 'stix'}):
        
        nrows, ncols = 1, 2
        kwargs_figure = dict(
            figsize=(ncols * 5, nrows * 5),
        )
        kwargs_subfigure = dict(
            wspace=0.01,
            hspace=0.0,
        )
        if nrows > ncols :
            kwargs_subfigure['height_ratios'] = (1, 0.9)
        
        figure = plt.figure(layout='none', **kwargs_figure)
        subfigs = figure.subfigures(nrows, ncols, **kwargs_subfigure)
        subfigs = subfigs.ravel(order='C')
             
        # =====================================================================================
        #                         RECONSTRUCTION METRIC vs ITERATION
        # =====================================================================================
        kwargs_labels = dict(fontsize=12, fontweight='semibold')
        kwargs_ticklabels = dict(labelsize=12)

        # fig = plot_dict['subfigs'][-2]
        fig = subfigs[0]
        axes = fig.subplots(
            2,
            2,
            sharey=True,
            gridspec_kw=dict(wspace=0.1, hspace=0.3, left=0.15, right=0.9, top=0.925, bottom=0.1),
        )
        axes = axes.ravel(order='C')

        kw_examples = {
            10: [
                dict(idx_freq=3, color='k', xpos=1, xytext=(0.6, 0.85)),
                dict(idx_freq=500, color='r', xpos=4, xytext=(0.7, 0.6)),
                dict(idx_freq=1000, color='b', xpos=8, xytext=(0.8, 0.2)),
            ],
            30: [
                dict(idx_freq=3, color='k', xpos=3, xytext=(0.25, 0.55)),
                dict(idx_freq=500, color='r', xpos=10, xytext=(0.55, 0.4)),
                dict(idx_freq=1000, color='b', xpos=20, xytext=(0.75, 0.25)),
            ],
            50: [
                dict(idx_freq=3, color='k', xpos=2, xytext=(0.2, 0.6)),
                dict(idx_freq=500, color='r', xpos=12, xytext=(0.4, 0.4)),
                dict(idx_freq=1000, color='b', xpos=22, xytext=(0.7, 0.25)),
            ],
            100: [
                dict(idx_freq=3, color='k', xpos=5, xytext=(0.2, 0.5)),
                dict(idx_freq=500, color='r', xpos=30, xytext=(0.45, 0.3)),
                dict(idx_freq=1000, color='b', xpos=60, xytext=(0.75, 0.2)),
            ],
        }

        for i, (niter, df) in enumerate(df_dict.items()):
            xtick_interval = 10 if niter > 10 else 2
            xtick_interval = 25 if niter >= 100 else xtick_interval

            df_subset = df[df.niterations != 0]  #
            df_subset = df_subset[[col for col in df_subset.columns if col not in columns]].T

            ax_iter = df_subset.plot(
                ax=axes[i],
                c='lightgrey',
                alpha=0.2,
                legend=False,
                xticks=np.arange(0, niter + xtick_interval + 1, xtick_interval),
                xlabel='iteration (#)' if i >= 2 else None,
                xlim=(0, niter),
                ylabel=r'$J_k$' if i % 2 == 0 else None,
                ylim=(-0.002, 0.12),
            )

            ax_iter.tick_params(**kwargs_ticklabels)
            ax_iter.xaxis.label.set_size(kwargs_ticklabels.get('labelsize'))
            ax_iter.yaxis.label.set_size(kwargs_ticklabels.get('labelsize') + 2)
            ax_iter.text(
                0.5,
                1.0,
                f'Iterations: {niter}',
                fontsize=11,
                fontweight='semibold',
                ha='center',
                va='center',
                transform=ax_iter.transAxes,
            )

            ax_iter.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
            ax_iter.tick_params(axis='y', which='minor', left=False)
            # ax_iter.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))

            kwargs_labels_freq = kwargs_labels.copy()
            kwargs_labels_freq.update(fontsize=kwargs_labels.get('fontsize', 12) - 1)
            for kw in kw_examples.get(niter):
                df_subset.iloc[:, kw['idx_freq']].plot(ax=ax_iter, c=kw['color'], legend=False)
                annotate_frequency(
                    ax_iter,
                    df_subset,
                    idx=kw['idx_freq'],
                    xpos=kw['xpos'],
                    xytext=kw['xytext'],
                    color=kw['color'],
                    **kwargs_labels_freq,
                )

            for loc in ['right', 'top']:
                ax_iter.spines[loc].set_visible(False)

        fig.text(
            0.05,
            0.975,
            'a)',
            transform=fig.transSubfigure,
            va="center",
            ha="center",
            fontsize=18,
            fontweight='semibold',
            family='Times New Roman'
        )

        # =====================================================================================
        #                                   RUNTIMES
        # =====================================================================================
        # fig = plot_dict['subfigs'][-1]
        fig = subfigs[1]
        ax_runtime = fig.subplots(
            1, 1, gridspec_kw=dict(left=0.15, right=0.9, bottom=0.2, top=0.85)
        )

        df_runtimes = pd.DataFrame.from_dict(runtimes, orient='index')
        df_runtimes.plot(ax=ax_runtime, kind='barh', color=['#bdbdbd', '#737373'], width=0.85)

        ax_runtime.set_xlim((0, 16000))
        ax_runtime.set_xlabel('Runtime (min)', fontsize=kwargs_labels.get('fontsize'))
        ax_runtime.set_ylabel('Total iterations (#)', fontsize=kwargs_labels.get('fontsize'))
        ax_runtime.tick_params(axis='y', left=False, labelsize=kwargs_labels.get('fontsize'))
        yticklabels = [t.get_text() for t in ax_runtime.get_yticklabels()]
        ax_runtime.set_yticklabels(yticklabels, weight='semibold')

        for i, p in enumerate(ax_runtime.patches):
            time = p.get_width()
            minutes = int(time // 60)
            seconds = time % 60
            time_fmt = f'{minutes:d} min {seconds:.0f} s' if seconds != 0 else f'{minutes:d} min'
            xpad = -400 if i in [6, 7] else 200
            ax_runtime.annotate(
                time_fmt,
                (p.get_width() + xpad, p.get_y() + p.get_height() / 2),
                va='center',
                ha='right' if i in [6, 7] else 'left',
                color='white' if i in [6, 7] else 'black',
                fontsize=11,
            )

        # print(ax_runtime.get_xticks())

        ax_runtime.xaxis.set_major_locator(mticker.MultipleLocator(1800))
        ax_runtime.xaxis.set_major_formatter(lambda x, pos: f'{x/60:.0f}')
        ax_runtime.set_xlim(0, 245 * 60)
        ax_runtime.invert_yaxis()

        for loc in ['right', 'top', 'left']:
            ax_runtime.spines[loc].set_visible(False)

        ax_runtime.text(
            0.5,
            1.05,
            f'Worker: {nworkers} | Threads: {nthreads}',
            fontsize=14,
            fontweight='semibold',
            ha='center',
            va='center',
            transform=ax_runtime.transAxes,
        )
        ax_runtime.legend(loc='upper right', fontsize=kwargs_labels.get('fontsize') - 2)

        fig.text(
            0.05,
            0.975,
            'b)',
            transform=fig.transSubfigure,
            va="center",
            ha="center",
            fontsize=18,
            fontweight='semibold',
            family='Times New Roman'
        )

        #%% save figure
        figure_number = 13
        figure.savefig(
            os.path.join(dir_fig, f"Figure-{figure_number:02d}_niterations_{dpi}dpi.png"),
            dpi=dpi,
            bbox_inches='tight',
        )
        # figure.savefig(
        #     os.path.join(dir_fig, f"Figure-{figure_number:02d}_niterations_{dpi}dpi.tiff"),
        #     dpi=dpi,
        #     bbox_inches='tight',
        # )
