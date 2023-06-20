"""
Create Figure illustrating interpolation results from different sparse transforms (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   08-11-2022

"""
import os
import sys
import time
import datetime
import itertools
import warnings
from functools import partial

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pywt
from pywt._thresholding import soft as _soft_threshold
from pywt._thresholding import hard as _hard_threshold
from pywt._thresholding import nn_garrote as _nn_garrote

import FFST  # shearlet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import curvelops
    SYSTEM = "LINUX"
    os.chdir("/mnt/d/scripts/packages/Warnke-et-al_2023_figures")
except ModuleNotFoundError:
    warnings.warn("Module `curvelops` not found, thus CURVELET transform is not available.")
    SYSTEM = "WINDOWS"

from functions_figures import _print_info_TOPAS_moratorium

xr.set_options(keep_attrs=True)

_print_info_TOPAS_moratorium()


#%% FUNCTIONS

# =================================================================================================
#                                           POCS
# =================================================================================================

def get_number_scales(x):
    """
    Compute number of shearlet scales based on input array shape.

    References
    ----------
    [^1]: [https://github.com/grlee77/PyShearlets/blob/master/FFST/_scalesShearsAndSpectra.py](https://github.com/grlee77/PyShearlets/blob/master/FFST/_scalesShearsAndSpectra.py)
    
    """
    scales = int(np.floor(0.5 * np.log2(np.max(x.shape))))
    return scales if scales >= 1 else 1


def _hard_threshold_perc(x, perc, sub):  # noqa
    """Wrapper for hard thresholding using percentile of abs(x)."""
    thresh = np.percentile(np.abs(x), perc)
    return _hard_threshold(x, thresh, sub)


def _soft_threshold_perc(x, perc, sub):  # noqa
    """Wrapper for soft thresholding using percentile of abs(x)."""
    thresh = np.percentile(np.abs(x), perc)
    return _soft_threshold(x, thresh, sub)


def _nn_garrote_perc(x, perc, sub):  # noqa
    """Wrapper for garrote thresholding using percentile of abs(x)."""
    thresh = np.percentile(np.abs(x), perc)
    return _nn_garrote(x, thresh, sub)


def threshold(data, thresh, sub=0, kind='soft'):
    """
    Apply user-defined threshold to input data (2D).

    Parameters
    ----------
    data : np.ndarray
        Input data.
    thresh : float, complex
        Threshold cut-off value.
    sub : int, float, optional
        Substitution value (default: `0`).
    kind : str, optional
        Threshold method:
            
          - `soft` (**default**)
          - `garrote`
          - `hard`
          - `soft-percentile`
          - `garrote-percentile`
          - `hard-percentile`

    Returns
    -------
    np.ndarray
        Updated input array using specified thresholding function.

    """
    data = np.asarray(data)

    if kind == 'soft':
        return _soft_threshold(data, thresh, sub)
    elif kind == 'hard':
        return _hard_threshold(data, thresh, sub)
    elif kind == 'soft-percentile':
        return _soft_threshold_perc(data, thresh, sub)
    elif kind == 'hard-percentile':
        return _hard_threshold_perc(data, thresh, sub)
    elif kind in ['garotte', 'garrote']:
        return _nn_garrote(data, thresh, sub)
    elif kind in ['garotte-percentile', 'garrote-percentile']:
        return _nn_garrote_perc(data, thresh, sub)


def threshold_wavelet(data, thresh, sub=0, kind='soft'):
    """
    Apply user-defined threshold to input data (2D).
    Compatible with output from `pywavelet.wavedec2` (multilevel Discrete Wavelet Transform).

    Parameters
    ----------
    data : np.ndarray
        Input data.
    thresh : float, complex
        Threshold cut-off value.
    sub : int, float, optional
        Substitution value (default: `0`).
    kind : str, optional
        Threshold method:
            
          - `soft` (**default**)
          - `garrote`
          - `hard`
          - `soft-percentile`
          - `garrote-percentile`
          - `hard-percentile`

    Returns
    -------
    np.ndarray
        Updated input array using specified thresholding function.

    """
    thresh = [list(d) for d in list(thresh)]
    dlen = len(data[-1])

    if kind == 'soft':
        return [
            [_soft_threshold(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind == 'hard':
        return [
            [_hard_threshold(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind == 'soft-percentile':
        return [
            [_soft_threshold_perc(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind == 'hard-percentile':
        return [
            [_hard_threshold_perc(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind in ['garotte', 'garrote']:
        return [
            [_nn_garrote(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind in ['garotte-percentile', 'garrote-percentile']:
        return [
            [_nn_garrote_perc(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]


def get_threshold_decay(
    thresh_model,
    niter: int,
    transform_kind: str = None,
    p_max: float = 0.99,
    p_min: float = 1e-3,
    x_fwd=None,
    kind: str = 'values',
):
    """
    Calculate iteration-based decay for thresholding function.
    Can be one of the following:
      
      - `values` (based on max value in data)
      - `factors` (for usage as multiplier).

    Parameters
    ----------
    thresh_model : str
        Thresholding decay function.
            
            - `linear`                  Gao et al. (2010)
            - `exponential`             Yang et al. (2012), Zhang et al. (2015), Zhao et al. (2021)
            - `data-driven`             Gao et al. (2013)
            - `inverse_proportional`    Ge et al. (2015)
    niter : int
        Maximum number of iterations.
    transform_kind : str
        Name of the specified transform (e.g. FFT, WAVELET, SHEARLET, CURVELET).
    p_max : float, optional
        Maximum regularization percentage (float).
    p_min : float, str, optional
        Minimum regularization percentage (float) or
        'adaptive': adaptive calculation of minimum threshold according to sparse coefficient.
    x_fwd : np.ndarray, optional
        Forward transformed input data (required for thresh_model=`data-driven` and kind=`values`).
    kind : str, optional
        Return either data `values` or multiplication `factors`.

    Returns
    -------
    tau : np.ndarray
        Array of decay values or factors (based on "kind" paramter).
    
    References
    ----------
    [^1]: Gao, J.-J., Chen, X.-H., Li, J.-Y., Liu, G.-C., & Ma, J. (2010).
        Irregular seismic data reconstruction based on exponential threshold model of POCS method.
        Applied Geophysics, 7(3), 229–238. [https://doi.org/10.1007/s11770-010-0246-5](https://doi.org/10.1007/s11770-010-0246-5)
    [^2]: Yang, P., Gao, J., & Chen, W. (2012).
        Curvelet-based POCS interpolation of nonuniformly sampled seismic records.
        Journal of Applied Geophysics, 79, 90–99. [https://doi.org/10.1016/j.jappgeo.2011.12.004](https://doi.org/10.1016/j.jappgeo.2011.12.004)
    [^3]: Zhang, H., Chen, X., & Li, H. (2015).
        3D seismic data reconstruction based on complex-valued curvelet transform in frequency domain.
        Journal of Applied Geophysics, 113, 64–73. [https://doi.org/10.1016/j.jappgeo.2014.12.004](https://doi.org/10.1016/j.jappgeo.2014.12.004)
    [^4]: Zhao, H., Yang, T., Ni, Y.-D., Liu, X.-G., Xu, Y.-P., Zhang, Y.-L., & Zhang, G.-R. (2021).
        Reconstruction method of irregular seismic data with adaptive thresholds based on different sparse transform bases.
        Applied Geophysics, 18(3), 345–360. [https://doi.org/10.1007/s11770-021-0903-5](https://doi.org/10.1007/s11770-021-0903-5)
    [^5]: Gao, J., Stanton, A., Naghizadeh, M., Sacchi, M. D., & Chen, X. (2013).
        Convergence improvement and noise attenuation considerations for beyond alias projection onto convex sets reconstruction.
        Geophysical Prospecting, 61, 138–151. [https://doi.org/10.1111/j.1365-2478.2012.01103.x](https://doi.org/10.1111/j.1365-2478.2012.01103.x)
    [^6]: Ge, Z.-J., Li, J.-Y., Pan, S.-L., & Chen, X.-H. (2015).
        A fast-convergence POCS seismic denoising and reconstruction method.
        Applied Geophysics, 12(2), 169–178. [https://doi.org/10.1007/s11770-015-0485-1](https://doi.org/10.1007/s11770-015-0485-1)

    """
    TRANSFORMS = ('FFT', 'WAVELET', 'SHEARLET', 'CURVELET', 'DCT')
    if transform_kind is None:
        pass
    elif transform_kind.upper() not in TRANSFORMS and (
        kind == 'values' or thresh_model == 'data-driven'
    ):
        raise ValueError(f'Unsupported transform. Please select one of: {TRANSFORMS}')
    elif transform_kind is not None:
        transform_kind = transform_kind.upper()

    if x_fwd is None and (kind == 'values' or thresh_model == 'data-driven'):
        raise ValueError(
            '`x_fwd` must be specified for thresh_model="data-driven" or kind="values"!'
        )

    # (A) inversely proportional threshold model (Ge et al., 2015)
    if all([s in thresh_model for s in ['inverse', 'proportional']]):
        if transform_kind == 'WAVELET':
            x_fwd_max = np.asarray([[np.abs(d).max() for d in level] for level in x_fwd])
            x_fwd_min = np.asarray([[np.abs(d).min() for d in level] for level in x_fwd])
            _iiter = np.arange(1, niter + 1)[:, None, None]
        elif transform_kind == 'SHEARLET':
            x_fwd_max = np.max(np.abs(x_fwd), axis=(0, 1))
            x_fwd_min = np.min(np.abs(x_fwd), axis=(0, 1))
            _iiter = np.arange(1, niter + 1)[:, None]
        elif transform_kind in ['FFT', 'CURVELET', 'DCT']:
            x_fwd_max = np.abs(x_fwd).max()
            x_fwd_min = np.abs(x_fwd).min()
            _iiter = np.arange(1, niter + 1)

        # arbitrary variable to adjust descent rate (most cases: 1 <= q <=3)
        q = thresh_model.split('-')[-1] if '-' in thresh_model else 1.0
        try:
            q = float(q)
        except:  # noqa
            q = 1.0

        a = (niter**q * (x_fwd_max - x_fwd_min)) / (niter**q - 1)
        b = (niter**q * x_fwd_min - x_fwd_max) / (niter**q - 1)
        return a / (_iiter**q) + b

    # (B) "classic" thresholding models
    if kind == 'values':
        # max (absolute) value in forward transformed data
        if transform_kind == 'WAVELET':
            # x_fwd_max = np.asarray([[np.abs(d).max() for d in level] for level in x_fwd])
            x_fwd_max = np.asarray([[d.max() for d in level] for level in x_fwd])
        elif transform_kind == 'SHEARLET':
            axis = (0, 1)
            # x_fwd_max = np.max(np.abs(x_fwd), axis=axis)
            x_fwd_max = np.max(x_fwd, axis=axis)
        elif transform_kind in ['FFT', 'CURVELET', 'DCT']:
            axis = None
            # x_fwd_max = np.abs(x_fwd).max()  # FIXME
            x_fwd_max = x_fwd.max()
        else:
            raise ValueError(
                '`transform_kind` must be specified for thresh_model="data-driven" or kind="values"!'
            )

        # min/max regularization factors
        #   adaptive calculation of minimum threshold (Zhao et al., 2021)
        if isinstance(p_min, str) and p_min == 'adaptive':
            # single-scale transform
            if transform_kind in ['FFT', 'DCT']:
                tau_min = 0.01 * np.sqrt(np.linalg.norm(x_fwd, axis=axis) ** 2 / x_fwd.size)

            # mulit-scale transform
            elif transform_kind in ['SHEARLET']:
                # calculate regularization factor `tau_min` for each scale
                nscales = get_number_scales(x_fwd)
                j = np.hstack(
                    (
                        np.array([0]),  # low-pass solution
                        np.repeat(
                            np.arange(1, nscales + 1), [2 ** (j + 2) for j in range(nscales)]
                        ),
                    )
                )
                tau_min = (
                    1
                    / 3
                    * np.median(  # noqa
                        np.log10(j + 1)
                        * np.sqrt(np.linalg.norm(x_fwd, axis=axis) ** 2 / x_fwd.size)
                    )
                )
            else:
                raise NotImplementedError(
                    f'p_min=`adaptive` is not implemented for {transform_kind} transform'
                )
        else:
            tau_min = p_min * x_fwd_max
        tau_max = p_max * x_fwd_max

    elif kind == 'factors':
        tau_max = p_max
        tau_min = p_min
    else:
        raise ValueError('Parameter `kind` only supports arguments "values" or "factors"')

    # --- iteration-based threshold factor ---
    _iiter = np.arange(1, niter + 1)

    if transform_kind == 'WAVELET':
        imultiplier = ((_iiter - 1) / (niter - 1))[:, None, None]
    elif transform_kind == 'SHEARLET':
        imultiplier = ((_iiter - 1) / (niter - 1))[:, None]
    elif transform_kind in ['FFT', 'CURVELET', 'DCT']:
        imultiplier = (_iiter - 1) / (niter - 1)
    elif transform_kind is None:
        imultiplier = (_iiter - 1) / (niter - 1)

    # --- thresholding operator ---
    if thresh_model == 'linear':
        tau = tau_max - (tau_max - tau_min) * imultiplier

    elif 'exponential' in thresh_model:
        q = float(thresh_model.split('-')[-1]) if '-' in thresh_model else 1.0  # Zhao et al. (2021)
        c = np.log(tau_min / tau_max)
        tau = tau_max * np.exp(c * imultiplier**q)

    elif thresh_model == 'data-driven' and transform_kind in ['FFT', 'DCT', 'CURVELET']:
        tau = np.zeros((_iiter.size,), dtype=x_fwd.dtype)
        idx = (x_fwd > tau_min) & (x_fwd < tau_max)
        v = np.sort(x_fwd[idx])[::-1]
        Nv = v.size
        tau[0] = v[0]
        tau[1:] = v[np.ceil((_iiter[1:] - 1) * (Nv - 1) / (niter - 1)).astype('int')]
    else:
        raise NotImplementedError(
            f'{thresh_model} is not implemented for {transform_kind} transform!'
        )

    return tau


def POCS_algorithm(
    x,
    mask,
    auxiliary_data=None,
    transform=None,
    itransform=None,
    transform_kind: str = None,
    niter: int = 50,
    thresh_op: str = 'hard',
    thresh_model: str = 'exponential',
    eps: float = 1e-9,
    alpha: int = 1.0,
    p_max: float = 0.99,
    p_min: float = 1e-5,
    sqrt_decay: str = False,
    decay_kind: str = 'values',
    verbose: bool = False,
    version: str = 'regular',
    results_dict: dict = None,
    path_results: str = None,
):
    """
    Interpolate sparse input grid using Point Onto Convex Sets (POCS) algorithm.
    Applying a user-specified **transform** method:
        
      - `FFT`
      - `Wavelet`
      - `Shearlet`
      - `Curvelet`

    Parameters
    ----------
    x : np.ndarray
        Sparse input data (2D).
    mask : np.ndarray
        Boolean mask of input data (`1`: data cell, `0`: nodata cell).
    auxiliary_data: np.ndarray
        Auxiliary data only required by `shearlet` transform.
    transform : callable
        Forward transform to apply.
    itransform : callable
        Inverse transform to apply.
    transform_kind : str
        Name of the specified transform.
    niter : int, optional
        Maximum number of iterations (default: `50`).
    thresh_op : str, optional
        Threshold operator (default: `soft`).
    thresh_model : str, optional
        Thresholding decay function.
            
            - `linear`                   Gao et al. (2010)
            - `exponential`              Yang et al. (2012), Zhang et al. (2015), Zhao et al. (2021)
            - `data-driven`              Gao et al. (2013)
            - `inverse_proportional`     Ge et al. (2015)
    eps : float, optional
        Covergence threshold (default: `1e-9`).
    alpha : float, optional
        Weighting factor to scale re-insertion of input data (default: `1.0`).
    sqrt_decay : bool, optional
        Use squared decay values for thresholding (default: `False`).
    decay_kind : str, optional
        Return either data "values" or multiplication "factors".
    verbose : bool, optional
        Print information about iteration steps (default: `False`).
    version : str, optional
        Version of POCS algorithm. One of the following:
            
            - `regular`     Abma and Kabir (2006), Yang et al. (2012)
            - `fast`        Yang et al. (2013), Gan et al (2015)
            - `adaptive`    Wang et al. (2015, 2016)
    results_dict : dict, optional
        If provided: return dict with total iterations, runtime (in seconds) and cost function.

    Returns
    -------
    x_inv : np.ndarray
        Reconstructed (i.e. interpolated) input data.

    References
    ----------
    [^1]: Gao, J.-J., Chen, X.-H., Li, J.-Y., Liu, G.-C., & Ma, J. (2010).
        Irregular seismic data reconstruction based on exponential threshold model of POCS method.
        Applied Geophysics, 7(3), 229–238. [https://doi.org/10.1007/s11770-010-0246-5](https://doi.org/10.1007/s11770-010-0246-5)
    [^2]: Yang, P., Gao, J., & Chen, W. (2012).
        Curvelet-based POCS interpolation of nonuniformly sampled seismic records.
        Journal of Applied Geophysics, 79, 90–99. [https://doi.org/10.1016/j.jappgeo.2011.12.004](https://doi.org/10.1016/j.jappgeo.2011.12.004)
    [^3]: Zhang, H., Chen, X., & Li, H. (2015).
        3D seismic data reconstruction based on complex-valued curvelet transform in frequency domain.
        Journal of Applied Geophysics, 113, 64–73. [https://doi.org/10.1016/j.jappgeo.2014.12.004](https://doi.org/10.1016/j.jappgeo.2014.12.004)
    [^4]: Zhao, H., Yang, T., Ni, Y.-D., Liu, X.-G., Xu, Y.-P., Zhang, Y.-L., & Zhang, G.-R. (2021).
        Reconstruction method of irregular seismic data with adaptive thresholds based on different sparse transform bases.
        Applied Geophysics, 18(3), 345–360. [https://doi.org/10.1007/s11770-021-0903-5](https://doi.org/10.1007/s11770-021-0903-5)
    [^5]: Gao, J., Stanton, A., Naghizadeh, M., Sacchi, M. D., & Chen, X. (2013).
        Convergence improvement and noise attenuation considerations for beyond alias projection onto convex sets reconstruction.
        Geophysical Prospecting, 61, 138–151. [https://doi.org/10.1111/j.1365-2478.2012.01103.x](https://doi.org/10.1111/j.1365-2478.2012.01103.x)
    [^6]: Ge, Z.-J., Li, J.-Y., Pan, S.-L., & Chen, X.-H. (2015).
        A fast-convergence POCS seismic denoising and reconstruction method.
        Applied Geophysics, 12(2), 169–178. [https://doi.org/10.1007/s11770-015-0485-1](https://doi.org/10.1007/s11770-015-0485-1)
    [^7]: Abma, R., & Kabir, N. (2006). 3D interpolation of irregular data with a POCS algorithm.
        Geophysics, 71(6), E91–E97. [https://doi.org/10.1190/1.2356088](https://doi.org/10.1190/1.2356088)
    [^8]: Yang, P., Gao, J., & Chen, W. (2013)
        On analysis-based two-step interpolation methods for randomly sampled seismic data.
        Computers & Geosciences, 51, 449–461. [https://doi.org/10.1016/j.cageo.2012.07.023](https://doi.org/10.1016/j.cageo.2012.07.023)
    [^9]: Gan, S., Wang, S., Chen, Y., Zhang, Y., & Jin, Z. (2015).
        Dealiased Seismic Data Interpolation Using Seislet Transform With Low-Frequency Constraint.
        IEEE Geoscience and Remote Sensing Letters, 12(10), 2150–2154. [https://doi.org/10.1109/LGRS.2015.2453119](https://doi.org/10.1109/LGRS.2015.2453119)
    [^10]:  Wang, B., Wu, R.-S., Chen, X., & Li, J. (2015).
        Simultaneous seismic data interpolation and denoising with a new adaptive method based on dreamlet transform.
        Geophysical Journal International, 201(2), 1182–1194. [https://doi.org/10.1093/gji/ggv072](https://doi.org/10.1093/gji/ggv072)
    [^11]: Wang, B., Chen, X., Li, J., & Cao, J. (2016).
        An Improved Weighted Projection Onto Convex Sets Method for Seismic Data Interpolation and Denoising.
        IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 9(1), 228–235.
        [https://doi.org/10.1109/jstars.2015.2496374](https://doi.org/10.1109/jstars.2015.2496374)

    """
    # sanity checks
    if np.max(mask) > 1:
        raise ValueError(f'mask should be quasi-boolean (0 or 1) but has maximum of {np.max(mask)}')

    if any(v is None for v in [transform, itransform]):
        raise ValueError('Forward and inverse transform function have to be supplied')

    TRANSFORMS = ('FFT', 'WAVELET', 'SHEARLET', 'CURVELET', 'DCT')
    if transform_kind.upper() not in TRANSFORMS:
        raise ValueError(f'Unsupported transform. Please select one of: {TRANSFORMS}')
    else:
        transform_kind = transform_kind.upper()

    if transform_kind == 'SHEARLET' and auxiliary_data is None:
        raise ValueError(
            f'{transform_kind} requires pre-computed shearlets in Fourier domain (Psi)'
        )

    # get input paramter
    is_complex_input = np.iscomplexobj(x)
    shape = x.shape
    original_shape = tuple(slice(s) for s in shape)

    if np.count_nonzero(x) == 0:
        niterations = 0
        runtime = 0
        cost = 0
        costs = [0]

        x_inv = x
    else:
        # initial forward transform
        if transform_kind == 'WAVELET':  # and isinstance(x_fwd, list):
            x_fwd = transform(x)[1:]  # exclude low-pass filter
        elif transform_kind == 'SHEARLET':  # and isinstance(x_fwd, tuple):
            x_fwd = transform(x, Psi=auxiliary_data)  # [0]   # output is like (ST, Psi)
        elif (
            transform_kind == 'CURVELET'
            and hasattr(transform, '__name__')
            and transform.__name__ == 'matvec'
        ):
            x_fwd = transform(x.ravel())
        else:
            x_fwd = transform(x)

        # get threshold decay array
        decay = get_threshold_decay(
            thresh_model=thresh_model,
            niter=niter,
            transform_kind=transform_kind,
            p_max=p_max,
            p_min=p_min,
            x_fwd=x_fwd,
            kind=decay_kind,
        )

        # init data variables
        x_old = x
        x_inv = x

        # init variable for improved convergence (Yang et al., 2013)
        if version == 'fast':
            v = 1

        t0 = time.perf_counter()
        if path_results is not None:
            costs = []

        for iiter in range(niter):
            if verbose:
                print(f'[Iteration: <{iiter+1:3d}>]')

            if version == 'regular':
                x_input = x_old
            elif version == 'fast':  # Yang et al. (2013)
                # improved convergence
                v1 = (1 + np.sqrt(1 + 4 * v**2)) / 2
                frac = (v - 1) / (v1 + 1)  # Gan et al. (2015)
                v = v1
                x_input = x_inv + frac * (x_inv - x_old)  # prediction
            elif version == 'adaptive':  # Wang et al. (2015, 2016)
                # init adaptive input data
                x_tmp = alpha * x + (1 - alpha * mask) * x_old
                x_input = x_tmp + (1 - alpha) * (x - mask * x_old)
                # x_input = x_inv + (1 - alpha) * (x - mask * x_old)

            # (1) forward transform
            if (
                transform_kind == 'CURVELET'
                and hasattr(transform, '__name__')
                and transform.__name__ == 'matvec'
            ):
                X = transform(x_input.ravel())
            elif transform_kind == 'WAVELET':
                X = transform(x_input)
                lowpass = X[0].copy()
                X = X[1:]
            elif transform_kind == 'SHEARLET':
                X = transform(x_input, Psi=auxiliary_data)
            else:
                X = transform(x_input)

            # (2) thresholding
            _decay = np.sqrt(decay[iiter]) if sqrt_decay else decay[iiter]
            if transform_kind == 'WAVELET' and isinstance(X, list):
                X_thresh = threshold_wavelet(X, _decay, kind=thresh_op)
            else:
                X_thresh = threshold(X, _decay, kind=thresh_op)

            # (3) inverse transform
            if (
                transform_kind == 'CURVELET'
                and hasattr(itransform, '__name__')
                and itransform.__name__ == 'rmatvec'
            ):
                x_inv = itransform(X_thresh).reshape(shape)
            elif transform_kind == 'WAVELET':
                x_inv = itransform([lowpass] + X_thresh)[original_shape]
            elif transform_kind == 'SHEARLET':
                x_inv = itransform(X_thresh, Psi=auxiliary_data)
            else:
                x_inv = itransform(X_thresh)

            # (4) apply mask (scaled by weighting factor)
            x_inv *= 1 - alpha * mask

            # (5) add original data (scaled by weighting factor)
            x_inv += x * alpha

            # cost function from Gao et al. (2013)
            cost = np.sum(np.abs(x_inv) - np.abs(x_old)) ** 2 / np.sum(np.abs(x_inv)) ** 2
            if path_results is not None:
                costs.append(cost)
            if verbose:
                print('[INFO]   cost:', cost)

            # set result from previous iteration as new input
            x_old = x_inv

            if iiter > 2 and cost < eps:
                break

        niterations = iiter + 1
        runtime = time.perf_counter() - t0

    if verbose:
        print('\n' + '-' * 20)
        print(f'# iterations:  {niterations:4d}')
        print(f'cost function: {cost}')
        print(f'runtime:       {runtime:.3f} s')
        print('-' * 20)

    if isinstance(results_dict, dict):
        results_dict['niterations'] = niterations
        results_dict['runtime'] = round(runtime, 3)
        results_dict['cost'] = cost

    if path_results is not None:
        with open(path_results, mode="a", newline='\n') as f:
            f.write(';'.join([str(i) for i in [niterations, runtime] + costs]) + '\n')

    if is_complex_input:
        return x_inv

    return np.real(x_inv)


POCS = partial(POCS_algorithm, version='regular')
FPOCS = partial(POCS_algorithm, version='fast')
APOCS = partial(POCS_algorithm, version='adaptive')

POCS_VERSIONS = {"POCS": POCS, "FPOCS": FPOCS, "APOCS": APOCS}


# =================================================================================================
#                                           MISC
# =================================================================================================

def split_by_chunks(dataset):
    """
    Split xarray.Dataset into sub-datasets (for netCDF export).
    Source: https://ncar.github.io/esds/posts/2020/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/#create-a-helper-function-to-split-a-dataset-into-sub-datasets

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to split.

    Yields
    ------
    generator (of xr.Datasets)
        Sub-datasets of input dataset.

    """
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]


def dataset_subsets(dataset, dim: str, size: int):
    """
    Generate dataset views of given `size` along `dim`.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to subset.
    dim : str
        Dimension along which to subset.
    size : int
        Size of subset along `dim`.

    Yields
    ------
    xr.Dataset
        Subset of input dataset.

    """
    indices = list(range(0, dataset[dim].size + size, size))
    for start, end in zip(indices[:-1], indices[1:]):
        # print(start, end)
        yield dataset[{dim: slice(start, end)}]


def create_file_path(ds, dim="twt", prefix=None, root_path="."):
    """Generate a file path when given an xarray dataset."""
    if prefix is None:
        prefix = datetime.datetime.today().strftime("%Y-%m-%d")
    try:
        start = ds[dim].data[0]
        end = ds[dim].data[-1]
    except IndexError:
        start = np.atleast_1d(ds[dim].data)[0]
        end = np.atleast_1d(ds[dim].data)[-1]
    return os.path.join(root_path, f"{prefix}_{start:.3f}_{end:.3f}.nc")


def sharpness_index(x: np.ndarray, return_percentage: bool = True) -> float:
    """Calclulate sharpness index."""
    gy, gx = np.gradient(x)
    s = np.sum(np.sum(np.sqrt(gx**2 + gy**2)) / x.size)
    if return_percentage:
        s *= 100
    return s


def plot_inset_wiggle(
    ax,
    data,
    twt_slice: slice = None,
    trace_slice: slice = None,
    scaler: float = 1.0,
    add_color: bool = False,
    twt_label_interval: float = None,
    kwargs_plot: dict = None,
):
    """Plot inset figure showing noise spike."""
    if kwargs_plot is None or kwargs_plot == {}:
        kwargs_plot = dict(c="black", alpha=1, lw=1)
    
    _dim = [d for d in data.dims if d != 'twt'][0]
    twt_slice = slice(None) if twt_slice is None else twt_slice
    trace_slice = slice(None) if trace_slice is None else trace_slice
    traces = data.sel({
        'twt': twt_slice,
        _dim: trace_slice,
    })
    nsamples, ntraces = traces.shape
    # print('nsamples, ntraces:', nsamples, ntraces)
    twt = traces['twt']
    dt = twt.attrs.get('dt', None)
    twt = twt.values
    # print('dt:', dt)
    
    traces = traces.to_numpy()
    
    y = np.arange(nsamples)
    for i in range(ntraces):
        ax.plot(traces[:, i] * scaler + (i + 1), y, **kwargs_plot)

    ax.set_ylim(0, nsamples)
    ax.invert_yaxis()
    
    if twt_label_interval is not None:
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(1, ntraces + 1)))
        ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(0, nsamples, int(twt_label_interval / dt))))
    yticks = ax.get_yticks()
    yticks = (yticks * dt + twt[0]).astype('int')
    ax.set_yticklabels(yticks)
    
    return ax


#%% MAIN
if __name__ == "__main__":

    dir_fig = (
        os.path.dirname(os.path.abspath(__file__))
        if SYSTEM == "WINDOWS" else "/mnt/d/scripts/packages/Warnke-et-al_2023_figures/Figure-11_results_transforms"
    )
    print(dir_fig)
    os.chdir(dir_fig)
    
    dpi = 300

    # === FREQUENCY ===
    cube_freq = xr.open_dataset("cube_center_IDW_env_5x5m_freq-0.075kHz.nc", engine="h5netcdf")  # cube_freq_0.075kHz.nc
    var_freq = "freq_env"
    x_freq = cube_freq[var_freq].load()

    # load fold into memory
    cube_freq["fold"].load()

    # create data mask from fold
    cube_freq["mask"] = cube_freq["fold"].where(cube_freq["fold"] <= 1, other=1)

    # === TIME ===
    cube_time = xr.open_dataset("cube_center_IDW_env_5x5m_twt-775ms.nc", engine="h5netcdf")  # cube_twt_775ms.nc
    var = "env"
    x_time = cube_time[var]
    shape = x_time.shape

    # load fold into memory
    cube_time["fold"].load()

    # create data mask from fold
    cube_time["mask"] = cube_time["fold"].where(cube_time["fold"] <= 1, other=1)
    
    #%% [TRACES] load data
    
    traces_time = xr.open_dataset("cube_time_selected_traces.nc", engine="h5netcdf")
    traces_freq = xr.open_dataset("cube_freq_selected_traces.nc", engine="h5netcdf")
    
    #%% initialize POCS parameter

    metadata = dict(
        niter=50,
        eps=1e-16,
        thresh_op="hard",  # hard, garrote, soft
        thresh_model="exponential-1",  # linear, exponential, data-driven, inverse-proportional
        decay_kind="values",
        p_max=0.99,  # max regularization percentage
        p_min=1e-4,  # 'adaptive' ,  # min regularization percentage 1e-4
        alpha=0.75,  # weighting factor
        sqrt_decay=False,  # apply np.sqrt to threshold decay
        version="fast",
        verbose=True,
        results_dict={},
    )

    # wavelet
    wavelet = "coif5"
    # shearlet
    Psi = FFST.scalesShearsAndSpectra(
        shape,
        numOfScales=None,
        realCoefficients=False if np.iscomplexobj(np.iscomplexobj(cube_freq[var_freq])) else True,
        fftshift_spectra=True,
    )
    # curvelet
    nbangles_coarse = 20  # default: 16
    allcurvelets = True
    DCTOp = curvelops.FDCT2D(
        shape, nbscales=None, nbangles_coarse=nbangles_coarse, allcurvelets=allcurvelets
    )

    transform_dict = {
        "FFT": (np.fft.fft2, np.fft.ifft2),
        "WAVELET": (
            partial(pywt.wavedec2, wavelet=wavelet, mode="smooth"),
            partial(pywt.waverec2, wavelet=wavelet, mode="smooth"),
        ),
        "SHEARLET": (FFST.shearletTransformSpect, FFST.inverseShearletTransformSpect),
        "CURVELET": (DCTOp.matvec, DCTOp.rmatvec),
    }

    #%% [POCS] transforms TIME

    # init outputs
    cubes_time = []
    metadata_time = []
    sharpness_time = {}

    for transform_name, transforms_op in transform_dict.items():

        # update metadata
        meta = metadata.copy()
        meta.update(
            transform_kind=transform_name,
            transform=transforms_op[0],
            itransform=transforms_op[1],
            path_results=os.path.join(dir_fig, f"TIME_{transform_name}_iterations.txt"),
        )

        aux = Psi if meta["transform_kind"] == "SHEARLET" else None
        input_core_dims = (
            [["iline", "xline"], ["iline", "xline"], ["iline", "xline", "lvl"]]
            if meta["transform_kind"] == "SHEARLET"
            else [["iline", "xline"], ["iline", "xline"], []]
        )
        cube_pocs_dask = (
            xr.apply_ufunc(
                POCS,
                x_time,
                cube_time["mask"],
                aux,
                input_core_dims=input_core_dims,
                output_core_dims=[["iline", "xline"]],
                vectorize=False,
                dask="parallelized",
                output_dtypes=[cube_time[var].dtype],
                kwargs=meta,
            )
            .to_dataset(name=f"{var}_interp")
            .compute()
        )
        # add fold DataArray
        cube_pocs_dask = cube_pocs_dask.assign(fold=cube_time.fold)

        for key in ["transform", "itransform"]:
            del meta[key]

        sharpness = sharpness_index(cube_pocs_dask[f"{var}_interp"].data)

        # add outputs to lists
        cubes_time.append(cube_pocs_dask)
        metadata_time.append(meta)
        sharpness_time[transform_name] = sharpness

    with open(os.path.join(dir_fig, "TIME_sharpness_indices.txt"), "w", newline="\n") as f:
        f.write("transform;sharpness_idx_perc" + "\n")
        for t, s in sharpness_time.items():
            f.write(f"{t};{s:02.2f}" + "\n")

    #%% [POCS] transforms FREQUENCY

    # init outputs
    cubes_freq = []
    metadata_freq = []
    sharpness_freq = {}

    for transform_name, transforms_op in transform_dict.items():

        # update metadata
        meta = metadata.copy()
        meta.update(
            transform_kind=transform_name,
            transform=transforms_op[0],
            itransform=transforms_op[1],
            path_results=os.path.join(dir_fig, f"FREQ_{transform_name}_iterations.txt"),
        )

        aux = Psi if meta["transform_kind"] == "SHEARLET" else None
        input_core_dims = (
            [["iline", "xline"], ["iline", "xline"], ["iline", "xline", "lvl"]]
            if meta["transform_kind"] == "SHEARLET"
            else [["iline", "xline"], ["iline", "xline"], []]
        )
        cube_pocs_dask = (
            xr.apply_ufunc(
                POCS,
                x_freq,
                cube_freq["mask"],
                aux,
                input_core_dims=input_core_dims,
                output_core_dims=[["iline", "xline"]],
                vectorize=False,
                dask="parallelized",
                output_dtypes=[cube_freq[var_freq].dtype],
                kwargs=meta,
            )
            .to_dataset(name=f"{var_freq}_interp")
            .compute()
        )
        # add fold DataArray
        cube_pocs_dask = cube_pocs_dask.assign(fold=cube_freq.fold)

        for key in ["transform", "itransform"]:
            del meta[key]

        sharpness = sharpness_index(np.abs(cube_pocs_dask[f"{var_freq}_interp"].data))

        # add outputs to lists
        cubes_freq.append(cube_pocs_dask)
        metadata_freq.append(meta)
        sharpness_freq[transform_name] = sharpness

    with open(os.path.join(dir_fig, "FREQ_sharpness_indices.txt"), "w", newline="\n") as f:
        f.write("transform;sharpness_idx_perc" + "\n")
        for t, s in sharpness_freq.items():
            f.write(f"{t};{s:02.2f}" + "\n")
            
    #%% [DEV]
    
    for cube_, name in zip(cubes_time, transform_dict.keys()):
        cube_.to_netcdf(os.path.join(dir_fig, f'{name}_time_interp.nc'), engine='h5netcdf')
    
    for cube_, name in zip(cubes_freq, transform_dict.keys()):
        cube_.to_netcdf(os.path.join(dir_fig, f'{name}_freq_interp.nc'), engine='h5netcdf', invalid_netcdf=True)
        
    import yaml
    
    for meta, name in zip(metadata_time, transform_dict.keys()):
        with open(os.path.join(dir_fig, f'{name}_time_metadata.yml'), 'w', newline='\n') as f:
            yaml.dump(meta, f)
            
    for meta, name in zip(metadata_freq, transform_dict.keys()):
        with open(os.path.join(dir_fig, f'{name}_freq_metadata.yml'), 'w', newline='\n') as f:
            yaml.dump(meta, f)
            
    with open(os.path.join(dir_fig, 'TIME_sharpness.yml'), 'w', newline='\n') as f:
        yaml.dump(sharpness_time, f)
        
    with open(os.path.join(dir_fig, 'FREQ_sharpness.yml'), 'w', newline='\n') as f:
        yaml.dump(sharpness_freq, f)
        
    #%% [DEV]
    import yaml
    
    names = ['FFT', 'WAVELET', 'SHEARLET', 'CURVELET']
    cubes_time = [
        xr.open_dataset(os.path.join(dir_fig, f'{name}_time_interp.nc'), engine='h5netcdf')
        for name in names
    ]
    
    cubes_freq = [
        xr.open_dataset(os.path.join(dir_fig, f'{name}_freq_interp.nc'), engine='h5netcdf')
        for name in names
    ]
    
    metadata_time = []
    metadata_freq = []
    for name in names:
        with open(os.path.join(dir_fig, f'{name}_time_metadata.yml'), 'r') as f:
            metadata_time.append(yaml.safe_load(f))
            
        with open(os.path.join(dir_fig, f'{name}_freq_metadata.yml'), 'r') as f:
            metadata_freq.append(yaml.safe_load(f))
    
    transform_dict = {
        "FFT": None,
        "WAVELET": None,
        "SHEARLET": None,
        "CURVELET": None,
    }

    #%% [PLOT] domains (time, freq)

    with mpl.rc_context({"font.family": "Arial", "mathtext.fontset": "stix"}):
        
        # fig = plt.figure(figsize=(5, 14))  # layout="constrained",
        # subfigs = fig.subfigures(2, 1, height_ratios=[4.5, 1], hspace=0.01)
        
        fig, axes = plt.subplots(
            5,
            2,
            layout="constrained",
            figsize=(5, 14.2),
            gridspec_kw=dict(hspace=0.02, wspace=0.02, height_ratios=(1, 1, 1, 1, 0.85)),
        )
        
        # subfigs[0].set_layout_engine(layout='constrained')
        # axes = subfigs[0].subplots(4, 2, gridspec_kw=dict(hspace=0.1, wspace=0.02))

        shrink = 0.85
        pad_cbar = 0.01
        kw_title = dict(fontsize=14, fontweight="semibold")

        for ax, name in zip(axes[:, 0], transform_dict.keys()):
            ax.text(
                -0.35,
                0.5,
                name,
                va="center",
                ha="center",
                rotation=90,
                rotation_mode="anchor",
                transform=ax.transAxes,
                **kw_title,
            )

        kw_title.update(pad=15)

        for i, (ax, label) in enumerate(
            zip(axes.ravel(), ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"])
        ):
            # if i % 2 == 1 and i < 8:
            #     kwargs = dict(color="white")
            # else:
            #     kwargs = dict(color="black")
            kwargs = dict(color="black")
            ax.text(
                -0.13,  # 0.1
                0.95,  # 0.93
                label,
                transform=ax.transAxes,
                backgroundcolor='white',
                va="center",
                ha="center",
                fontsize=18,
                fontweight="semibold",
                family="Times New Roman",
                **kwargs,
            )

        # ========== TIME ==========
        # axes_time = subfigs[0].subplots(4, 1, sharex=True, sharey=True,)
        # axes_time = axes_time.ravel(order='F')
        axes_time = axes[:, 0]

        for i, (cube_pocs_dask, meta) in enumerate(zip(cubes_time, metadata_time)):
            var = "env"
            kwargs_filt = dict(sigma=1, order=0)

            cube_pocs_dask[f"{var}_interp_filt"] = (
                ["iline", "xline"],
                gaussian_filter(cube_pocs_dask[f"{var}_interp"].data, **kwargs_filt),
            )

            im = cube_pocs_dask[f"{var}_interp"].T.plot(
                ax=axes_time[i],
                cmap="Greys",
                vmin=0,
                vmax=0.04,
                add_colorbar=False,
            )
            # cbar_kwargs=dict(extend='max', pad=0.02, label=''))
            axes_time[i].set_aspect("equal")
            axes_time[i].set_title("")
            axes_time[i].tick_params(
                axis="both", which="both", bottom=True, top=True, left=True, right=True
            )
            axes_time[i].set_ylabel("")

            axes_time[i].xaxis.set_label_position("top")
            if i > 0:
                axes_time[i].set_xlabel("")
                axes_time[i].tick_params(axis="x", labelbottom=False)
            else:
                axes_time[i].tick_params(axis="x", labelbottom=False, labeltop=True)

            loc_major = 100
            loc_minor = 25
            axes_time[i].xaxis.set_major_locator(mticker.MultipleLocator(loc_major))
            axes_time[i].xaxis.set_minor_locator(mticker.MultipleLocator(loc_minor))

            axes_time[i].yaxis.set_major_locator(mticker.MultipleLocator(loc_major))
            axes_time[i].yaxis.set_minor_locator(mticker.MultipleLocator(loc_minor))

        # title
        axes_time[0].set_title("Time domain", **kw_title)

        # colorbar
        cbar = fig.colorbar(
            im,
            ax=axes_time[:-1].tolist(),
            shrink=shrink,
            pad=pad_cbar,
            orientation="horizontal",
            location="bottom",
            extend="max",
            label="Envelope",
        )
        
        # ========== FREQUENCY ==========
        # axes_freq = subfigs[1].subplots(4, 2, sharex=True, sharey=True,
        #                                 gridspec_kw=dict(hspace=0.06))
        axes_freq = axes[:, 1]

        for i, (cube_pocs_dask, meta) in enumerate(zip(cubes_freq, metadata_freq)):
            var = "freq_env"
            kwargs_filt = dict(sigma=1, order=0)

            cube_pocs_dask[f"{var}_interp_filt"] = (
                ["iline", "xline"],
                (
                    gaussian_filter(cube_pocs_dask[f"{var}_interp"].data.real, **kwargs_filt)
                    + 1j
                    * gaussian_filter(cube_pocs_dask[f"{var}_interp"].data.imag, **kwargs_filt)
                ),
            )

            im = xr.apply_ufunc(np.abs, cube_pocs_dask[f"{var}_interp"]).T.plot(
                ax=axes_freq[i],
                cmap="inferno",
                vmin=0,
                vmax=0.5,
                add_colorbar=False,
            )

            axes_freq[i].set_aspect("equal")
            axes_freq[i].set_title("")
            axes_freq[i].tick_params(
                axis="both", which="both", bottom=True, top=True, left=True, right=True
            )
            # axes_freq[i].set_ylabel('')
            axes_freq[i].yaxis.set_label_position("right")
            axes_freq[i].tick_params(axis="y", labelleft=False, labelright=True)
            # if i < 3:
            #     axes_freq[i].set_xlabel('')
            #     axes_freq[i].tick_params(axis='x', labelbottom=False)

            axes_freq[i].xaxis.set_label_position("top")
            if i > 0:
                axes_freq[i].set_xlabel("")
                axes_freq[i].tick_params(axis="x", labelbottom=False)
            else:
                axes_freq[i].tick_params(axis="x", labelbottom=False, labeltop=True)

            loc_major = 100
            loc_minor = 25
            axes_freq[i].xaxis.set_major_locator(mticker.MultipleLocator(loc_major))
            axes_freq[i].xaxis.set_minor_locator(mticker.MultipleLocator(loc_minor))

            axes_freq[i].yaxis.set_major_locator(mticker.MultipleLocator(loc_major))
            axes_freq[i].yaxis.set_minor_locator(mticker.MultipleLocator(loc_minor))

        # title
        axes_freq[0].set_title("Frequency domain", **kw_title)

        # colorbar
        cbar = fig.colorbar(
            im,
            ax=axes_freq[:-1].tolist(),
            shrink=shrink,
            pad=pad_cbar,
            orientation="horizontal",
            location="bottom",
            extend="max",
            label="Amplitude",
        )
        
        # =====================================================================================
        axes_wiggle = axes[-1, :]
        # # axes_wiggle_legend = axes[-2, :]
        # gs = axes[-2, 0].get_gridspec()
        # # remove the underlying axes
        # for ax in axes[-2, :]:
        #     ax.remove()
        # axes_wiggle_legend = fig.add_subplot(gs[-2, :])
        
        slice_twt = slice(750, 765)
        
        kwargs_wiggle = dict(
            scaler=3,
            twt_slice=None,
            trace_slice=None,
            twt_label_interval=5,
        )
        colors_wiggles = ['black', 'blue', 'green', 'red']
        
        for i, _transform in enumerate(list(traces_time['transform'].values)):
            label = _transform.capitalize() if _transform != 'FFT' else _transform
            plot_inset_wiggle(
                axes_wiggle[0],
                traces_time['env_interp'].sel(transform=_transform, twt=slice_twt),
                kwargs_plot=dict(lw=1, c=colors_wiggles[i], label=label),
                **kwargs_wiggle
            )
        
        for i, _transform in enumerate(list(traces_freq['transform'].values)):
            label = _transform.lower() if _transform != 'FFT' else _transform
            plot_inset_wiggle(
                axes_wiggle[1],
                traces_freq['env'].sel(transform=_transform, twt=slice_twt),
                kwargs_plot=dict(lw=1, c=colors_wiggles[i], label=label),
                **kwargs_wiggle
            )
                    
        axes_wiggle[0].set_ylabel('TWT (ms)')
        axes_wiggle[1].tick_params(axis="y", which="both", labelleft=False)
        for axis in axes_wiggle:
            axis.spines[['right', 'top', 'bottom']].set_visible(False)
            axis.tick_params(
                axis="x", which="both", bottom=False, labelbottom=True
            )
            axis.set_xticklabels(['R', 'O', 'R'], fontweight='bold')
            
        for i, (ax, label) in enumerate(zip(axes_wiggle.ravel(), ["i)", "j)"])):
            ax.text(
                -0.15 + i * 0.04,
                0.95,
                label,
                transform=ax.transAxes,
                va="center",
                ha="center",
                fontsize=18,
                fontweight="semibold",
                family="Times New Roman",
                color="black",
                bbox=dict(facecolor='white', edgecolor='white', pad=10.0) if "i" in label else None
            )
        
        kwargs_legend = dict(
            fontsize=11,
            # handlelength=2,
            # handleheight=3,
            loc='outside lower center',
            ncols=4,
            mode='expand',
            frameon=False,
            # bbox_to_anchor=(0.5, 0.0),
            # bbox_transform=axes_wiggle_legend.transAxes
        )
            
        handles, labels = axes_wiggle[0].get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        legend = fig.legend(*zip(*unique), **kwargs_legend)
        # legend = axes_wiggle[0].legend(bbox_to_anchor=(0.5, -0.10), loc='lower center',
        #                                bbox_transform=fig.transFigure, **kwargs_legend)
        for line in legend.get_lines():
            line.set_linewidth(2.5)
        
        #%% save figure
        figure_number = 11
        plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_transforms_{dpi}dpi.png"), dpi=dpi)  # , bbox_inches="tight"
        # plt.savefig(os.path.join(dir_fig, f"Figure-{figure_number:02d}_transforms.tiff"), dpi=dpi)
