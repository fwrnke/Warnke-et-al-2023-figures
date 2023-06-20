"""
Create Figure showing 3D view of sparse/interpolated cube (Warnke et al., 2023).

@author: fwrnke
@email:  fwar378@aucklanduni.ac.nz
@date:   2022-11-17

"""
import os
import sys

import numpy as np
import xarray as xr
import pyvista as pv
import rioxarray
import pyproj
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions_figures import (
    depth2twt,
    _print_info_TOPAS_moratorium
)

_print_info_TOPAS_moratorium()

#%% FUNCTIONS

def check_crs(crs_ref, crs_alt, x, y, verbose: bool = False):  # noqa
    if crs_ref.equals(crs_alt):
        if verbose:
            print("[INFO]    Identical Coordinate Reference Systems. Nothing to worry about.")
        # create X and Y DataArrays to avoid matrix sampling
        x = xr.DataArray(data=x, dims="pos")
        y = xr.DataArray(data=y, dims="pos")
    else:
        if verbose:
            print(
                "[WARNING]    Different Coordinate Reference Systems!",
                "Transformning sampling locations to bathymetry CRS.",
            )
        transformer = pyproj.Transformer.from_crs(crs_ref, crs_alt, always_xy=False)
        xx, yy = transformer.transform(x, y)
        # create X and Y DataArrays to avoid matrix sampling
        x = xr.DataArray(data=xx, dims="pos")
        y = xr.DataArray(data=yy, dims="pos")

    return x, y


def sample_seafloor_horizon_from_bathymetry(
    path_bathy,
    cube: xr.Dataset = None,
    x=None,
    y=None,
    crs_cube=None,
    crs_coords=None,
    return_bathy: bool = False,
    verbose: bool = False,
):  # noqa

    if (cube is None) and (x is None and y is None):
        print("[WARNING]    Either `cube` or `x` and `y` have to be provided!")
        return None

    # check which input
    if cube is not None:
        use_cube = True
        crs_cube_internal = cube.attrs.get("spatial_ref", None)
    else:
        use_cube = False

    # check cube CRS
    if use_cube and crs_cube_internal is not None:
        crs_cube = pyproj.crs.CRS(crs_cube_internal)
    elif use_cube and crs_cube_internal is None and crs_cube is not None:
        crs_cube = pyproj.crs.CRS(crs_cube)
    elif use_cube and crs_cube_internal is None and crs_cube is None:
        print("[WARNING]    No CRS found in cube and no alternative specified. Check your input!")
        return None

    if not use_cube and (x is not None and y is not None):
        crs_coords = pyproj.crs.CRS(crs_coords)

    # bathymetry
    bathy = rioxarray.open_rasterio(path_bathy)
    crs_bathy = bathy.rio.crs
    bathy_xsize = np.diff(bathy.x.values).mean()
    bathy_ysize = np.diff(bathy.y.values).mean()
    bathy_unit = "m" if crs_bathy.is_projected else "deg"
    if verbose:
        print("[INFO]    MBES bathymetry GeoTIFF")
        print(f"[INFO]    CRS:      {crs_bathy.to_string()}")
        print(f"[INFO]    bin size: {bathy_xsize} {bathy_unit} x {bathy_ysize} {bathy_unit}")

    if use_cube:
        x_bins = cube["x"].values.flatten()
        y_bins = cube["y"].values.flatten()

        x_pos, y_pos = check_crs(crs_cube, crs_bathy, x_bins, y_bins, verbose=True)
    else:
        x_pos, y_pos = check_crs(crs_coords, crs_bathy, x, y, verbose=True)

    if verbose:
        print("[INFO]    Sampling MBES bathymetry at bin center locations")
    depths_sampled = bathy.sel(x=x_pos, y=y_pos, method="nearest").values

    if use_cube:
        depths_sampled = depths_sampled.reshape(cube["x"].shape)

    if return_bathy:
        return (depths_sampled, bathy)

    return depths_sampled


def get_data_meshes(
    cube_sel: xr.Dataset,
    var: str = 'env',
    subsample_factor: int = 1,
    da_mbes: xr.DataArray = None,
    grid_mbes=None,
    clip_volume: bool = True,
    clip_kwargs: dict = None,
    other=np.nan,
):
    """Extract data mesh from xarray.Dataset."""
    # transpose for 3D visualization
    cube_data = cube_sel[var].transpose('iline', 'xline', 'twt')

    # clip water column using sampled bathymetry
    if da_mbes is not None:
        cube_data = cube_data.where(cube_data.twt > da_mbes * -1, other=other)

    # clip box from cube
    if clip_volume:
        # define maximum inline/xline/twt values of clipped box
        if clip_kwargs is None:
            clip_kwargs = dict(iline=860, xline=790, twt=795)
        il_max = clip_kwargs.get(
            'iline', cube_sel['iline'].isel(dict(iline=cube_sel['iline'].size // 2))
        )
        xl_max = clip_kwargs.get(
            'xline', cube_sel['xline'].isel(dict(xline=cube_sel['xline'].size // 2))
        )
        twt_max = clip_kwargs.get('twt', cube_sel['twt'].isel(dict(twt=cube_sel['twt'].size // 2)))

        # define masks
        clip_il = cube_sel.iline < il_max
        clip_xl = cube_sel.xline < xl_max
        clip_twt = cube_sel.twt < twt_max
        # apply mask
        cube_data = cube_data.where(~(clip_il & clip_xl & clip_twt), other=other)

        # load clipped subset seperately
        cube_subset = cube_data.sel(
            iline=slice(cube_sel.iline.data.min(), il_max),
            xline=slice(cube_sel.xline.data.min(), xl_max),
            twt=slice(cube_sel.twt.data.min(), twt_max),
        )
        # reverse time axis & load data into RAM
        cube_subset_data = cube_subset.data[..., ::-1].compute()

    # subsample data for easier visulization
    if subsample_factor > 1:
        data = cube_data.data[..., ::subsample_factor]
        if CLIP_BOX:
            cube_subset_data = cube_subset_data[..., ::subsample_factor]
            # cube_subset_data = cube_subset[..., ::subsample_factor]
    else:
        data = cube_data.data

    # reverse time axis & load data into RAM
    data = data[..., ::-1].compute()
    # cube_subset_data = cube_subset.data[..., ::-1].compute()

    # clip seafloor surface
    if grid_mbes is not None and clip_volume:
        bounds = grid_mbes.bounds
        bound_il = float(il_max)
        bound_xl = float(xl_max)
        grid_mbes_clip = grid_mbes.clip_box(
            bounds=(bounds[0], bound_il, bounds[2], bound_xl, bounds[4], bounds[5])
        )
    else:
        grid_mbes_clip = grid_mbes

    # [CUBE] setup grid
    dt = cube_sel.coords['twt'].attrs.get(
        'dt',
        float(f'{float(cube_sel.twt[1].astype("float32") - cube_sel.twt[0].astype("float32")):g}'),
    )

    spacing = (1, 1, dt * subsample_factor)
    dims = np.asarray(data.shape) + 1
    origin = (float(cube_sel.iline[0]), float(cube_sel.xline[0]), float(cube_sel.twt[-1] * -1))

    grid = pv.UniformGrid(
        dimensions=dims,  # (424,  570, 1352)
        spacing=spacing,  # (1, 1, dt * subsample_factor)
        origin=origin,
    )  # (605, 463, 740)

    grid.cell_data[var] = data.reshape(grid.n_cells, order='F')  # [...,::-1]

    # [CUBE] setup subset
    dims_subset = np.asarray(cube_subset_data.shape) + 1
    origin_subset = (
        float(cube_subset.iline[0]),
        float(cube_subset.xline[0]),
        float(cube_subset.twt[-1] * -1),
    )  # * -1
    grid_subset = pv.UniformGrid(dims=dims_subset, spacing=spacing, origin=origin_subset)
    grid_subset.cell_data[var] = cube_subset_data.reshape(grid_subset.n_cells, order='F')

    return grid, grid_subset, grid_mbes_clip


#%% MAIN

if __name__ == '__main__':

    dir_work = r'E:\PhD\processing\TOPAS\TAN2006\pockmarks_3D\cube_center'
    dir_fig = os.path.dirname(os.path.abspath(__file__))

    path_bathy = 'E:/PhD/processing/gis/mbes/TAN2006_pseudo-3D_high-res_5m_CUBE_smoothed-3x3.tif'

    fig_name = os.path.basename(dir_fig)
    subset = 'cube_center'
    var = 'env'

    # general settings
    subsample_factor = 4
    BATHY = True
    BATHY_CLIP = True
    CLIP_BOX = True

    # (1) sparse cube
    file = 'cube_center_IDW_env_5x5m_0+05ms_twt-il-xl_preproc.nc'  # sparse

    # open cube
    path_cube = os.path.join(dir_work, file)
    cube = xr.open_dataset(
        path_cube, chunks={'twt': 1, 'iline': -1, 'xline': -1}, engine='h5netcdf'
    )
    cube['twt'] = np.around(cube['twt'].astype('float64'), 3)

    # restrict cube
    slice_twt = slice(740, 875)
    slice_iline = slice(cube.iline.min() + 5, cube.iline.max())  # - 5)  # indices!
    slice_xline = slice(cube.xline.min() + 5, cube.xline.max())  # - 5)  # indices!
    cube_sel = cube.sel(twt=slice_twt, iline=slice_iline, xline=slice_xline)

    # (2) interpolated cube
    file = (
        'cube_center_IDW_env_5x5m_0+05ms_twt-il-xl_preproc_up-2-trunc_FFT_hard_niter-50_gaussian-1_interp-freq.nc'
    )

    # open cube
    path_cube_interp = os.path.join(dir_work, file)
    cube_interp = xr.open_dataset(
        path_cube_interp, chunks={'twt': 1, 'iline': -1, 'xline': -1}, engine='h5netcdf'
    )
    cube_interp['twt'] = np.around(cube_interp['twt'].astype('float64'), 3)

    # restrict cube
    slice_twt = slice(740, 875)
    slice_iline = slice(cube_interp.iline.min() + 5, cube_interp.iline.max())  # - 5)  # indices!
    slice_xline = slice(cube_interp.xline.min() + 5, cube_interp.xline.max())  # - 5)  # indices!
    cube_interp_sel = cube_interp.sel(twt=slice_twt, iline=slice_iline, xline=slice_xline)

    #%% [SEAFLOOR] sample bathymetry

    vel = 1515

    if BATHY:
        depth_sampled, bathy = sample_seafloor_horizon_from_bathymetry(
            path_bathy, cube=cube_sel, return_bathy=True, verbose=True
        )
        depth_sampled_twt = depth2twt(depth_sampled, v=vel, units='ms')  # * 1000  # to ms

        # smooth sampled bathymetry
        depth_sampled_twt_smooth = gaussian_filter(depth_sampled_twt, sigma=1)
        depth_sampled_twt_smooth_da = xr.DataArray(
            depth_sampled_twt_smooth, coords={'iline': cube_sel.iline, 'xline': cube_sel.xline}
        )

        # create mesh
        xx, yy = np.meshgrid(cube_sel['iline'].values, cube_sel['xline'].values)
        grid_seafloor = pv.StructuredGrid(
            xx.astype('float32'),
            yy.astype('float32'),
            np.flipud(np.rot90(depth_sampled_twt_smooth)).astype('float32')
            # xx.T.astype('float32'), yy.T.astype('float32'), np.fliplr(depth_sampled_twt).astype('float32')
        )
        grid_seafloor_bathy = grid_seafloor.elevation()

    #%% [CUBE] Load SPARSE data

    grid, grid_subset, grid_mbes = get_data_meshes(
        cube_sel,
        var=var,
        subsample_factor=subsample_factor,
        da_mbes=depth_sampled_twt_smooth_da,
        grid_mbes=grid_seafloor_bathy,
        clip_volume=CLIP_BOX,
    )

    #%% [CUBE] Load INTERPOLATED data

    grid_interp, grid_subset_interp, grid_mbes_interp = get_data_meshes(
        cube_interp_sel,
        var=var,
        subsample_factor=subsample_factor,
        da_mbes=depth_sampled_twt_smooth_da,
        grid_mbes=grid_seafloor_bathy,
        clip_volume=CLIP_BOX,
    )

    #%% [PLOTTING]

    # MESH: grid, grid_subset, grid_seafloor

    # settings
    # KIND = 'volume'  # 'slices', 'volume'
    SCREENSHOT = True
    scaler = 2 if SCREENSHOT else 1
    window_size = (np.array([1024, 768]) * np.array([2 * scaler, 1 * scaler])).astype('int16')
    show_scalar_bar = True
    show_axes_bounds = True
    show_titles = False
    clims_topas = {
        'env': [0, 0.04],
        'env_balanced': [0, 2],
    }
    clims_mbes = {
        'cube_center': [-775, -745],
        'cube_east': [-770, -740],
    }
    if 'env' in file:
        _type = 'env_balanced' if 'balanced' in file else 'env'
    else:
        _type = 'amp_balanced' if 'balanced' in file else 'amp'
    clim = clims_topas.get(_type, None)

    kw_labels = dict(position='upper_left', font_size=20 * scaler, font='arial')
    kw_axes = dict(xlabel='iline', ylabel='xline', zlabel='twt')
    kw_axes_labels = dict(
        xlabel='iline',
        ylabel='xline',
        zlabel='twt',
        font_size=20 * scaler,
        bold=False,
        location='outer',
        ticks='outside',
    )

    # set theme
    theme = pv.themes.DocumentTheme()
    theme.font.title_size = 24 * scaler
    theme.font.label_size = 20 * scaler
    theme.font.family = 'arial'
    pv.set_plot_theme(theme)

    pos = [(181.03861151746042, 5.205647267208329, -1336.4028545201932),
           (784.5696837344781, 704.8474192977201, -1666.294778045524),
           (0.22515196795109613, 0.24982713361623174, 0.9417499639696547)]
    zoom = 0.9

    # create plotter
    p = pv.Plotter(shape=(1, 2), border=False, off_screen=SCREENSHOT, window_size=list(window_size))

    # ========== SPARSE ==========
    p.subplot(0, 0)

    # add CUBE
    p.add_mesh(
        grid,
        cmap='Greys',
        clim=clim,
        culling='back',
        nan_opacity=0,
        nan_color=pv.Color('black', opacity=0.0),
        show_scalar_bar=False,
        scalar_bar_args=dict(
            n_labels=4, width=0.3, title='envelope', fmt='%.2f', position_x=0.05, position_y=0.05
        ),
    )
    # add CUBE SUBSET
    p.add_mesh(
        grid_subset,
        cmap='Greys',
        clim=clim,
        nan_opacity=0,
        nan_color=pv.Color('black', opacity=0.0),
        show_scalar_bar=False,
    )

    # add SEAFLOOR SURFACE
    if BATHY:
        p.add_mesh(
            grid_mbes,
            cmap='viridis',
            clim=clims_mbes.get(subset, [-765, -745]),
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args=dict(
                n_labels=5,
                width=0.3,
                title='seafloor depth (ms)',
                fmt='%.0f',
                position_x=0.65,
                position_y=0.05,
            ),
        )

    # add AUX STUFF
    if show_axes_bounds:
        # p.add_axes(**kw_axes)
        p.show_bounds(**kw_axes_labels)

    # set perspective (x: iline, y: xline, z: time)
    p.set_scale(zscale=2)
    p.camera_position = pos
    p.camera.zoom(zoom)

    # p.add_title('sparse cube')
    p.add_text('a)', **kw_labels)

    # ========== INTERPOLATED ==========
    p.subplot(0, 1)
    # add CUBE
    p.add_mesh(
        grid_interp,
        cmap='Greys',
        clim=clim,
        culling='back',
        nan_opacity=0,
        nan_color=pv.Color('black', opacity=0.0),
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=dict(
            n_labels=4, width=0.3, title='envelope', fmt='%.2f', position_x=0.05, position_y=0.05
        ),
    )
    # add CUBE SUBSET
    p.add_mesh(
        grid_subset_interp,
        cmap='Greys',
        clim=clim,
        nan_opacity=0,
        nan_color=pv.Color('black', opacity=0.0),
        show_scalar_bar=False,
    )

    # add SEAFLOOR SURFACE
    if BATHY:
        p.add_mesh(
            grid_mbes_interp,
            cmap='viridis',
            clim=clims_mbes.get(subset, [-765, -745]),
            show_scalar_bar=False,
            scalar_bar_args=dict(
                n_labels=5,
                width=0.3,
                title='seafloor depth (ms)',
                fmt='%.0f',
                position_x=0.65,
                position_y=0.05,
            ),
        )

    # add AUX STUFF
    if show_axes_bounds:
        p.show_bounds(**kw_axes_labels)

    # set perspective (x: iline, y: xline, z: time)
    p.set_scale(zscale=2)
    p.camera_position = pos
    p.camera.zoom(zoom)

    p.add_text('b)', **kw_labels)

    if SCREENSHOT:
        # save screenshot
        suffix = '_with_cmap' if show_scalar_bar else ''
        suffix += '_with_axes' if show_axes_bounds else ''
        p.screenshot(
            os.path.join(dir_fig, f'figure_results_sparse_interp_3D{suffix}.png'),
            transparent_background=False,
            return_img=False,
        )
    else:
        p.show(full_screen=False)
        
    print(p.camera_position)
