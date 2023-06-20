# Warnke et al. (2023) _GEOPHYSICS_ - Reproducible figures

This repository provides the Python code and the data requried to reproduce (most) figures of the publication:

```
Warnke et al. (2023) Pseudo-3D cubes from densely spaced subbottom profiles via Projection Onto Convex Sets interpolation: an open-source workflow applied to a pockmark field. GEOPHYSICS
```

> **Note**
> 
> **Figures 5, 8, 9 and 13** can be reproduced using only the included ASCII auxiliary files, while **Figures 10, 11, 12, 14, and 15** depend on additional binary netCDF files. **Figure 1** was created using QGIS and **Figures 2, 3, and 7** were designed using Inkscape.

> **Warning**
> 
> The source data for **Figures 4, 6, and 16** are not yet available as the corresponding TOPAS subbottom profiler dataset will not be made publicly available until after an **initial moratorium**.

## Installation

In order to create the figures using the included code, please download and unzip the repository locally.

### [Optional] Separate dependecy installation

> **Warning**
> 
> We highly recommend using some kind of **virtual environment** when installing the required dependencies!
> Potential options are `conda`/`mamba` (*our recommendation*) or `virtualenv` ([more information](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)).

The required dependencies can be installed using `conda`:

```bash
>>> conda install -f environment.yml            # minimal dependencies for Figures 8, 9, 13
>>> conda install -f environment_data.yml       # additional dependencies for Figures 3, 4, 6, 10, 12-15
>>> conda install -f environment_tide.yml       # additional dependencies for Figure 5
>>> conda install -f environment_transforms.yml # additional dependencies for Figure 11
>>> conda install -f environment_3D_viz.yml     # additional dependencies for Figure 16
>>> conda install -f environment_complete.yml   # complete dependencies

>>> conda activate figures  # activate new env (default name: "figures")
```

or `pip`:

```bash
>>> pip install -r requirements.txt            # minimal dependencies for Figures 8, 9, 13
>>> pip install -r requirements_data.txt       # additional dependencies for Figures 3, 4, 6, 10, 12, 14, 15
>>> pip install -r requirements_tide.txt       # additional dependencies for Figure 5
>>> pip install -r requirements_transforms.txt # additional dependencies for Figure 11
>>> pip install -r requirements_3D_viz.txt     # additional dependencies for Figure 16
>>> pip install -r requirements_complete.txt   # complete dependencies
```

### Using `setup.py`

All dependencies required to reproduce the figures in this repository can be installed via:

```bash
>>> cd ./Warnke-et-al_2023_GEOPHYSICS_figures   # navigate into unzipped folder
>>> pip install .              # minimal dependencies for Figures 8, 9, 13
>>> pip install ".[data]"      # dditional dependencies for Figures 3, 4, 6, 10, 12, 14, 15
>>> pip install ".[tide]"       # additional dependencies for Figure 5
>>> pip install ".[transforms]" # additional dependencies for Figure 11
>>> pip install ".[3d]"         # additional dependencies for Figure 16
>>> pip install ".[complete]"   # complete dependencies
```

## Producing figures

> **Note**
> 
> Figures 8 and 9 can be reproduced using the included auxiliary data `TOPAS_metadata.aux`.
> 
> Figure 5 requires the additional installation of _tpxo-tide-prediction_ and its dependencies!

To reproduce **Figures 8, 9 and 13** run the following code with your activated virtual environment (here `figures`):

```bash
(figures) >>> python ./Figure-08_methods_resample_env/figure_methods_resample_envelope.py
(figures) >>> python ./Figure-09_results_vertical_offsets/figure_results_vertical_offsets.py
(figures) >>> python ./Figure-13_results_niterations/figure_results_niterations.py
```

To reproduce **Figure 5** run (requires additional dependencies!):

```bash
(figures) >>> python ./Figure-05_methods_tide_compensation/figure_methods_tide_compensation.py {path/to/TPXO9_atlas/files.nc}
```
