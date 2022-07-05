# iso_grid_optimizer
Geometry optimization loops for H-bar and isotropic solar cell front electrodes with central sink and circular outer envelope.

This repository contains companion code necessary to replicate experiments and figures presented in:
Nakano-Baker, O., Boyd, C., Cramer, C., Brush, L. & MacKenzie, J. D. Isotropic Grids Revisited: A Numerical Study of Solar Cell Electrode Geometries. IEEE Trans. Electron Devices 69, 3783–3790 (2022). DOI: 10.1109/TED.2022.3174810


# Usage
## Notebooks
Included Jupyter notebooks can be used to recreate publication figures (IsoGrid Figures.ipynb) or explore the relationship between multiple layers of isotropic grid patterns (angled_isogrids_plots.ipynb)

## Experiment
To run the numerical experiments used to generate these figures, run the script ./experiments/run_numerical_circle.py as:

> python run_numerical_circle.py --recipe_file '../recipes/center_circle.csv' --log_dir './my_log_dir/'

Material parameters can be adjusted by changing the target csv recipe file.

## Just the models
To query model directly in Python, invoke the functions defined in ./gridgraph/numerical_circle_cell.py. For H-bar and isotropic grids respectively, the functions perform numerical simulation of the power output of a circular solar cell with optimal line thickness and pitch under the provided params.

simulate_H(params, force=None)

simulate_iso(params, force=None)

    params - dictionary of material and simulation parameters
    force - optional tuple containing (width, pitch). If provided, no
    optimization is performed, instead the power output is estimated at the
    provided grid geometry.

    returns - Power [W], width[cm], pitch[cm]
