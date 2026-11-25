# Hello world: Gaussian toy example

A minimal end-to-end NSBI example on a five-dimensional Gaussian toy dataset: train two density-ratio networks (signal vs reference, background vs reference), build an unbinned workspace by hand and fit the signal strength.

Generate the datasets with `python generate_distributions.py` (2M background and 1M signal events per node by default; use `--n_bkg` / `--n_sig` to adjust). Parquets land in `./dataframes/`, and per-event weights are auto-scaled so the total yields stay fixed regardless of `--n_bkg` / `--n_sig`. Alternatively, download pre-generated datasets from https://cernbox.cern.ch/s/xRPGQluLZJij1vp and set `BASE_PATH` in each notebook.

The generating script has been developed by Malin Horstmann and slightly adapted to this example.

Notebooks
---

1. `1_visualize_data.ipynb` - look at the signal and background distributions.
2. `2a_SigvsRef_training.ipynb` - train the signal vs reference density ratio and run the calibration and reweighting closure tests. The reference is an equal mixture of signal and background.
3. `2b_BkgvsRef_training.ipynb` - the same for background vs reference.
4. `3_parameter_fitting.ipynb` - evaluate the trained networks on the observed (Asimov) dataset, write the workspace with its `observations` block and fit the signal strength.

Installation
---

- Clone the GitHub repository locally using `git clone git@github.com:iris-hep/nsbi-lhc-toolkit.git --depth=1 && cd nsbi-lhc-toolkit`.
- Run `pixi install -e nsbi-env` if you are using CPU or Mac and `pixi install -e nsbi-env-gpu` if you have access to CUDA-supported GPU.
- Install the kernel using `pixi run -e nsbi-env-gpu python -m ipykernel install --user --name nsbi-env-gpu --display-name "Python (pixi: nsbi-env)"` if you are running on GPU or `pixi run -e nsbi-env python -m ipykernel install --user --name nsbi-env --display-name "Python (pixi: nsbi-env)"` if you are running on CPU or Mac.
- Go to `examples/helloworld_gaussian_toy/` and start running the notebooks. Make sure to select the kernel `Python (pixi: nsbi-env)` before you run.
