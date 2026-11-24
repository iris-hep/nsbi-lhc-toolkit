# Model Building and Fitting Example

This standalone example demonstrates how to build an unbinned statistical model and perform hypothesis testing using pre-computed density ratios. It is **independent** of the training pipeline — you only need:

1. A fit configuration YAML file (`config_fit_nsbi.yml`)
2. Pre-computed density ratio `.npy` files (nominal + systematic variations)
3. Observed data: a ROOT tree flagged `Data: True`, plus the per-event weights of its events in the unbinned region (this example uses the Asimov dataset as its observation)
4. Input ROOT files (for binned histogram channels)

## Quick Start

```python
from nsbi_common_utils import workspace_builder, models, inference

# 1. Build workspace from config
ws = workspace_builder.WorkspaceBuilder(config_path="config_fit_nsbi.yml").build()

# 2. Initialize the statistical model (JAX JIT-compiled NLL)
model = models.sbi_parametric_model(workspace=ws, measurement_to_fit="my_measurement")

# 3. Fit
params, init_vals = model.get_model_parameters()
fitter = inference.inference(
    model_nll=model.model,
    initial_values=init_vals,
    list_parameters=params,
    num_unconstrained_params=model.num_unconstrained_param
)
fitter.perform_fit()
```

## What you need to provide

### Fit configuration (`config_fit_nsbi.yml`)

This YAML file defines:
- **Measurement**: which parameters to fit and the parameter of interest (POI)
- **Samples**: the physics processes (e.g. signal, backgrounds) with paths to ROOT files, and the observed dataset flagged `Data: True`
- **NormFactors**: free normalization parameters per sample
- **Systematics**: nuisance parameters with paths to up/down variation ROOT files
- **Regions**: analysis regions — binned (control regions) and unbinned (signal region with density ratios)
- **TrainedModels**: pre-computed density ratios and event weights for each unbinned region

```yaml
TrainedModels:
- RegionName: SR
  Weights: <path/to>/weights.npy
  Models:
    - Name: signal
      Nominal:
        Ratios: <path/to>/ratio_signal.npy
      Systematics:
        - SystName: JES
          RatiosUp: <path/to>/ratio_signal_JES_Up.npy
          RatiosDn: <path/to>/ratio_signal_JES_Dn.npy
```

### Density ratio files

Each `.npy` file is a 1D array of shape `(n_events,)` containing the ensemble-averaged density ratio `p_c(x) / p_ref(x)` for each event in the dataset. These are produced by the training + evaluation pipeline (Stages 1-3), but can also be provided from any external source.

### Data directory

All paths in the configs are written as `${NSBI_DATA_DIR:-./saved_datasets}`. Set `NSBI_DATA_DIR` to the same data directory the `FAIR_universe_Higgs_tautau` example uses, which holds its pipeline outputs (or the downloaded bundle):

```shell
export NSBI_DATA_DIR=/path/to/saved_datasets
```

Expected contents:

```
saved_datasets/
  asimov_weights.npy         # Per-event weights for the unbinned region
  dataset_with_obs.root      # Observed data (here: the Asimov dataset)
  dataset_nominal.root       # Nominal MC (for binned channels)
  dataset_JES_up.root        # Systematic variation ROOT files
  ...
  output_training_nominal/
    output_ratios_<sample>/
      ratio_<sample>.npy     # Nominal density ratios per sample
  output_training_systematics/
    output_ratios_<sample>_<syst>_<dir>/
      ratio_<sample>.npy     # Systematic density ratios
```

## Notebooks

- `1_workspace_building.ipynb` builds the NSBI and histogram workspaces from the config files and saves them as JSON under `workspaces/`.
- `2_parameter_fitting.ipynb` loads the workspaces, initializes `sbi_parametric_model` for each, minimizes the NLL via `iminuit`, runs profile likelihood scans and compares NSBI vs histogram-only sensitivity.
