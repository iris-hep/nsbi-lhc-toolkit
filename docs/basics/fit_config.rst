Writing a Fit Configuration
===========================

The fit configuration YAML file is the central input to the entire SBI workflow. It defines the measurement, samples, systematic uncertainties, analysis regions, training features, and pointers to trained density-ratio models. The :class:`~nsbi_common_utils.configuration.ConfigManager` reads this file and exposes its contents through a set of accessor methods that are consumed at every stage of the pipeline:

- **Data preparation** — the ``datasets`` module reads sample paths, ROOT tree names, weight branches, and systematic variation files from the config to load and organise the data.
- **Preselection training** — training features and feature scaling lists are retrieved via ``ConfigManager.get_training_features()``.
- **Density-ratio training** — basis processes (``get_basis_samples()``), the reference hypothesis (``get_reference_samples()``), training features, and region filters (``get_channel_filters()``) are all read from the config.
- **Systematic uncertainty training** — the ``Systematics`` block and ``get_samples_in_syst_for_training()`` determine which processes are affected by each variation and where the varied ROOT files are located.
- **Evaluation** — Asimov weight paths (``get_channel_asimov_weight_path()``) and trained model metadata come from the config.
- **Workspace building** — the :class:`~nsbi_common_utils.workspace_builder.WorkspaceBuilder` assembles the full statistical model from all of the above, producing a workspace dictionary that the statistical model consumes.

This page is the canonical reference for the YAML format. For the Python API, see :doc:`/api/workspace_builder`. For a hands-on walkthrough, see :doc:`model_building_example`.

Measurement and parameters
--------------------------

The ``General`` block names the measurement and declares which parameters enter the fit:

.. code-block:: yaml

   General:
     Measurement:
       Name: higgs_tautau_signal_strength
       POI: mu_htautau
       ParametersToFit:
         - mu_htautau
         - mu_ztautau
         - mu_ttbar
         - TES
         - JES

- **POI** — the parameter of interest (signal strength).
- **ParametersToFit** — only these parameters appear in the likelihood. Omitting a parameter here effectively fixes it at its nominal value.

Samples
-------

Each row in ``Samples`` is a physics process read from a ROOT file:

.. code-block:: yaml

   Samples:
     - Name: htautau
       Tree: tree_htautau
       SamplePath: <path/to>/dataset_nominal.root
       Weight: weights
       UseAsBasis: True

- **UseAsBasis** — this process gets its own density-ratio network. This tells the workflow, that the sample forms one of the basis of the parametric morphing formula used in the statistical model.

The observed dataset is one more entry in ``Samples``, flagged with ``Data: True``:

.. code-block:: yaml

   Samples:
     - Name: obs
       Tree: tree_obs
       SamplePath: <path/to>/dataset_with_obs.root
       Data: True

- **Data: True** — this sample is the observation. In binned regions its histogram is the observed data, in unbinned regions its events are the ones the density ratios are evaluated on. Only one sample can be flagged as data. ``Weight`` is optional and defaults to one per event.
- If no sample is flagged, the expected (Asimov) dataset built from the MC samples is used as the observation.

Which samples form the reference :math:`p_{\text{ref}}` for the density ratio :math:`r(x) = p / p_{\text{ref}}` is configured at the workflow level (``config.pipeline.yaml`` in the example, under ``neural_likelihood_ratio_estimation.reference_priors``) and consumed by :meth:`nsbi_common_utils.datasets.datasets.prepare_basis_training_dataset`. See that method's docstring for the full spec (auto-yield, numeric, ``{cap: M}``, exclude). The previous ``UseAsReference: True/False`` field on each sample is deprecated.

Normalization factors
---------------------

Free multiplicative parameters (signal strengths, background norms):

.. code-block:: yaml

   NormFactors:
     - Name: mu_htautau
       Samples: htautau
       Nominal: 1
       Bounds: [0, 10]

These are **unconstrained** in the fit — no Gaussian penalty term.

Systematic uncertainties
------------------------

Shape + normalization uncertainties point to the up/down varied ROOT files:

.. code-block:: yaml

   Systematics:
     - Name: JES
       Type: NormPlusShape
       Nominal: 0
       Samples: [htautau, ztautau, ttbar]
       Up:
         - SampleName: htautau
           Path: <path/to>/dataset_JES_up.root
           Tree: tree_htautau
           Weight: weights
       Dn:
         - SampleName: htautau
           Path: <path/to>/dataset_JES_dn.root
           Tree: tree_htautau
           Weight: weights

- **Nominal: 0** — the nuisance parameter starts at zero (the Gaussian constraint is centred here).
- The workspace builder computes variation ratios (varied / nominal) automatically from the histograms.

Analysis regions
----------------

Regions define event selections and whether the channel is binned or unbinned:

.. code-block:: yaml

   Regions:
     # Binned control region
     - Name: CR
       Filter: presel_score < -1.0
       Variable: presel_score
       Type: binned
       Binning: [-8.75, -4.0, -2.5, -1.0]

     # Unbinned SBI signal region
     - Name: SR
       Filter: presel_score >= 1.0 & presel_score <= 4.5
       Type: unbinned

For **binned** regions, the workspace builder histograms every sample using ``Variable`` and ``Binning``.
For **unbinned** regions, the density-ratio models are listed separately under ``TrainedModels``.

Trained models
--------------

``TrainedModels`` links each unbinned region to its density-ratio networks and the per-event ratios they produce:

.. code-block:: yaml

   TrainedModels:
     - RegionName: SR
       Weights: <path/to>/weights.npy
       Models:
         - Name: htautau
           Nominal:
             Models: <path/to>/models_htautau/
             Ratios: <path/to>/ratio_htautau.npy
           Systematics:
             - SystName: JES
               ModelsUp: <path/to>/models_htautau_JES_Up/
               ModelsDn: <path/to>/models_htautau_JES_Dn/
               RatiosUp: <path/to>/ratio_htautau_JES_Up.npy
               RatiosDn: <path/to>/ratio_htautau_JES_Dn.npy

- **RegionName** — name assigned to the unbinned region.
- **Weights** — per-event weights of the observed (or Asimov) events in the region.
- **Models** / **ModelsUp** / **ModelsDn** — directories the trained networks are written to and loaded from.
- **Ratios** / **RatiosUp** / **RatiosDn** — per-event density ratios ``p_c(x) / p_ref(x)`` evaluated on the observed events, one ``.npy`` array of shape ``(n_events,)`` per sample and variation.

Observed data
-------------

The workspace carries an ``observations`` entry per region, built from the ``Data: True`` sample (or the Asimov expectation):

.. code-block:: python

   "observations": [
       {"name": "CR", "data": [n_1, ..., n_k]},
       {"name": "SR",
        "data": [N_obs],
        "weights": "<path/to>/weights.npy",
        "ratios": {"htautau": "<path/to>/ratio_htautau.npy",
                   "ttbar":   "<path/to>/ratio_ttbar.npy",
                   "ztautau": "<path/to>/ratio_ztautau.npy"}},
   ]

Binned regions hold the observed histogram. Unbinned regions hold the total observed yield together with the ``Weights`` and nominal ``Ratios`` paths from ``TrainedModels``.

Training features
-----------------

Which branches from the ROOT file are used as NN inputs:

.. code-block:: yaml

   TrainingFeatures:
     - DER_mass_transverse_met_lep
     - log_DER_mass_vis
     - log_DER_pt_h

   TrainingFeaturesToStandardize:
     - DER_mass_transverse_met_lep
     - log_DER_mass_vis

Features listed in ``TrainingFeaturesToStandardize`` are z-scored before being passed to the network. Features not listed are passed through unchanged.
