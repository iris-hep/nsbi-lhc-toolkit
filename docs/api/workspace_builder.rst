Workspace Builder API
=====================

The workspace builder converts a human-readable YAML fit configuration into a
pyhf-like workspace dictionary that :class:`~nsbi_common_utils.models.sbi_parametric_model.sbi_parametric_model`
consumes directly. This is the bridge between your analysis definition and the
statistical model.

Typical usage
-------------

.. code-block:: python

   from nsbi_common_utils import workspace_builder, models, inference

   # 1. Build workspace from fit config
   ws = workspace_builder.WorkspaceBuilder(config_path="config_fit_nsbi.yml").build()

   # 2. Optionally serialise / reload (avoids re-reading ROOT files)
   builder.dump_workspace(ws, "workspace.json")
   ws = workspace_builder.WorkspaceBuilder.load_workspace("workspace.json")

   # 3. Pass to the statistical model
   model = models.sbi_parametric_model(workspace=ws, measurement_to_fit="my_measurement")

For details on the YAML configuration format consumed by the builder, see :doc:`/basics/fit_config`.
For a hands-on walkthrough, see :doc:`/basics/model_building_example`.

API Reference
-------------

.. currentmodule:: nsbi_common_utils.workspace_builder

.. autoclass:: WorkspaceBuilder
   :members: build, dump_workspace, load_workspace, channels, observations, measurements, normfactor_modifiers, normplusshape_modifiers, sys_modifiers
   :undoc-members: False
   :show-inheritance:
   :exclude-members: __init__
