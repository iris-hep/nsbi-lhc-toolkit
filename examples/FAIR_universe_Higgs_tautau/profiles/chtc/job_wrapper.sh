#!/bin/bash

set -e

export HOME=$(pwd)

# Site-specific data location.
export NSBI_DATA_DIR=/projects/Physics_Cranmer/saved_datasets

STAGED_DATA_ZIP=/projects/Physics_Cranmer/sbi_studies/FAIR_Universe_HiggsML_data.zip
if [ -f "$STAGED_DATA_ZIP" ]; then
    ln -sf "$STAGED_DATA_ZIP" examples/FAIR_universe_Higgs_tautau/data.zip
fi

# The container ships snakemake + deps via pixi; here we (re)install the package
# from the transferred `src/` so the EP runs the current source. 
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
python -m pip install --no-deps --user -e .

exec snakemake "$@"
