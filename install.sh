#!/bin/bash
# Creates (or reactivates) the oulu-thermal conda environment.

ENVS=$(conda info --envs | awk '{print $1}')
if [[ $ENVS = *"oulu-thermal"* ]]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate oulu-thermal
else
    echo "Creating conda environment oulu-thermal..."
    conda env create -f environment.yml
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate oulu-thermal
fi
