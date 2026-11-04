#!/usr/bin/env bash
#SBATCH --nodes=1

echo "Running on: $SLURMD_NODENAME"
date

ml PyTorch/2.13.0-foss-2024a-CUDA-12.6.0 torchvision/0.28.0-foss-2024a-CUDA-12.6.0
ml unload protobuf-python
cd "${FARL_REPO_ROOT:?FARL_REPO_ROOT was not provided by the launch script}"
source "${FARL_VENV:?FARL_VENV was not provided by the launch script}/bin/activate"

python -m blueprint.run \
    "$1" \
    --exp_name "$2" \
    --blob_root "${FARL_BLOB_ROOT:-./blob}"

exitcode=$?

date

exit "$exitcode"
