#!/bin/bash
#SBATCH --nodes=1

date

ml PyTorch/2.13.0-foss-2024a-CUDA-12.6.0 torchvision/0.28.0-foss-2024a-CUDA-12.6.0
ml unload protobuf-python
cd /mnt/personal/bezdealb/FaRL/
source ${3}/bin/activate

python -m blueprint.run \
        $1 \
        --exp_name $2 \
        --blob_root ./blob

exitcode=$?

date

exit "$exitcode"
