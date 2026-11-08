#!/usr/bin/env bash
#SBATCH --nodes=1

echo "Running on: $SLURMD_NODENAME"
date

ml PyTorch/2.13.0-foss-2024a-CUDA-12.6.0 torchvision/0.28.0-foss-2024a-CUDA-12.6.0
ml unload protobuf-python
cd "${FARL_REPO_ROOT:?FARL_REPO_ROOT was not provided by the launch script}"
source "${FARL_VENV:?FARL_VENV was not provided by the launch script}/bin/activate"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CONFIG_PATH EXPERIMENT_NAME" >&2
    exit 2
fi

blob_root="${FARL_BLOB_ROOT:?FARL_BLOB_ROOT was not provided by the launch script}"
if [[ -n "${FARL_TEMP_BLOB_ROOT:-}" ]]; then
    temporary_blob_root="$FARL_TEMP_BLOB_ROOT"
    data_source="$blob_root/data"
    data_destination="$temporary_blob_root/data"

    if [[ ! -d "$data_source" ]]; then
        echo "blob data directory does not exist: $data_source" >&2
        exit 2
    fi
    mkdir -p "$temporary_blob_root" || exit 1
    if [[ -e "$data_destination" || -L "$data_destination" ]]; then
        echo "temporary blob data directory already exists: $data_destination" >&2
        exit 2
    fi
    mkdir -p "$data_destination" || exit 1
    shopt -s nullglob
    data_files=("$data_source"/*.*.zip)
    if (( ${#data_files[@]} == 0 )); then
        echo "no data files found in: $data_source" >&2
        exit 2
    fi
    cp -a "${data_files[@]}" "$data_destination"/ || exit 1

    for source_path in "$blob_root"/*; do
        [[ -d "$source_path" ]] || continue
        [[ "$(basename "$source_path")" == "data" ]] && continue
        link_path="$temporary_blob_root/$(basename "$source_path")"
        if [[ -e "$link_path" || -L "$link_path" ]]; then
            echo "temporary blob link already exists: $link_path" >&2
            exit 2
        fi
        ln -s "$source_path" "$link_path" || exit 1
    done
    blob_root="$temporary_blob_root"
fi

python -m blueprint.run \
    "$1" \
    --exp_name "$2" \
    --blob_root "$blob_root"

exitcode=$?

date

exit "$exitcode"
