#!/usr/bin/env python3

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from .generate_launch_yaml import save_launch_yaml

DATASETS: Final = ("aflw19", "ibug300w", "wflw", "celebm", "lapa")
RUN_TYPES: Final = ("frozen", "finetune")
TOTAL_BATCH_SIZE: Final = 40


@dataclass(frozen=True)
class ResourceSpec:
    run_type: str
    dataset: str
    queue: str
    duration: str
    num_gpus: int = 1
    cpus: int = 10
    memory: str = "64G"
    venv: str = ".venv"
    train_data_ratio: float | None = None
    input_resolution: int = 224


def _parse_args(run_names: list[str], resource_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit all FaRL runs for one checkpoint."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--configs-dir", type=Path, default=Path(__file__).parent.parent / "configs"
    )
    parser.add_argument(
        "--resource-config",
        type=Path,
        default=resource_config,
        help="YAML file containing per-run Slurm resources",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=run_names,
        default=run_names,
        help="named runs to submit; defaults to all runs",
    )
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        default=Path(__file__).with_name("train.sbatch"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate configs without submitting Slurm jobs",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.experiment_name:
        raise ValueError("experiment name must not be empty")
    if not args.checkpoint_path:
        raise ValueError("checkpoint path must not be empty")
    if not args.sbatch_script.is_file():
        raise FileNotFoundError(f"sbatch script does not exist: {args.sbatch_script}")
    if not args.resource_config.is_file():
        raise FileNotFoundError(
            f"resource config does not exist: {args.resource_config}"
        )


def _load_resource_specs(path: Path) -> dict[str, ResourceSpec]:
    with path.open(encoding="utf-8") as resource_file:
        raw_config = yaml.safe_load(resource_file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("resource config must contain a mapping at its top level")

    specs = {}
    valid_fields = set(ResourceSpec.__dataclass_fields__)
    for run_name, settings in raw_config.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{run_name!r} must contain a run mapping")
        missing_fields = {"run_type", "dataset"} - set(settings)
        if missing_fields:
            raise ValueError(
                f"{run_name!r} is missing required fields: {sorted(missing_fields)}"
            )
        unknown_fields = set(settings) - valid_fields
        if unknown_fields:
            raise ValueError(
                f"unknown resource fields for {run_name}: {sorted(unknown_fields)}"
            )
        if settings["run_type"] not in RUN_TYPES:
            raise ValueError(
                f"unsupported run type for {run_name}: {settings['run_type']!r}"
            )
        if settings["dataset"] not in DATASETS:
            raise ValueError(
                f"unsupported dataset for {run_name}: {settings['dataset']!r}"
            )
        specs[run_name] = ResourceSpec(**settings)
    return specs


def _validate_resource_spec(resource_spec: ResourceSpec) -> None:
    if resource_spec.num_gpus <= 0 or TOTAL_BATCH_SIZE % resource_spec.num_gpus != 0:
        raise ValueError(
            f"number of GPUs must be a positive divisor of {TOTAL_BATCH_SIZE}"
        )


def _sbatch_options(resource_spec: ResourceSpec) -> list[str]:
    return [
        "--partition",
        resource_spec.queue,
        "--time",
        resource_spec.duration,
        "--gres",
        f"gpu:{resource_spec.num_gpus}",
        "--cpus-per-task",
        str(resource_spec.cpus),
        "--mem",
        resource_spec.memory,
    ]


def main() -> None:
    resource_config_path = Path(__file__).with_name("runs.yaml")
    resource_config_parser = argparse.ArgumentParser(add_help=False)
    resource_config_parser.add_argument("--resource-config", type=Path)
    resource_config_args, _ = resource_config_parser.parse_known_args()
    if resource_config_args.resource_config is not None:
        resource_config_path = resource_config_args.resource_config
    resource_specs = _load_resource_specs(resource_config_path)
    args = _parse_args(list(resource_specs), resource_config_path)
    _validate_args(args)
    selected_specs = {}
    for run_name in args.runs:
        resource_spec = resource_specs[run_name]
        _validate_resource_spec(resource_spec)
        selected_specs[run_name] = resource_spec

    experiment_dir = args.configs_dir / args.experiment_name
    if experiment_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing experiment directory: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True)

    for run_name in args.runs:
        resource_spec = selected_specs[run_name]
        config_path = experiment_dir / f"{run_name}.yaml"
        save_launch_yaml(
            output_path=config_path,
            dataset=resource_spec.dataset,
            input_resolution=resource_spec.input_resolution,
            num_gpus=resource_spec.num_gpus,
            mode="frozen" if resource_spec.run_type == "frozen" else "finetune",
            checkpoint_path=args.checkpoint_path,
            train_data_ratio=resource_spec.train_data_ratio,
        )

        command = [
            "sbatch",
            *_sbatch_options(resource_spec),
            str(args.sbatch_script),
            str(config_path),
            f"{args.experiment_name}_{run_name}",
            resource_spec.venv,
        ]
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)
            print(f"submitted {run_name}: {config_path}")


if __name__ == "__main__":
    main()
