#!/usr/bin/env python3

import argparse
import shlex
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
    launch_priority: int = 0


def _parse_args(run_names: list[str], resource_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit all FaRL runs for one checkpoint."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--mail-user",
        default=None,
        help="send mail when a job fails and when all jobs have finished",
    )
    parser.add_argument(
        "--notify-partition",
        default="cpufast",
        help="partition for the final notification job (default: %(default)s)",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=Path(__file__).parent.parent / "configs"
    )
    parser.add_argument(
        "--blob-root",
        type=Path,
        default=Path("./blob"),
        help="blob directory containing outputs, states, and logs",
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
    parser.add_argument(
        "--ignore-priorities",
        action="store_true",
        help="submit jobs without Slurm priority dependencies",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> bool:
    if not args.experiment_name:
        raise ValueError("experiment name must not be empty")
    if not args.checkpoint_path:
        raise ValueError("checkpoint path must not be empty")
    if not args.dry_run and not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint_path}")
    if not args.sbatch_script.is_file():
        raise FileNotFoundError(f"sbatch script does not exist: {args.sbatch_script}")
    if not args.resource_config.is_file():
        raise FileNotFoundError(
            f"resource config does not exist: {args.resource_config}"
        )
    if args.mail_user is None:
        answer = input("Warning: no --mail-user was provided. Continue? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return False
    return True


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


def _sbatch_options(
    resource_spec: ResourceSpec,
    job_name: str,
    log_dir: Path,
    repo_root: Path,
    blob_root: Path,
    mail_user: str | None,
    dependency_job_ids: list[str] | None = None,
) -> list[str]:
    log_path = log_dir / f"{job_name}.log"
    options = [
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        str(log_path),
        "--error",
        str(log_path),
        "--export",
        "ALL,"
        f"FARL_REPO_ROOT={repo_root},"
        f"FARL_VENV={resource_spec.venv},"
        f"FARL_BLOB_ROOT={blob_root}",
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
    if mail_user:
        options.extend(["--mail-type", "FAIL,TIME_LIMIT", "--mail-user", mail_user])
    if dependency_job_ids:
        options.extend(["--dependency", f"after:{':'.join(dependency_job_ids)}"])
    return options


def _priority_groups(
    run_names: list[str], resource_specs: dict[str, ResourceSpec]
) -> dict[int, list[str]]:
    priority_groups: dict[int, list[str]] = {}
    for run_name in run_names:
        priority = resource_specs[run_name].launch_priority
        priority_groups.setdefault(priority, []).append(run_name)
    return priority_groups


def _job_id_from_submission(result: subprocess.CompletedProcess[str]) -> str:
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id:
        raise RuntimeError(
            f"sbatch did not return a job ID; stderr: {result.stderr.strip()}"
        )
    return job_id


def main() -> None:
    resource_config_path = Path(__file__).with_name("runs.yaml")
    resource_config_parser = argparse.ArgumentParser(add_help=False)
    resource_config_parser.add_argument("--resource-config", type=Path)
    resource_config_args, _ = resource_config_parser.parse_known_args()
    if resource_config_args.resource_config is not None:
        resource_config_path = resource_config_args.resource_config
    resource_specs = _load_resource_specs(resource_config_path)
    args = _parse_args(list(resource_specs), resource_config_path)
    if not _validate_args(args):
        return
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
    log_dir = args.blob_root / "logs" / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    commands_text = ""
    all_job_ids: list[str] = []

    priority_groups = _priority_groups(args.runs, selected_specs)
    previous_priority_job_ids: list[str] = []
    for priority in sorted(priority_groups, reverse=True):
        current_priority_job_ids = []
        for run_name in priority_groups[priority]:
            resource_spec = selected_specs[run_name]
            config_path = experiment_dir / f"{run_name}.yaml"
            save_launch_yaml(
                output_path=config_path,
                dataset=resource_spec.dataset,
                input_resolution=resource_spec.input_resolution,
                num_gpus=resource_spec.num_gpus,
                mode="frozen" if resource_spec.run_type == "frozen" else "finetune",
                checkpoint_path=str(Path(args.checkpoint_path).resolve()),
                train_data_ratio=resource_spec.train_data_ratio,
            )

            dependency_job_ids = (
                None if args.ignore_priorities else previous_priority_job_ids or None
            )
            command = [
                "sbatch",
                *_sbatch_options(
                    resource_spec,
                    job_name=f"{args.experiment_name}_{run_name}",
                    log_dir=log_dir,
                    repo_root=repo_root,
                    blob_root=args.blob_root,
                    mail_user=args.mail_user,
                    dependency_job_ids=dependency_job_ids,
                ),
                str(args.sbatch_script),
                str(config_path),
                args.experiment_name,
            ]
            command_text = shlex.join(command)
            commands_text += f"{command_text}\n"
            if args.dry_run:
                print(command_text)
            else:
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                job_id = _job_id_from_submission(result)
                current_priority_job_ids.append(job_id)
                all_job_ids.append(job_id)
                print(f"submitted {run_name} ({job_id}): {config_path}")
        previous_priority_job_ids = current_priority_job_ids

    if args.mail_user:
        notify_command = [
            "sbatch",
            "--parsable",
            "--job-name",
            f"notify_{args.experiment_name}",
            "--output",
            "/dev/null",
            "--partition",
            args.notify_partition,
            "--mail-type",
            "END,FAIL",
            "--mail-user",
            args.mail_user,
            "--dependency",
            f"afterany:{':'.join(all_job_ids) if all_job_ids else '<job ids>'}",
            "--wrap",
            "echo Jobs finished",
        ]
        commands_text += f"{shlex.join(notify_command)}\n"
        if args.dry_run:
            print(shlex.join(notify_command))
        else:
            notify_result = subprocess.run(
                notify_command, check=True, capture_output=True, text=True
            )
            print(
                f"submitted notification job ({_job_id_from_submission(notify_result)})"
            )

    if not args.dry_run:
        (log_dir / "commands.txt").write_text(commands_text, encoding="utf-8")


if __name__ == "__main__":
    main()
