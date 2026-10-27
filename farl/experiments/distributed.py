import os
import random
import socket
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from packaging import version

from blueprint.builder import Builder
from blueprint.ml import DistributedGPURun


def _single_thread_run(gpu_rank, num_gpus, distributed_run):
    """
    A modified version of blueprint.ml.distributed._single_thread_run
    """

    rank = distributed_run.node_rank * num_gpus + gpu_rank
    world_size = num_gpus * distributed_run.num_nodes
    disable_distributed = os.environ.get("DISABLE_DISTRIBUTED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if disable_distributed:
        init_method = None
    elif distributed_run.num_nodes == 1:
        address = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "10315")
        init_method = f"tcp://{address}:{port}"
    elif "AZ_BATCHAI_PYTORCH_INIT_METHOD" in os.environ:
        init_method = os.environ["AZ_BATCHAI_PYTORCH_INIT_METHOD"]
    elif "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        init_method = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
    else:
        raise RuntimeError("Failed to create the init_method")

    if not disable_distributed:
        dist.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=world_size
        )
        print(f"DistributedGPURun: init_process_group: {rank}/{world_size}")

    torch.cuda.set_device(gpu_rank)
    seed = distributed_run.seed0 + rank
    torch.manual_seed(seed)
    random.seed(seed + 100)
    np.random.seed(seed + 200)

    builder: Builder = deepcopy(distributed_run.builder_snapshot)
    local_run: Any = builder.parse_cfg_object(distributed_run.local_run)
    local_run()

    if not disable_distributed:
        if version.parse(torch.__version__) >= version.parse("1.8.0"):
            dist.barrier(device_ids=[gpu_rank])
        else:
            dist.barrier()
        dist.destroy_process_group()


class CustomDistributedGPURun(DistributedGPURun):
    def __call__(self):
        num_gpus = torch.cuda.device_count()
        disable_distributed = os.environ.get("DISABLE_DISTRIBUTED", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_distributed:
            if num_gpus != 1 or self.num_nodes != 1:
                raise RuntimeError(
                    "DISABLE_DISTRIBUTED requires exactly one GPU on one node"
                )
            _single_thread_run(0, 1, self)
            return

        original_port = os.environ.get("MASTER_PORT")
        selected_socket = None
        if self.num_nodes == 1 and original_port in (None, "0"):
            address = os.environ.get("MASTER_ADDR", "localhost")
            selected_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            selected_socket.bind((address, 0))
            os.environ["MASTER_PORT"] = str(selected_socket.getsockname()[1])
            selected_socket.close()

        try:
            spawn = getattr(torch.multiprocessing, "spawn")
            spawn(_single_thread_run, args=(num_gpus, self), nprocs=num_gpus, join=True)
        finally:
            if selected_socket is not None:
                if original_port is None:
                    os.environ.pop("MASTER_PORT", None)
                else:
                    os.environ["MASTER_PORT"] = original_port
