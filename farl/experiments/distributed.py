import os
import random
import socket
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from blueprint.builder import Builder
from blueprint.ml import DistributedGPURun
from torch.multiprocessing.spawn import ProcessRaisedException


_MAX_PORT_SELECTION_ATTEMPTS = 10


def _single_thread_run(
    gpu_rank, num_gpus, distributed_run, port_conflict_event=None
):
    """
    A modified version of blueprint.ml.distributed._single_thread_run
    """

    rank = distributed_run.node_rank * num_gpus + gpu_rank
    world_size = num_gpus * distributed_run.num_nodes
    if distributed_run.num_nodes == 1:
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

    try:
        dist.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=world_size
        )
    except dist.DistNetworkError:
        if port_conflict_event is not None:
            port_conflict_event.set()
        raise
    print(f"DistributedGPURun: init_process_group: {rank}/{world_size}")

    torch.cuda.set_device(gpu_rank)
    seed = distributed_run.seed0 + rank
    torch.manual_seed(seed)
    random.seed(seed + 100)
    np.random.seed(seed + 200)

    builder: Builder = deepcopy(distributed_run.builder_snapshot)
    local_run: Any = builder.parse_cfg_object(distributed_run.local_run)
    local_run()

    dist.barrier(device_ids=[gpu_rank])
    dist.destroy_process_group()


class CustomDistributedGPURun(DistributedGPURun):
    def __call__(self):
        num_gpus = torch.cuda.device_count()
        original_port = os.environ.get("MASTER_PORT")
        select_port = self.num_nodes == 1 and original_port in (None, "0")
        attempts = _MAX_PORT_SELECTION_ATTEMPTS if select_port else 1

        for attempt in range(attempts):
            port_conflict_event = (
                torch.multiprocessing.get_context("spawn").Event()
                if select_port
                else None
            )
            selected_socket = None
            if select_port:
                address = os.environ.get("MASTER_ADDR", "localhost")
                selected_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    selected_socket.bind((address, 0))
                    os.environ["MASTER_PORT"] = str(
                        selected_socket.getsockname()[1]
                    )
                finally:
                    selected_socket.close()

            try:
                spawn = getattr(torch.multiprocessing, "spawn")
                spawn(
                    _single_thread_run,
                    args=(num_gpus, self, port_conflict_event),
                    nprocs=num_gpus,
                    join=True,
                )
                return
            except ProcessRaisedException:
                if (
                    not select_port
                    or not port_conflict_event.is_set()
                    or attempt == attempts - 1
                ):
                    raise
            finally:
                if select_port:
                    if original_port is None:
                        os.environ.pop("MASTER_PORT", None)
                    else:
                        os.environ["MASTER_PORT"] = original_port
