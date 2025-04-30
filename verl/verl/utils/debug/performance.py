import torch
import torch.distributed as dist
import logging


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)
