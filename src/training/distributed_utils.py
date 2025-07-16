"""
Distributed training utilities for PyTorch DDP.
Developed for healthcare AI research - demonstration purposes only.
"""

import os
import torch
import torch.distributed as dist


def init_distributed_mode():
    """Initialize distributed training mode."""
    dist_url = 'env://'
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return 0, 1, 0

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    
    print(f'| distributed init (rank {rank}): {dist_url}', flush=True)
    dist.init_process_group(
        backend=dist_backend, 
        init_method=dist_url,
        world_size=world_size, 
        rank=rank
    )
    dist.barrier()
    
    return rank, world_size, gpu


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get the world size for distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank for distributed training."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0
