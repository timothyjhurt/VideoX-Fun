import importlib.util

from .flux2_xfuser import Flux2MultiGPUsAttnProcessor2_0
from .flux_xfuser import FluxMultiGPUsAttnProcessor2_0
from .fsdp import shard_model
from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    get_world_group, init_distributed_environment,
                    initialize_model_parallel, sequence_parallel_all_gather,
                    sequence_parallel_chunk, set_multi_gpus_devices,
                    xFuserLongContextAttention)
from .qwen_xfuser import QwenImageMultiGPUsAttnProcessor2_0
from .z_image_xfuser import ZMultiGPUsSingleStreamAttnProcessor