# vectoradd_v2
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Simple vector addition using pure PyTorch.
    """
    A, B, C = data  # A and B are vectors, C is pre-allocated output
    # Element-wise vector addition
    torch.add(A, B, out=C)
    return C
