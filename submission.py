# vectoradd_v2
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    # alignment hints
    tl.multiple_of(offsets, 32)
    tl.multiple_of(a_ptr, 32)
    tl.multiple_of(b_ptr, 32)
    tl.multiple_of(c_ptr, 32)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    A, B, C = data
    n = A.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](A, B, C, n, BLOCK_SIZE, num_warps=8, num_stages=3)
    return C
