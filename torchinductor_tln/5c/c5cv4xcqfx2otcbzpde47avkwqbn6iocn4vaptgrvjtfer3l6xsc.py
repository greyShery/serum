
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_114', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_114(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 40960
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 10)
        r0_3 = r0_index // 10
        tmp0 = tl.load(in_ptr0 + (r0_2 + 10*x0 + 320*r0_3 + 1310720*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
