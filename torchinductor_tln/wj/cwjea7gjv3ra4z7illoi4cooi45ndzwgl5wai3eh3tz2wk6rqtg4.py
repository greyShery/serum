
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 40)
        r0_3 = r0_index // 40
        tmp0 = tl.load(in_ptr0 + (r0_2 + 40*x0 + 1280*r0_3 + 81920*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 40*x4), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp12[:, None]
    tmp10 = tmp13[:, None]
    tmp11 = tmp14[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
    tl.store(out_ptr1 + (x4), tmp10, xmask)
