
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_82', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_82(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 5120
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
    tmp21_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 64
        r0_2 = (r0_index % 64)
        r0_5 = r0_index
        tmp0 = r0_3 + 80*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1280, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r0_2 + 64*(r0_3 + 80*x0) + 81920*x1), r0_mask & xmask & tmp4, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (1280*r0_2 + 81920*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp4, tmp11, tmp12)
        tmp14 = tmp0 >= tmp3
        tmp15 = tl.full([1, 1], 2560, tl.int64)
        tmp16 = tmp0 < tmp15
        tmp17 = tl.load(in_ptr3 + (r0_2 + 64*((-1280) + r0_3 + 80*x0) + 81920*x1), r0_mask & xmask & tmp14, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.where(tmp4, tmp13, tmp17)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight, roffset == 0
        )
        tmp21_mean = tl.where(r0_mask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(r0_mask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(r0_mask & xmask, tmp21_weight_next, tmp21_weight)
        tl.store(out_ptr0 + (r0_5 + 5120*x4), tmp18, r0_mask & xmask)
    tmp24, tmp25, tmp26 = triton_helpers.welford(tmp21_mean, tmp21_m2, tmp21_weight, 1)
    tmp21 = tmp24[:, None]
    tmp22 = tmp25[:, None]
    tmp23 = tmp26[:, None]
    tl.store(out_ptr1 + (x4), tmp21, xmask)
    tl.store(out_ptr2 + (x4), tmp22, xmask)
