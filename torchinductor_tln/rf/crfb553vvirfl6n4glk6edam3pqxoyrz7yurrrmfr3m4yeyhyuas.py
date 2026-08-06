
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_layer_norm_76', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_native_layer_norm_76(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 1280
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = 1.0
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(r0_mask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(r0_mask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(r0_mask & xmask, tmp11_weight_next, tmp11_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp14[:, None]
    tmp12 = tmp15[:, None]
    tmp13 = tmp16[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = 1.0
        tmp21 = tmp19 * tmp20
        tmp24 = tmp22 + tmp23
        tmp25 = tmp21 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 - tmp11
        tmp28 = 1280.0
        tmp29 = (tmp12 / tmp28)
        tmp30 = 1e-05
        tmp31 = tmp29 + tmp30
        tmp32 = libdevice.rsqrt(tmp31)
        tmp33 = tmp27 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 * tmp35
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp36 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1280*x0), tmp40, xmask & r0_mask)
