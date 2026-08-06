
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_layer_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_native_layer_norm_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 640, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = tl.where(r0_mask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 640.0
    tmp28 = (tmp25 / tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 640*x0), tmp39, r0_mask)
