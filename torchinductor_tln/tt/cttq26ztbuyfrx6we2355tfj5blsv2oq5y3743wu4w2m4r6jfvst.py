
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 320)
    x2 = xindex // 1310720
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 320*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 40960.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp25, None)
