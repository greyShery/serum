
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_87', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_87(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 20480) % 16)
    x1 = ((xindex // 1280) % 16)
    x0 = (xindex % 1280)
    x3 = xindex // 327680
    x5 = xindex
    tmp10 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 1280*tmp8 + 10240*tmp4 + 81920*x3), None).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr2 + (x0 + 1280*tmp8 + 10240*tmp4 + 81920*x3), None).to(tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp19, None)
