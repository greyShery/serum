
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_92', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_92(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 61440
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1920)
    y1 = yindex // 1920
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 60), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 60), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 15360.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 1920*x2 + 491520*y1), tmp19, xmask)
    tl.store(out_ptr2 + (y0 + 1920*x2 + 491520*y1), tmp0, xmask)
