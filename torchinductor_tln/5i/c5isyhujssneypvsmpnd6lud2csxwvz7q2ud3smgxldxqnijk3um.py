
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 320
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4096)
    y1 = yindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x2 + 320*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 4096*x2 + 1310720*y1), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2 + 320*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 320*y3), tmp10, ymask & xmask)
