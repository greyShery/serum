
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 320)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp4, tmp6, 0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = -9.210340371976184
    tmp12 = tmp10 * tmp11
    tmp13 = 0.00625
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp8 * tmp15
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tl_math.sin(tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1], 320, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (0))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tl.where(tmp22, tmp26, 0)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = (-160) + x0
    tmp30 = tmp29.to(tl.float32)
    tmp31 = -9.210340371976184
    tmp32 = tmp30 * tmp31
    tmp33 = 0.00625
    tmp34 = tmp32 * tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = tmp28 * tmp35
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tl_math.cos(tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp22, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp21, tmp41)
    tl.store(out_ptr0 + (x2), tmp42, xmask)
