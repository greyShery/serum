# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fa/cfallvkd5hd7uqgrr43rbzkglb3tlye42ckg2i5lxkt5vo2m2pwr.py
# Topologically Sorted Source Nodes: [sample_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x2 + 16384*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/st/cstbgnmejvpxogsztrx2uc4iu2tu3rgkdy7k44c7okf7updhndsg.py
# Topologically Sorted Source Nodes: [sample_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/2d/c2drijwqw6t5sh5po5a7hcaotyrtcscrkcdfdcl2bhfhxtuf3udh.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type_11, var_mean
# Graph fragment:
#   %convert_element_type_11 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_11, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_2 = async_compile.triton('triton_red_fused_native_group_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 10)
        r0_3 = r0_index // 10
        tmp0 = tl.load(in_ptr0 + (r0_2 + 10*x0 + 320*r0_3 + 1310720*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 10*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/4c/c4ceb3lxz35klwlbqar5r5lcwertj36b3a4vzjqva373r2ycmfuz.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add_2, mul_6
#   hidden_states_1 => convert_element_type_16, mul_7, sigmoid_1
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_8), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_5), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_1), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_3 = async_compile.triton('triton_poi_fused_native_group_norm_silu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 320)
    x2 = xindex // 1310720
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 40960.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/7q/c7qaajpc6hj44yljhb2ucdmpft2mvo3kmndrjym5joqu6b3kpfye.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_1 => convert_element_type_16, mul_7, sigmoid_1
#   hidden_states_2 => convolution_1
# Graph fragment:
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_1), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.float16), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_16, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_4 = async_compile.triton('triton_poi_fused_convolution_silu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/g2/cg2gr6fkiowltpxvafoelgtx5cg3wmd5nsfnsjd7cnnwt6cgc6h4.py
# Topologically Sorted Source Nodes: [emb_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   emb_3 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sin, %cos], -1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/xz/cxzni763t4edugrg3ryh6pq7nhggppsctgjq2qs4nh57ghcf4zhg.py
# Topologically Sorted Source Nodes: [emb_4, t_emb], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   emb_4 => cat_1
#   t_emb => convert_element_type_2
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6], -1), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.float16), kwargs = {})
triton_poi_fused__to_copy_cat_6 = async_compile.triton('triton_poi_fused__to_copy_cat_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 320)
    x1 = xindex // 320
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (160 + 320*x1 + (x0)), xmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 320, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (320*x1 + ((-160) + x0)), xmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/lf/clfgh6cdhq5jzslktk2fxif6u2ds26rkglcezyv23lv2mpnndbcp.py
# Topologically Sorted Source Nodes: [sample, sample_1], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   sample => add_tensor_103
#   sample_1 => convert_element_type_6, convert_element_type_7, mul_4, sigmoid
# Graph fragment:
#   %add_tensor_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_103, %arg3_1), kwargs = {})
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_103, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.float16), kwargs = {})
triton_poi_fused_addmm_silu_7 = async_compile.triton('triton_poi_fused_addmm_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/xh/cxhaopy7rufp2dtedsmljnj6yq4pzl6uc52n56ugmyimbbsknsqq.py
# Topologically Sorted Source Nodes: [sample_2, temb, temb_2], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   sample_2 => add_tensor_102
#   temb => convert_element_type_17, convert_element_type_18, mul_8, sigmoid_2
#   temb_2 => convert_element_type_75, convert_element_type_76, mul_27, sigmoid_5
# Graph fragment:
#   %add_tensor_102 : [num_users=22] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_102, %arg5_1), kwargs = {})
#   %convert_element_type_17 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_17,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %sigmoid_2), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.float16), kwargs = {})
#   %convert_element_type_75 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_75,), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_75, %sigmoid_5), kwargs = {})
#   %convert_element_type_76 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.float16), kwargs = {})
triton_poi_fused_addmm_silu_8 = async_compile.triton('triton_poi_fused_addmm_silu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_8(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fm/cfmcfdytobpnnpwd4x72bxqv4kjvyy5unecwd3efx2zxdnj5otf4.py
# Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_4 => convert_element_type_22, var_mean_1
# Graph fragment:
#   %convert_element_type_22 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_22, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_9 = async_compile.triton('triton_red_fused_native_group_norm_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    x4 = xindex
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 10)
        r0_3 = r0_index // 10
        tmp0 = tl.load(in_ptr0 + (r0_2 + 10*x0 + 320*r0_3 + 1310720*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 10*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 10*x4), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 10*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/tt/cttq26ztbuyfrx6we2355tfj5blsv2oq5y3743wu4w2m4r6jfvst.py
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_4 => add_5, mul_10
#   hidden_states_5 => convert_element_type_27, mul_11, sigmoid_3
# Graph fragment:
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %unsqueeze_16), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_13), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_5,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %sigmoid_3), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_10 = async_compile.triton('triton_poi_fused_native_group_norm_silu_10', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/rg/crg72iyjn5iuu2xa3hvjnj5zdej3uke7aoaewaziddu75ibb4sdi.py
# Topologically Sorted Source Nodes: [hidden_states_8], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_8 => convert_element_type_28, var_mean_2
# Graph fragment:
#   %convert_element_type_28 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_11 = async_compile.triton('triton_red_fused_native_group_norm_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 10)
        r0_3 = r0_index // 10
        tmp0 = tl.load(in_ptr0 + (r0_2 + 10*x0 + 320*r0_3 + 1310720*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 10*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 10*x0 + 320*r0_3 + 1310720*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 10*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
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
    tl.store(out_ptr0 + (x4), tmp11, xmask)
    tl.store(out_ptr1 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/od/codijdmhhvcrjh35n5vmk6tbh3ggdkc7qnmw7jzmjg5zjjr3cb3t.py
# Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_10 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 320)
    x1 = ((xindex // 320) % 4096)
    x2 = xindex // 1310720
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 320*x1 + 20480*(((x1 % 64)) // 64) + 1310720*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 320*x1 + 20480*(((x1 % 64)) // 64) + 1310720*x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 40960.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/4b/c4bv7342pu37563f2xbymr4gx7uynamgyydw6ivpckno4qp7sum3.py
# Topologically Sorted Source Nodes: [hidden_states_10, norm_hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_9
#   norm_hidden_states => add_10, add_11, convert_element_type_34, convert_element_type_35, mul_14, mul_15, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %arg22_1), kwargs = {})
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_34, %getitem_7), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %arg23_1), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %arg24_1), kwargs = {})
#   %convert_element_type_35 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float16), kwargs = {})
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel):
    xnumel = 131072
    XBLOCK: tl.constexpr = 1
    r0_numel = 320
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 320*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 320, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 320.0
    tmp22 = (tmp19 / tmp21)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 320*x0), tmp33, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ho/choxgeivbkaqfgd2gkvxiim2vhmkejfapc6bgvznvxkzmcq2fbpm.py
# Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_16, hidden_states_17, norm_hidden_states_1], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_9
#   hidden_states_16 => div_2
#   hidden_states_17 => add_12
#   norm_hidden_states_1 => add_13, add_14, convert_element_type_45, convert_element_type_46, mul_16, mul_17, rsqrt_4, sub_4, var_mean_4
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %arg22_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_20, 1.0), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %add_9), kwargs = {})
#   %convert_element_type_45 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_45, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_45, %getitem_18), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_17, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg30_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg31_1), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_14, torch.float16), kwargs = {})
triton_per_fused_add_div_native_layer_norm_14 = async_compile.triton('triton_per_fused_add_div_native_layer_norm_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_layer_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_native_layer_norm_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel):
    xnumel = 131072
    XBLOCK: tl.constexpr = 1
    r0_numel = 320
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 320*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 320*x0), r0_mask, other=0.0).to(tl.float32)
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
    tmp17 = tl.full([1], 320, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = tl.where(r0_mask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 320.0
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
    tl.store(out_ptr2 + (r0_1 + 320*x0), tmp39, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/yq/cyq42azvfjcolrdejq5jrkdit22tyczlmfgelwwvzehzl2njovcj.py
# Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_16, hidden_states_17, hidden_states_23, hidden_states_24, norm_hidden_states_2], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_9
#   hidden_states_16 => div_2
#   hidden_states_17 => add_12
#   hidden_states_23 => div_3
#   hidden_states_24 => add_15
#   norm_hidden_states_2 => add_16, add_17, convert_element_type_56, convert_element_type_57, mul_18, mul_19, rsqrt_5, sub_5, var_mean_5
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %arg22_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_20, 1.0), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %add_9), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_32, 1.0), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %add_12), kwargs = {})
#   %convert_element_type_56 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_56, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_56, %getitem_29), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg37_1), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg38_1), kwargs = {})
#   %convert_element_type_57 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.float16), kwargs = {})
triton_per_fused_add_div_native_layer_norm_15 = async_compile.triton('triton_per_fused_add_div_native_layer_norm_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_layer_norm_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_native_layer_norm_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel):
    xnumel = 131072
    XBLOCK: tl.constexpr = 1
    r0_numel = 320
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 320*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (r0_1 + 320*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 320*x0), r0_mask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp3
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp4 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 320, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [R0_BLOCK])
    tmp29 = tl.where(r0_mask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp14 - tmp24
    tmp32 = 320.0
    tmp33 = (tmp30 / tmp32)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 * tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 320*x0), tmp13, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 320*x0), tmp44, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/zz/czzgp2uxucyokoes7gcl3gv7o2chfwavft4kcdjcgnkfifckejql.py
# Topologically Sorted Source Nodes: [gelu, hidden_states_27], Original ATen: [aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu => add_18, convert_element_type_61, convert_element_type_62, erf, mul_20, mul_21, mul_22
#   hidden_states_27 => mul_23
# Graph fragment:
#   %convert_element_type_61 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_31, torch.float32), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, 0.5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_21,), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %add_18), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.float16), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_30, %convert_element_type_62), kwargs = {})
triton_poi_fused_gelu_mul_16 = async_compile.triton('triton_poi_fused_gelu_mul_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_mul_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 167772160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1280)
    x1 = xindex // 1280
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2560*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1280 + x0 + 2560*x1), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (1280 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ud/cudl3fscyj4nnt3tonogpcqegpqtq6b6jirzlkzlwuzubtjujpj2.py
# Topologically Sorted Source Nodes: [hidden_states_30], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   hidden_states_30 => add_19
# Graph fragment:
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %add_15), kwargs = {})
triton_poi_fused_add_17 = async_compile.triton('triton_poi_fused_add_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ja/cjaguvpun4uvrb3z5mv73pojvrf5uuyy6lheqaxrhpbdbbngqwpi.py
# Topologically Sorted Source Nodes: [sample_3, hidden_states_5, hidden_states_7, add_1, output_tensor, hidden_states_32, output], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_1 => add_6
#   hidden_states_32 => clone_5
#   hidden_states_5 => convert_element_type_27, mul_11, sigmoid_3
#   hidden_states_7 => convolution_2
#   output => add_20
#   output_tensor => div_1
#   sample_3 => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_5,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %sigmoid_3), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.float16), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_27, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, 1.0), kwargs = {})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_5, %div_1), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_18 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 320*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 320*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2 + 320*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tl.store(out_ptr0 + (y0 + 4096*x2 + 1310720*y1), tmp12, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fy/cfyknxobiyunvy7evppmu3asy2rnqhso2brotexcpk2n3z37n6fc.py
# Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_33 => convert_element_type_69, var_mean_6
# Graph fragment:
#   %convert_element_type_69 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_40, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_69, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_19 = async_compile.triton('triton_red_fused_native_group_norm_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_19(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 40960
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 40960*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/23/c23vht4ode2jtf3fc4dvlqwjrojxetmoh23qonr5ohtrb5wlrugg.py
# Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_33 => add_22, mul_25
#   hidden_states_34 => convert_element_type_74, mul_26, sigmoid_4
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_41, %unsqueeze_28), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_25), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_4), kwargs = {})
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_20 = async_compile.triton('triton_poi_fused_native_group_norm_silu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 10), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 10), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 40960.0
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
    tl.store(out_ptr1 + (y0 + 320*x2 + 1310720*y1), tmp19, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/5s/c5sbalokrvbkpujyb4wvwxzajf64ktzgnodk5ubpscrq7pc3vhzs.py
# Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_41 => convert_element_type_86, var_mean_8
# Graph fragment:
#   %convert_element_type_86 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_44, torch.float32), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_86, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_21 = async_compile.triton('triton_red_fused_native_group_norm_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 40960
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_2 = (r0_index % 4096)
        r0_3 = r0_index // 4096
        tmp0 = tl.load(in_ptr0 + (r0_5 + 40960*x4), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 10*x0 + 320*r0_2 + 1310720*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_3 + 10*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/j4/cj4dpna4664kvzsn5v2lm5ykmgymbgpeiqvbtt7u72on63nxkv4q.py
# Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_43 => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_46,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_22 = async_compile.triton('triton_poi_fused_clone_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 320
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4096)
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*(((y0 % 64)) // 64) + 4096*x2 + 1310720*y1), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 320*y0 + 20480*(((y0 % 64)) // 64) + 1310720*y1), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*y1 + (x2 // 10)), ymask & xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*y1 + (x2 // 10)), ymask & xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 40960.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 320*y3), tmp23, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/5i/c5isyhujssneypvsmpnd6lud2csxwvz7q2ud3smgxldxqnijk3um.py
# Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_40, add_7, output_tensor_1, hidden_states_65, output_1], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_7 => add_26
#   hidden_states_38 => convert_element_type_85, mul_30, sigmoid_6
#   hidden_states_40 => convolution_4
#   hidden_states_65 => clone_11
#   output_1 => add_40
#   output_tensor_1 => div_4
# Graph fragment:
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_25,), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, %sigmoid_6), kwargs = {})
#   %convert_element_type_85 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_30, torch.float16), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_85, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %convolution_4), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_26, 1.0), kwargs = {})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_47,), kwargs = {memory_format: torch.contiguous_format})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_11, %div_4), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_23 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_23', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/em/cembjhymt3v4ktcgoz2b66jv76ikqvwkibzhmycmzyjnkjdidfpd.py
# Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_66 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_40, %arg81_1, %arg82_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 327680*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3k/c3kxkzzmu5izawlx3hfsbdjwz6hcg45muk5txkfq53qsjcte4d4b.py
# Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_67 => convert_element_type_127, var_mean_12
# Graph fragment:
#   %convert_element_type_127 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_80, torch.float32), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_127, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_25 = async_compile.triton('triton_red_fused_native_group_norm_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_25(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 10240
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 10240*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/r6/cr6unmecagxnw4bizuj5whlxry5v3wqi4t4ffmfamnem4albbfi5.py
# Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, input_tensor], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_67 => add_42, mul_44
#   hidden_states_68 => convert_element_type_132, mul_45, sigmoid_7
#   input_tensor => convolution_8
# Graph fragment:
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_81, %unsqueeze_48), kwargs = {})
#   %add_42 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_45), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_42,), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %sigmoid_7), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_45, torch.float16), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_5, %arg93_1, %arg94_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_26 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 10), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 10), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 10240.0
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
    tl.store(out_ptr1 + (y0 + 320*x2 + 327680*y1), tmp19, xmask)
    tl.store(out_ptr2 + (y0 + 320*x2 + 327680*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/pb/cpbzteucxq6way7yod2sjkmgzca3zlfy6woq4ea2rx26umginnii.py
# Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_68 => convert_element_type_132, mul_45, sigmoid_7
#   hidden_states_69 => convolution_6
# Graph fragment:
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_42,), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %sigmoid_7), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_45, torch.float16), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_132, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_27 = async_compile.triton('triton_poi_fused_convolution_silu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 204800
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/mb/cmbpbgy6ohcvqnikux7q66gca4hztl3reac5bwagjx4yfljlclxb.py
# Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => convert_element_type_138, var_mean_13
# Graph fragment:
#   %convert_element_type_138 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_82, torch.float32), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_138, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_28 = async_compile.triton('triton_red_fused_native_group_norm_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 20480
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
        r0_2 = (r0_index % 20)
        r0_3 = r0_index // 20
        tmp0 = tl.load(in_ptr0 + (r0_2 + 20*x0 + 640*r0_3 + 655360*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 20*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 20*x4), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 20*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/g2/cg2brn4jk4iu7yyr3wqpfoeuec5s5q5cqswzh5fd6ws7w5rht35h.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_71 => add_45, mul_48
#   hidden_states_72 => convert_element_type_143, mul_49, sigmoid_9
# Graph fragment:
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_83, %unsqueeze_56), kwargs = {})
#   %add_45 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %unsqueeze_53), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_45,), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %sigmoid_9), kwargs = {})
#   %convert_element_type_143 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_49, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_29 = async_compile.triton('triton_poi_fused_native_group_norm_silu_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20971520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 640)
    x2 = xindex // 655360
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 640*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (32*x2 + (x0 // 20)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (32*x2 + (x0 // 20)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 20480.0
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/x7/cx77sx2jhcvwhhv2iv236j435vuugcecpawou7xc5376tu3huwqb.py
# Topologically Sorted Source Nodes: [hidden_states_72, hidden_states_74], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_72 => convert_element_type_143, mul_49, sigmoid_9
#   hidden_states_74 => convolution_7
# Graph fragment:
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_45,), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %sigmoid_9), kwargs = {})
#   %convert_element_type_143 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_49, torch.float16), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_143, %arg91_1, %arg92_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_30 = async_compile.triton('triton_poi_fused_convolution_silu_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 409600
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 640*x2 + 5760*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/br/cbr5cq3x64foqo2nj6f6q4e5w54ny4t5onxb6tpnltzlxldb6yks.py
# Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_75 => convert_element_type_144, var_mean_14
# Graph fragment:
#   %convert_element_type_144 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_84, torch.float32), kwargs = {})
#   %var_mean_14 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_144, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_31 = async_compile.triton('triton_red_fused_native_group_norm_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 20480
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 20)
        r0_3 = r0_index // 20
        tmp0 = tl.load(in_ptr0 + (r0_2 + 20*x0 + 640*r0_3 + 655360*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 20*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 20*x0 + 640*r0_3 + 655360*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 20*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
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
    tl.store(out_ptr0 + (x4), tmp11, xmask)
    tl.store(out_ptr1 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/hk/chkd2tfwnsrm26l4nrwu5iqwhzks3ttvaapl5zwhqfqhbrnxrwjj.py
# Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_77 => clone_13
# Graph fragment:
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_86,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_32 = async_compile.triton('triton_poi_fused_clone_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20971520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 640)
    x1 = ((xindex // 640) % 1024)
    x2 = xindex // 655360
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 640*x1 + 20480*(((x1 % 32)) // 32) + 655360*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 640*x1 + 20480*(((x1 % 32)) // 32) + 655360*x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 20)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (32*x2 + (x0 // 20)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 20480.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3h/c3hugp4c52zgqg7q2urxmr6jooamc2tlpqufwemf26pcfbicjgcv.py
# Topologically Sorted Source Nodes: [hidden_states_77, norm_hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_77 => add_49
#   norm_hidden_states_6 => add_50, add_51, convert_element_type_150, convert_element_type_151, mul_52, mul_53, rsqrt_15, sub_15, var_mean_15
# Graph fragment:
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_88, %arg98_1), kwargs = {})
#   %convert_element_type_150 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_49, torch.float32), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_150, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_150, %getitem_71), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_70, 1e-05), kwargs = {})
#   %rsqrt_15 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_50,), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_15), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %arg99_1), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %arg100_1), kwargs = {})
#   %convert_element_type_151 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.float16), kwargs = {})
triton_per_fused_add_native_layer_norm_33 = async_compile.triton('triton_per_fused_add_native_layer_norm_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel):
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
    tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 640, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 640.0
    tmp22 = (tmp19 / tmp21)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 640*x0), tmp33, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/pj/cpjbbp6mpu2jaxnsl7xbwoyvqksnubmbnyuwan6zg56nkkth3s5z.py
# Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_83, hidden_states_84, norm_hidden_states_7], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_77 => add_49
#   hidden_states_83 => div_8
#   hidden_states_84 => add_52
#   norm_hidden_states_7 => add_53, add_54, convert_element_type_161, convert_element_type_162, mul_54, mul_55, rsqrt_16, sub_16, var_mean_16
# Graph fragment:
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_88, %arg98_1), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_100, 1.0), kwargs = {})
#   %add_52 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_8, %add_49), kwargs = {})
#   %convert_element_type_161 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_161, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_161, %getitem_82), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_81, 1e-05), kwargs = {})
#   %rsqrt_16 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %rsqrt_16), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %arg106_1), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %arg107_1), kwargs = {})
#   %convert_element_type_162 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_54, torch.float16), kwargs = {})
triton_per_fused_add_div_native_layer_norm_34 = async_compile.triton('triton_per_fused_add_div_native_layer_norm_34', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/yg/cygf7twbrmzeih3emwji23veuxq65rdiukshquu6wpkkqrkcv4uf.py
# Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_83, hidden_states_84, hidden_states_90, hidden_states_91, norm_hidden_states_8], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_77 => add_49
#   hidden_states_83 => div_8
#   hidden_states_84 => add_52
#   hidden_states_90 => div_9
#   hidden_states_91 => add_55
#   norm_hidden_states_8 => add_56, add_57, convert_element_type_172, convert_element_type_173, mul_56, mul_57, rsqrt_17, sub_17, var_mean_17
# Graph fragment:
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_88, %arg98_1), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_100, 1.0), kwargs = {})
#   %add_52 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_8, %add_49), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_112, 1.0), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_9, %add_52), kwargs = {})
#   %convert_element_type_172 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.float32), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_172, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_172, %getitem_93), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_92, 1e-05), kwargs = {})
#   %rsqrt_17 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_56,), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %rsqrt_17), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %arg113_1), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %arg114_1), kwargs = {})
#   %convert_element_type_173 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.float16), kwargs = {})
triton_per_fused_add_div_native_layer_norm_35 = async_compile.triton('triton_per_fused_add_div_native_layer_norm_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_layer_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_native_layer_norm_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp3
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp4 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 640, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [R0_BLOCK])
    tmp29 = tl.where(r0_mask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp14 - tmp24
    tmp32 = 640.0
    tmp33 = (tmp30 / tmp32)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 * tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp13, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 640*x0), tmp44, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/zf/czfmlivsjvdof6ekgxkysjcpbmeqnisty7g7jbhp2tpjg2623pt5.py
# Topologically Sorted Source Nodes: [gelu_2, hidden_states_94], Original ATen: [aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_2 => add_58, convert_element_type_177, convert_element_type_178, erf_2, mul_58, mul_59, mul_60
#   hidden_states_94 => mul_61
# Graph fragment:
#   %convert_element_type_177 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_95, torch.float32), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_177, 0.5), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_177, 0.7071067811865476), kwargs = {})
#   %erf_2 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_59,), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %add_58), kwargs = {})
#   %convert_element_type_178 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.float16), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_94, %convert_element_type_178), kwargs = {})
triton_poi_fused_gelu_mul_36 = async_compile.triton('triton_poi_fused_gelu_mul_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_mul_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 83886080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5120*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2560 + x0 + 5120*x1), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (2560 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/jo/cjozzucuclbds5tfprei7agaeyokmoa6jm72vdo5neu3g7nrqhb5.py
# Topologically Sorted Source Nodes: [hidden_states_97], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   hidden_states_97 => add_59
# Graph fragment:
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_116, %add_55), kwargs = {})
triton_poi_fused_add_37 = async_compile.triton('triton_poi_fused_add_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_37(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20971520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/tv/ctvgwiodvio4uutatqj5uorxo6yydnjinc666uwxporr32qqfhn4.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_72, hidden_states_74, add_13, output_tensor_2, hidden_states_99, output_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_13 => add_46
#   hidden_states_72 => convert_element_type_143, mul_49, sigmoid_9
#   hidden_states_74 => convolution_7
#   hidden_states_99 => clone_17
#   input_tensor => convolution_8
#   output_2 => add_60
#   output_tensor_2 => div_7
# Graph fragment:
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_5, %arg93_1, %arg94_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_45,), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %sigmoid_9), kwargs = {})
#   %convert_element_type_143 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_49, torch.float16), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_143, %arg91_1, %arg92_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %convolution_7), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_46, 1.0), kwargs = {})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_70,), kwargs = {memory_format: torch.contiguous_format})
#   %add_60 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_17, %div_7), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_38 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 640*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 640*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2 + 640*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tl.store(out_ptr0 + (y0 + 1024*x2 + 655360*y1), tmp12, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/uc/cucryuv5iimdbvk42ymw7ktsovpxp3exqqmlqxbuf4xfpgry6wm5.py
# Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_100 => convert_element_type_185, var_mean_18
# Graph fragment:
#   %convert_element_type_185 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_120, torch.float32), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_185, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_39 = async_compile.triton('triton_red_fused_native_group_norm_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_39(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 20480
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 20480*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/hm/chmsivwvhzjfzznnogxelnndkdqyf5ap53gfa6sj5226c2jzjnfa.py
# Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_100 => add_62, mul_63
#   hidden_states_101 => convert_element_type_190, mul_64, sigmoid_10
# Graph fragment:
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_121, %unsqueeze_68), kwargs = {})
#   %add_62 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %unsqueeze_65), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_62,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_62, %sigmoid_10), kwargs = {})
#   %convert_element_type_190 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_64, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_40 = async_compile.triton('triton_poi_fused_native_group_norm_silu_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 20), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 20), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 20480.0
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
    tl.store(out_ptr1 + (y0 + 640*x2 + 655360*y1), tmp19, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/de/cdeclovsaoyvv7higk2hr3ylpdodpes3swaiayuarviep7dltqaj.py
# Topologically Sorted Source Nodes: [hidden_states_108], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_108 => convert_element_type_202, var_mean_20
# Graph fragment:
#   %convert_element_type_202 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_124, torch.float32), kwargs = {})
#   %var_mean_20 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_202, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_41 = async_compile.triton('triton_red_fused_native_group_norm_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 20480
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_2 = (r0_index % 1024)
        r0_3 = r0_index // 1024
        tmp0 = tl.load(in_ptr0 + (r0_5 + 20480*x4), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 20*x0 + 640*r0_2 + 655360*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_3 + 20*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/sh/cshpyn7kktbsiwwx7ta5xxu2e4djawvsks4wg334o7kdosujcien.py
# Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_110 => clone_19
# Graph fragment:
#   %clone_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_126,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_42 = async_compile.triton('triton_poi_fused_clone_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*(((y0 % 32)) // 32) + 1024*x2 + 655360*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 640*y0 + 20480*(((y0 % 32)) // 32) + 655360*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*y1 + (x2 // 20)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*y1 + (x2 // 20)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 20480.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 640*y3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/65/c65b4ryb52gfuww54f5thyai5rh5u55pkw7x2hfrdcajddvua62v.py
# Topologically Sorted Source Nodes: [hidden_states_105, hidden_states_107, add_19, output_tensor_3, hidden_states_132, output_3], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_19 => add_66
#   hidden_states_105 => convert_element_type_201, mul_68, sigmoid_12
#   hidden_states_107 => convolution_10
#   hidden_states_132 => clone_23
#   output_3 => add_80
#   output_tensor_3 => div_10
# Graph fragment:
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_65,), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_65, %sigmoid_12), kwargs = {})
#   %convert_element_type_201 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_68, torch.float16), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_201, %arg129_1, %arg130_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %convolution_10), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_66, 1.0), kwargs = {})
#   %clone_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_93,), kwargs = {memory_format: torch.contiguous_format})
#   %add_80 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_23, %div_10), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_43 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_out_ptr0 + (x2 + 640*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 1024*x2 + 655360*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2 + 640*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 640*y3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/eq/ceq6apsuych7ixse3345pksguskbadf2bkqgkndbcbb2kjqdqtn5.py
# Topologically Sorted Source Nodes: [hidden_states_133], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_133 => convolution_11
# Graph fragment:
#   %convolution_11 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_80, %arg157_1, %arg158_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_44 = async_compile.triton('triton_poi_fused_convolution_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_44(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 640*x2 + 163840*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 256*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/5t/c5tmey7twzmwaijlkndf3i27a2w2ubpld6w3rmrw2bjxwky6332m.py
# Topologically Sorted Source Nodes: [hidden_states_134], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_134 => convert_element_type_243, var_mean_24
# Graph fragment:
#   %convert_element_type_243 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_160, torch.float32), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_243, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_45 = async_compile.triton('triton_red_fused_native_group_norm_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_45(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 5120*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/rr/crrq764wq2ncnd66zyjbprdy3ldzbjidj2nnil7isljbvbkygsow.py
# Topologically Sorted Source Nodes: [hidden_states_134, hidden_states_135, input_tensor_1], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_134 => add_82, mul_82
#   hidden_states_135 => convert_element_type_248, mul_83, sigmoid_13
#   input_tensor_1 => convolution_14
# Graph fragment:
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_161, %unsqueeze_88), kwargs = {})
#   %add_82 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %unsqueeze_85), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_82,), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %sigmoid_13), kwargs = {})
#   %convert_element_type_248 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_83, torch.float16), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_11, %arg169_1, %arg170_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_46 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 20), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 20), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 5120.0
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
    tl.store(out_ptr1 + (y0 + 640*x2 + 163840*y1), tmp19, xmask)
    tl.store(out_ptr2 + (y0 + 640*x2 + 163840*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/gv/cgv3kc6c7uvnpk3qamwmovukd32dvc7yl3wxffslbcw7nuwp7vmr.py
# Topologically Sorted Source Nodes: [hidden_states_135, hidden_states_136], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_135 => convert_element_type_248, mul_83, sigmoid_13
#   hidden_states_136 => convolution_12
# Graph fragment:
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_82,), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %sigmoid_13), kwargs = {})
#   %convert_element_type_248 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_83, torch.float16), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_248, %arg161_1, %arg162_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_47 = async_compile.triton('triton_poi_fused_convolution_silu_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 819200
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 640*x2 + 5760*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/4g/c4gw7hjkve27mneooj7t7tkbomsevbhrpa7a2g5mds3dupepgcyl.py
# Topologically Sorted Source Nodes: [hidden_states_138], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_138 => convert_element_type_254, var_mean_25
# Graph fragment:
#   %convert_element_type_254 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_162, torch.float32), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_254, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_48 = async_compile.triton('triton_red_fused_native_group_norm_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 10240
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
        tmp0 = tl.load(in_ptr0 + (r0_2 + 40*x0 + 1280*r0_3 + 327680*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/n5/cn5chjjj2r56uafeerdwljjmdhve3z47azsxklpbqoxjx3oofpc5.py
# Topologically Sorted Source Nodes: [hidden_states_138, hidden_states_139], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_138 => add_85, mul_86
#   hidden_states_139 => convert_element_type_259, mul_87, sigmoid_15
# Graph fragment:
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_163, %unsqueeze_96), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_93), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_85,), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %sigmoid_15), kwargs = {})
#   %convert_element_type_259 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_87, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_49 = async_compile.triton('triton_poi_fused_native_group_norm_silu_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1280)
    x2 = xindex // 327680
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 1280*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 10240.0
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/nw/cnwiwhca36ybw4cftxyd7uoss52xmanbi37txpuul7wskgmlxm2d.py
# Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_141], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_139 => convert_element_type_259, mul_87, sigmoid_15
#   hidden_states_141 => convolution_13
# Graph fragment:
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_85,), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %sigmoid_15), kwargs = {})
#   %convert_element_type_259 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_87, torch.float16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_259, %arg167_1, %arg168_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_50 = async_compile.triton('triton_poi_fused_convolution_silu_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_50(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1638400
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 1280*x2 + 11520*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/xk/cxkh6hnwlcnxrem2utfjxkwjhcnkziqcagfy5lgqtbevc2yhbir6.py
# Topologically Sorted Source Nodes: [hidden_states_142], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_142 => convert_element_type_260, var_mean_26
# Graph fragment:
#   %convert_element_type_260 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_164, torch.float32), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_260, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_51 = async_compile.triton('triton_red_fused_native_group_norm_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 10240
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 40)
        r0_3 = r0_index // 40
        tmp0 = tl.load(in_ptr0 + (r0_2 + 40*x0 + 1280*r0_3 + 327680*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 40*x0 + 1280*r0_3 + 327680*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
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
    tl.store(out_ptr0 + (x4), tmp11, xmask)
    tl.store(out_ptr1 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/xf/cxfbfyuonqcmx2x25uwdiymf65z6qqp5sowc4di6gtqlpswpp3ta.py
# Topologically Sorted Source Nodes: [hidden_states_144], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_144 => clone_25
# Graph fragment:
#   %clone_25 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_166,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_52 = async_compile.triton('triton_poi_fused_clone_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1280)
    x1 = ((xindex // 1280) % 256)
    x2 = xindex // 327680
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1280*x1 + 20480*(((x1 % 16)) // 16) + 327680*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 1280*x1 + 20480*(((x1 % 16)) // 16) + 327680*x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 10240.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/qe/cqezwnmwdion5balv42whrwunv6yoxzu6wxbtcevqkfqcpqk42vj.py
# Topologically Sorted Source Nodes: [hidden_states_144, norm_hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_144 => add_89
#   norm_hidden_states_12 => add_90, add_91, convert_element_type_266, convert_element_type_267, mul_90, mul_91, rsqrt_27, sub_27, var_mean_27
# Graph fragment:
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_168, %arg174_1), kwargs = {})
#   %convert_element_type_266 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_89, torch.float32), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_266, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_266, %getitem_135), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_134, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_90,), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %arg175_1), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %arg176_1), kwargs = {})
#   %convert_element_type_267 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_91, torch.float16), kwargs = {})
triton_red_fused_add_native_layer_norm_53 = async_compile.triton('triton_red_fused_add_native_layer_norm_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 1280
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask, tmp5_weight_next, tmp5_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp5
        tmp16 = 1280.0
        tmp17 = (tmp6 / tmp16)
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1280*x0), tmp28, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fn/cfnl3s5uvyfvbcna3nxpb2rvcp5fabogwgndlsttmemydlskh6zj.py
# Topologically Sorted Source Nodes: [hidden_states_144, hidden_states_150, hidden_states_151, norm_hidden_states_13], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_144 => add_89
#   hidden_states_150 => div_14
#   hidden_states_151 => add_92
#   norm_hidden_states_13 => add_93, add_94, convert_element_type_277, convert_element_type_278, mul_92, mul_93, rsqrt_28, sub_28, var_mean_28
# Graph fragment:
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_168, %arg174_1), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_180, 1.0), kwargs = {})
#   %add_92 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_14, %add_89), kwargs = {})
#   %convert_element_type_277 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_92, torch.float32), kwargs = {})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_277, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_277, %getitem_146), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_145, 1e-05), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_93,), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_28), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_92, %arg182_1), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %arg183_1), kwargs = {})
#   %convert_element_type_278 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_94, torch.float16), kwargs = {})
triton_red_fused_add_div_native_layer_norm_54 = async_compile.triton('triton_red_fused_add_div_native_layer_norm_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_layer_norm_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_native_layer_norm_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 1280
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp11_mean = tl.where(r0_mask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(r0_mask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(r0_mask, tmp11_weight_next, tmp11_weight)
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
        tmp17 = tl.load(in_ptr0 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tl.store(out_ptr2 + (r0_1 + 1280*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/od/codmylqgxjd7lxu334fi5mfispqx4hzwp55rjjgfg5muferm7gio.py
# Topologically Sorted Source Nodes: [hidden_states_144, hidden_states_150, hidden_states_151, hidden_states_157, hidden_states_158, norm_hidden_states_14], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_144 => add_89
#   hidden_states_150 => div_14
#   hidden_states_151 => add_92
#   hidden_states_157 => div_15
#   hidden_states_158 => add_95
#   norm_hidden_states_14 => add_96, add_97, convert_element_type_288, convert_element_type_289, mul_94, mul_95, rsqrt_29, sub_29, var_mean_29
# Graph fragment:
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_168, %arg174_1), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_180, 1.0), kwargs = {})
#   %add_92 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_14, %add_89), kwargs = {})
#   %div_15 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_192, 1.0), kwargs = {})
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_15, %add_92), kwargs = {})
#   %convert_element_type_288 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_95, torch.float32), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_288, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_288, %getitem_157), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_156, 1e-05), kwargs = {})
#   %rsqrt_29 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_96,), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %rsqrt_29), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %arg189_1), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %arg190_1), kwargs = {})
#   %convert_element_type_289 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_97, torch.float16), kwargs = {})
triton_red_fused_add_div_native_layer_norm_55 = async_compile.triton('triton_red_fused_add_div_native_layer_norm_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_layer_norm_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_native_layer_norm_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 1280
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp16_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = 1.0
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7 * tmp3
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tmp4 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight, roffset == 0
        )
        tmp16_mean = tl.where(r0_mask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(r0_mask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(r0_mask, tmp16_weight_next, tmp16_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1280*x0), tmp13, r0_mask)
    tmp19, tmp20, tmp21 = triton_helpers.welford(tmp16_mean, tmp16_m2, tmp16_weight, 1)
    tmp16 = tmp19[:, None]
    tmp17 = tmp20[:, None]
    tmp18 = tmp21[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp22 = tl.load(in_out_ptr0 + (r0_1 + 1280*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp16
        tmp25 = 1280.0
        tmp26 = (tmp17 / tmp25)
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1280*x0), tmp37, r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/bv/cbvpaf46dke4qxsgvlokabopb4tbai63bkk2spxqxn5uyutzuspo.py
# Topologically Sorted Source Nodes: [gelu_4, hidden_states_161], Original ATen: [aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_4 => add_98, convert_element_type_293, convert_element_type_294, erf_4, mul_96, mul_97, mul_98
#   hidden_states_161 => mul_99
# Graph fragment:
#   %convert_element_type_293 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_159, torch.float32), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_293, 0.5), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_293, 0.7071067811865476), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_97,), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %add_98), kwargs = {})
#   %convert_element_type_294 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_98, torch.float16), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_158, %convert_element_type_294), kwargs = {})
triton_poi_fused_gelu_mul_56 = async_compile.triton('triton_poi_fused_gelu_mul_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_mul_56(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5120)
    x1 = xindex // 5120
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 10240*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (5120 + x0 + 10240*x1), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (5120 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/oo/cooxkq6anukcdpum54vgswborwik53sabs2jdknd24clzus5gn6m.py
# Topologically Sorted Source Nodes: [hidden_states_164], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   hidden_states_164 => add_99
# Graph fragment:
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_196, %add_95), kwargs = {})
triton_poi_fused_add_57 = async_compile.triton('triton_poi_fused_add_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_57(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ku/ckukinozunx4nzsdfpgkvdbk4lyhyvja7a522agoae4hcsorvowq.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_139, hidden_states_141, add_25, output_tensor_4, hidden_states_166, output_4], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_25 => add_86
#   hidden_states_139 => convert_element_type_259, mul_87, sigmoid_15
#   hidden_states_141 => convolution_13
#   hidden_states_166 => clone_29
#   input_tensor_1 => convolution_14
#   output_4 => add_100
#   output_tensor_4 => div_13
# Graph fragment:
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_11, %arg169_1, %arg170_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_85,), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %sigmoid_15), kwargs = {})
#   %convert_element_type_259 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_87, torch.float16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_259, %arg167_1, %arg168_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %div_13 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_86, 1.0), kwargs = {})
#   %clone_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_116,), kwargs = {memory_format: torch.contiguous_format})
#   %add_100 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_29, %div_13), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_58 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tl.store(out_ptr0 + (y0 + 256*x2 + 327680*y1), tmp12, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/hi/chizhbb66klavleacyuimsro23otrohyz6sdphdut3f7wgcq74fd.py
# Topologically Sorted Source Nodes: [hidden_states_167, hidden_states_168], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_167 => add_102, mul_101
#   hidden_states_168 => convert_element_type_306, mul_102, sigmoid_16
# Graph fragment:
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_201, %unsqueeze_108), kwargs = {})
#   %add_102 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_105), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_102,), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_102, %sigmoid_16), kwargs = {})
#   %convert_element_type_306 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_102, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_59 = async_compile.triton('triton_poi_fused_native_group_norm_silu_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 40), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 40), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 10240.0
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
    tl.store(out_ptr1 + (y0 + 1280*x2 + 327680*y1), tmp19, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/mr/cmrwxxxigiyk4iynmqu4gitmc3bh2aktxudko2lcwyxtplezpqfj.py
# Topologically Sorted Source Nodes: [hidden_states_175], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_175 => convert_element_type_318, var_mean_32
# Graph fragment:
#   %convert_element_type_318 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_204, torch.float32), kwargs = {})
#   %var_mean_32 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_318, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_60 = async_compile.triton('triton_red_fused_native_group_norm_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_60(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 10240
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_2 = (r0_index % 256)
        r0_3 = r0_index // 256
        tmp0 = tl.load(in_ptr0 + (r0_5 + 10240*x4), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 40*x0 + 1280*r0_2 + 327680*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_3 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/hk/chkposale2uinhag4z6vcmzvka66nuxcwkz7bqbqjopflze5rci4.py
# Topologically Sorted Source Nodes: [hidden_states_177], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_177 => clone_31
# Graph fragment:
#   %clone_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_206,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_61 = async_compile.triton('triton_poi_fused_clone_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*(((y0 % 16)) // 16) + 256*x2 + 327680*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 1280*y0 + 20480*(((y0 % 16)) // 16) + 327680*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*y1 + (x2 // 40)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*y1 + (x2 // 40)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 10240.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 1280*y3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/77/c77dsun7vrq5g3htpju3mhvxjrvjh6zfc5xfd2goufhhouh344i5.py
# Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_174, add_31, output_tensor_5, hidden_states_199, output_5], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_31 => add_106
#   hidden_states_172 => convert_element_type_317, mul_106, sigmoid_18
#   hidden_states_174 => convolution_16
#   hidden_states_199 => clone_35
#   output_5 => add_120
#   output_tensor_5 => div_16
# Graph fragment:
#   %sigmoid_18 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_105,), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, %sigmoid_18), kwargs = {})
#   %convert_element_type_317 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_106, torch.float16), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_317, %arg205_1, %arg206_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_100, %convolution_16), kwargs = {})
#   %div_16 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_106, 1.0), kwargs = {})
#   %clone_35 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_139,), kwargs = {memory_format: torch.contiguous_format})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_35, %div_16), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_62 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 256*x2 + 327680*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1280*y3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fa/cfavuqpgosoir25vxzk6vrckrd7z6qenceh2c2r3blthhrtbvl2g.py
# Topologically Sorted Source Nodes: [hidden_states_201], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_201 => convert_element_type_359, var_mean_36
# Graph fragment:
#   %convert_element_type_359 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_240, torch.float32), kwargs = {})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_359, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_63 = async_compile.triton('triton_red_fused_native_group_norm_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_63(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 40)
        r0_3 = r0_index // 40
        tmp0 = tl.load(in_ptr0 + (r0_2 + 40*x0 + 1280*r0_3 + 81920*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/4q/c4qn5bzpsdrb3i2skg6fnsyqpotobegpkwlabrvpipxvnxlralyn.py
# Topologically Sorted Source Nodes: [hidden_states_201, hidden_states_202], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_201 => add_122, mul_120
#   hidden_states_202 => convert_element_type_364, mul_121, sigmoid_19
# Graph fragment:
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_241, %unsqueeze_128), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_125), kwargs = {})
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_19), kwargs = {})
#   %convert_element_type_364 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_121, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_64 = async_compile.triton('triton_poi_fused_native_group_norm_silu_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1280)
    x2 = xindex // 81920
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 2560.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3o/c3o4s7xcg7yi3nltyqac33yuzyei2st7v6bfkreemf2szix4d2f5.py
# Topologically Sorted Source Nodes: [sample_2, temb_12, temb_14, temb_16], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   sample_2 => add_tensor_102
#   temb_12 => convert_element_type_365, convert_element_type_366, mul_122, sigmoid_20
#   temb_14 => convert_element_type_382, convert_element_type_383, mul_129, sigmoid_23
#   temb_16 => convert_element_type_399, convert_element_type_400, mul_136, sigmoid_26
# Graph fragment:
#   %add_tensor_102 : [num_users=22] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_102, %arg5_1), kwargs = {})
#   %convert_element_type_365 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_365,), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_365, %sigmoid_20), kwargs = {})
#   %convert_element_type_366 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_122, torch.float16), kwargs = {})
#   %convert_element_type_382 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_382,), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_382, %sigmoid_23), kwargs = {})
#   %convert_element_type_383 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_129, torch.float16), kwargs = {})
#   %convert_element_type_399 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_26 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_399,), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_399, %sigmoid_26), kwargs = {})
#   %convert_element_type_400 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_136, torch.float16), kwargs = {})
triton_poi_fused_addmm_silu_65 = async_compile.triton('triton_poi_fused_addmm_silu_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_65(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
    tl.store(out_ptr2 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ky/cky3gwuk6fhuv64464nd7qeax6j5omvrojteg6yj4ziihvf5l65k.py
# Topologically Sorted Source Nodes: [hidden_states_205], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_205 => convert_element_type_370, var_mean_37
# Graph fragment:
#   %convert_element_type_370 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_242, torch.float32), kwargs = {})
#   %var_mean_37 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_370, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_66 = async_compile.triton('triton_red_fused_native_group_norm_66', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
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
        tmp0 = tl.load(in_ptr0 + (r0_2 + 40*x0 + 1280*r0_3 + 81920*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/jp/cjp5dpqumlzibz6n5hantfeoysvvs6uhr5aojmr5yte2h2hoxghh.py
# Topologically Sorted Source Nodes: [hidden_states_205, hidden_states_206], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_205 => add_125, mul_124
#   hidden_states_206 => convert_element_type_375, mul_125, sigmoid_21
# Graph fragment:
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_243, %unsqueeze_136), kwargs = {})
#   %add_125 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %unsqueeze_133), kwargs = {})
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_125,), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_125, %sigmoid_21), kwargs = {})
#   %convert_element_type_375 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_125, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_67 = async_compile.triton('triton_poi_fused_native_group_norm_silu_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1280)
    x2 = xindex // 81920
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 1280*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 2560.0
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/2a/c2a7lsuigbs3vhsdpgsspgofp5mxq2sdzeh6q5jwqv2ooesejpyd.py
# Topologically Sorted Source Nodes: [hidden_states_209], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_209 => convert_element_type_376, var_mean_38
# Graph fragment:
#   %convert_element_type_376 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_244, torch.float32), kwargs = {})
#   %var_mean_38 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_376, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_68 = async_compile.triton('triton_red_fused_native_group_norm_68', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 40)
        r0_3 = r0_index // 40
        tmp0 = tl.load(in_ptr0 + (r0_2 + 40*x0 + 1280*r0_3 + 81920*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 40*x0 + 1280*r0_3 + 81920*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
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
    tl.store(out_ptr0 + (x4), tmp11, xmask)
    tl.store(out_ptr1 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ml/cmlajkhbjt5tvd3it3y6bqnrxoass6oomun7wivluucftxxfdptm.py
# Topologically Sorted Source Nodes: [hidden_states_209, hidden_states_210], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_209 => add_128, mul_127
#   hidden_states_210 => convert_element_type_381, mul_128, sigmoid_22
# Graph fragment:
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_245, %unsqueeze_142), kwargs = {})
#   %add_128 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %unsqueeze_139), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_128,), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %sigmoid_22), kwargs = {})
#   %convert_element_type_381 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_128, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_69 = async_compile.triton('triton_poi_fused_native_group_norm_silu_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1280)
    x2 = xindex // 81920
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (32*x2 + (x0 // 40)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 2560.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/yk/cykt5ivn5ev6pkpn3eonk26j6onrh34xgehsldtyzzvnbnuchcl4.py
# Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_206, hidden_states_208, add_37, output_tensor_6, hidden_states_214, hidden_states_216, add_39, output_tensor_7], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_37 => add_126
#   add_39 => add_132
#   hidden_states_200 => convolution_17
#   hidden_states_206 => convert_element_type_375, mul_125, sigmoid_21
#   hidden_states_208 => convolution_19
#   hidden_states_214 => convert_element_type_392, mul_132, sigmoid_24
#   hidden_states_216 => convolution_21
#   output_tensor_6 => div_19
#   output_tensor_7 => div_20
# Graph fragment:
#   %convolution_17 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_120, %arg233_1, %arg234_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_125,), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_125, %sigmoid_21), kwargs = {})
#   %convert_element_type_375 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_125, torch.float16), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_375, %arg243_1, %arg244_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %convolution_19), kwargs = {})
#   %div_19 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_126, 1.0), kwargs = {})
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_24), kwargs = {})
#   %convert_element_type_392 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_132, torch.float16), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_392, %arg253_1, %arg254_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_19, %convolution_21), kwargs = {})
#   %div_20 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_132, 1.0), kwargs = {})
triton_poi_fused_add_convolution_div_silu_70 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x2 + 1280*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp7
    tl.store(out_ptr0 + (y0 + 64*x2 + 81920*y1), tmp13, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/tz/ctzipwvvkapgnncj337q2amcp5m4cfuk6us6hfeztpbxnoxonhic.py
# Topologically Sorted Source Nodes: [hidden_states_217], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_217 => convert_element_type_393, var_mean_40
# Graph fragment:
#   %convert_element_type_393 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_248, torch.float32), kwargs = {})
#   %var_mean_40 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_393, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_71 = async_compile.triton('triton_red_fused_native_group_norm_71', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_71(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/jn/cjnpgbeyzdxxqli6ci5qnm7qzjtcatmnp5k2rxy54ov4rduesdci.py
# Topologically Sorted Source Nodes: [hidden_states_217, hidden_states_218], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_217 => add_134, mul_134
#   hidden_states_218 => convert_element_type_398, mul_135, sigmoid_25
# Graph fragment:
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_249, %unsqueeze_156), kwargs = {})
#   %add_134 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_153), kwargs = {})
#   %sigmoid_25 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_134,), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_134, %sigmoid_25), kwargs = {})
#   %convert_element_type_398 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_135, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_72 = async_compile.triton('triton_poi_fused_native_group_norm_silu_72', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 40), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 40), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 2560.0
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
    tl.store(out_ptr1 + (y0 + 1280*x2 + 81920*y1), tmp19, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3l/c3lwzplkjjznzppjdcxxjj2hrhtibnsmdeh6nqujs2pswgx2acnx.py
# Topologically Sorted Source Nodes: [hidden_states_225], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_225 => convert_element_type_410, var_mean_42
# Graph fragment:
#   %convert_element_type_410 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_252, torch.float32), kwargs = {})
#   %var_mean_42 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_410, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_73 = async_compile.triton('triton_red_fused_native_group_norm_73', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_73(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_2 = (r0_index % 64)
        r0_3 = r0_index // 64
        tmp0 = tl.load(in_ptr0 + (r0_5 + 2560*x4), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 40*x0 + 1280*r0_2 + 81920*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_3 + 40*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/mt/cmt3up2axl5o75qao52gjbxfm3xvsa6eaft37y2mlxjuy6zzp5qj.py
# Topologically Sorted Source Nodes: [hidden_states_227], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_227 => clone_39
# Graph fragment:
#   %clone_39 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_254,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_74 = async_compile.triton('triton_poi_fused_clone_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8*(((y0 % 8)) // 8) + 64*x2 + 81920*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 1280*y0 + 10240*(((y0 % 8)) // 8) + 81920*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*y1 + (x2 // 40)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*y1 + (x2 // 40)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 2560.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 1280*y3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/t4/ct4p2pndy4irctkgzi5v4wyoo6yqbcq4rgpfssajsycolbdabz4v.py
# Topologically Sorted Source Nodes: [hidden_states_227, norm_hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_227 => add_141
#   norm_hidden_states_18 => add_142, add_143, convert_element_type_416, convert_element_type_417, mul_142, mul_143, rsqrt_43, sub_43, var_mean_43
# Graph fragment:
#   %add_141 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_256, %arg268_1), kwargs = {})
#   %convert_element_type_416 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_141, torch.float32), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_416, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_416, %getitem_207), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_206, 1e-05), kwargs = {})
#   %rsqrt_43 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_142,), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %rsqrt_43), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %arg269_1), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %arg270_1), kwargs = {})
#   %convert_element_type_417 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_143, torch.float16), kwargs = {})
triton_red_fused_add_native_layer_norm_75 = async_compile.triton('triton_red_fused_add_native_layer_norm_75', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_75(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp5
        tmp16 = 1280.0
        tmp17 = (tmp6 / tmp16)
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1280*x0), tmp28, xmask & r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/rf/crfb553vvirfl6n4glk6edam3pqxoyrz7yurrrmfr3m4yeyhyuas.py
# Topologically Sorted Source Nodes: [hidden_states_227, hidden_states_233, hidden_states_234, norm_hidden_states_19], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_227 => add_141
#   hidden_states_233 => div_22
#   hidden_states_234 => add_144
#   norm_hidden_states_19 => add_145, add_146, convert_element_type_427, convert_element_type_428, mul_144, mul_145, rsqrt_44, sub_44, var_mean_44
# Graph fragment:
#   %add_141 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_256, %arg268_1), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_268, 1.0), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_22, %add_141), kwargs = {})
#   %convert_element_type_427 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_144, torch.float32), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_427, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_427, %getitem_218), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_217, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_145,), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %rsqrt_44), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_144, %arg276_1), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_145, %arg277_1), kwargs = {})
#   %convert_element_type_428 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_146, torch.float16), kwargs = {})
triton_red_fused_add_div_native_layer_norm_76 = async_compile.triton('triton_red_fused_add_div_native_layer_norm_76', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3r/c3rh74jlfo6vhlgw63jvmii2yz2ai575jbma227kpt7arurd6tmk.py
# Topologically Sorted Source Nodes: [hidden_states_227, hidden_states_233, hidden_states_234, hidden_states_240, hidden_states_241, norm_hidden_states_20], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_227 => add_141
#   hidden_states_233 => div_22
#   hidden_states_234 => add_144
#   hidden_states_240 => div_23
#   hidden_states_241 => add_147
#   norm_hidden_states_20 => add_148, add_149, convert_element_type_438, convert_element_type_439, mul_146, mul_147, rsqrt_45, sub_45, var_mean_45
# Graph fragment:
#   %add_141 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_256, %arg268_1), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_268, 1.0), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_22, %add_141), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_280, 1.0), kwargs = {})
#   %add_147 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_23, %add_144), kwargs = {})
#   %convert_element_type_438 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_147, torch.float32), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_438, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_438, %getitem_229), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_228, 1e-05), kwargs = {})
#   %rsqrt_45 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_148,), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %rsqrt_45), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %arg283_1), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_147, %arg284_1), kwargs = {})
#   %convert_element_type_439 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_149, torch.float16), kwargs = {})
triton_red_fused_add_div_native_layer_norm_77 = async_compile.triton('triton_red_fused_add_div_native_layer_norm_77', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_layer_norm_77', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_native_layer_norm_77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp16_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = 1.0
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7 * tmp3
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tmp4 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight, roffset == 0
        )
        tmp16_mean = tl.where(r0_mask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(r0_mask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(r0_mask & xmask, tmp16_weight_next, tmp16_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1280*x0), tmp13, xmask & r0_mask)
    tmp19, tmp20, tmp21 = triton_helpers.welford(tmp16_mean, tmp16_m2, tmp16_weight, 1)
    tmp16 = tmp19[:, None]
    tmp17 = tmp20[:, None]
    tmp18 = tmp21[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp22 = tl.load(in_out_ptr0 + (r0_1 + 1280*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp16
        tmp25 = 1280.0
        tmp26 = (tmp17 / tmp25)
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1280*x0), tmp37, xmask & r0_mask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/mn/cmnaiukr3ylghmhmwywe73kp4es72rnsivxhsvohjaejgbp3pwrd.py
# Topologically Sorted Source Nodes: [gelu_6, hidden_states_244], Original ATen: [aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu_6 => add_150, convert_element_type_443, convert_element_type_444, erf_6, mul_148, mul_149, mul_150
#   hidden_states_244 => mul_151
# Graph fragment:
#   %convert_element_type_443 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_231, torch.float32), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_443, 0.5), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_443, 0.7071067811865476), kwargs = {})
#   %erf_6 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_149,), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %add_150), kwargs = {})
#   %convert_element_type_444 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_150, torch.float16), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_230, %convert_element_type_444), kwargs = {})
triton_poi_fused_gelu_mul_78 = async_compile.triton('triton_poi_fused_gelu_mul_78', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_78', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_mul_78(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5120)
    x1 = xindex // 5120
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 10240*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (5120 + x0 + 10240*x1), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (5120 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/cy/ccyyzqd6jfxibuhbusmyogqsim67lgnopc6gmdhgjjmlqv5wyr6l.py
# Topologically Sorted Source Nodes: [hidden_states_247], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   hidden_states_247 => add_151
# Graph fragment:
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_284, %add_147), kwargs = {})
triton_poi_fused_add_79 = async_compile.triton('triton_poi_fused_add_79', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_79', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_79(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fa/cfatcmmlqt6d3gpf3fsccm7p3oqkodanq43euuponh7ugitc3az3.py
# Topologically Sorted Source Nodes: [hidden_states_222, hidden_states_224, add_41, output_tensor_8, hidden_states_249, output_6], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_41 => add_138
#   hidden_states_222 => convert_element_type_409, mul_139, sigmoid_27
#   hidden_states_224 => convolution_23
#   hidden_states_249 => clone_43
#   output_6 => add_152
#   output_tensor_8 => div_21
# Graph fragment:
#   %sigmoid_27 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_27), kwargs = {})
#   %convert_element_type_409 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_139, torch.float16), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_409, %arg263_1, %arg264_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_20, %convolution_23), kwargs = {})
#   %div_21 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_138, 1), kwargs = {})
#   %clone_43 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_164,), kwargs = {memory_format: torch.contiguous_format})
#   %add_152 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_43, %div_21), kwargs = {})
triton_poi_fused_add_clone_convolution_div_silu_80 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_silu_80', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_silu_80', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_silu_80(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1280*x2 + 81920*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 64*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (y0 + 1280*x2 + 81920*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tl.store(out_ptr0 + (x2 + 64*y3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/cw/ccwgnqbz7su5rencynlkpk5yqvvbrdaz7nbosxrpwhubem67gixq.py
# Topologically Sorted Source Nodes: [sample_2, temb_18, temb_20, temb_22, temb_24], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   sample_2 => add_tensor_102
#   temb_18 => convert_element_type_457, convert_element_type_458, mul_155, sigmoid_29
#   temb_20 => convert_element_type_474, convert_element_type_475, mul_162, sigmoid_32
#   temb_22 => convert_element_type_491, convert_element_type_492, mul_169, sigmoid_35
#   temb_24 => convert_element_type_508, convert_element_type_509, mul_176, sigmoid_38
# Graph fragment:
#   %add_tensor_102 : [num_users=22] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_102, %arg5_1), kwargs = {})
#   %convert_element_type_457 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_29 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_457,), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_457, %sigmoid_29), kwargs = {})
#   %convert_element_type_458 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_155, torch.float16), kwargs = {})
#   %convert_element_type_474 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_32 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_474,), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_474, %sigmoid_32), kwargs = {})
#   %convert_element_type_475 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_162, torch.float16), kwargs = {})
#   %convert_element_type_491 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_491,), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_491, %sigmoid_35), kwargs = {})
#   %convert_element_type_492 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_169, torch.float16), kwargs = {})
#   %convert_element_type_508 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_38 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_508,), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_508, %sigmoid_38), kwargs = {})
#   %convert_element_type_509 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_176, torch.float16), kwargs = {})
triton_poi_fused_addmm_silu_81 = async_compile.triton('triton_poi_fused_addmm_silu_81', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_81', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_81(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
    tl.store(out_ptr2 + (x2), tmp6, None)
    tl.store(out_ptr3 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/vz/cvz7ltdu4jactcwplp7i3duqv6onp5qrzu47edel5xqqkcgc4dw2.py
# Topologically Sorted Source Nodes: [hidden_states_258, hidden_states_259], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_258 => cat_2
#   hidden_states_259 => convert_element_type_468, var_mean_48
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%div_24, %div_20], 1), kwargs = {})
#   %convert_element_type_468 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_292, torch.float32), kwargs = {})
#   %var_mean_48 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_468, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_82 = async_compile.triton('triton_red_fused_cat_native_group_norm_82', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
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
        tmp5 = tl.load(in_ptr0 + (r0_2 + 64*(r0_3 + 80*x0) + 81920*x1), xmask & r0_mask & tmp4, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp17 = tl.load(in_ptr3 + (r0_2 + 64*((-1280) + r0_3 + 80*x0) + 81920*x1), xmask & r0_mask & tmp14, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.where(tmp4, tmp13, tmp17)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight, roffset == 0
        )
        tmp21_mean = tl.where(r0_mask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(r0_mask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(r0_mask & xmask, tmp21_weight_next, tmp21_weight)
        tl.store(out_ptr0 + (r0_5 + 5120*x4), tmp18, xmask & r0_mask)
    tmp24, tmp25, tmp26 = triton_helpers.welford(tmp21_mean, tmp21_m2, tmp21_weight, 1)
    tmp21 = tmp24[:, None]
    tmp22 = tmp25[:, None]
    tmp23 = tmp26[:, None]
    tl.store(out_ptr1 + (x4), tmp21, xmask)
    tl.store(out_ptr2 + (x4), tmp22, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/7t/c7tyhgalkqqi4bwpoawuhnoirue23ppkgmvapm6s556h3d4hiifk.py
# Topologically Sorted Source Nodes: [hidden_states_259, hidden_states_260, input_tensor_2], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_259 => add_160, mul_160
#   hidden_states_260 => convert_element_type_473, mul_161, sigmoid_31
#   input_tensor_2 => convolution_28
# Graph fragment:
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_293, %unsqueeze_190), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_187), kwargs = {})
#   %sigmoid_31 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_160,), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, %sigmoid_31), kwargs = {})
#   %convert_element_type_473 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_161, torch.float16), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg311_1, %arg312_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_83 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_83', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_83', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_83(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 81920
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2560)
    y1 = yindex // 2560
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 80), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 80), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 5120.0
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
    tl.store(out_ptr1 + (y0 + 2560*x2 + 163840*y1), tmp19, ymask & xmask)
    tl.store(out_ptr2 + (y0 + 2560*x2 + 163840*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/mc/cmc3ilhjlvsetav7lamxxcvpphpd7fgmhdbxuyfrsjanptdar2zu.py
# Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_260 => convert_element_type_473, mul_161, sigmoid_31
#   hidden_states_261 => convolution_26
# Graph fragment:
#   %sigmoid_31 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_160,), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, %sigmoid_31), kwargs = {})
#   %convert_element_type_473 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_161, torch.float16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_473, %arg303_1, %arg304_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_84 = async_compile.triton('triton_poi_fused_convolution_silu_84', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_84', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_84(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3276800
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2560)
    y1 = yindex // 2560
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 2560*x2 + 23040*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3m/c3mhvz675xtp4ayywkckyu4n2vvkmnfx34bcwmehlgyng7kascjn.py
# Topologically Sorted Source Nodes: [hidden_states_267, hidden_states_268], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_267 => cat_3
#   hidden_states_268 => convert_element_type_485, var_mean_50
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%div_25, %div_19], 1), kwargs = {})
#   %convert_element_type_485 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_296, torch.float32), kwargs = {})
#   %var_mean_50 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_485, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_85 = async_compile.triton('triton_red_fused_cat_native_group_norm_85', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_85', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_85(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp33_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp33_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp33_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
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
        tmp5 = tl.load(in_ptr0 + (1280*r0_2 + 81920*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (1280*r0_2 + 81920*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = 1.0
        tmp13 = tmp11 * tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp4, tmp13, tmp14)
        tmp16 = tmp0 >= tmp3
        tmp17 = tl.full([1, 1], 2560, tl.int64)
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr4 + (1280*r0_2 + 81920*x1 + ((-1280) + r0_3 + 80*x0)), xmask & r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr5 + ((-1280) + r0_3 + 80*x0), xmask & r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.load(in_ptr6 + (1280*r0_2 + 81920*x1 + ((-1280) + r0_3 + 80*x0)), xmask & r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr7 + ((-1280) + r0_3 + 80*x0), xmask & r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp21 + tmp24
        tmp26 = 1.0
        tmp27 = tmp25 * tmp26
        tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
        tmp29 = tl.where(tmp16, tmp27, tmp28)
        tmp30 = tl.where(tmp4, tmp15, tmp29)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp33_mean_next, tmp33_m2_next, tmp33_weight_next = triton_helpers.welford_reduce(
            tmp32, tmp33_mean, tmp33_m2, tmp33_weight, roffset == 0
        )
        tmp33_mean = tl.where(r0_mask & xmask, tmp33_mean_next, tmp33_mean)
        tmp33_m2 = tl.where(r0_mask & xmask, tmp33_m2_next, tmp33_m2)
        tmp33_weight = tl.where(r0_mask & xmask, tmp33_weight_next, tmp33_weight)
        tl.store(out_ptr0 + (r0_5 + 5120*x4), tmp30, xmask & r0_mask)
    tmp36, tmp37, tmp38 = triton_helpers.welford(tmp33_mean, tmp33_m2, tmp33_weight, 1)
    tmp33 = tmp36[:, None]
    tmp34 = tmp37[:, None]
    tmp35 = tmp38[:, None]
    tl.store(out_ptr1 + (x4), tmp33, xmask)
    tl.store(out_ptr2 + (x4), tmp34, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/r7/cr7y4bmmjqkrvkufyi3oc5p4izdn3s3df6svskcyws7awycwstqc.py
# Topologically Sorted Source Nodes: [hidden_states_276, hidden_states_277], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_276 => cat_4
#   hidden_states_277 => convert_element_type_502, var_mean_52
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%div_26, %convolution_17], 1), kwargs = {})
#   %convert_element_type_502 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_300, torch.float32), kwargs = {})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_502, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_86 = async_compile.triton('triton_red_fused_cat_native_group_norm_86', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_86', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_86(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
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
        tmp5 = tl.load(in_ptr0 + (1280*r0_2 + 81920*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (1280*r0_2 + 81920*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = 1.0
        tmp13 = tmp11 * tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp4, tmp13, tmp14)
        tmp16 = tmp0 >= tmp3
        tmp17 = tl.full([1, 1], 2560, tl.int64)
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr4 + (1280*r0_2 + 81920*x1 + ((-1280) + r0_3 + 80*x0)), xmask & r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr5 + ((-1280) + r0_3 + 80*x0), xmask & r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
        tmp23 = tl.where(tmp16, tmp21, tmp22)
        tmp24 = tl.where(tmp4, tmp15, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
        tl.store(out_ptr0 + (r0_5 + 5120*x4), tmp24, xmask & r0_mask)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp30[:, None]
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/l2/cl2pelpseci76bkw35pvln5tps26n5d64fvnlsocfu7imsuxgvsc.py
# Topologically Sorted Source Nodes: [input_tensor_4, hidden_states_282, hidden_states_284, add_53, output_tensor_12, hidden_states_285], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_53 => add_176
#   hidden_states_282 => convert_element_type_518, mul_179, sigmoid_39
#   hidden_states_284 => convolution_33
#   hidden_states_285 => _unsafe_index, convert_element_type_519, convert_element_type_524
#   input_tensor_4 => convolution_34
#   output_tensor_12 => div_27
# Graph fragment:
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %arg335_1, %arg336_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_39 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_175,), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, %sigmoid_39), kwargs = {})
#   %convert_element_type_518 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_179, torch.float16), kwargs = {})
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_518, %arg333_1, %arg334_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_34, %convolution_33), kwargs = {})
#   %div_27 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_176, 1.0), kwargs = {})
#   %convert_element_type_519 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_27, torch.float32), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_519, [None, None, %unsqueeze_227, %convert_element_type_523]), kwargs = {})
#   %convert_element_type_524 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index, torch.float16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_87 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_87', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/6c/c6cmirne7yd2qqy6uryq33xto5awamqk3e5nnrcq3ly57xwf5wub.py
# Topologically Sorted Source Nodes: [hidden_states_287, hidden_states_288], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_287 => cat_5
#   hidden_states_288 => convert_element_type_525, var_mean_54
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_35, %add_120], 1), kwargs = {})
#   %convert_element_type_525 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_304, torch.float32), kwargs = {})
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_525, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_88 = async_compile.triton('triton_red_fused_cat_native_group_norm_88', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_88', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_88(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 20480
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
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 256
        r0_2 = (r0_index % 256)
        r0_5 = r0_index
        tmp0 = r0_3 + 80*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1280, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (1280*r0_2 + 327680*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 2560, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr2 + (1280*r0_2 + 327680*x1 + ((-1280) + r0_3 + 80*x0)), xmask & r0_mask & tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.where(tmp4, tmp9, tmp13)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(r0_mask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(r0_mask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(r0_mask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r0_5 + 20480*x4), tmp14, xmask & r0_mask)
    tmp20, tmp21, tmp22 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp20[:, None]
    tmp18 = tmp21[:, None]
    tmp19 = tmp22[:, None]
    tl.store(out_ptr1 + (x4), tmp17, xmask)
    tl.store(out_ptr2 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/g6/cg6c76xrtavs4yxzotcinqgjk7ugkusgz2in7imfiqt5lhfywagb.py
# Topologically Sorted Source Nodes: [hidden_states_288, hidden_states_289, input_tensor_5], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_288 => add_182, mul_185
#   hidden_states_289 => convert_element_type_530, mul_186, sigmoid_40
#   input_tensor_5 => convolution_38
# Graph fragment:
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_305, %unsqueeze_233), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_230), kwargs = {})
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_182,), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %sigmoid_40), kwargs = {})
#   %convert_element_type_530 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_186, torch.float16), kwargs = {})
#   %convolution_38 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %arg349_1, %arg350_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_89 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_89', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_89', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_89(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 81920
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2560)
    y1 = yindex // 2560
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 80), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 80), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 20480.0
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
    tl.store(out_ptr1 + (y0 + 2560*x2 + 655360*y1), tmp19, ymask & xmask)
    tl.store(out_ptr2 + (y0 + 2560*x2 + 655360*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3e/c3ekvrnw7epj74swnuch3wdt43wdddt3ucxhdezt53dqob3tgad7.py
# Topologically Sorted Source Nodes: [hidden_states_321, hidden_states_322], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_321 => cat_6
#   hidden_states_322 => convert_element_type_583, var_mean_60
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_200, %add_100], 1), kwargs = {})
#   %convert_element_type_583 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_344, torch.float32), kwargs = {})
#   %var_mean_60 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_583, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_90 = async_compile.triton('triton_red_fused_cat_native_group_norm_90', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_90', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_90(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 20480
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
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 256
        r0_2 = (r0_index % 256)
        r0_5 = r0_index
        tmp0 = r0_3 + 80*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1280, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (1280*r0_2 + 327680*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (1280*r0_2 + 327680*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.load(in_ptr4 + (1280*r0_2 + 327680*x1 + (r0_3 + 80*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (r0_3 + 80*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1, 1], 2560, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr6 + (r0_2 + 256*((-1280) + r0_3 + 80*x0) + 327680*x1), xmask & r0_mask & tmp20, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.where(tmp4, tmp19, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
        tl.store(out_ptr0 + (r0_5 + 20480*x4), tmp24, xmask & r0_mask)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp30[:, None]
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/uw/cuwqd2of2csqvcgyohsdt5eibdn6ada24rpv2ldwu7hcqpo4xccx.py
# Topologically Sorted Source Nodes: [hidden_states_355, hidden_states_356], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_355 => cat_7
#   hidden_states_356 => convert_element_type_641, var_mean_66
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_220, %convolution_11], 1), kwargs = {})
#   %convert_element_type_641 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_384, torch.float32), kwargs = {})
#   %var_mean_66 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_641, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_91 = async_compile.triton('triton_red_fused_cat_native_group_norm_91', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 16384},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_91', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_91(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 15360
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
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 256
        r0_2 = (r0_index % 256)
        r0_5 = r0_index
        tmp0 = r0_3 + 60*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1280, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (1280*r0_2 + 327680*x1 + (r0_3 + 60*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 60*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (1280*r0_2 + 327680*x1 + (r0_3 + 60*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 60*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.load(in_ptr4 + (1280*r0_2 + 327680*x1 + (r0_3 + 60*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (r0_3 + 60*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1, 1], 1920, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr6 + (r0_2 + 256*((-1280) + r0_3 + 60*x0) + 163840*x1), xmask & r0_mask & tmp20, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.where(tmp4, tmp19, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
        tl.store(out_ptr0 + (r0_5 + 15360*x4), tmp24, xmask & r0_mask)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp30[:, None]
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/id/cidvv4jgaxc7idsku52oda5qllka5osxngvdjwsb7e2gd3wnsqmo.py
# Topologically Sorted Source Nodes: [hidden_states_356, hidden_states_357, input_tensor_7], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_356 => add_222, mul_223
#   hidden_states_357 => convert_element_type_646, mul_224, sigmoid_46
#   input_tensor_7 => convolution_44
# Graph fragment:
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_385, %unsqueeze_273), kwargs = {})
#   %add_222 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_223, %unsqueeze_270), kwargs = {})
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_222,), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_222, %sigmoid_46), kwargs = {})
#   %convert_element_type_646 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_224, torch.float16), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_7, %arg425_1, %arg426_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_92 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_92', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/tq/ctqcppbww3ixer4u3e2ktpsb2ykjufihoym2kgti26k2m4ycr6jl.py
# Topologically Sorted Source Nodes: [hidden_states_357, hidden_states_358], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_357 => convert_element_type_646, mul_224, sigmoid_46
#   hidden_states_358 => convolution_42
# Graph fragment:
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_222,), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_222, %sigmoid_46), kwargs = {})
#   %convert_element_type_646 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_224, torch.float16), kwargs = {})
#   %convolution_42 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_646, %arg417_1, %arg418_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_93 = async_compile.triton('triton_poi_fused_convolution_silu_93', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_93', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_93(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2457600
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1920)
    y1 = yindex // 1920
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 1920*x2 + 17280*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/su/csulsijws5ljp4lxmncirxucqvmylj7mkbvlie6b3o3kezpbcd3t.py
# Topologically Sorted Source Nodes: [sample_2, temb_30], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   sample_2 => add_tensor_102
#   temb_30 => convert_element_type_647, convert_element_type_648, mul_225, sigmoid_47
# Graph fragment:
#   %add_tensor_102 : [num_users=22] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_102, %arg5_1), kwargs = {})
#   %convert_element_type_647 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_102, torch.float32), kwargs = {})
#   %sigmoid_47 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_647,), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_647, %sigmoid_47), kwargs = {})
#   %convert_element_type_648 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_225, torch.float16), kwargs = {})
triton_poi_fused_addmm_silu_94 = async_compile.triton('triton_poi_fused_addmm_silu_94', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_94', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_94(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1280)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ga/cgak2z5mezu6kb66qg6wcajwz7gncgvckk5afypjyyvlygvuhzr6.py
# Topologically Sorted Source Nodes: [input_tensor_7, hidden_states_361, hidden_states_363, add_67, output_tensor_15, hidden_states_388, output_9, hidden_states_389], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten._to_copy, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_67 => add_226
#   hidden_states_361 => convert_element_type_657, mul_228, sigmoid_48
#   hidden_states_363 => convolution_43
#   hidden_states_388 => clone_65
#   hidden_states_389 => _unsafe_index_1, convert_element_type_699, convert_element_type_704
#   input_tensor_7 => convolution_44
#   output_9 => add_240
#   output_tensor_15 => div_34
# Graph fragment:
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_7, %arg425_1, %arg426_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_48 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_225,), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_225, %sigmoid_48), kwargs = {})
#   %convert_element_type_657 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_228, torch.float16), kwargs = {})
#   %convolution_43 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_657, %arg423_1, %arg424_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_44, %convolution_43), kwargs = {})
#   %div_34 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_226, 1.0), kwargs = {})
#   %clone_65 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_237,), kwargs = {memory_format: torch.contiguous_format})
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_65, %div_34), kwargs = {})
#   %convert_element_type_699 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_240, torch.float32), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_699, [None, None, %unsqueeze_288, %convert_element_type_703]), kwargs = {})
#   %convert_element_type_704 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_1, torch.float16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_95 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_95', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_95', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_95(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 32
    x2 = (xindex % 32)
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    x4 = xindex
    y5 = yindex
    tmp10 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + 1280*tmp8 + 20480*tmp4 + 327680*y1), xmask).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr2 + (y0 + 1280*tmp8 + 20480*tmp4 + 327680*y1), xmask).to(tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.load(in_ptr4 + (y0 + 1280*tmp8 + 20480*tmp4 + 327680*y1), xmask).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 1280*x4 + 1310720*y1), tmp23, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/th/cthye56da7nj7wkp6pi26v53xjrg5ymjyjoj5ricx37szm3wuiuv.py
# Topologically Sorted Source Nodes: [hidden_states_391, hidden_states_392], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_391 => cat_8
#   hidden_states_392 => convert_element_type_705, var_mean_72
# Graph fragment:
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_45, %add_80], 1), kwargs = {})
#   %convert_element_type_705 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_424, torch.float32), kwargs = {})
#   %var_mean_72 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_705, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_96 = async_compile.triton('triton_red_fused_cat_native_group_norm_96', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 65536},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_96', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_96(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 61440
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
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 1024
        r0_2 = (r0_index % 1024)
        r0_5 = r0_index
        tmp0 = r0_3 + 60*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1280, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (1280*r0_2 + 1310720*x1 + (r0_3 + 60*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 60*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 1920, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr2 + (640*r0_2 + 655360*x1 + ((-1280) + r0_3 + 60*x0)), xmask & r0_mask & tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.where(tmp4, tmp9, tmp13)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(r0_mask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(r0_mask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(r0_mask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r0_5 + 61440*x4), tmp14, xmask & r0_mask)
    tmp20, tmp21, tmp22 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp20[:, None]
    tmp18 = tmp21[:, None]
    tmp19 = tmp22[:, None]
    tl.store(out_ptr1 + (x4), tmp17, xmask)
    tl.store(out_ptr2 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/a7/ca744zyzebcr2hoea5t4iyt2mqsgchqp4knvluzlxymklaht2kd5.py
# Topologically Sorted Source Nodes: [hidden_states_392, hidden_states_393, input_tensor_8], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_392 => add_246, mul_246
#   hidden_states_393 => convert_element_type_710, mul_247, sigmoid_49
#   input_tensor_8 => convolution_48
# Graph fragment:
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_425, %unsqueeze_294), kwargs = {})
#   %add_246 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %unsqueeze_291), kwargs = {})
#   %sigmoid_49 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_246,), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_246, %sigmoid_49), kwargs = {})
#   %convert_element_type_710 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_247, torch.float16), kwargs = {})
#   %convolution_48 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %arg465_1, %arg466_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_97 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_97', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_97', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_97(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 61440
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 60), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 60), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 61440.0
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
    tl.store(out_ptr1 + (y0 + 1920*x2 + 1966080*y1), tmp19, xmask)
    tl.store(out_ptr2 + (y0 + 1920*x2 + 1966080*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/md/cmdd2v3ahjdmktblewhmwjqpx76f4hx2tshmf4mbbxkorfsxbpqh.py
# Topologically Sorted Source Nodes: [hidden_states_393, hidden_states_394], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_393 => convert_element_type_710, mul_247, sigmoid_49
#   hidden_states_394 => convolution_46
# Graph fragment:
#   %sigmoid_49 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_246,), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_246, %sigmoid_49), kwargs = {})
#   %convert_element_type_710 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_247, torch.float16), kwargs = {})
#   %convolution_46 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_710, %arg457_1, %arg458_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_98 = async_compile.triton('triton_poi_fused_convolution_silu_98', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_98', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_98(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1228800
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1920)
    y1 = yindex // 1920
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 1920*x2 + 17280*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/qs/cqssuhnli5rgpy6bhz2i5jfcpq6eixfwwfofetwm4gyp4coqoccs.py
# Topologically Sorted Source Nodes: [hidden_states_425, hidden_states_426], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_425 => cat_9
#   hidden_states_426 => convert_element_type_763, var_mean_78
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_264, %add_60], 1), kwargs = {})
#   %convert_element_type_763 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_464, torch.float32), kwargs = {})
#   %var_mean_78 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_763, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_99 = async_compile.triton('triton_red_fused_cat_native_group_norm_99', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 65536},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_99', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_99(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    x4 = xindex
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 1024
        r0_2 = (r0_index % 1024)
        r0_5 = r0_index
        tmp0 = r0_3 + 40*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 640, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (640*r0_2 + 655360*x1 + (r0_3 + 40*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 40*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (640*r0_2 + 655360*x1 + (r0_3 + 40*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 40*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.load(in_ptr4 + (640*r0_2 + 655360*x1 + (r0_3 + 40*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (r0_3 + 40*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1, 1], 1280, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr6 + (r0_2 + 1024*((-640) + r0_3 + 40*x0) + 655360*x1), xmask & r0_mask & tmp20, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.where(tmp4, tmp19, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
        tl.store(out_ptr0 + (r0_5 + 40960*x4), tmp24, xmask & r0_mask)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp30[:, None]
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/fs/cfstpj6xa75uzq55vvrnntdoxrz5ousvmk3vl4srtnijpbphx4rf.py
# Topologically Sorted Source Nodes: [hidden_states_426, hidden_states_427, input_tensor_9], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_426 => add_266, mul_265
#   hidden_states_427 => convert_element_type_768, mul_266, sigmoid_52
#   input_tensor_9 => convolution_51
# Graph fragment:
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_465, %unsqueeze_314), kwargs = {})
#   %add_266 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_265, %unsqueeze_311), kwargs = {})
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_266,), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_266, %sigmoid_52), kwargs = {})
#   %convert_element_type_768 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_266, torch.float16), kwargs = {})
#   %convolution_51 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_9, %arg503_1, %arg504_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_100 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_100', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_100', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_100(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 40), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 40), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 40960.0
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
    tl.store(out_ptr1 + (y0 + 1280*x2 + 1310720*y1), tmp19, xmask)
    tl.store(out_ptr2 + (y0 + 1280*x2 + 1310720*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/sy/csy34erc3qc4asgf2yaohhjui3qb72b2h3ft3iocnhz3ydbgdv6a.py
# Topologically Sorted Source Nodes: [hidden_states_427, hidden_states_428], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_427 => convert_element_type_768, mul_266, sigmoid_52
#   hidden_states_428 => convolution_49
# Graph fragment:
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_266,), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_266, %sigmoid_52), kwargs = {})
#   %convert_element_type_768 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_266, torch.float16), kwargs = {})
#   %convolution_49 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_768, %arg495_1, %arg496_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_101 = async_compile.triton('triton_poi_fused_convolution_silu_101', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_101', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_101(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 819200
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 1280*x2 + 11520*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/io/ciokvwb4eacxlioheccbvmrbfp5oa2tkv4ebocmygvufuuj3wfku.py
# Topologically Sorted Source Nodes: [hidden_states_459, hidden_states_460], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_459 => cat_10
#   hidden_states_460 => convert_element_type_821, var_mean_84
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_284, %convolution_5], 1), kwargs = {})
#   %convert_element_type_821 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_504, torch.float32), kwargs = {})
#   %var_mean_84 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_821, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_102 = async_compile.triton('triton_red_fused_cat_native_group_norm_102', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_102', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_102(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 30720
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
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 1024
        r0_2 = (r0_index % 1024)
        r0_5 = r0_index
        tmp0 = r0_3 + 30*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 640, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (640*r0_2 + 655360*x1 + (r0_3 + 30*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 30*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (640*r0_2 + 655360*x1 + (r0_3 + 30*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 30*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.load(in_ptr4 + (640*r0_2 + 655360*x1 + (r0_3 + 30*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (r0_3 + 30*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1, 1], 960, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr6 + (r0_2 + 1024*((-640) + r0_3 + 30*x0) + 327680*x1), xmask & r0_mask & tmp20, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.where(tmp4, tmp19, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
        tl.store(out_ptr0 + (r0_5 + 30720*x4), tmp24, xmask & r0_mask)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp30[:, None]
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ud/cudgxowd47qgkltibvj7sjdhk2fg2gxwmwxl7v7rhke7wk5wolum.py
# Topologically Sorted Source Nodes: [hidden_states_460, hidden_states_461, input_tensor_10], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_460 => add_286, mul_284
#   hidden_states_461 => convert_element_type_826, mul_285, sigmoid_55
#   input_tensor_10 => convolution_54
# Graph fragment:
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_505, %unsqueeze_334), kwargs = {})
#   %add_286 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_331), kwargs = {})
#   %sigmoid_55 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_286,), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_286, %sigmoid_55), kwargs = {})
#   %convert_element_type_826 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_285, torch.float16), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %arg541_1, %arg542_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_103 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_103', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_103', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_103(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 960)
    y1 = yindex // 960
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 30), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 30), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 30720.0
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
    tl.store(out_ptr1 + (y0 + 960*x2 + 983040*y1), tmp19, xmask)
    tl.store(out_ptr2 + (y0 + 960*x2 + 983040*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/3r/c3rpl3ju6o6qxoefit7wgugzngilocgj4crzvsujsfcvkegxh2g5.py
# Topologically Sorted Source Nodes: [hidden_states_461, hidden_states_462], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_461 => convert_element_type_826, mul_285, sigmoid_55
#   hidden_states_462 => convolution_52
# Graph fragment:
#   %sigmoid_55 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_286,), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_286, %sigmoid_55), kwargs = {})
#   %convert_element_type_826 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_285, torch.float16), kwargs = {})
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_826, %arg533_1, %arg534_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_104 = async_compile.triton('triton_poi_fused_convolution_silu_104', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_104', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_104(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 614400
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 960)
    y1 = yindex // 960
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 960*x2 + 8640*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/wy/cwygwu4k5v3i3n6vrwu7p7ydayaxcmh2a7ldwpwf6esbsj7oyoko.py
# Topologically Sorted Source Nodes: [input_tensor_10, hidden_states_465, hidden_states_467, add_85, output_tensor_18, hidden_states_492, output_12, hidden_states_493], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten._to_copy, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_85 => add_290
#   hidden_states_465 => convert_element_type_837, mul_289, sigmoid_57
#   hidden_states_467 => convolution_53
#   hidden_states_492 => clone_83
#   hidden_states_493 => _unsafe_index_2, convert_element_type_879, convert_element_type_884
#   input_tensor_10 => convolution_54
#   output_12 => add_304
#   output_tensor_18 => div_43
# Graph fragment:
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %arg541_1, %arg542_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_57 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_289,), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_289, %sigmoid_57), kwargs = {})
#   %convert_element_type_837 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_289, torch.float16), kwargs = {})
#   %convolution_53 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_837, %arg539_1, %arg540_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_290 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_54, %convolution_53), kwargs = {})
#   %div_43 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_290, 1.0), kwargs = {})
#   %clone_83 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_306,), kwargs = {memory_format: torch.contiguous_format})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_83, %div_43), kwargs = {})
#   %convert_element_type_879 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_304, torch.float32), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_879, [None, None, %unsqueeze_349, %convert_element_type_883]), kwargs = {})
#   %convert_element_type_884 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_2, torch.float16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_105 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_105', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_105', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_105(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex // 64
    x2 = (xindex % 64)
    y0 = (yindex % 640)
    y1 = yindex // 640
    x4 = xindex
    y5 = yindex
    tmp10 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + 640*tmp8 + 20480*tmp4 + 655360*y1), None).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr2 + (y0 + 640*tmp8 + 20480*tmp4 + 655360*y1), None).to(tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.load(in_ptr4 + (y0 + 640*tmp8 + 20480*tmp4 + 655360*y1), None).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 640*x4 + 2621440*y1), tmp23, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ck/cckltxa52tziozb2iivwridnwhnlvy65klf3jxrlqhkjmhgyzjg6.py
# Topologically Sorted Source Nodes: [hidden_states_495, hidden_states_496], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_495 => cat_11
#   hidden_states_496 => convert_element_type_885, var_mean_90
# Graph fragment:
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_55, %add_40], 1), kwargs = {})
#   %convert_element_type_885 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_544, torch.float32), kwargs = {})
#   %var_mean_90 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_885, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_106 = async_compile.triton('triton_red_fused_cat_native_group_norm_106', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 131072},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_106', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_106(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 122880
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
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 4096
        r0_2 = (r0_index % 4096)
        r0_5 = r0_index
        tmp0 = r0_3 + 30*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 640, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (640*r0_2 + 2621440*x1 + (r0_3 + 30*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 30*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 960, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr2 + (320*r0_2 + 1310720*x1 + ((-640) + r0_3 + 30*x0)), xmask & r0_mask & tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.where(tmp4, tmp9, tmp13)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(r0_mask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(r0_mask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(r0_mask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r0_5 + 122880*x4), tmp14, xmask & r0_mask)
    tmp20, tmp21, tmp22 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp20[:, None]
    tmp18 = tmp21[:, None]
    tmp19 = tmp22[:, None]
    tl.store(out_ptr1 + (x4), tmp17, xmask)
    tl.store(out_ptr2 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/c4/cc4vggxkkrsmfqknrb4rw7fbecwpxztrii62blkx53yeqi6k266f.py
# Topologically Sorted Source Nodes: [hidden_states_496, hidden_states_497, input_tensor_11], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_496 => add_310, mul_307
#   hidden_states_497 => convert_element_type_890, mul_308, sigmoid_58
#   input_tensor_11 => convolution_58
# Graph fragment:
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_545, %unsqueeze_355), kwargs = {})
#   %add_310 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_307, %unsqueeze_352), kwargs = {})
#   %sigmoid_58 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_310,), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_310, %sigmoid_58), kwargs = {})
#   %convert_element_type_890 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_308, torch.float16), kwargs = {})
#   %convolution_58 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg581_1, %arg582_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_107 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_107', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_107', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_107(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 960)
    y1 = yindex // 960
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 30), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 30), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 122880.0
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
    tl.store(out_ptr1 + (y0 + 960*x2 + 3932160*y1), tmp19, None)
    tl.store(out_ptr2 + (y0 + 960*x2 + 3932160*y1), tmp0, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/bt/cbtn44aqe6vf63vxfsolbjcs7koybbood245icmv4rjgkf6xkowx.py
# Topologically Sorted Source Nodes: [hidden_states_497, hidden_states_498], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_497 => convert_element_type_890, mul_308, sigmoid_58
#   hidden_states_498 => convolution_56
# Graph fragment:
#   %sigmoid_58 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_310,), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_310, %sigmoid_58), kwargs = {})
#   %convert_element_type_890 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_308, torch.float16), kwargs = {})
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_890, %arg573_1, %arg574_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_108 = async_compile.triton('triton_poi_fused_convolution_silu_108', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_108', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_108(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 307200
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 960)
    y1 = yindex // 960
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 960*x2 + 8640*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/rv/crvdnemfww2nyov3kywdxqc24aeh3jpsv5ay4yykxcu3twefwzzo.py
# Topologically Sorted Source Nodes: [hidden_states_529, hidden_states_530], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_529 => cat_12
#   hidden_states_530 => convert_element_type_943, var_mean_96
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_328, %add_20], 1), kwargs = {})
#   %convert_element_type_943 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_584, torch.float32), kwargs = {})
#   %var_mean_96 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_943, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_109 = async_compile.triton('triton_red_fused_cat_native_group_norm_109', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 131072},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_109', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_109(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 81920
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
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 4096
        r0_2 = (r0_index % 4096)
        r0_5 = r0_index
        tmp0 = r0_3 + 20*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 320, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (320*r0_2 + 1310720*x1 + (r0_3 + 20*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 20*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (320*r0_2 + 1310720*x1 + (r0_3 + 20*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 20*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.load(in_ptr4 + (320*r0_2 + 1310720*x1 + (r0_3 + 20*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (r0_3 + 20*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1, 1], 640, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr6 + (r0_2 + 4096*((-320) + r0_3 + 20*x0) + 1310720*x1), xmask & r0_mask & tmp20, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.where(tmp4, tmp19, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
        tl.store(out_ptr0 + (r0_5 + 81920*x4), tmp24, xmask & r0_mask)
    tmp30, tmp31, tmp32 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp30[:, None]
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/go/cgouu6g4w3kszfilvtg6ccuofobdtaaq3qypbfxq5jqxnziynanq.py
# Topologically Sorted Source Nodes: [hidden_states_530, hidden_states_531, input_tensor_12], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_530 => add_330, mul_326
#   hidden_states_531 => convert_element_type_948, mul_327, sigmoid_61
#   input_tensor_12 => convolution_61
# Graph fragment:
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_585, %unsqueeze_375), kwargs = {})
#   %add_330 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_372), kwargs = {})
#   %sigmoid_61 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_330,), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_330, %sigmoid_61), kwargs = {})
#   %convert_element_type_948 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_327, torch.float16), kwargs = {})
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_12, %arg619_1, %arg620_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_native_group_norm_silu_110 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_110', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_110', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_110(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3 // 20), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y3 // 20), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 81920.0
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
    tl.store(out_ptr1 + (y0 + 640*x2 + 2621440*y1), tmp19, None)
    tl.store(out_ptr2 + (y0 + 640*x2 + 2621440*y1), tmp0, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/is/cisgigyulqyhmokghycfymk547gwjwauf6qhrxr2atce3bvkwcio.py
# Topologically Sorted Source Nodes: [hidden_states_531, hidden_states_532], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_531 => convert_element_type_948, mul_327, sigmoid_61
#   hidden_states_532 => convolution_59
# Graph fragment:
#   %sigmoid_61 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_330,), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_330, %sigmoid_61), kwargs = {})
#   %convert_element_type_948 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_327, torch.float16), kwargs = {})
#   %convolution_59 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_948, %arg611_1, %arg612_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_111 = async_compile.triton('triton_poi_fused_convolution_silu_111', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_111', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_111(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 204800
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 640*x2 + 5760*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/kz/ckzmv6tgx75fep5mxpiof3lt5ty4pndgvotlsrxd7jmperc6bv6u.py
# Topologically Sorted Source Nodes: [hidden_states_563, hidden_states_564], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_563 => cat_13
#   hidden_states_564 => convert_element_type_1001, var_mean_102
# Graph fragment:
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_348, %convolution], 1), kwargs = {})
#   %convert_element_type_1001 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_624, torch.float32), kwargs = {})
#   %var_mean_102 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1001, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_group_norm_112 = async_compile.triton('triton_red_fused_cat_native_group_norm_112', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 131072},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_group_norm_112', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_group_norm_112(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 81920
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
    tmp31_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp31_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp31_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 4096
        r0_2 = (r0_index % 4096)
        r0_5 = r0_index
        tmp0 = r0_3 + 20*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 320, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (320*r0_2 + 1310720*x1 + (r0_3 + 20*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (r0_3 + 20*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr2 + (320*r0_2 + 1310720*x1 + (r0_3 + 20*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_3 + 20*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.load(in_ptr4 + (320*r0_2 + 1310720*x1 + (r0_3 + 20*x0)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (r0_3 + 20*x0), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1, 1], 640, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr6 + (320*r0_2 + 1310720*x1 + ((-320) + r0_3 + 20*x0)), xmask & r0_mask & tmp20, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr7 + ((-320) + r0_3 + 20*x0), xmask & r0_mask & tmp20, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
        tmp27 = tl.where(tmp20, tmp25, tmp26)
        tmp28 = tl.where(tmp4, tmp19, tmp27)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp31_mean_next, tmp31_m2_next, tmp31_weight_next = triton_helpers.welford_reduce(
            tmp30, tmp31_mean, tmp31_m2, tmp31_weight, roffset == 0
        )
        tmp31_mean = tl.where(r0_mask & xmask, tmp31_mean_next, tmp31_mean)
        tmp31_m2 = tl.where(r0_mask & xmask, tmp31_m2_next, tmp31_m2)
        tmp31_weight = tl.where(r0_mask & xmask, tmp31_weight_next, tmp31_weight)
        tl.store(out_ptr0 + (r0_5 + 81920*x4), tmp28, xmask & r0_mask)
    tmp34, tmp35, tmp36 = triton_helpers.welford(tmp31_mean, tmp31_m2, tmp31_weight, 1)
    tmp31 = tmp34[:, None]
    tmp32 = tmp35[:, None]
    tmp33 = tmp36[:, None]
    tl.store(out_ptr1 + (x4), tmp31, xmask)
    tl.store(out_ptr2 + (x4), tmp32, xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/ph/cphnrn5tqltvo3ulj2fyn75js57fduizeku4x4bbrhyl44up5f55.py
# Topologically Sorted Source Nodes: [sample_4], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   sample_4 => convert_element_type_1059
# Graph fragment:
#   %convert_element_type_1059 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_664, torch.float32), kwargs = {})
triton_poi_fused_native_group_norm_113 = async_compile.triton('triton_poi_fused_native_group_norm_113', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_113', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_113(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/5c/c5cv4xcqfx2otcbzpde47avkwqbn6iocn4vaptgrvjtfer3l6xsc.py
# Topologically Sorted Source Nodes: [sample_4], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   sample_4 => var_mean_108
# Graph fragment:
#   %var_mean_108 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1059, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_114 = async_compile.triton('triton_red_fused_native_group_norm_114', '''
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
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/jv/cjv2wfhnagmyzhitqninchz4kbq5h66yo6c2m73kcfpqankd7s3a.py
# Topologically Sorted Source Nodes: [sample_4, sample_5], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   sample_4 => add_370, mul_364
#   sample_5 => convert_element_type_1064, mul_365, sigmoid_67
# Graph fragment:
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_665, %unsqueeze_415), kwargs = {})
#   %add_370 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_364, %unsqueeze_412), kwargs = {})
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_370,), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_370, %sigmoid_67), kwargs = {})
#   %convert_element_type_1064 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_365, torch.float16), kwargs = {})
triton_poi_fused_native_group_norm_silu_115 = async_compile.triton('triton_poi_fused_native_group_norm_silu_115', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_115', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_115(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 320)
    x2 = xindex // 1310720
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (32*x2 + (x0 // 10)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 - tmp1
    tmp4 = 40960.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/a2/ca2kwgamwbzb7ke2oql6zeh7jiui5q4xu2tkiza7ojtbmqwflcdv.py
# Topologically Sorted Source Nodes: [sample_5, sample_6], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_5 => convert_element_type_1064, mul_365, sigmoid_67
#   sample_6 => convolution_65
# Graph fragment:
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_370,), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_370, %sigmoid_67), kwargs = {})
#   %convert_element_type_1064 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_365, torch.float16), kwargs = {})
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_1064, %arg687_1, %arg688_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_116 = async_compile.triton('triton_poi_fused_convolution_silu_116', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_116', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_116(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, ymask & xmask)
''', device_str='cuda')


# kernel path: /data2/fzx/SERUM/torchinductor_tln/tc/ctcbmu3cpt5glnx3aj4e2gzcs3i5s7jjglfoa2fhu625n75u6ti3.py
# Topologically Sorted Source Nodes: [sample_5, sample_6], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_5 => convert_element_type_1064, mul_365, sigmoid_67
#   sample_6 => convolution_65
# Graph fragment:
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_370,), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_370, %sigmoid_67), kwargs = {})
#   %convert_element_type_1064 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_365, torch.float16), kwargs = {})
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_1064, %arg687_1, %arg688_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_117 = async_compile.triton('triton_poi_fused_convolution_silu_117', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_117', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '75044612F558001C776DE40EF3B9C7996A8444FEF19555030BDC0BC1BE7B21BB', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_117(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16384*y1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp2, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(arg1_1, (), ())
    assert_size_stride(arg2_1, (1280, 320), (320, 1))
    assert_size_stride(arg3_1, (1280, ), (1, ))
    assert_size_stride(arg4_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg5_1, (1280, ), (1, ))
    assert_size_stride(arg6_1, (32, 77, 1024), (78848, 1024, 1))
    assert_size_stride(arg7_1, (320, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg8_1, (320, ), (1, ))
    assert_size_stride(arg9_1, (320, ), (1, ))
    assert_size_stride(arg10_1, (320, ), (1, ))
    assert_size_stride(arg11_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg12_1, (320, ), (1, ))
    assert_size_stride(arg13_1, (320, 1280), (1280, 1))
    assert_size_stride(arg14_1, (320, ), (1, ))
    assert_size_stride(arg15_1, (320, ), (1, ))
    assert_size_stride(arg16_1, (320, ), (1, ))
    assert_size_stride(arg17_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg18_1, (320, ), (1, ))
    assert_size_stride(arg19_1, (320, ), (1, ))
    assert_size_stride(arg20_1, (320, ), (1, ))
    assert_size_stride(arg21_1, (320, 320), (320, 1))
    assert_size_stride(arg22_1, (320, ), (1, ))
    assert_size_stride(arg23_1, (320, ), (1, ))
    assert_size_stride(arg24_1, (320, ), (1, ))
    assert_size_stride(arg25_1, (320, 320), (320, 1))
    assert_size_stride(arg26_1, (320, 320), (320, 1))
    assert_size_stride(arg27_1, (320, 320), (320, 1))
    assert_size_stride(arg28_1, (320, 320), (320, 1))
    assert_size_stride(arg29_1, (320, ), (1, ))
    assert_size_stride(arg30_1, (320, ), (1, ))
    assert_size_stride(arg31_1, (320, ), (1, ))
    assert_size_stride(arg32_1, (320, 320), (320, 1))
    assert_size_stride(arg33_1, (320, 1024), (1024, 1))
    assert_size_stride(arg34_1, (320, 1024), (1024, 1))
    assert_size_stride(arg35_1, (320, 320), (320, 1))
    assert_size_stride(arg36_1, (320, ), (1, ))
    assert_size_stride(arg37_1, (320, ), (1, ))
    assert_size_stride(arg38_1, (320, ), (1, ))
    assert_size_stride(arg39_1, (2560, 320), (320, 1))
    assert_size_stride(arg40_1, (2560, ), (1, ))
    assert_size_stride(arg41_1, (320, 1280), (1280, 1))
    assert_size_stride(arg42_1, (320, ), (1, ))
    assert_size_stride(arg43_1, (320, 320), (320, 1))
    assert_size_stride(arg44_1, (320, ), (1, ))
    assert_size_stride(arg45_1, (320, ), (1, ))
    assert_size_stride(arg46_1, (320, ), (1, ))
    assert_size_stride(arg47_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg48_1, (320, ), (1, ))
    assert_size_stride(arg49_1, (320, 1280), (1280, 1))
    assert_size_stride(arg50_1, (320, ), (1, ))
    assert_size_stride(arg51_1, (320, ), (1, ))
    assert_size_stride(arg52_1, (320, ), (1, ))
    assert_size_stride(arg53_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg54_1, (320, ), (1, ))
    assert_size_stride(arg55_1, (320, ), (1, ))
    assert_size_stride(arg56_1, (320, ), (1, ))
    assert_size_stride(arg57_1, (320, 320), (320, 1))
    assert_size_stride(arg58_1, (320, ), (1, ))
    assert_size_stride(arg59_1, (320, ), (1, ))
    assert_size_stride(arg60_1, (320, ), (1, ))
    assert_size_stride(arg61_1, (320, 320), (320, 1))
    assert_size_stride(arg62_1, (320, 320), (320, 1))
    assert_size_stride(arg63_1, (320, 320), (320, 1))
    assert_size_stride(arg64_1, (320, 320), (320, 1))
    assert_size_stride(arg65_1, (320, ), (1, ))
    assert_size_stride(arg66_1, (320, ), (1, ))
    assert_size_stride(arg67_1, (320, ), (1, ))
    assert_size_stride(arg68_1, (320, 320), (320, 1))
    assert_size_stride(arg69_1, (320, 1024), (1024, 1))
    assert_size_stride(arg70_1, (320, 1024), (1024, 1))
    assert_size_stride(arg71_1, (320, 320), (320, 1))
    assert_size_stride(arg72_1, (320, ), (1, ))
    assert_size_stride(arg73_1, (320, ), (1, ))
    assert_size_stride(arg74_1, (320, ), (1, ))
    assert_size_stride(arg75_1, (2560, 320), (320, 1))
    assert_size_stride(arg76_1, (2560, ), (1, ))
    assert_size_stride(arg77_1, (320, 1280), (1280, 1))
    assert_size_stride(arg78_1, (320, ), (1, ))
    assert_size_stride(arg79_1, (320, 320), (320, 1))
    assert_size_stride(arg80_1, (320, ), (1, ))
    assert_size_stride(arg81_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg82_1, (320, ), (1, ))
    assert_size_stride(arg83_1, (320, ), (1, ))
    assert_size_stride(arg84_1, (320, ), (1, ))
    assert_size_stride(arg85_1, (640, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg86_1, (640, ), (1, ))
    assert_size_stride(arg87_1, (640, 1280), (1280, 1))
    assert_size_stride(arg88_1, (640, ), (1, ))
    assert_size_stride(arg89_1, (640, ), (1, ))
    assert_size_stride(arg90_1, (640, ), (1, ))
    assert_size_stride(arg91_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg92_1, (640, ), (1, ))
    assert_size_stride(arg93_1, (640, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg94_1, (640, ), (1, ))
    assert_size_stride(arg95_1, (640, ), (1, ))
    assert_size_stride(arg96_1, (640, ), (1, ))
    assert_size_stride(arg97_1, (640, 640), (640, 1))
    assert_size_stride(arg98_1, (640, ), (1, ))
    assert_size_stride(arg99_1, (640, ), (1, ))
    assert_size_stride(arg100_1, (640, ), (1, ))
    assert_size_stride(arg101_1, (640, 640), (640, 1))
    assert_size_stride(arg102_1, (640, 640), (640, 1))
    assert_size_stride(arg103_1, (640, 640), (640, 1))
    assert_size_stride(arg104_1, (640, 640), (640, 1))
    assert_size_stride(arg105_1, (640, ), (1, ))
    assert_size_stride(arg106_1, (640, ), (1, ))
    assert_size_stride(arg107_1, (640, ), (1, ))
    assert_size_stride(arg108_1, (640, 640), (640, 1))
    assert_size_stride(arg109_1, (640, 1024), (1024, 1))
    assert_size_stride(arg110_1, (640, 1024), (1024, 1))
    assert_size_stride(arg111_1, (640, 640), (640, 1))
    assert_size_stride(arg112_1, (640, ), (1, ))
    assert_size_stride(arg113_1, (640, ), (1, ))
    assert_size_stride(arg114_1, (640, ), (1, ))
    assert_size_stride(arg115_1, (5120, 640), (640, 1))
    assert_size_stride(arg116_1, (5120, ), (1, ))
    assert_size_stride(arg117_1, (640, 2560), (2560, 1))
    assert_size_stride(arg118_1, (640, ), (1, ))
    assert_size_stride(arg119_1, (640, 640), (640, 1))
    assert_size_stride(arg120_1, (640, ), (1, ))
    assert_size_stride(arg121_1, (640, ), (1, ))
    assert_size_stride(arg122_1, (640, ), (1, ))
    assert_size_stride(arg123_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg124_1, (640, ), (1, ))
    assert_size_stride(arg125_1, (640, 1280), (1280, 1))
    assert_size_stride(arg126_1, (640, ), (1, ))
    assert_size_stride(arg127_1, (640, ), (1, ))
    assert_size_stride(arg128_1, (640, ), (1, ))
    assert_size_stride(arg129_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg130_1, (640, ), (1, ))
    assert_size_stride(arg131_1, (640, ), (1, ))
    assert_size_stride(arg132_1, (640, ), (1, ))
    assert_size_stride(arg133_1, (640, 640), (640, 1))
    assert_size_stride(arg134_1, (640, ), (1, ))
    assert_size_stride(arg135_1, (640, ), (1, ))
    assert_size_stride(arg136_1, (640, ), (1, ))
    assert_size_stride(arg137_1, (640, 640), (640, 1))
    assert_size_stride(arg138_1, (640, 640), (640, 1))
    assert_size_stride(arg139_1, (640, 640), (640, 1))
    assert_size_stride(arg140_1, (640, 640), (640, 1))
    assert_size_stride(arg141_1, (640, ), (1, ))
    assert_size_stride(arg142_1, (640, ), (1, ))
    assert_size_stride(arg143_1, (640, ), (1, ))
    assert_size_stride(arg144_1, (640, 640), (640, 1))
    assert_size_stride(arg145_1, (640, 1024), (1024, 1))
    assert_size_stride(arg146_1, (640, 1024), (1024, 1))
    assert_size_stride(arg147_1, (640, 640), (640, 1))
    assert_size_stride(arg148_1, (640, ), (1, ))
    assert_size_stride(arg149_1, (640, ), (1, ))
    assert_size_stride(arg150_1, (640, ), (1, ))
    assert_size_stride(arg151_1, (5120, 640), (640, 1))
    assert_size_stride(arg152_1, (5120, ), (1, ))
    assert_size_stride(arg153_1, (640, 2560), (2560, 1))
    assert_size_stride(arg154_1, (640, ), (1, ))
    assert_size_stride(arg155_1, (640, 640), (640, 1))
    assert_size_stride(arg156_1, (640, ), (1, ))
    assert_size_stride(arg157_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg158_1, (640, ), (1, ))
    assert_size_stride(arg159_1, (640, ), (1, ))
    assert_size_stride(arg160_1, (640, ), (1, ))
    assert_size_stride(arg161_1, (1280, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg162_1, (1280, ), (1, ))
    assert_size_stride(arg163_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg164_1, (1280, ), (1, ))
    assert_size_stride(arg165_1, (1280, ), (1, ))
    assert_size_stride(arg166_1, (1280, ), (1, ))
    assert_size_stride(arg167_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg168_1, (1280, ), (1, ))
    assert_size_stride(arg169_1, (1280, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg170_1, (1280, ), (1, ))
    assert_size_stride(arg171_1, (1280, ), (1, ))
    assert_size_stride(arg172_1, (1280, ), (1, ))
    assert_size_stride(arg173_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg174_1, (1280, ), (1, ))
    assert_size_stride(arg175_1, (1280, ), (1, ))
    assert_size_stride(arg176_1, (1280, ), (1, ))
    assert_size_stride(arg177_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg178_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg179_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg180_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg181_1, (1280, ), (1, ))
    assert_size_stride(arg182_1, (1280, ), (1, ))
    assert_size_stride(arg183_1, (1280, ), (1, ))
    assert_size_stride(arg184_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg185_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg187_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg188_1, (1280, ), (1, ))
    assert_size_stride(arg189_1, (1280, ), (1, ))
    assert_size_stride(arg190_1, (1280, ), (1, ))
    assert_size_stride(arg191_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg192_1, (10240, ), (1, ))
    assert_size_stride(arg193_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg194_1, (1280, ), (1, ))
    assert_size_stride(arg195_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg196_1, (1280, ), (1, ))
    assert_size_stride(arg197_1, (1280, ), (1, ))
    assert_size_stride(arg198_1, (1280, ), (1, ))
    assert_size_stride(arg199_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg200_1, (1280, ), (1, ))
    assert_size_stride(arg201_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg202_1, (1280, ), (1, ))
    assert_size_stride(arg203_1, (1280, ), (1, ))
    assert_size_stride(arg204_1, (1280, ), (1, ))
    assert_size_stride(arg205_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg206_1, (1280, ), (1, ))
    assert_size_stride(arg207_1, (1280, ), (1, ))
    assert_size_stride(arg208_1, (1280, ), (1, ))
    assert_size_stride(arg209_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg210_1, (1280, ), (1, ))
    assert_size_stride(arg211_1, (1280, ), (1, ))
    assert_size_stride(arg212_1, (1280, ), (1, ))
    assert_size_stride(arg213_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg214_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg215_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg216_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg217_1, (1280, ), (1, ))
    assert_size_stride(arg218_1, (1280, ), (1, ))
    assert_size_stride(arg219_1, (1280, ), (1, ))
    assert_size_stride(arg220_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg221_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg222_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg223_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg224_1, (1280, ), (1, ))
    assert_size_stride(arg225_1, (1280, ), (1, ))
    assert_size_stride(arg226_1, (1280, ), (1, ))
    assert_size_stride(arg227_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg228_1, (10240, ), (1, ))
    assert_size_stride(arg229_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg230_1, (1280, ), (1, ))
    assert_size_stride(arg231_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg232_1, (1280, ), (1, ))
    assert_size_stride(arg233_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg234_1, (1280, ), (1, ))
    assert_size_stride(arg235_1, (1280, ), (1, ))
    assert_size_stride(arg236_1, (1280, ), (1, ))
    assert_size_stride(arg237_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg238_1, (1280, ), (1, ))
    assert_size_stride(arg239_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg240_1, (1280, ), (1, ))
    assert_size_stride(arg241_1, (1280, ), (1, ))
    assert_size_stride(arg242_1, (1280, ), (1, ))
    assert_size_stride(arg243_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg244_1, (1280, ), (1, ))
    assert_size_stride(arg245_1, (1280, ), (1, ))
    assert_size_stride(arg246_1, (1280, ), (1, ))
    assert_size_stride(arg247_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg248_1, (1280, ), (1, ))
    assert_size_stride(arg249_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg250_1, (1280, ), (1, ))
    assert_size_stride(arg251_1, (1280, ), (1, ))
    assert_size_stride(arg252_1, (1280, ), (1, ))
    assert_size_stride(arg253_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg254_1, (1280, ), (1, ))
    assert_size_stride(arg255_1, (1280, ), (1, ))
    assert_size_stride(arg256_1, (1280, ), (1, ))
    assert_size_stride(arg257_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg258_1, (1280, ), (1, ))
    assert_size_stride(arg259_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg260_1, (1280, ), (1, ))
    assert_size_stride(arg261_1, (1280, ), (1, ))
    assert_size_stride(arg262_1, (1280, ), (1, ))
    assert_size_stride(arg263_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg264_1, (1280, ), (1, ))
    assert_size_stride(arg265_1, (1280, ), (1, ))
    assert_size_stride(arg266_1, (1280, ), (1, ))
    assert_size_stride(arg267_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg268_1, (1280, ), (1, ))
    assert_size_stride(arg269_1, (1280, ), (1, ))
    assert_size_stride(arg270_1, (1280, ), (1, ))
    assert_size_stride(arg271_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg272_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg273_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg274_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg275_1, (1280, ), (1, ))
    assert_size_stride(arg276_1, (1280, ), (1, ))
    assert_size_stride(arg277_1, (1280, ), (1, ))
    assert_size_stride(arg278_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg279_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg280_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg281_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg282_1, (1280, ), (1, ))
    assert_size_stride(arg283_1, (1280, ), (1, ))
    assert_size_stride(arg284_1, (1280, ), (1, ))
    assert_size_stride(arg285_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg286_1, (10240, ), (1, ))
    assert_size_stride(arg287_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg288_1, (1280, ), (1, ))
    assert_size_stride(arg289_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg290_1, (1280, ), (1, ))
    assert_size_stride(arg291_1, (1280, ), (1, ))
    assert_size_stride(arg292_1, (1280, ), (1, ))
    assert_size_stride(arg293_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg294_1, (1280, ), (1, ))
    assert_size_stride(arg295_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg296_1, (1280, ), (1, ))
    assert_size_stride(arg297_1, (1280, ), (1, ))
    assert_size_stride(arg298_1, (1280, ), (1, ))
    assert_size_stride(arg299_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg300_1, (1280, ), (1, ))
    assert_size_stride(arg301_1, (2560, ), (1, ))
    assert_size_stride(arg302_1, (2560, ), (1, ))
    assert_size_stride(arg303_1, (1280, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg304_1, (1280, ), (1, ))
    assert_size_stride(arg305_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg306_1, (1280, ), (1, ))
    assert_size_stride(arg307_1, (1280, ), (1, ))
    assert_size_stride(arg308_1, (1280, ), (1, ))
    assert_size_stride(arg309_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg310_1, (1280, ), (1, ))
    assert_size_stride(arg311_1, (1280, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg312_1, (1280, ), (1, ))
    assert_size_stride(arg313_1, (2560, ), (1, ))
    assert_size_stride(arg314_1, (2560, ), (1, ))
    assert_size_stride(arg315_1, (1280, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg316_1, (1280, ), (1, ))
    assert_size_stride(arg317_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg318_1, (1280, ), (1, ))
    assert_size_stride(arg319_1, (1280, ), (1, ))
    assert_size_stride(arg320_1, (1280, ), (1, ))
    assert_size_stride(arg321_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg322_1, (1280, ), (1, ))
    assert_size_stride(arg323_1, (1280, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg324_1, (1280, ), (1, ))
    assert_size_stride(arg325_1, (2560, ), (1, ))
    assert_size_stride(arg326_1, (2560, ), (1, ))
    assert_size_stride(arg327_1, (1280, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg328_1, (1280, ), (1, ))
    assert_size_stride(arg329_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg330_1, (1280, ), (1, ))
    assert_size_stride(arg331_1, (1280, ), (1, ))
    assert_size_stride(arg332_1, (1280, ), (1, ))
    assert_size_stride(arg333_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg334_1, (1280, ), (1, ))
    assert_size_stride(arg335_1, (1280, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg336_1, (1280, ), (1, ))
    assert_size_stride(arg337_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg338_1, (1280, ), (1, ))
    assert_size_stride(arg339_1, (2560, ), (1, ))
    assert_size_stride(arg340_1, (2560, ), (1, ))
    assert_size_stride(arg341_1, (1280, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg342_1, (1280, ), (1, ))
    assert_size_stride(arg343_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg344_1, (1280, ), (1, ))
    assert_size_stride(arg345_1, (1280, ), (1, ))
    assert_size_stride(arg346_1, (1280, ), (1, ))
    assert_size_stride(arg347_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg348_1, (1280, ), (1, ))
    assert_size_stride(arg349_1, (1280, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg350_1, (1280, ), (1, ))
    assert_size_stride(arg351_1, (1280, ), (1, ))
    assert_size_stride(arg352_1, (1280, ), (1, ))
    assert_size_stride(arg353_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg354_1, (1280, ), (1, ))
    assert_size_stride(arg355_1, (1280, ), (1, ))
    assert_size_stride(arg356_1, (1280, ), (1, ))
    assert_size_stride(arg357_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg358_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg359_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg360_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg361_1, (1280, ), (1, ))
    assert_size_stride(arg362_1, (1280, ), (1, ))
    assert_size_stride(arg363_1, (1280, ), (1, ))
    assert_size_stride(arg364_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg365_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg366_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg367_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg368_1, (1280, ), (1, ))
    assert_size_stride(arg369_1, (1280, ), (1, ))
    assert_size_stride(arg370_1, (1280, ), (1, ))
    assert_size_stride(arg371_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg372_1, (10240, ), (1, ))
    assert_size_stride(arg373_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg374_1, (1280, ), (1, ))
    assert_size_stride(arg375_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg376_1, (1280, ), (1, ))
    assert_size_stride(arg377_1, (2560, ), (1, ))
    assert_size_stride(arg378_1, (2560, ), (1, ))
    assert_size_stride(arg379_1, (1280, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg380_1, (1280, ), (1, ))
    assert_size_stride(arg381_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg382_1, (1280, ), (1, ))
    assert_size_stride(arg383_1, (1280, ), (1, ))
    assert_size_stride(arg384_1, (1280, ), (1, ))
    assert_size_stride(arg385_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg386_1, (1280, ), (1, ))
    assert_size_stride(arg387_1, (1280, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg388_1, (1280, ), (1, ))
    assert_size_stride(arg389_1, (1280, ), (1, ))
    assert_size_stride(arg390_1, (1280, ), (1, ))
    assert_size_stride(arg391_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg392_1, (1280, ), (1, ))
    assert_size_stride(arg393_1, (1280, ), (1, ))
    assert_size_stride(arg394_1, (1280, ), (1, ))
    assert_size_stride(arg395_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg396_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg397_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg398_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg399_1, (1280, ), (1, ))
    assert_size_stride(arg400_1, (1280, ), (1, ))
    assert_size_stride(arg401_1, (1280, ), (1, ))
    assert_size_stride(arg402_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg403_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg404_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg405_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg406_1, (1280, ), (1, ))
    assert_size_stride(arg407_1, (1280, ), (1, ))
    assert_size_stride(arg408_1, (1280, ), (1, ))
    assert_size_stride(arg409_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg410_1, (10240, ), (1, ))
    assert_size_stride(arg411_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg412_1, (1280, ), (1, ))
    assert_size_stride(arg413_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg414_1, (1280, ), (1, ))
    assert_size_stride(arg415_1, (1920, ), (1, ))
    assert_size_stride(arg416_1, (1920, ), (1, ))
    assert_size_stride(arg417_1, (1280, 1920, 3, 3), (17280, 9, 3, 1))
    assert_size_stride(arg418_1, (1280, ), (1, ))
    assert_size_stride(arg419_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg420_1, (1280, ), (1, ))
    assert_size_stride(arg421_1, (1280, ), (1, ))
    assert_size_stride(arg422_1, (1280, ), (1, ))
    assert_size_stride(arg423_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg424_1, (1280, ), (1, ))
    assert_size_stride(arg425_1, (1280, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg426_1, (1280, ), (1, ))
    assert_size_stride(arg427_1, (1280, ), (1, ))
    assert_size_stride(arg428_1, (1280, ), (1, ))
    assert_size_stride(arg429_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg430_1, (1280, ), (1, ))
    assert_size_stride(arg431_1, (1280, ), (1, ))
    assert_size_stride(arg432_1, (1280, ), (1, ))
    assert_size_stride(arg433_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg434_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg435_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg436_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg437_1, (1280, ), (1, ))
    assert_size_stride(arg438_1, (1280, ), (1, ))
    assert_size_stride(arg439_1, (1280, ), (1, ))
    assert_size_stride(arg440_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg441_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg442_1, (1280, 1024), (1024, 1))
    assert_size_stride(arg443_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg444_1, (1280, ), (1, ))
    assert_size_stride(arg445_1, (1280, ), (1, ))
    assert_size_stride(arg446_1, (1280, ), (1, ))
    assert_size_stride(arg447_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg448_1, (10240, ), (1, ))
    assert_size_stride(arg449_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg450_1, (1280, ), (1, ))
    assert_size_stride(arg451_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg452_1, (1280, ), (1, ))
    assert_size_stride(arg453_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg454_1, (1280, ), (1, ))
    assert_size_stride(arg455_1, (1920, ), (1, ))
    assert_size_stride(arg456_1, (1920, ), (1, ))
    assert_size_stride(arg457_1, (640, 1920, 3, 3), (17280, 9, 3, 1))
    assert_size_stride(arg458_1, (640, ), (1, ))
    assert_size_stride(arg459_1, (640, 1280), (1280, 1))
    assert_size_stride(arg460_1, (640, ), (1, ))
    assert_size_stride(arg461_1, (640, ), (1, ))
    assert_size_stride(arg462_1, (640, ), (1, ))
    assert_size_stride(arg463_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg464_1, (640, ), (1, ))
    assert_size_stride(arg465_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg466_1, (640, ), (1, ))
    assert_size_stride(arg467_1, (640, ), (1, ))
    assert_size_stride(arg468_1, (640, ), (1, ))
    assert_size_stride(arg469_1, (640, 640), (640, 1))
    assert_size_stride(arg470_1, (640, ), (1, ))
    assert_size_stride(arg471_1, (640, ), (1, ))
    assert_size_stride(arg472_1, (640, ), (1, ))
    assert_size_stride(arg473_1, (640, 640), (640, 1))
    assert_size_stride(arg474_1, (640, 640), (640, 1))
    assert_size_stride(arg475_1, (640, 640), (640, 1))
    assert_size_stride(arg476_1, (640, 640), (640, 1))
    assert_size_stride(arg477_1, (640, ), (1, ))
    assert_size_stride(arg478_1, (640, ), (1, ))
    assert_size_stride(arg479_1, (640, ), (1, ))
    assert_size_stride(arg480_1, (640, 640), (640, 1))
    assert_size_stride(arg481_1, (640, 1024), (1024, 1))
    assert_size_stride(arg482_1, (640, 1024), (1024, 1))
    assert_size_stride(arg483_1, (640, 640), (640, 1))
    assert_size_stride(arg484_1, (640, ), (1, ))
    assert_size_stride(arg485_1, (640, ), (1, ))
    assert_size_stride(arg486_1, (640, ), (1, ))
    assert_size_stride(arg487_1, (5120, 640), (640, 1))
    assert_size_stride(arg488_1, (5120, ), (1, ))
    assert_size_stride(arg489_1, (640, 2560), (2560, 1))
    assert_size_stride(arg490_1, (640, ), (1, ))
    assert_size_stride(arg491_1, (640, 640), (640, 1))
    assert_size_stride(arg492_1, (640, ), (1, ))
    assert_size_stride(arg493_1, (1280, ), (1, ))
    assert_size_stride(arg494_1, (1280, ), (1, ))
    assert_size_stride(arg495_1, (640, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg496_1, (640, ), (1, ))
    assert_size_stride(arg497_1, (640, 1280), (1280, 1))
    assert_size_stride(arg498_1, (640, ), (1, ))
    assert_size_stride(arg499_1, (640, ), (1, ))
    assert_size_stride(arg500_1, (640, ), (1, ))
    assert_size_stride(arg501_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg502_1, (640, ), (1, ))
    assert_size_stride(arg503_1, (640, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg504_1, (640, ), (1, ))
    assert_size_stride(arg505_1, (640, ), (1, ))
    assert_size_stride(arg506_1, (640, ), (1, ))
    assert_size_stride(arg507_1, (640, 640), (640, 1))
    assert_size_stride(arg508_1, (640, ), (1, ))
    assert_size_stride(arg509_1, (640, ), (1, ))
    assert_size_stride(arg510_1, (640, ), (1, ))
    assert_size_stride(arg511_1, (640, 640), (640, 1))
    assert_size_stride(arg512_1, (640, 640), (640, 1))
    assert_size_stride(arg513_1, (640, 640), (640, 1))
    assert_size_stride(arg514_1, (640, 640), (640, 1))
    assert_size_stride(arg515_1, (640, ), (1, ))
    assert_size_stride(arg516_1, (640, ), (1, ))
    assert_size_stride(arg517_1, (640, ), (1, ))
    assert_size_stride(arg518_1, (640, 640), (640, 1))
    assert_size_stride(arg519_1, (640, 1024), (1024, 1))
    assert_size_stride(arg520_1, (640, 1024), (1024, 1))
    assert_size_stride(arg521_1, (640, 640), (640, 1))
    assert_size_stride(arg522_1, (640, ), (1, ))
    assert_size_stride(arg523_1, (640, ), (1, ))
    assert_size_stride(arg524_1, (640, ), (1, ))
    assert_size_stride(arg525_1, (5120, 640), (640, 1))
    assert_size_stride(arg526_1, (5120, ), (1, ))
    assert_size_stride(arg527_1, (640, 2560), (2560, 1))
    assert_size_stride(arg528_1, (640, ), (1, ))
    assert_size_stride(arg529_1, (640, 640), (640, 1))
    assert_size_stride(arg530_1, (640, ), (1, ))
    assert_size_stride(arg531_1, (960, ), (1, ))
    assert_size_stride(arg532_1, (960, ), (1, ))
    assert_size_stride(arg533_1, (640, 960, 3, 3), (8640, 9, 3, 1))
    assert_size_stride(arg534_1, (640, ), (1, ))
    assert_size_stride(arg535_1, (640, 1280), (1280, 1))
    assert_size_stride(arg536_1, (640, ), (1, ))
    assert_size_stride(arg537_1, (640, ), (1, ))
    assert_size_stride(arg538_1, (640, ), (1, ))
    assert_size_stride(arg539_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg540_1, (640, ), (1, ))
    assert_size_stride(arg541_1, (640, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg542_1, (640, ), (1, ))
    assert_size_stride(arg543_1, (640, ), (1, ))
    assert_size_stride(arg544_1, (640, ), (1, ))
    assert_size_stride(arg545_1, (640, 640), (640, 1))
    assert_size_stride(arg546_1, (640, ), (1, ))
    assert_size_stride(arg547_1, (640, ), (1, ))
    assert_size_stride(arg548_1, (640, ), (1, ))
    assert_size_stride(arg549_1, (640, 640), (640, 1))
    assert_size_stride(arg550_1, (640, 640), (640, 1))
    assert_size_stride(arg551_1, (640, 640), (640, 1))
    assert_size_stride(arg552_1, (640, 640), (640, 1))
    assert_size_stride(arg553_1, (640, ), (1, ))
    assert_size_stride(arg554_1, (640, ), (1, ))
    assert_size_stride(arg555_1, (640, ), (1, ))
    assert_size_stride(arg556_1, (640, 640), (640, 1))
    assert_size_stride(arg557_1, (640, 1024), (1024, 1))
    assert_size_stride(arg558_1, (640, 1024), (1024, 1))
    assert_size_stride(arg559_1, (640, 640), (640, 1))
    assert_size_stride(arg560_1, (640, ), (1, ))
    assert_size_stride(arg561_1, (640, ), (1, ))
    assert_size_stride(arg562_1, (640, ), (1, ))
    assert_size_stride(arg563_1, (5120, 640), (640, 1))
    assert_size_stride(arg564_1, (5120, ), (1, ))
    assert_size_stride(arg565_1, (640, 2560), (2560, 1))
    assert_size_stride(arg566_1, (640, ), (1, ))
    assert_size_stride(arg567_1, (640, 640), (640, 1))
    assert_size_stride(arg568_1, (640, ), (1, ))
    assert_size_stride(arg569_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg570_1, (640, ), (1, ))
    assert_size_stride(arg571_1, (960, ), (1, ))
    assert_size_stride(arg572_1, (960, ), (1, ))
    assert_size_stride(arg573_1, (320, 960, 3, 3), (8640, 9, 3, 1))
    assert_size_stride(arg574_1, (320, ), (1, ))
    assert_size_stride(arg575_1, (320, 1280), (1280, 1))
    assert_size_stride(arg576_1, (320, ), (1, ))
    assert_size_stride(arg577_1, (320, ), (1, ))
    assert_size_stride(arg578_1, (320, ), (1, ))
    assert_size_stride(arg579_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg580_1, (320, ), (1, ))
    assert_size_stride(arg581_1, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg582_1, (320, ), (1, ))
    assert_size_stride(arg583_1, (320, ), (1, ))
    assert_size_stride(arg584_1, (320, ), (1, ))
    assert_size_stride(arg585_1, (320, 320), (320, 1))
    assert_size_stride(arg586_1, (320, ), (1, ))
    assert_size_stride(arg587_1, (320, ), (1, ))
    assert_size_stride(arg588_1, (320, ), (1, ))
    assert_size_stride(arg589_1, (320, 320), (320, 1))
    assert_size_stride(arg590_1, (320, 320), (320, 1))
    assert_size_stride(arg591_1, (320, 320), (320, 1))
    assert_size_stride(arg592_1, (320, 320), (320, 1))
    assert_size_stride(arg593_1, (320, ), (1, ))
    assert_size_stride(arg594_1, (320, ), (1, ))
    assert_size_stride(arg595_1, (320, ), (1, ))
    assert_size_stride(arg596_1, (320, 320), (320, 1))
    assert_size_stride(arg597_1, (320, 1024), (1024, 1))
    assert_size_stride(arg598_1, (320, 1024), (1024, 1))
    assert_size_stride(arg599_1, (320, 320), (320, 1))
    assert_size_stride(arg600_1, (320, ), (1, ))
    assert_size_stride(arg601_1, (320, ), (1, ))
    assert_size_stride(arg602_1, (320, ), (1, ))
    assert_size_stride(arg603_1, (2560, 320), (320, 1))
    assert_size_stride(arg604_1, (2560, ), (1, ))
    assert_size_stride(arg605_1, (320, 1280), (1280, 1))
    assert_size_stride(arg606_1, (320, ), (1, ))
    assert_size_stride(arg607_1, (320, 320), (320, 1))
    assert_size_stride(arg608_1, (320, ), (1, ))
    assert_size_stride(arg609_1, (640, ), (1, ))
    assert_size_stride(arg610_1, (640, ), (1, ))
    assert_size_stride(arg611_1, (320, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg612_1, (320, ), (1, ))
    assert_size_stride(arg613_1, (320, 1280), (1280, 1))
    assert_size_stride(arg614_1, (320, ), (1, ))
    assert_size_stride(arg615_1, (320, ), (1, ))
    assert_size_stride(arg616_1, (320, ), (1, ))
    assert_size_stride(arg617_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg618_1, (320, ), (1, ))
    assert_size_stride(arg619_1, (320, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg620_1, (320, ), (1, ))
    assert_size_stride(arg621_1, (320, ), (1, ))
    assert_size_stride(arg622_1, (320, ), (1, ))
    assert_size_stride(arg623_1, (320, 320), (320, 1))
    assert_size_stride(arg624_1, (320, ), (1, ))
    assert_size_stride(arg625_1, (320, ), (1, ))
    assert_size_stride(arg626_1, (320, ), (1, ))
    assert_size_stride(arg627_1, (320, 320), (320, 1))
    assert_size_stride(arg628_1, (320, 320), (320, 1))
    assert_size_stride(arg629_1, (320, 320), (320, 1))
    assert_size_stride(arg630_1, (320, 320), (320, 1))
    assert_size_stride(arg631_1, (320, ), (1, ))
    assert_size_stride(arg632_1, (320, ), (1, ))
    assert_size_stride(arg633_1, (320, ), (1, ))
    assert_size_stride(arg634_1, (320, 320), (320, 1))
    assert_size_stride(arg635_1, (320, 1024), (1024, 1))
    assert_size_stride(arg636_1, (320, 1024), (1024, 1))
    assert_size_stride(arg637_1, (320, 320), (320, 1))
    assert_size_stride(arg638_1, (320, ), (1, ))
    assert_size_stride(arg639_1, (320, ), (1, ))
    assert_size_stride(arg640_1, (320, ), (1, ))
    assert_size_stride(arg641_1, (2560, 320), (320, 1))
    assert_size_stride(arg642_1, (2560, ), (1, ))
    assert_size_stride(arg643_1, (320, 1280), (1280, 1))
    assert_size_stride(arg644_1, (320, ), (1, ))
    assert_size_stride(arg645_1, (320, 320), (320, 1))
    assert_size_stride(arg646_1, (320, ), (1, ))
    assert_size_stride(arg647_1, (640, ), (1, ))
    assert_size_stride(arg648_1, (640, ), (1, ))
    assert_size_stride(arg649_1, (320, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg650_1, (320, ), (1, ))
    assert_size_stride(arg651_1, (320, 1280), (1280, 1))
    assert_size_stride(arg652_1, (320, ), (1, ))
    assert_size_stride(arg653_1, (320, ), (1, ))
    assert_size_stride(arg654_1, (320, ), (1, ))
    assert_size_stride(arg655_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg656_1, (320, ), (1, ))
    assert_size_stride(arg657_1, (320, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg658_1, (320, ), (1, ))
    assert_size_stride(arg659_1, (320, ), (1, ))
    assert_size_stride(arg660_1, (320, ), (1, ))
    assert_size_stride(arg661_1, (320, 320), (320, 1))
    assert_size_stride(arg662_1, (320, ), (1, ))
    assert_size_stride(arg663_1, (320, ), (1, ))
    assert_size_stride(arg664_1, (320, ), (1, ))
    assert_size_stride(arg665_1, (320, 320), (320, 1))
    assert_size_stride(arg666_1, (320, 320), (320, 1))
    assert_size_stride(arg667_1, (320, 320), (320, 1))
    assert_size_stride(arg668_1, (320, 320), (320, 1))
    assert_size_stride(arg669_1, (320, ), (1, ))
    assert_size_stride(arg670_1, (320, ), (1, ))
    assert_size_stride(arg671_1, (320, ), (1, ))
    assert_size_stride(arg672_1, (320, 320), (320, 1))
    assert_size_stride(arg673_1, (320, 1024), (1024, 1))
    assert_size_stride(arg674_1, (320, 1024), (1024, 1))
    assert_size_stride(arg675_1, (320, 320), (320, 1))
    assert_size_stride(arg676_1, (320, ), (1, ))
    assert_size_stride(arg677_1, (320, ), (1, ))
    assert_size_stride(arg678_1, (320, ), (1, ))
    assert_size_stride(arg679_1, (2560, 320), (320, 1))
    assert_size_stride(arg680_1, (2560, ), (1, ))
    assert_size_stride(arg681_1, (320, 1280), (1280, 1))
    assert_size_stride(arg682_1, (320, ), (1, ))
    assert_size_stride(arg683_1, (320, 320), (320, 1))
    assert_size_stride(arg684_1, (320, ), (1, ))
    assert_size_stride(arg685_1, (320, ), (1, ))
    assert_size_stride(arg686_1, (320, ), (1, ))
    assert_size_stride(arg687_1, (4, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg688_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(2):
        torch.cuda.set_device(2)
        buf0 = empty_strided_cuda((32, 4, 64, 64), (16384, 1, 256, 4), torch.float16)
        # Topologically Sorted Source Nodes: [sample_3], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_0.run(arg0_1, buf0, 128, 4096, stream=stream2)
        del arg0_1
        buf1 = empty_strided_cuda((320, 4, 3, 3), (36, 1, 12, 4), torch.float16)
        # Topologically Sorted Source Nodes: [sample_3], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_1.run(arg7_1, buf1, 1280, 9, stream=stream2)
        del arg7_1
        # Topologically Sorted Source Nodes: [sample_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf3 = empty_strided_cuda((32, 32, 1, 1), (32, 1, 1024, 1024), torch.float32)
        buf4 = empty_strided_cuda((32, 32, 1, 1), (32, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_2.run(buf2, arg8_1, buf3, buf4, 1024, 40960, stream=stream2)
        buf7 = empty_strided_cuda((32, 320, 64, 64), (1310720, 1, 20480, 320), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_3.run(buf2, arg8_1, buf3, buf4, arg9_1, arg10_1, buf7, 41943040, stream=stream2)
        del arg10_1
        del arg9_1
        buf8 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg11_1, buf8, 102400, 9, stream=stream2)
        del arg11_1
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf10 = empty_strided_cuda((32, 320), (320, 1), torch.float32)
        # Topologically Sorted Source Nodes: [emb_3], Original ATen: [aten.cat]
        stream2 = get_raw_stream(2)
        triton_poi_fused_cat_5.run(arg1_1, buf10, 10240, stream=stream2)
        del arg1_1
        buf11 = empty_strided_cuda((32, 320), (320, 1), torch.float16)
        # Topologically Sorted Source Nodes: [emb_4, t_emb], Original ATen: [aten.cat, aten._to_copy]
        stream2 = get_raw_stream(2)
        triton_poi_fused__to_copy_cat_6.run(buf10, buf11, 10240, stream=stream2)
        del buf10
        buf12 = empty_strided_cuda((32, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [emb_4, t_emb, sample], Original ATen: [aten.cat, aten._to_copy, aten.addmm]
        extern_kernels.mm(buf11, reinterpret_tensor(arg2_1, (320, 1280), (1, 320), 0), out=buf12)
        del arg2_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [sample, sample_1], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_7.run(buf13, arg3_1, 40960, stream=stream2)
        del arg3_1
        buf14 = empty_strided_cuda((32, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [sample, sample_1, sample_2], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf13, reinterpret_tensor(arg4_1, (1280, 1280), (1, 1280), 0), out=buf14)
        del arg4_1
        buf15 = buf13; del buf13  # reuse
        buf75 = empty_strided_cuda((32, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [sample_2, temb, temb_2], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_8.run(buf14, arg5_1, buf15, buf75, 40960, stream=stream2)
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb, linear_2], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf15, reinterpret_tensor(arg13_1, (1280, 320), (1, 1280), 0), out=buf16)
        del arg13_1
        buf17 = buf4; del buf4  # reuse
        buf18 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_9.run(buf9, arg12_1, buf16, arg14_1, buf17, buf18, 1024, 40960, stream=stream2)
        buf21 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_10.run(buf9, arg12_1, buf16, arg14_1, buf17, buf18, arg15_1, arg16_1, buf21, 41943040, stream=stream2)
        del arg12_1
        del arg14_1
        del arg15_1
        del arg16_1
        buf22 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_7], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg17_1, buf22, 102400, 9, stream=stream2)
        del arg17_1
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_7], Original ATen: [aten.silu, aten.convolution]
        buf23 = extern_kernels.convolution(buf21, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf24 = buf18; del buf18  # reuse
        buf25 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_8], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_11.run(buf2, arg8_1, buf23, arg18_1, buf24, buf25, 1024, 40960, stream=stream2)
        buf27 = reinterpret_tensor(buf21, (32, 4096, 320), (1310720, 320, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_12.run(buf2, arg8_1, buf23, arg18_1, buf24, buf25, arg19_1, arg20_1, buf27, 41943040, stream=stream2)
        del arg19_1
        del arg20_1
        buf28 = reinterpret_tensor(buf9, (131072, 320), (320, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (131072, 320), (320, 1), 0), reinterpret_tensor(arg21_1, (320, 320), (1, 320), 0), out=buf28)
        del arg21_1
        buf32 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, norm_hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_13.run(buf28, arg22_1, arg23_1, arg24_1, buf32, 131072, 320, stream=stream2)
        del arg23_1
        del arg24_1
        buf33 = empty_strided_cuda((131072, 320), (320, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (131072, 320), (320, 1), 0), reinterpret_tensor(arg25_1, (320, 320), (1, 320), 0), out=buf33)
        del arg25_1
        buf34 = empty_strided_cuda((131072, 320), (320, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (131072, 320), (320, 1), 0), reinterpret_tensor(arg26_1, (320, 320), (1, 320), 0), out=buf34)
        del arg26_1
        buf35 = empty_strided_cuda((131072, 320), (320, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (131072, 320), (320, 1), 0), reinterpret_tensor(arg27_1, (320, 320), (1, 320), 0), out=buf35)
        del arg27_1
        del buf32
        # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf36 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf33, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf34, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf35, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), scale=0.125)
        del buf33
        buf37 = buf36[0]
        assert_size_stride(buf37, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf36
        buf42 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf37, (131072, 320), (320, 1), 0), reinterpret_tensor(arg28_1, (320, 320), (1, 320), 0), out=buf42)
        del arg28_1
        buf46 = reinterpret_tensor(buf37, (32, 4096, 320), (1310720, 320, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_16, hidden_states_17, norm_hidden_states_1], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_14.run(buf42, arg29_1, buf28, arg22_1, arg30_1, arg31_1, buf46, 131072, 320, stream=stream2)
        del arg30_1
        del arg31_1
        buf47 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [query_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (131072, 320), (320, 1), 0), reinterpret_tensor(arg32_1, (320, 320), (1, 320), 0), out=buf47)
        del arg32_1
        del buf46
        buf48 = empty_strided_cuda((2464, 320), (320, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg33_1, (1024, 320), (1, 1024), 0), out=buf48)
        del arg33_1
        buf49 = empty_strided_cuda((2464, 320), (320, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg34_1, (1024, 320), (1, 1024), 0), out=buf49)
        del arg34_1
        # Topologically Sorted Source Nodes: [hidden_states_18], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf50 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf47, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf48, (32, 5, 77, 64), (24640, 64, 320, 1), 0), reinterpret_tensor(buf49, (32, 5, 77, 64), (24640, 64, 320, 1), 0), scale=0.125)
        buf51 = buf50[0]
        assert_size_stride(buf51, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf50
        buf56 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf51, (131072, 320), (320, 1), 0), reinterpret_tensor(arg35_1, (320, 320), (1, 320), 0), out=buf56)
        del arg35_1
        buf57 = reinterpret_tensor(buf56, (32, 4096, 320), (1310720, 320, 1), 0); del buf56  # reuse
        buf61 = reinterpret_tensor(buf51, (32, 4096, 320), (1310720, 320, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_16, hidden_states_17, hidden_states_23, hidden_states_24, norm_hidden_states_2], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_15.run(buf57, arg36_1, buf42, arg29_1, buf28, arg22_1, arg37_1, arg38_1, buf61, 131072, 320, stream=stream2)
        del arg22_1
        del arg29_1
        del arg36_1
        del arg37_1
        del arg38_1
        buf62 = empty_strided_cuda((131072, 2560), (2560, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf61, (131072, 320), (320, 1), 0), reinterpret_tensor(arg39_1, (320, 2560), (1, 320), 0), out=buf62)
        del arg39_1
        buf63 = empty_strided_cuda((32, 4096, 1280), (5242880, 1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [gelu, hidden_states_27], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_16.run(buf62, arg40_1, buf63, 167772160, stream=stream2)
        del arg40_1
        buf64 = reinterpret_tensor(buf61, (131072, 320), (320, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf63, (131072, 1280), (1280, 1), 0), reinterpret_tensor(arg41_1, (1280, 320), (1, 1280), 0), out=buf64)
        del arg41_1
        buf65 = reinterpret_tensor(buf64, (32, 4096, 320), (1310720, 320, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_30], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_17.run(buf65, arg42_1, buf57, 41943040, stream=stream2)
        del arg42_1
        buf66 = reinterpret_tensor(buf57, (131072, 320), (320, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf65, (131072, 320), (320, 1), 0), reinterpret_tensor(arg43_1, (320, 320), (1, 320), 0), out=buf66)
        del arg43_1
        buf67 = reinterpret_tensor(buf65, (32, 320, 64, 64), (1310720, 4096, 64, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [sample_3, hidden_states_5, hidden_states_7, add_1, output_tensor, hidden_states_32, output], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_18.run(buf66, arg44_1, buf2, arg8_1, buf23, arg18_1, buf67, 131072, 320, stream=stream2)
        del arg18_1
        del arg44_1
        buf68 = buf25; del buf25  # reuse
        buf69 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_19.run(buf67, buf68, buf69, 1024, 40960, stream=stream2)
        buf72 = reinterpret_tensor(buf66, (32, 320, 64, 64), (1310720, 1, 20480, 320), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_20.run(buf67, buf68, buf69, arg45_1, arg46_1, buf72, 10240, 4096, stream=stream2)
        del arg45_1
        del arg46_1
        buf73 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg47_1, buf73, 102400, 9, stream=stream2)
        del arg47_1
        # Topologically Sorted Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.silu, aten.convolution]
        buf74 = extern_kernels.convolution(buf72, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf76 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_2, linear_15], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf75, reinterpret_tensor(arg49_1, (1280, 320), (1, 1280), 0), out=buf76)
        del arg49_1
        buf77 = buf69; del buf69  # reuse
        buf78 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_9.run(buf74, arg48_1, buf76, arg50_1, buf77, buf78, 1024, 40960, stream=stream2)
        buf81 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_10.run(buf74, arg48_1, buf76, arg50_1, buf77, buf78, arg51_1, arg52_1, buf81, 41943040, stream=stream2)
        del arg48_1
        del arg50_1
        del arg51_1
        del arg52_1
        buf82 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg53_1, buf82, 102400, 9, stream=stream2)
        del arg53_1
        # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
        buf83 = extern_kernels.convolution(buf81, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf84 = buf78; del buf78  # reuse
        buf85 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_21.run(buf67, buf83, arg54_1, buf84, buf85, 1024, 40960, stream=stream2)
        buf87 = reinterpret_tensor(buf81, (32, 4096, 320), (1310720, 320, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_22.run(buf67, buf83, arg54_1, buf84, buf85, arg55_1, arg56_1, buf87, 131072, 320, stream=stream2)
        del arg55_1
        del arg56_1
        buf88 = reinterpret_tensor(buf74, (131072, 320), (320, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (131072, 320), (320, 1), 0), reinterpret_tensor(arg57_1, (320, 320), (1, 320), 0), out=buf88)
        del arg57_1
        buf92 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43, norm_hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_13.run(buf88, arg58_1, arg59_1, arg60_1, buf92, 131072, 320, stream=stream2)
        del arg59_1
        del arg60_1
        buf93 = reinterpret_tensor(buf23, (131072, 320), (320, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [query_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (131072, 320), (320, 1), 0), reinterpret_tensor(arg61_1, (320, 320), (1, 320), 0), out=buf93)
        del arg61_1
        buf94 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [key_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (131072, 320), (320, 1), 0), reinterpret_tensor(arg62_1, (320, 320), (1, 320), 0), out=buf94)
        del arg62_1
        buf95 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [value_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (131072, 320), (320, 1), 0), reinterpret_tensor(arg63_1, (320, 320), (1, 320), 0), out=buf95)
        del arg63_1
        del buf92
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf96 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf93, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf94, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf95, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), scale=0.125)
        del buf93
        buf97 = buf96[0]
        assert_size_stride(buf97, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf96
        buf102 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf97, (131072, 320), (320, 1), 0), reinterpret_tensor(arg64_1, (320, 320), (1, 320), 0), out=buf102)
        del arg64_1
        buf106 = reinterpret_tensor(buf97, (32, 4096, 320), (1310720, 320, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_49, hidden_states_50, norm_hidden_states_4], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_14.run(buf102, arg65_1, buf88, arg58_1, arg66_1, arg67_1, buf106, 131072, 320, stream=stream2)
        del arg66_1
        del arg67_1
        buf107 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [query_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (131072, 320), (320, 1), 0), reinterpret_tensor(arg68_1, (320, 320), (1, 320), 0), out=buf107)
        del arg68_1
        del buf106
        buf108 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [key_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 320), (1, 1024), 0), out=buf108)
        del arg69_1
        buf109 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [value_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg70_1, (1024, 320), (1, 1024), 0), out=buf109)
        del arg70_1
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf110 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf107, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf108, (32, 5, 77, 64), (24640, 64, 320, 1), 0), reinterpret_tensor(buf109, (32, 5, 77, 64), (24640, 64, 320, 1), 0), scale=0.125)
        buf111 = buf110[0]
        assert_size_stride(buf111, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf110
        buf116 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf111, (131072, 320), (320, 1), 0), reinterpret_tensor(arg71_1, (320, 320), (1, 320), 0), out=buf116)
        del arg71_1
        buf117 = reinterpret_tensor(buf116, (32, 4096, 320), (1310720, 320, 1), 0); del buf116  # reuse
        buf121 = reinterpret_tensor(buf111, (32, 4096, 320), (1310720, 320, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_49, hidden_states_50, hidden_states_56, hidden_states_57, norm_hidden_states_5], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_15.run(buf117, arg72_1, buf102, arg65_1, buf88, arg58_1, arg73_1, arg74_1, buf121, 131072, 320, stream=stream2)
        del arg58_1
        del arg65_1
        del arg72_1
        del arg73_1
        del arg74_1
        del buf102
        del buf88
        buf122 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf121, (131072, 320), (320, 1), 0), reinterpret_tensor(arg75_1, (320, 2560), (1, 320), 0), out=buf122)
        del arg75_1
        buf123 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [gelu_1, hidden_states_60], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_16.run(buf122, arg76_1, buf123, 167772160, stream=stream2)
        del arg76_1
        buf124 = reinterpret_tensor(buf121, (131072, 320), (320, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf123, (131072, 1280), (1280, 1), 0), reinterpret_tensor(arg77_1, (1280, 320), (1, 1280), 0), out=buf124)
        del arg77_1
        buf125 = reinterpret_tensor(buf124, (32, 4096, 320), (1310720, 320, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_63], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_17.run(buf125, arg78_1, buf117, 41943040, stream=stream2)
        del arg78_1
        buf126 = reinterpret_tensor(buf117, (131072, 320), (320, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf125, (131072, 320), (320, 1), 0), reinterpret_tensor(arg79_1, (320, 320), (1, 320), 0), out=buf126)
        del arg79_1
        buf127 = reinterpret_tensor(buf126, (32, 320, 64, 64), (1310720, 1, 20480, 320), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_40, add_7, output_tensor_1, hidden_states_65, output_1], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_23.run(buf127, arg80_1, buf67, buf83, arg54_1, 131072, 320, stream=stream2)
        del arg54_1
        del arg80_1
        buf128 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg81_1, buf128, 102400, 9, stream=stream2)
        del arg81_1
        # Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf127, buf128, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (32, 320, 32, 32), (327680, 1, 10240, 320))
        buf130 = empty_strided_cuda((32, 320, 32, 32), (327680, 1024, 32, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_24.run(buf129, arg82_1, buf130, 10240, 1024, stream=stream2)
        del arg82_1
        buf131 = buf85; del buf85  # reuse
        buf132 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_25.run(buf130, buf131, buf132, 1024, 10240, stream=stream2)
        buf135 = buf129; del buf129  # reuse
        buf143 = empty_strided_cuda((32, 320, 32, 32), (327680, 1, 10240, 320), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, input_tensor], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_26.run(buf130, buf131, buf132, arg83_1, arg84_1, buf135, buf143, 10240, 1024, stream=stream2)
        del arg83_1
        del arg84_1
        buf136 = empty_strided_cuda((640, 320, 3, 3), (2880, 1, 960, 320), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_27.run(arg85_1, buf136, 204800, 9, stream=stream2)
        del arg85_1
        # Topologically Sorted Source Nodes: [hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution]
        buf137 = extern_kernels.convolution(buf135, buf136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf138 = buf75; del buf75  # reuse
        buf200 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_4, temb_6], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_8.run(buf14, arg5_1, buf138, buf200, 40960, stream=stream2)
        buf139 = empty_strided_cuda((32, 640), (640, 1), torch.float16)
        # Topologically Sorted Source Nodes: [sample_2, temb_4, linear_28], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf138, reinterpret_tensor(arg87_1, (1280, 640), (1, 1280), 0), out=buf139)
        del arg87_1
        buf140 = buf132; del buf132  # reuse
        buf141 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_28.run(buf137, arg86_1, buf139, arg88_1, buf140, buf141, 1024, 20480, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del arg93_1
        buf146 = empty_strided_cuda((32, 640, 32, 32), (655360, 1, 20480, 640), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_29.run(buf137, arg86_1, buf139, arg88_1, buf140, buf141, arg89_1, arg90_1, buf146, 20971520, stream=stream2)
        del arg86_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf147 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_72, hidden_states_74], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg91_1, buf147, 409600, 9, stream=stream2)
        del arg91_1
        # Topologically Sorted Source Nodes: [hidden_states_72, hidden_states_74], Original ATen: [aten.silu, aten.convolution]
        buf148 = extern_kernels.convolution(buf146, buf147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf149 = buf141; del buf141  # reuse
        buf150 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_31.run(buf144, arg94_1, buf148, arg92_1, buf149, buf150, 1024, 20480, stream=stream2)
        buf152 = reinterpret_tensor(buf146, (32, 1024, 640), (655360, 640, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_32.run(buf144, arg94_1, buf148, arg92_1, buf149, buf150, arg95_1, arg96_1, buf152, 20971520, stream=stream2)
        del arg95_1
        del arg96_1
        buf153 = reinterpret_tensor(buf137, (32768, 640), (640, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (32768, 640), (640, 1), 0), reinterpret_tensor(arg97_1, (640, 640), (1, 640), 0), out=buf153)
        del arg97_1
        buf157 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, norm_hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_33.run(buf153, arg98_1, arg99_1, arg100_1, buf157, 32768, 640, stream=stream2)
        del arg100_1
        del arg99_1
        buf158 = empty_strided_cuda((32768, 640), (640, 1), torch.float16)
        # Topologically Sorted Source Nodes: [query_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (32768, 640), (640, 1), 0), reinterpret_tensor(arg101_1, (640, 640), (1, 640), 0), out=buf158)
        del arg101_1
        buf159 = empty_strided_cuda((32768, 640), (640, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (32768, 640), (640, 1), 0), reinterpret_tensor(arg102_1, (640, 640), (1, 640), 0), out=buf159)
        del arg102_1
        buf160 = empty_strided_cuda((32768, 640), (640, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (32768, 640), (640, 1), 0), reinterpret_tensor(arg103_1, (640, 640), (1, 640), 0), out=buf160)
        del arg103_1
        del buf157
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf161 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf158, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf159, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf160, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), scale=0.125)
        del buf158
        buf162 = buf161[0]
        assert_size_stride(buf162, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf161
        buf167 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf162, (32768, 640), (640, 1), 0), reinterpret_tensor(arg104_1, (640, 640), (1, 640), 0), out=buf167)
        del arg104_1
        buf171 = reinterpret_tensor(buf162, (32, 1024, 640), (655360, 640, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_83, hidden_states_84, norm_hidden_states_7], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_34.run(buf167, arg105_1, buf153, arg98_1, arg106_1, arg107_1, buf171, 32768, 640, stream=stream2)
        del arg106_1
        del arg107_1
        buf172 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [query_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (32768, 640), (640, 1), 0), reinterpret_tensor(arg108_1, (640, 640), (1, 640), 0), out=buf172)
        del arg108_1
        del buf171
        buf173 = empty_strided_cuda((2464, 640), (640, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 640), (1, 1024), 0), out=buf173)
        del arg109_1
        buf174 = empty_strided_cuda((2464, 640), (640, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg110_1, (1024, 640), (1, 1024), 0), out=buf174)
        del arg110_1
        # Topologically Sorted Source Nodes: [hidden_states_85], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf175 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf172, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf173, (32, 10, 77, 64), (49280, 64, 640, 1), 0), reinterpret_tensor(buf174, (32, 10, 77, 64), (49280, 64, 640, 1), 0), scale=0.125)
        buf176 = buf175[0]
        assert_size_stride(buf176, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf175
        buf181 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf176, (32768, 640), (640, 1), 0), reinterpret_tensor(arg111_1, (640, 640), (1, 640), 0), out=buf181)
        del arg111_1
        buf182 = reinterpret_tensor(buf181, (32, 1024, 640), (655360, 640, 1), 0); del buf181  # reuse
        buf186 = reinterpret_tensor(buf176, (32, 1024, 640), (655360, 640, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_83, hidden_states_84, hidden_states_90, hidden_states_91, norm_hidden_states_8], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_35.run(buf182, arg112_1, buf167, arg105_1, buf153, arg98_1, arg113_1, arg114_1, buf186, 32768, 640, stream=stream2)
        del arg105_1
        del arg112_1
        del arg113_1
        del arg114_1
        del arg98_1
        del buf153
        buf187 = reinterpret_tensor(buf123, (32768, 5120), (5120, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf186, (32768, 640), (640, 1), 0), reinterpret_tensor(arg115_1, (640, 5120), (1, 640), 0), out=buf187)
        del arg115_1
        buf188 = empty_strided_cuda((32, 1024, 2560), (2621440, 2560, 1), torch.float16)
        # Topologically Sorted Source Nodes: [gelu_2, hidden_states_94], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_36.run(buf187, arg116_1, buf188, 83886080, stream=stream2)
        del arg116_1
        buf189 = reinterpret_tensor(buf186, (32768, 640), (640, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf188, (32768, 2560), (2560, 1), 0), reinterpret_tensor(arg117_1, (2560, 640), (1, 2560), 0), out=buf189)
        del arg117_1
        buf190 = reinterpret_tensor(buf189, (32, 1024, 640), (655360, 640, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_97], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_37.run(buf190, arg118_1, buf182, 20971520, stream=stream2)
        del arg118_1
        buf191 = reinterpret_tensor(buf182, (32768, 640), (640, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_98], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf190, (32768, 640), (640, 1), 0), reinterpret_tensor(arg119_1, (640, 640), (1, 640), 0), out=buf191)
        del arg119_1
        buf192 = reinterpret_tensor(buf190, (32, 640, 32, 32), (655360, 1024, 32, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_72, hidden_states_74, add_13, output_tensor_2, hidden_states_99, output_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_38.run(buf191, arg120_1, buf144, arg94_1, buf148, arg92_1, buf192, 32768, 640, stream=stream2)
        del arg120_1
        del arg92_1
        del arg94_1
        buf193 = buf150; del buf150  # reuse
        buf194 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_39.run(buf192, buf193, buf194, 1024, 20480, stream=stream2)
        buf197 = reinterpret_tensor(buf191, (32, 640, 32, 32), (655360, 1, 20480, 640), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_40.run(buf192, buf193, buf194, arg121_1, arg122_1, buf197, 20480, 1024, stream=stream2)
        del arg121_1
        del arg122_1
        buf198 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg123_1, buf198, 409600, 9, stream=stream2)
        del arg123_1
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
        buf199 = extern_kernels.convolution(buf197, buf198, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf201 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_6, linear_41], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf200, reinterpret_tensor(arg125_1, (1280, 640), (1, 1280), 0), out=buf201)
        del arg125_1
        buf202 = buf194; del buf194  # reuse
        buf203 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_28.run(buf199, arg124_1, buf201, arg126_1, buf202, buf203, 1024, 20480, stream=stream2)
        buf206 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_105], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_29.run(buf199, arg124_1, buf201, arg126_1, buf202, buf203, arg127_1, arg128_1, buf206, 20971520, stream=stream2)
        del arg124_1
        del arg126_1
        del arg127_1
        del arg128_1
        buf207 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105, hidden_states_107], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg129_1, buf207, 409600, 9, stream=stream2)
        del arg129_1
        # Topologically Sorted Source Nodes: [hidden_states_105, hidden_states_107], Original ATen: [aten.silu, aten.convolution]
        buf208 = extern_kernels.convolution(buf206, buf207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf209 = buf203; del buf203  # reuse
        buf210 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_41.run(buf192, buf208, arg130_1, buf209, buf210, 1024, 20480, stream=stream2)
        buf212 = reinterpret_tensor(buf206, (32, 1024, 640), (655360, 640, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_42.run(buf192, buf208, arg130_1, buf209, buf210, arg131_1, arg132_1, buf212, 32768, 640, stream=stream2)
        del arg131_1
        del arg132_1
        buf213 = reinterpret_tensor(buf199, (32768, 640), (640, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (32768, 640), (640, 1), 0), reinterpret_tensor(arg133_1, (640, 640), (1, 640), 0), out=buf213)
        del arg133_1
        buf217 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110, norm_hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_33.run(buf213, arg134_1, arg135_1, arg136_1, buf217, 32768, 640, stream=stream2)
        del arg135_1
        del arg136_1
        buf218 = reinterpret_tensor(buf148, (32768, 640), (640, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [query_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (32768, 640), (640, 1), 0), reinterpret_tensor(arg137_1, (640, 640), (1, 640), 0), out=buf218)
        del arg137_1
        buf219 = reinterpret_tensor(buf144, (32768, 640), (640, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [key_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (32768, 640), (640, 1), 0), reinterpret_tensor(arg138_1, (640, 640), (1, 640), 0), out=buf219)
        del arg138_1
        buf220 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [value_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (32768, 640), (640, 1), 0), reinterpret_tensor(arg139_1, (640, 640), (1, 640), 0), out=buf220)
        del arg139_1
        del buf217
        # Topologically Sorted Source Nodes: [hidden_states_111], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf221 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf218, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf219, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf220, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), scale=0.125)
        del buf218
        buf222 = buf221[0]
        assert_size_stride(buf222, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf221
        buf227 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf222, (32768, 640), (640, 1), 0), reinterpret_tensor(arg140_1, (640, 640), (1, 640), 0), out=buf227)
        del arg140_1
        buf231 = reinterpret_tensor(buf222, (32, 1024, 640), (655360, 640, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_116, hidden_states_117, norm_hidden_states_10], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_34.run(buf227, arg141_1, buf213, arg134_1, arg142_1, arg143_1, buf231, 32768, 640, stream=stream2)
        del arg142_1
        del arg143_1
        buf232 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [query_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (32768, 640), (640, 1), 0), reinterpret_tensor(arg144_1, (640, 640), (1, 640), 0), out=buf232)
        del arg144_1
        del buf231
        buf233 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [key_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg145_1, (1024, 640), (1, 1024), 0), out=buf233)
        del arg145_1
        buf234 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [value_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg146_1, (1024, 640), (1, 1024), 0), out=buf234)
        del arg146_1
        # Topologically Sorted Source Nodes: [hidden_states_118], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf235 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf232, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf233, (32, 10, 77, 64), (49280, 64, 640, 1), 0), reinterpret_tensor(buf234, (32, 10, 77, 64), (49280, 64, 640, 1), 0), scale=0.125)
        buf236 = buf235[0]
        assert_size_stride(buf236, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf235
        buf241 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf236, (32768, 640), (640, 1), 0), reinterpret_tensor(arg147_1, (640, 640), (1, 640), 0), out=buf241)
        del arg147_1
        buf242 = reinterpret_tensor(buf241, (32, 1024, 640), (655360, 640, 1), 0); del buf241  # reuse
        buf246 = reinterpret_tensor(buf236, (32, 1024, 640), (655360, 640, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_116, hidden_states_117, hidden_states_123, hidden_states_124, norm_hidden_states_11], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_35.run(buf242, arg148_1, buf227, arg141_1, buf213, arg134_1, arg149_1, arg150_1, buf246, 32768, 640, stream=stream2)
        del arg134_1
        del arg141_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf213
        buf247 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_125], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf246, (32768, 640), (640, 1), 0), reinterpret_tensor(arg151_1, (640, 5120), (1, 640), 0), out=buf247)
        del arg151_1
        buf248 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [gelu_3, hidden_states_127], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_36.run(buf247, arg152_1, buf248, 83886080, stream=stream2)
        del arg152_1
        buf249 = reinterpret_tensor(buf246, (32768, 640), (640, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_129], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf248, (32768, 2560), (2560, 1), 0), reinterpret_tensor(arg153_1, (2560, 640), (1, 2560), 0), out=buf249)
        del arg153_1
        buf250 = reinterpret_tensor(buf249, (32, 1024, 640), (655360, 640, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_37.run(buf250, arg154_1, buf242, 20971520, stream=stream2)
        del arg154_1
        buf251 = reinterpret_tensor(buf242, (32768, 640), (640, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf250, (32768, 640), (640, 1), 0), reinterpret_tensor(arg155_1, (640, 640), (1, 640), 0), out=buf251)
        del arg155_1
        buf252 = reinterpret_tensor(buf251, (32, 640, 32, 32), (655360, 1, 20480, 640), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105, hidden_states_107, add_19, output_tensor_3, hidden_states_132, output_3], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_43.run(buf252, arg156_1, buf192, buf208, arg130_1, 32768, 640, stream=stream2)
        del arg130_1
        del arg156_1
        buf253 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_133], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg157_1, buf253, 409600, 9, stream=stream2)
        del arg157_1
        # Topologically Sorted Source Nodes: [hidden_states_133], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf252, buf253, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (32, 640, 16, 16), (163840, 1, 10240, 640))
        buf255 = empty_strided_cuda((32, 640, 16, 16), (163840, 256, 16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_133], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_44.run(buf254, arg158_1, buf255, 20480, 256, stream=stream2)
        del arg158_1
        buf256 = buf210; del buf210  # reuse
        buf257 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_134], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_45.run(buf255, buf256, buf257, 1024, 5120, stream=stream2)
        buf260 = buf254; del buf254  # reuse
        buf268 = empty_strided_cuda((32, 640, 16, 16), (163840, 1, 10240, 640), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_134, hidden_states_135, input_tensor_1], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_46.run(buf255, buf256, buf257, arg159_1, arg160_1, buf260, buf268, 20480, 256, stream=stream2)
        del arg159_1
        del arg160_1
        buf261 = empty_strided_cuda((1280, 640, 3, 3), (5760, 1, 1920, 640), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_135, hidden_states_136], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_47.run(arg161_1, buf261, 819200, 9, stream=stream2)
        del arg161_1
        # Topologically Sorted Source Nodes: [hidden_states_135, hidden_states_136], Original ATen: [aten.silu, aten.convolution]
        buf262 = extern_kernels.convolution(buf260, buf261, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf263 = buf200; del buf200  # reuse
        buf325 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_8, temb_10], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_8.run(buf14, arg5_1, buf263, buf325, 40960, stream=stream2)
        buf264 = empty_strided_cuda((32, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [sample_2, temb_8, linear_54], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf263, reinterpret_tensor(arg163_1, (1280, 1280), (1, 1280), 0), out=buf264)
        del arg163_1
        buf265 = buf257; del buf257  # reuse
        buf266 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_138], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_48.run(buf262, arg162_1, buf264, arg164_1, buf265, buf266, 1024, 10240, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        del arg169_1
        buf271 = reinterpret_tensor(buf143, (32, 1280, 16, 16), (327680, 1, 20480, 1280), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_138, hidden_states_139], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_49.run(buf262, arg162_1, buf264, arg164_1, buf265, buf266, arg165_1, arg166_1, buf271, 10485760, stream=stream2)
        del arg162_1
        del arg164_1
        del arg165_1
        del arg166_1
        buf272 = empty_strided_cuda((1280, 1280, 3, 3), (11520, 1, 3840, 1280), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_141], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg167_1, buf272, 1638400, 9, stream=stream2)
        del arg167_1
        # Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_141], Original ATen: [aten.silu, aten.convolution]
        buf273 = extern_kernels.convolution(buf271, buf272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf274 = buf266; del buf266  # reuse
        buf275 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_142], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_51.run(buf269, arg170_1, buf273, arg168_1, buf274, buf275, 1024, 10240, stream=stream2)
        buf277 = reinterpret_tensor(buf271, (32, 256, 1280), (327680, 1280, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_52.run(buf269, arg170_1, buf273, arg168_1, buf274, buf275, arg171_1, arg172_1, buf277, 10485760, stream=stream2)
        del arg171_1
        del arg172_1
        buf278 = reinterpret_tensor(buf262, (8192, 1280), (1280, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg173_1, (1280, 1280), (1, 1280), 0), out=buf278)
        del arg173_1
        buf282 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144, norm_hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_native_layer_norm_53.run(buf278, arg174_1, arg175_1, arg176_1, buf282, 8192, 1280, stream=stream2)
        del arg175_1
        del arg176_1
        buf283 = reinterpret_tensor(buf135, (8192, 1280), (1280, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [query_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg177_1, (1280, 1280), (1, 1280), 0), out=buf283)
        del arg177_1
        buf284 = empty_strided_cuda((8192, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg178_1, (1280, 1280), (1, 1280), 0), out=buf284)
        del arg178_1
        buf285 = empty_strided_cuda((8192, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg179_1, (1280, 1280), (1, 1280), 0), out=buf285)
        del arg179_1
        del buf282
        # Topologically Sorted Source Nodes: [hidden_states_145], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf286 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf283, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf284, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf285, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), scale=0.125)
        del buf283
        buf287 = buf286[0]
        assert_size_stride(buf287, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf286
        buf292 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf287, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg180_1, (1280, 1280), (1, 1280), 0), out=buf292)
        del arg180_1
        buf296 = reinterpret_tensor(buf287, (32, 256, 1280), (327680, 1280, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144, hidden_states_150, hidden_states_151, norm_hidden_states_13], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_54.run(buf292, arg181_1, buf278, arg174_1, arg182_1, arg183_1, buf296, 8192, 1280, stream=stream2)
        del arg182_1
        del arg183_1
        buf297 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [query_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg184_1, (1280, 1280), (1, 1280), 0), out=buf297)
        del arg184_1
        del buf296
        buf298 = empty_strided_cuda((2464, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1280), (1, 1024), 0), out=buf298)
        del arg185_1
        buf299 = empty_strided_cuda((2464, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 1280), (1, 1024), 0), out=buf299)
        del arg186_1
        # Topologically Sorted Source Nodes: [hidden_states_152], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf300 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf297, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf298, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), reinterpret_tensor(buf299, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), scale=0.125)
        buf301 = buf300[0]
        assert_size_stride(buf301, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf300
        buf306 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_155], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf301, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg187_1, (1280, 1280), (1, 1280), 0), out=buf306)
        del arg187_1
        buf307 = reinterpret_tensor(buf306, (32, 256, 1280), (327680, 1280, 1), 0); del buf306  # reuse
        buf311 = reinterpret_tensor(buf301, (32, 256, 1280), (327680, 1280, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_144, hidden_states_150, hidden_states_151, hidden_states_157, hidden_states_158, norm_hidden_states_14], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_55.run(buf307, arg188_1, buf292, arg181_1, buf278, arg174_1, arg189_1, arg190_1, buf311, 8192, 1280, stream=stream2)
        del arg174_1
        del arg181_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf278
        buf312 = reinterpret_tensor(buf248, (8192, 10240), (10240, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf311, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg191_1, (1280, 10240), (1, 1280), 0), out=buf312)
        del arg191_1
        buf313 = reinterpret_tensor(buf83, (32, 256, 5120), (1310720, 5120, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [gelu_4, hidden_states_161], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_56.run(buf312, arg192_1, buf313, 41943040, stream=stream2)
        del arg192_1
        buf314 = reinterpret_tensor(buf311, (8192, 1280), (1280, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf313, (8192, 5120), (5120, 1), 0), reinterpret_tensor(arg193_1, (5120, 1280), (1, 5120), 0), out=buf314)
        del arg193_1
        buf315 = reinterpret_tensor(buf314, (32, 256, 1280), (327680, 1280, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_164], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_57.run(buf315, arg194_1, buf307, 10485760, stream=stream2)
        del arg194_1
        buf316 = reinterpret_tensor(buf307, (8192, 1280), (1280, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf315, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg195_1, (1280, 1280), (1, 1280), 0), out=buf316)
        del arg195_1
        buf317 = reinterpret_tensor(buf315, (32, 1280, 16, 16), (327680, 256, 16, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_139, hidden_states_141, add_25, output_tensor_4, hidden_states_166, output_4], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_58.run(buf316, arg196_1, buf269, arg170_1, buf273, arg168_1, buf317, 8192, 1280, stream=stream2)
        del arg168_1
        del arg170_1
        del arg196_1
        buf318 = buf275; del buf275  # reuse
        buf319 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_167], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_25.run(buf317, buf318, buf319, 1024, 10240, stream=stream2)
        buf322 = reinterpret_tensor(buf316, (32, 1280, 16, 16), (327680, 1, 20480, 1280), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_167, hidden_states_168], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_59.run(buf317, buf318, buf319, arg197_1, arg198_1, buf322, 40960, 256, stream=stream2)
        del arg197_1
        del arg198_1
        buf323 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_168, hidden_states_169], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg199_1, buf323, 1638400, 9, stream=stream2)
        del arg199_1
        # Topologically Sorted Source Nodes: [hidden_states_168, hidden_states_169], Original ATen: [aten.silu, aten.convolution]
        buf324 = extern_kernels.convolution(buf322, buf323, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf326 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_10, linear_67], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf325, reinterpret_tensor(arg201_1, (1280, 1280), (1, 1280), 0), out=buf326)
        del arg201_1
        buf327 = buf319; del buf319  # reuse
        buf328 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_171], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_48.run(buf324, arg200_1, buf326, arg202_1, buf327, buf328, 1024, 10240, stream=stream2)
        buf331 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_171, hidden_states_172], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_49.run(buf324, arg200_1, buf326, arg202_1, buf327, buf328, arg203_1, arg204_1, buf331, 10485760, stream=stream2)
        del arg200_1
        del arg202_1
        del arg203_1
        del arg204_1
        buf332 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_174], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg205_1, buf332, 1638400, 9, stream=stream2)
        del arg205_1
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_174], Original ATen: [aten.silu, aten.convolution]
        buf333 = extern_kernels.convolution(buf331, buf332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf334 = buf328; del buf328  # reuse
        buf335 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_175], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_60.run(buf317, buf333, arg206_1, buf334, buf335, 1024, 10240, stream=stream2)
        buf337 = reinterpret_tensor(buf331, (32, 256, 1280), (327680, 1280, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_61.run(buf317, buf333, arg206_1, buf334, buf335, arg207_1, arg208_1, buf337, 8192, 1280, stream=stream2)
        del arg207_1
        del arg208_1
        buf338 = reinterpret_tensor(buf324, (8192, 1280), (1280, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg209_1, (1280, 1280), (1, 1280), 0), out=buf338)
        del arg209_1
        buf342 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177, norm_hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_native_layer_norm_53.run(buf338, arg210_1, arg211_1, arg212_1, buf342, 8192, 1280, stream=stream2)
        del arg211_1
        del arg212_1
        buf343 = reinterpret_tensor(buf273, (8192, 1280), (1280, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [query_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg213_1, (1280, 1280), (1, 1280), 0), out=buf343)
        del arg213_1
        buf344 = reinterpret_tensor(buf269, (8192, 1280), (1280, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [key_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg214_1, (1280, 1280), (1, 1280), 0), out=buf344)
        del arg214_1
        buf345 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [value_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg215_1, (1280, 1280), (1, 1280), 0), out=buf345)
        del arg215_1
        del buf342
        # Topologically Sorted Source Nodes: [hidden_states_178], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf346 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf343, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf344, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf345, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), scale=0.125)
        del buf343
        buf347 = buf346[0]
        assert_size_stride(buf347, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf346
        buf352 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_181], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf347, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg216_1, (1280, 1280), (1, 1280), 0), out=buf352)
        del arg216_1
        buf356 = reinterpret_tensor(buf347, (32, 256, 1280), (327680, 1280, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177, hidden_states_183, hidden_states_184, norm_hidden_states_16], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_54.run(buf352, arg217_1, buf338, arg210_1, arg218_1, arg219_1, buf356, 8192, 1280, stream=stream2)
        del arg218_1
        del arg219_1
        buf357 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [query_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg220_1, (1280, 1280), (1, 1280), 0), out=buf357)
        del arg220_1
        del buf356
        buf358 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [key_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg221_1, (1024, 1280), (1, 1024), 0), out=buf358)
        del arg221_1
        buf359 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [value_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg222_1, (1024, 1280), (1, 1024), 0), out=buf359)
        del arg222_1
        # Topologically Sorted Source Nodes: [hidden_states_185], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf360 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf357, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf358, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), reinterpret_tensor(buf359, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), scale=0.125)
        buf361 = buf360[0]
        assert_size_stride(buf361, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf360
        buf366 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_188], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf361, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg223_1, (1280, 1280), (1, 1280), 0), out=buf366)
        del arg223_1
        buf367 = reinterpret_tensor(buf366, (32, 256, 1280), (327680, 1280, 1), 0); del buf366  # reuse
        buf371 = reinterpret_tensor(buf361, (32, 256, 1280), (327680, 1280, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177, hidden_states_183, hidden_states_184, hidden_states_190, hidden_states_191, norm_hidden_states_17], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_55.run(buf367, arg224_1, buf352, arg217_1, buf338, arg210_1, arg225_1, arg226_1, buf371, 8192, 1280, stream=stream2)
        del arg210_1
        del arg217_1
        del arg224_1
        del arg225_1
        del arg226_1
        del buf338
        del buf352
        buf372 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_192], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf371, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg227_1, (1280, 10240), (1, 1280), 0), out=buf372)
        del arg227_1
        buf373 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [gelu_5, hidden_states_194], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_56.run(buf372, arg228_1, buf373, 41943040, stream=stream2)
        del arg228_1
        buf374 = reinterpret_tensor(buf371, (8192, 1280), (1280, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_196], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf373, (8192, 5120), (5120, 1), 0), reinterpret_tensor(arg229_1, (5120, 1280), (1, 5120), 0), out=buf374)
        del arg229_1
        buf375 = reinterpret_tensor(buf374, (32, 256, 1280), (327680, 1280, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_197], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_57.run(buf375, arg230_1, buf367, 10485760, stream=stream2)
        del arg230_1
        buf376 = reinterpret_tensor(buf367, (8192, 1280), (1280, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_198], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf375, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg231_1, (1280, 1280), (1, 1280), 0), out=buf376)
        del arg231_1
        buf377 = reinterpret_tensor(buf376, (32, 1280, 16, 16), (327680, 1, 20480, 1280), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_174, add_31, output_tensor_5, hidden_states_199, output_5], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_62.run(buf377, arg232_1, buf317, buf333, arg206_1, 8192, 1280, stream=stream2)
        del arg206_1
        del arg232_1
        buf378 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_200], Original ATen: [aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg233_1, buf378, 1638400, 9, stream=stream2)
        del arg233_1
        # Topologically Sorted Source Nodes: [hidden_states_200], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf377, buf378, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf380 = buf335; del buf335  # reuse
        buf381 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_201], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_63.run(buf379, arg234_1, buf380, buf381, 1024, 2560, stream=stream2)
        buf384 = empty_strided_cuda((32, 1280, 8, 8), (81920, 1, 10240, 1280), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_201, hidden_states_202], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_64.run(buf379, arg234_1, buf380, buf381, arg235_1, arg236_1, buf384, 2621440, stream=stream2)
        del arg235_1
        del arg236_1
        buf385 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_202, hidden_states_203], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg237_1, buf385, 1638400, 9, stream=stream2)
        del arg237_1
        # Topologically Sorted Source Nodes: [hidden_states_202, hidden_states_203], Original ATen: [aten.silu, aten.convolution]
        buf386 = extern_kernels.convolution(buf384, buf385, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf387 = buf326; del buf326  # reuse
        buf403 = buf325; del buf325  # reuse
        buf420 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_12, temb_14, temb_16], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_65.run(buf14, arg5_1, buf387, buf403, buf420, 40960, stream=stream2)
        buf388 = empty_strided_cuda((32, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [sample_2, temb_12, linear_80], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf387, reinterpret_tensor(arg239_1, (1280, 1280), (1, 1280), 0), out=buf388)
        del arg239_1
        buf389 = buf381; del buf381  # reuse
        buf390 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_205], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf386, arg238_1, buf388, arg240_1, buf389, buf390, 1024, 2560, stream=stream2)
        buf393 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_205, hidden_states_206], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf386, arg238_1, buf388, arg240_1, buf389, buf390, arg241_1, arg242_1, buf393, 2621440, stream=stream2)
        del arg238_1
        del arg240_1
        del arg241_1
        del arg242_1
        buf394 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_208], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg243_1, buf394, 1638400, 9, stream=stream2)
        del arg243_1
        # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_208], Original ATen: [aten.silu, aten.convolution]
        buf395 = extern_kernels.convolution(buf393, buf394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf396 = buf390; del buf390  # reuse
        buf397 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_209], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_68.run(buf379, arg234_1, buf395, arg244_1, buf396, buf397, 1024, 2560, stream=stream2)
        buf400 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_209, hidden_states_210], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_69.run(buf379, arg234_1, buf395, arg244_1, buf396, buf397, arg245_1, arg246_1, buf400, 2621440, stream=stream2)
        del arg245_1
        del arg246_1
        buf401 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg247_1, buf401, 1638400, 9, stream=stream2)
        del arg247_1
        # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211], Original ATen: [aten.silu, aten.convolution]
        buf402 = extern_kernels.convolution(buf400, buf401, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf404 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_14, linear_81], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf403, reinterpret_tensor(arg249_1, (1280, 1280), (1, 1280), 0), out=buf404)
        del arg249_1
        buf405 = buf397; del buf397  # reuse
        buf406 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_213], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf402, arg248_1, buf404, arg250_1, buf405, buf406, 1024, 2560, stream=stream2)
        buf409 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_213, hidden_states_214], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf402, arg248_1, buf404, arg250_1, buf405, buf406, arg251_1, arg252_1, buf409, 2621440, stream=stream2)
        del arg248_1
        del arg250_1
        del arg251_1
        del arg252_1
        buf410 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_214, hidden_states_216], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg253_1, buf410, 1638400, 9, stream=stream2)
        del arg253_1
        # Topologically Sorted Source Nodes: [hidden_states_214, hidden_states_216], Original ATen: [aten.silu, aten.convolution]
        buf411 = extern_kernels.convolution(buf409, buf410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf412 = reinterpret_tensor(buf409, (32, 1280, 8, 8), (81920, 64, 8, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_206, hidden_states_208, add_37, output_tensor_6, hidden_states_214, hidden_states_216, add_39, output_tensor_7], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_convolution_div_silu_70.run(buf379, arg234_1, buf395, arg244_1, buf411, arg254_1, buf412, 2048, 1280, stream=stream2)
        del arg254_1
        buf413 = buf406; del buf406  # reuse
        buf414 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_217], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_71.run(buf412, buf413, buf414, 1024, 2560, stream=stream2)
        buf417 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_217, hidden_states_218], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_72.run(buf412, buf413, buf414, arg255_1, arg256_1, buf417, 40960, 64, stream=stream2)
        del arg255_1
        del arg256_1
        buf418 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_218, hidden_states_219], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg257_1, buf418, 1638400, 9, stream=stream2)
        del arg257_1
        # Topologically Sorted Source Nodes: [hidden_states_218, hidden_states_219], Original ATen: [aten.silu, aten.convolution]
        buf419 = extern_kernels.convolution(buf417, buf418, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf421 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_16, linear_82], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf420, reinterpret_tensor(arg259_1, (1280, 1280), (1, 1280), 0), out=buf421)
        del arg259_1
        buf422 = buf414; del buf414  # reuse
        buf423 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_221], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf419, arg258_1, buf421, arg260_1, buf422, buf423, 1024, 2560, stream=stream2)
        buf426 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_221, hidden_states_222], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf419, arg258_1, buf421, arg260_1, buf422, buf423, arg261_1, arg262_1, buf426, 2621440, stream=stream2)
        del arg258_1
        del arg260_1
        del arg261_1
        del arg262_1
        buf427 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_222, hidden_states_224], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg263_1, buf427, 1638400, 9, stream=stream2)
        del arg263_1
        # Topologically Sorted Source Nodes: [hidden_states_222, hidden_states_224], Original ATen: [aten.silu, aten.convolution]
        buf428 = extern_kernels.convolution(buf426, buf427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf429 = buf423; del buf423  # reuse
        buf430 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_225], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_73.run(buf412, buf428, arg264_1, buf429, buf430, 1024, 2560, stream=stream2)
        buf432 = reinterpret_tensor(buf426, (32, 64, 1280), (81920, 1280, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_227], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_74.run(buf412, buf428, arg264_1, buf429, buf430, arg265_1, arg266_1, buf432, 2048, 1280, stream=stream2)
        del arg265_1
        del arg266_1
        buf433 = reinterpret_tensor(buf419, (2048, 1280), (1280, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_227], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg267_1, (1280, 1280), (1, 1280), 0), out=buf433)
        del arg267_1
        buf437 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_227, norm_hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_native_layer_norm_75.run(buf433, arg268_1, arg269_1, arg270_1, buf437, 2048, 1280, stream=stream2)
        del arg269_1
        del arg270_1
        buf438 = reinterpret_tensor(buf402, (2048, 1280), (1280, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [query_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg271_1, (1280, 1280), (1, 1280), 0), out=buf438)
        del arg271_1
        buf439 = reinterpret_tensor(buf386, (2048, 1280), (1280, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [key_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg272_1, (1280, 1280), (1, 1280), 0), out=buf439)
        del arg272_1
        buf440 = empty_strided_cuda((2048, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg273_1, (1280, 1280), (1, 1280), 0), out=buf440)
        del arg273_1
        del buf437
        # Topologically Sorted Source Nodes: [hidden_states_228], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf441 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf438, (32, 20, 64, 64), (81920, 64, 1280, 1), 0), reinterpret_tensor(buf439, (32, 20, 64, 64), (81920, 64, 1280, 1), 0), reinterpret_tensor(buf440, (32, 20, 64, 64), (81920, 64, 1280, 1), 0), scale=0.125)
        del buf438
        buf442 = buf441[0]
        assert_size_stride(buf442, (32, 20, 64, 64), (81920, 64, 1280, 1))
        del buf441
        buf447 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_231], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf442, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg274_1, (1280, 1280), (1, 1280), 0), out=buf447)
        del arg274_1
        buf451 = reinterpret_tensor(buf442, (32, 64, 1280), (81920, 1280, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_227, hidden_states_233, hidden_states_234, norm_hidden_states_19], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_76.run(buf447, arg275_1, buf433, arg268_1, arg276_1, arg277_1, buf451, 2048, 1280, stream=stream2)
        del arg276_1
        del arg277_1
        buf452 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [query_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg278_1, (1280, 1280), (1, 1280), 0), out=buf452)
        del arg278_1
        del buf451
        buf453 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [key_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1280), (1, 1024), 0), out=buf453)
        del arg279_1
        buf454 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [value_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg280_1, (1024, 1280), (1, 1024), 0), out=buf454)
        del arg280_1
        # Topologically Sorted Source Nodes: [hidden_states_235], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf455 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf452, (32, 20, 64, 64), (81920, 64, 1280, 1), 0), reinterpret_tensor(buf453, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), reinterpret_tensor(buf454, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), scale=0.125)
        buf456 = buf455[0]
        assert_size_stride(buf456, (32, 20, 64, 64), (81920, 64, 1280, 1))
        del buf455
        buf461 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_238], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf456, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg281_1, (1280, 1280), (1, 1280), 0), out=buf461)
        del arg281_1
        buf462 = reinterpret_tensor(buf461, (32, 64, 1280), (81920, 1280, 1), 0); del buf461  # reuse
        buf466 = reinterpret_tensor(buf456, (32, 64, 1280), (81920, 1280, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_227, hidden_states_233, hidden_states_234, hidden_states_240, hidden_states_241, norm_hidden_states_20], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_77.run(buf462, arg282_1, buf447, arg275_1, buf433, arg268_1, arg283_1, arg284_1, buf466, 2048, 1280, stream=stream2)
        del arg268_1
        del arg275_1
        del arg282_1
        del arg283_1
        del arg284_1
        del buf433
        del buf447
        buf467 = reinterpret_tensor(buf208, (2048, 10240), (10240, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_242], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf466, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg285_1, (1280, 10240), (1, 1280), 0), out=buf467)
        del arg285_1
        buf468 = reinterpret_tensor(buf333, (32, 64, 5120), (327680, 5120, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [gelu_6, hidden_states_244], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_78.run(buf467, arg286_1, buf468, 10485760, stream=stream2)
        del arg286_1
        buf469 = reinterpret_tensor(buf466, (2048, 1280), (1280, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_246], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf468, (2048, 5120), (5120, 1), 0), reinterpret_tensor(arg287_1, (5120, 1280), (1, 5120), 0), out=buf469)
        del arg287_1
        buf470 = reinterpret_tensor(buf469, (32, 64, 1280), (81920, 1280, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_247], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_79.run(buf470, arg288_1, buf462, 2621440, stream=stream2)
        del arg288_1
        buf471 = reinterpret_tensor(buf462, (2048, 1280), (1280, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_248], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf470, (2048, 1280), (1280, 1), 0), reinterpret_tensor(arg289_1, (1280, 1280), (1, 1280), 0), out=buf471)
        del arg289_1
        buf472 = reinterpret_tensor(buf470, (32, 1280, 8, 8), (81920, 64, 8, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_222, hidden_states_224, add_41, output_tensor_8, hidden_states_249, output_6], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_clone_convolution_div_silu_80.run(buf471, arg290_1, buf412, buf428, arg264_1, buf472, 40960, 64, stream=stream2)
        del arg264_1
        del arg290_1
        del buf428
        buf473 = buf430; del buf430  # reuse
        buf474 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_250], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_71.run(buf472, buf473, buf474, 1024, 2560, stream=stream2)
        buf477 = reinterpret_tensor(buf471, (32, 1280, 8, 8), (81920, 1, 10240, 1280), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_72.run(buf472, buf473, buf474, arg291_1, arg292_1, buf477, 40960, 64, stream=stream2)
        del arg291_1
        del arg292_1
        buf478 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_251, hidden_states_252], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg293_1, buf478, 1638400, 9, stream=stream2)
        del arg293_1
        # Topologically Sorted Source Nodes: [hidden_states_251, hidden_states_252], Original ATen: [aten.silu, aten.convolution]
        buf479 = extern_kernels.convolution(buf477, buf478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf480 = buf421; del buf421  # reuse
        buf497 = buf420; del buf420  # reuse
        buf516 = buf403; del buf403  # reuse
        buf535 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_18, temb_20, temb_22, temb_24], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_81.run(buf14, arg5_1, buf480, buf497, buf516, buf535, 40960, stream=stream2)
        buf481 = empty_strided_cuda((32, 1280), (1280, 1), torch.float16)
        # Topologically Sorted Source Nodes: [sample_2, temb_18, linear_95], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf480, reinterpret_tensor(arg295_1, (1280, 1280), (1, 1280), 0), out=buf481)
        del arg295_1
        del buf480
        buf482 = buf474; del buf474  # reuse
        buf483 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_254], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf479, arg294_1, buf481, arg296_1, buf482, buf483, 1024, 2560, stream=stream2)
        buf486 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_254, hidden_states_255], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf479, arg294_1, buf481, arg296_1, buf482, buf483, arg297_1, arg298_1, buf486, 2621440, stream=stream2)
        del arg294_1
        del arg296_1
        del arg297_1
        del arg298_1
        del buf479
        buf487 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_255, hidden_states_257], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg299_1, buf487, 1638400, 9, stream=stream2)
        del arg299_1
        # Topologically Sorted Source Nodes: [hidden_states_255, hidden_states_257], Original ATen: [aten.silu, aten.convolution]
        buf488 = extern_kernels.convolution(buf486, buf487, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del buf486
        buf489 = reinterpret_tensor(buf268, (32, 2560, 8, 8), (163840, 64, 8, 1), 0); del buf268  # reuse
        buf490 = buf483; del buf483  # reuse
        buf491 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_258, hidden_states_259], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_82.run(buf472, buf488, arg300_1, buf412, buf489, buf490, buf491, 1024, 5120, stream=stream2)
        del arg300_1
        del buf412
        del buf472
        buf494 = reinterpret_tensor(buf260, (32, 2560, 8, 8), (163840, 1, 20480, 2560), 0); del buf260  # reuse
        buf502 = empty_strided_cuda((32, 2560, 8, 8), (163840, 1, 20480, 2560), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_259, hidden_states_260, input_tensor_2], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_83.run(buf489, buf490, buf491, arg301_1, arg302_1, buf494, buf502, 81920, 64, stream=stream2)
        del arg301_1
        del arg302_1
        buf495 = empty_strided_cuda((1280, 2560, 3, 3), (23040, 1, 7680, 2560), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_84.run(arg303_1, buf495, 3276800, 9, stream=stream2)
        del arg303_1
        # Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261], Original ATen: [aten.silu, aten.convolution]
        buf496 = extern_kernels.convolution(buf494, buf495, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf498 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_20, linear_96], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf497, reinterpret_tensor(arg305_1, (1280, 1280), (1, 1280), 0), out=buf498)
        del arg305_1
        del buf497
        buf499 = buf491; del buf491  # reuse
        buf500 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_263], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf496, arg304_1, buf498, arg306_1, buf499, buf500, 1024, 2560, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_2], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, arg311_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del arg311_1
        buf505 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_263, hidden_states_264], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf496, arg304_1, buf498, arg306_1, buf499, buf500, arg307_1, arg308_1, buf505, 2621440, stream=stream2)
        del arg304_1
        del arg306_1
        del arg307_1
        del arg308_1
        del buf496
        buf506 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_264, hidden_states_266], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg309_1, buf506, 1638400, 9, stream=stream2)
        del arg309_1
        # Topologically Sorted Source Nodes: [hidden_states_264, hidden_states_266], Original ATen: [aten.silu, aten.convolution]
        buf507 = extern_kernels.convolution(buf505, buf506, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del buf505
        buf508 = reinterpret_tensor(buf502, (32, 2560, 8, 8), (163840, 64, 8, 1), 0); del buf502  # reuse
        buf509 = buf500; del buf500  # reuse
        buf510 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_267, hidden_states_268], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_85.run(buf503, arg312_1, buf507, arg310_1, buf379, arg234_1, buf395, arg244_1, buf508, buf509, buf510, 1024, 5120, stream=stream2)
        del arg244_1
        del arg310_1
        del arg312_1
        del buf395
        del buf503
        buf513 = buf494; del buf494  # reuse
        buf521 = reinterpret_tensor(buf489, (32, 2560, 8, 8), (163840, 1, 20480, 2560), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_268, hidden_states_269, input_tensor_3], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_83.run(buf508, buf509, buf510, arg313_1, arg314_1, buf513, buf521, 81920, 64, stream=stream2)
        del arg313_1
        del arg314_1
        buf514 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_269, hidden_states_270], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_84.run(arg315_1, buf514, 3276800, 9, stream=stream2)
        del arg315_1
        # Topologically Sorted Source Nodes: [hidden_states_269, hidden_states_270], Original ATen: [aten.silu, aten.convolution]
        buf515 = extern_kernels.convolution(buf513, buf514, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        buf517 = buf498; del buf498  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_22, linear_97], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf516, reinterpret_tensor(arg317_1, (1280, 1280), (1, 1280), 0), out=buf517)
        del arg317_1
        buf518 = buf510; del buf510  # reuse
        buf519 = buf509; del buf509  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_272], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf515, arg316_1, buf517, arg318_1, buf518, buf519, 1024, 2560, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_3], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del arg323_1
        buf524 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_272, hidden_states_273], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf515, arg316_1, buf517, arg318_1, buf518, buf519, arg319_1, arg320_1, buf524, 2621440, stream=stream2)
        del arg316_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf515
        buf525 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_273, hidden_states_275], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg321_1, buf525, 1638400, 9, stream=stream2)
        del arg321_1
        # Topologically Sorted Source Nodes: [hidden_states_273, hidden_states_275], Original ATen: [aten.silu, aten.convolution]
        buf526 = extern_kernels.convolution(buf524, buf525, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del buf524
        buf527 = reinterpret_tensor(buf521, (32, 2560, 8, 8), (163840, 64, 8, 1), 0); del buf521  # reuse
        buf528 = buf519; del buf519  # reuse
        buf529 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_276, hidden_states_277], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_86.run(buf522, arg324_1, buf526, arg322_1, buf379, arg234_1, buf527, buf528, buf529, 1024, 5120, stream=stream2)
        del arg234_1
        del arg322_1
        del arg324_1
        del buf379
        del buf522
        buf532 = buf513; del buf513  # reuse
        buf540 = reinterpret_tensor(buf508, (32, 2560, 8, 8), (163840, 1, 20480, 2560), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_277, hidden_states_278, input_tensor_4], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_83.run(buf527, buf528, buf529, arg325_1, arg326_1, buf532, buf540, 81920, 64, stream=stream2)
        del arg325_1
        del arg326_1
        del buf527
        buf533 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_278, hidden_states_279], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_84.run(arg327_1, buf533, 3276800, 9, stream=stream2)
        del arg327_1
        # Topologically Sorted Source Nodes: [hidden_states_278, hidden_states_279], Original ATen: [aten.silu, aten.convolution]
        buf534 = extern_kernels.convolution(buf532, buf533, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del buf532
        buf536 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_24, linear_98], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf535, reinterpret_tensor(arg329_1, (1280, 1280), (1, 1280), 0), out=buf536)
        del arg329_1
        buf537 = buf529; del buf529  # reuse
        buf538 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_281], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_66.run(buf534, arg328_1, buf536, arg330_1, buf537, buf538, 1024, 2560, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_4], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, arg335_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del arg335_1
        del buf540
        buf543 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_281, hidden_states_282], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_67.run(buf534, arg328_1, buf536, arg330_1, buf537, buf538, arg331_1, arg332_1, buf543, 2621440, stream=stream2)
        del arg328_1
        del arg330_1
        del arg331_1
        del arg332_1
        del buf534
        buf544 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_282, hidden_states_284], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg333_1, buf544, 1638400, 9, stream=stream2)
        del arg333_1
        # Topologically Sorted Source Nodes: [hidden_states_282, hidden_states_284], Original ATen: [aten.silu, aten.convolution]
        buf545 = extern_kernels.convolution(buf543, buf544, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (32, 1280, 8, 8), (81920, 1, 10240, 1280))
        del buf543
        buf546 = reinterpret_tensor(buf468, (32, 1280, 16, 16), (327680, 1, 20480, 1280), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_4, hidden_states_282, hidden_states_284, add_53, output_tensor_12, hidden_states_285], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        stream2 = get_raw_stream(2)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_div_silu_87.run(buf541, arg336_1, buf545, arg334_1, buf546, 10485760, stream=stream2)
        del arg334_1
        del arg336_1
        del buf541
        del buf545
        buf547 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_4, hidden_states_282, hidden_states_284, add_53, output_tensor_12, hidden_states_285, hidden_states_286], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg337_1, buf547, 1638400, 9, stream=stream2)
        del arg337_1
        # Topologically Sorted Source Nodes: [input_tensor_4, hidden_states_282, hidden_states_284, add_53, output_tensor_12, hidden_states_285, hidden_states_286], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten._to_copy, aten._unsafe_index]
        buf548 = extern_kernels.convolution(buf546, buf547, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf549 = reinterpret_tensor(buf467, (32, 2560, 16, 16), (655360, 256, 16, 1), 0); del buf467  # reuse
        buf550 = buf538; del buf538  # reuse
        buf551 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_287, hidden_states_288], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_88.run(buf548, arg338_1, buf377, buf549, buf550, buf551, 1024, 20480, stream=stream2)
        del arg338_1
        buf554 = reinterpret_tensor(buf250, (32, 2560, 16, 16), (655360, 1, 40960, 2560), 0); del buf250  # reuse
        buf562 = reinterpret_tensor(buf227, (32, 2560, 16, 16), (655360, 1, 40960, 2560), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_288, hidden_states_289, input_tensor_5], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_89.run(buf549, buf550, buf551, arg339_1, arg340_1, buf554, buf562, 81920, 256, stream=stream2)
        del arg339_1
        del arg340_1
        buf555 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_289, hidden_states_290], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_84.run(arg341_1, buf555, 3276800, 9, stream=stream2)
        del arg341_1
        # Topologically Sorted Source Nodes: [hidden_states_289, hidden_states_290], Original ATen: [aten.silu, aten.convolution]
        buf556 = extern_kernels.convolution(buf554, buf555, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf557 = buf536; del buf536  # reuse
        buf619 = buf535; del buf535  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_26, temb_28], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_8.run(buf14, arg5_1, buf557, buf619, 40960, stream=stream2)
        buf558 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_26, linear_99], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf557, reinterpret_tensor(arg343_1, (1280, 1280), (1, 1280), 0), out=buf558)
        del arg343_1
        del buf557
        buf559 = buf551; del buf551  # reuse
        buf560 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_292], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_48.run(buf556, arg342_1, buf558, arg344_1, buf559, buf560, 1024, 10240, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_5], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, arg349_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        del arg349_1
        buf565 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_292, hidden_states_293], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_49.run(buf556, arg342_1, buf558, arg344_1, buf559, buf560, arg345_1, arg346_1, buf565, 10485760, stream=stream2)
        del arg342_1
        del arg344_1
        del arg345_1
        del arg346_1
        buf566 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_293, hidden_states_295], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg347_1, buf566, 1638400, 9, stream=stream2)
        del arg347_1
        # Topologically Sorted Source Nodes: [hidden_states_293, hidden_states_295], Original ATen: [aten.silu, aten.convolution]
        buf567 = extern_kernels.convolution(buf565, buf566, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf568 = buf560; del buf560  # reuse
        buf569 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_296], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_51.run(buf563, arg350_1, buf567, arg348_1, buf568, buf569, 1024, 10240, stream=stream2)
        buf571 = reinterpret_tensor(buf565, (32, 256, 1280), (327680, 1280, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_298], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_52.run(buf563, arg350_1, buf567, arg348_1, buf568, buf569, arg351_1, arg352_1, buf571, 10485760, stream=stream2)
        del arg351_1
        del arg352_1
        buf572 = reinterpret_tensor(buf556, (8192, 1280), (1280, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_298], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf571, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg353_1, (1280, 1280), (1, 1280), 0), out=buf572)
        del arg353_1
        buf576 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_298, norm_hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_native_layer_norm_53.run(buf572, arg354_1, arg355_1, arg356_1, buf576, 8192, 1280, stream=stream2)
        del arg355_1
        del arg356_1
        buf577 = reinterpret_tensor(buf377, (8192, 1280), (1280, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [query_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg357_1, (1280, 1280), (1, 1280), 0), out=buf577)
        del arg357_1
        buf578 = reinterpret_tensor(buf546, (8192, 1280), (1280, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [key_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg358_1, (1280, 1280), (1, 1280), 0), out=buf578)
        del arg358_1
        buf579 = reinterpret_tensor(buf375, (8192, 1280), (1280, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [value_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg359_1, (1280, 1280), (1, 1280), 0), out=buf579)
        del arg359_1
        del buf576
        # Topologically Sorted Source Nodes: [hidden_states_299], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf580 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf577, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf578, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf579, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), scale=0.125)
        del buf577
        buf581 = buf580[0]
        assert_size_stride(buf581, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf580
        buf586 = buf579; del buf579  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_302], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf581, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg360_1, (1280, 1280), (1, 1280), 0), out=buf586)
        del arg360_1
        buf590 = reinterpret_tensor(buf581, (32, 256, 1280), (327680, 1280, 1), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_298, hidden_states_304, hidden_states_305, norm_hidden_states_22], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_54.run(buf586, arg361_1, buf572, arg354_1, arg362_1, arg363_1, buf590, 8192, 1280, stream=stream2)
        del arg362_1
        del arg363_1
        buf591 = buf578; del buf578  # reuse
        # Topologically Sorted Source Nodes: [query_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf590, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg364_1, (1280, 1280), (1, 1280), 0), out=buf591)
        del arg364_1
        del buf590
        buf592 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [key_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg365_1, (1024, 1280), (1, 1024), 0), out=buf592)
        del arg365_1
        buf593 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [value_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg366_1, (1024, 1280), (1, 1024), 0), out=buf593)
        del arg366_1
        # Topologically Sorted Source Nodes: [hidden_states_306], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf594 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf591, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf592, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), reinterpret_tensor(buf593, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), scale=0.125)
        buf595 = buf594[0]
        assert_size_stride(buf595, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf594
        buf600 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_309], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf595, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg367_1, (1280, 1280), (1, 1280), 0), out=buf600)
        del arg367_1
        buf601 = reinterpret_tensor(buf600, (32, 256, 1280), (327680, 1280, 1), 0); del buf600  # reuse
        buf605 = reinterpret_tensor(buf595, (32, 256, 1280), (327680, 1280, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_298, hidden_states_304, hidden_states_305, hidden_states_311, hidden_states_312, norm_hidden_states_23], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_55.run(buf601, arg368_1, buf586, arg361_1, buf572, arg354_1, arg369_1, arg370_1, buf605, 8192, 1280, stream=stream2)
        del arg354_1
        del arg361_1
        del arg368_1
        del arg369_1
        del arg370_1
        del buf572
        del buf586
        buf606 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_313], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf605, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg371_1, (1280, 10240), (1, 1280), 0), out=buf606)
        del arg371_1
        buf607 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [gelu_7, hidden_states_315], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_56.run(buf606, arg372_1, buf607, 41943040, stream=stream2)
        del arg372_1
        buf608 = reinterpret_tensor(buf605, (8192, 1280), (1280, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_317], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf607, (8192, 5120), (5120, 1), 0), reinterpret_tensor(arg373_1, (5120, 1280), (1, 5120), 0), out=buf608)
        del arg373_1
        buf609 = reinterpret_tensor(buf608, (32, 256, 1280), (327680, 1280, 1), 0); del buf608  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_318], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_57.run(buf609, arg374_1, buf601, 10485760, stream=stream2)
        del arg374_1
        buf610 = reinterpret_tensor(buf601, (8192, 1280), (1280, 1), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_319], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf609, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg375_1, (1280, 1280), (1, 1280), 0), out=buf610)
        del arg375_1
        del buf609
        buf611 = reinterpret_tensor(buf562, (32, 2560, 16, 16), (655360, 256, 16, 1), 0); del buf562  # reuse
        buf612 = buf569; del buf569  # reuse
        buf613 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_321, hidden_states_322], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_90.run(buf610, arg376_1, buf563, arg350_1, buf567, arg348_1, buf317, buf611, buf612, buf613, 1024, 20480, stream=stream2)
        del arg348_1
        del arg350_1
        del arg376_1
        buf616 = buf554; del buf554  # reuse
        buf624 = reinterpret_tensor(buf549, (32, 2560, 16, 16), (655360, 1, 40960, 2560), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_322, hidden_states_323, input_tensor_6], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_89.run(buf611, buf612, buf613, arg377_1, arg378_1, buf616, buf624, 81920, 256, stream=stream2)
        del arg377_1
        del arg378_1
        buf617 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_323, hidden_states_324], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_84.run(arg379_1, buf617, 3276800, 9, stream=stream2)
        del arg379_1
        # Topologically Sorted Source Nodes: [hidden_states_323, hidden_states_324], Original ATen: [aten.silu, aten.convolution]
        buf618 = extern_kernels.convolution(buf616, buf617, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        del buf617
        buf620 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_28, linear_112], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf619, reinterpret_tensor(arg381_1, (1280, 1280), (1, 1280), 0), out=buf620)
        del arg381_1
        buf621 = buf613; del buf613  # reuse
        buf622 = buf612; del buf612  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_326], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_48.run(buf618, arg380_1, buf620, arg382_1, buf621, buf622, 1024, 10240, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_6], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf624, arg387_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf625, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        del arg387_1
        buf627 = reinterpret_tensor(buf610, (32, 1280, 16, 16), (327680, 1, 20480, 1280), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_326, hidden_states_327], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_49.run(buf618, arg380_1, buf620, arg382_1, buf621, buf622, arg383_1, arg384_1, buf627, 10485760, stream=stream2)
        del arg380_1
        del arg382_1
        del arg383_1
        del arg384_1
        buf628 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_327, hidden_states_329], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg385_1, buf628, 1638400, 9, stream=stream2)
        del arg385_1
        # Topologically Sorted Source Nodes: [hidden_states_327, hidden_states_329], Original ATen: [aten.silu, aten.convolution]
        buf629 = extern_kernels.convolution(buf627, buf628, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf629, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf630 = buf622; del buf622  # reuse
        buf631 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_330], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_51.run(buf625, arg388_1, buf629, arg386_1, buf630, buf631, 1024, 10240, stream=stream2)
        buf633 = reinterpret_tensor(buf627, (32, 256, 1280), (327680, 1280, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_332], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_52.run(buf625, arg388_1, buf629, arg386_1, buf630, buf631, arg389_1, arg390_1, buf633, 10485760, stream=stream2)
        del arg389_1
        del arg390_1
        buf634 = reinterpret_tensor(buf618, (8192, 1280), (1280, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_332], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf633, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg391_1, (1280, 1280), (1, 1280), 0), out=buf634)
        del arg391_1
        buf638 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_332, norm_hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_native_layer_norm_53.run(buf634, arg392_1, arg393_1, arg394_1, buf638, 8192, 1280, stream=stream2)
        del arg393_1
        del arg394_1
        buf639 = reinterpret_tensor(buf567, (8192, 1280), (1280, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [query_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg395_1, (1280, 1280), (1, 1280), 0), out=buf639)
        del arg395_1
        buf640 = reinterpret_tensor(buf563, (8192, 1280), (1280, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [key_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg396_1, (1280, 1280), (1, 1280), 0), out=buf640)
        del arg396_1
        buf641 = reinterpret_tensor(buf317, (8192, 1280), (1280, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [value_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg397_1, (1280, 1280), (1, 1280), 0), out=buf641)
        del arg397_1
        del buf638
        # Topologically Sorted Source Nodes: [hidden_states_333], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf642 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf639, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf640, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf641, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), scale=0.125)
        del buf639
        buf643 = buf642[0]
        assert_size_stride(buf643, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf642
        buf648 = buf641; del buf641  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_336], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf643, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg398_1, (1280, 1280), (1, 1280), 0), out=buf648)
        del arg398_1
        buf652 = reinterpret_tensor(buf643, (32, 256, 1280), (327680, 1280, 1), 0); del buf643  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_332, hidden_states_338, hidden_states_339, norm_hidden_states_25], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_54.run(buf648, arg399_1, buf634, arg392_1, arg400_1, arg401_1, buf652, 8192, 1280, stream=stream2)
        del arg400_1
        del arg401_1
        buf653 = buf640; del buf640  # reuse
        # Topologically Sorted Source Nodes: [query_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg402_1, (1280, 1280), (1, 1280), 0), out=buf653)
        del arg402_1
        del buf652
        buf654 = buf593; del buf593  # reuse
        # Topologically Sorted Source Nodes: [key_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg403_1, (1024, 1280), (1, 1024), 0), out=buf654)
        del arg403_1
        buf655 = buf592; del buf592  # reuse
        # Topologically Sorted Source Nodes: [value_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg404_1, (1024, 1280), (1, 1024), 0), out=buf655)
        del arg404_1
        # Topologically Sorted Source Nodes: [hidden_states_340], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf656 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf653, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf654, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), reinterpret_tensor(buf655, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), scale=0.125)
        buf657 = buf656[0]
        assert_size_stride(buf657, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf656
        buf662 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_343], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf657, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg405_1, (1280, 1280), (1, 1280), 0), out=buf662)
        del arg405_1
        buf663 = reinterpret_tensor(buf662, (32, 256, 1280), (327680, 1280, 1), 0); del buf662  # reuse
        buf667 = reinterpret_tensor(buf657, (32, 256, 1280), (327680, 1280, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_332, hidden_states_338, hidden_states_339, hidden_states_345, hidden_states_346, norm_hidden_states_26], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_55.run(buf663, arg406_1, buf648, arg399_1, buf634, arg392_1, arg407_1, arg408_1, buf667, 8192, 1280, stream=stream2)
        del arg392_1
        del arg399_1
        del arg406_1
        del arg407_1
        del arg408_1
        del buf634
        del buf648
        buf668 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_347], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf667, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg409_1, (1280, 10240), (1, 1280), 0), out=buf668)
        del arg409_1
        buf669 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [gelu_8, hidden_states_349], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_56.run(buf668, arg410_1, buf669, 41943040, stream=stream2)
        del arg410_1
        buf670 = reinterpret_tensor(buf667, (8192, 1280), (1280, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_351], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf669, (8192, 5120), (5120, 1), 0), reinterpret_tensor(arg411_1, (5120, 1280), (1, 5120), 0), out=buf670)
        del arg411_1
        buf671 = reinterpret_tensor(buf670, (32, 256, 1280), (327680, 1280, 1), 0); del buf670  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_352], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_57.run(buf671, arg412_1, buf663, 10485760, stream=stream2)
        del arg412_1
        buf672 = reinterpret_tensor(buf663, (8192, 1280), (1280, 1), 0); del buf663  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_353], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf671, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg413_1, (1280, 1280), (1, 1280), 0), out=buf672)
        del arg413_1
        buf673 = empty_strided_cuda((32, 1920, 16, 16), (491520, 256, 16, 1), torch.float16)
        buf674 = buf631; del buf631  # reuse
        buf675 = buf630; del buf630  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_355, hidden_states_356], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_91.run(buf672, arg414_1, buf625, arg388_1, buf629, arg386_1, buf255, buf673, buf674, buf675, 1024, 15360, stream=stream2)
        del arg386_1
        del arg388_1
        del arg414_1
        del buf255
        buf678 = empty_strided_cuda((32, 1920, 16, 16), (491520, 1, 30720, 1920), torch.float16)
        buf686 = empty_strided_cuda((32, 1920, 16, 16), (491520, 1, 30720, 1920), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_356, hidden_states_357, input_tensor_7], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_92.run(buf673, buf674, buf675, arg415_1, arg416_1, buf678, buf686, 61440, 256, stream=stream2)
        del arg415_1
        del arg416_1
        del buf673
        buf679 = empty_strided_cuda((1280, 1920, 3, 3), (17280, 1, 5760, 1920), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_357, hidden_states_358], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_93.run(arg417_1, buf679, 2457600, 9, stream=stream2)
        del arg417_1
        # Topologically Sorted Source Nodes: [hidden_states_357, hidden_states_358], Original ATen: [aten.silu, aten.convolution]
        buf680 = extern_kernels.convolution(buf678, buf679, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf680, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        del buf678
        del buf679
        buf681 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_30], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_94.run(buf14, arg5_1, buf681, 40960, stream=stream2)
        buf682 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_30, linear_125], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf681, reinterpret_tensor(arg419_1, (1280, 1280), (1, 1280), 0), out=buf682)
        del arg419_1
        buf683 = buf675; del buf675  # reuse
        buf684 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_360], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_48.run(buf680, arg418_1, buf682, arg420_1, buf683, buf684, 1024, 10240, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_7], Original ATen: [aten.convolution]
        buf687 = extern_kernels.convolution(buf686, arg425_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf687, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        del arg425_1
        del buf686
        buf689 = reinterpret_tensor(buf672, (32, 1280, 16, 16), (327680, 1, 20480, 1280), 0); del buf672  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_360, hidden_states_361], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_49.run(buf680, arg418_1, buf682, arg420_1, buf683, buf684, arg421_1, arg422_1, buf689, 10485760, stream=stream2)
        del arg418_1
        del arg420_1
        del arg421_1
        del arg422_1
        buf690 = buf628; del buf628  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_361, hidden_states_363], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg423_1, buf690, 1638400, 9, stream=stream2)
        del arg423_1
        # Topologically Sorted Source Nodes: [hidden_states_361, hidden_states_363], Original ATen: [aten.silu, aten.convolution]
        buf691 = extern_kernels.convolution(buf689, buf690, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (32, 1280, 16, 16), (327680, 1, 20480, 1280))
        buf692 = buf684; del buf684  # reuse
        buf693 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_364], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_51.run(buf687, arg426_1, buf691, arg424_1, buf692, buf693, 1024, 10240, stream=stream2)
        buf695 = reinterpret_tensor(buf689, (32, 256, 1280), (327680, 1280, 1), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_366], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_52.run(buf687, arg426_1, buf691, arg424_1, buf692, buf693, arg427_1, arg428_1, buf695, 10485760, stream=stream2)
        del arg427_1
        del arg428_1
        buf696 = reinterpret_tensor(buf680, (8192, 1280), (1280, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_366], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg429_1, (1280, 1280), (1, 1280), 0), out=buf696)
        del arg429_1
        buf700 = buf695; del buf695  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_366, norm_hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_native_layer_norm_53.run(buf696, arg430_1, arg431_1, arg432_1, buf700, 8192, 1280, stream=stream2)
        del arg431_1
        del arg432_1
        buf701 = reinterpret_tensor(buf629, (8192, 1280), (1280, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [query_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg433_1, (1280, 1280), (1, 1280), 0), out=buf701)
        del arg433_1
        buf702 = reinterpret_tensor(buf625, (8192, 1280), (1280, 1), 0); del buf625  # reuse
        # Topologically Sorted Source Nodes: [key_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg434_1, (1280, 1280), (1, 1280), 0), out=buf702)
        del arg434_1
        buf703 = reinterpret_tensor(buf671, (8192, 1280), (1280, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [value_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg435_1, (1280, 1280), (1, 1280), 0), out=buf703)
        del arg435_1
        del buf700
        # Topologically Sorted Source Nodes: [hidden_states_367], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf704 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf701, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf702, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf703, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), scale=0.125)
        del buf701
        buf705 = buf704[0]
        assert_size_stride(buf705, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf704
        buf710 = buf703; del buf703  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_370], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf705, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg436_1, (1280, 1280), (1, 1280), 0), out=buf710)
        del arg436_1
        buf714 = reinterpret_tensor(buf705, (32, 256, 1280), (327680, 1280, 1), 0); del buf705  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_366, hidden_states_372, hidden_states_373, norm_hidden_states_28], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_54.run(buf710, arg437_1, buf696, arg430_1, arg438_1, arg439_1, buf714, 8192, 1280, stream=stream2)
        del arg438_1
        del arg439_1
        buf715 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [query_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf714, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg440_1, (1280, 1280), (1, 1280), 0), out=buf715)
        del arg440_1
        del buf714
        buf716 = buf655; del buf655  # reuse
        # Topologically Sorted Source Nodes: [key_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg441_1, (1024, 1280), (1, 1024), 0), out=buf716)
        del arg441_1
        buf717 = buf654; del buf654  # reuse
        # Topologically Sorted Source Nodes: [value_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg442_1, (1024, 1280), (1, 1024), 0), out=buf717)
        del arg442_1
        # Topologically Sorted Source Nodes: [hidden_states_374], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf718 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf715, (32, 20, 256, 64), (327680, 64, 1280, 1), 0), reinterpret_tensor(buf716, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), reinterpret_tensor(buf717, (32, 20, 77, 64), (98560, 64, 1280, 1), 0), scale=0.125)
        del buf716
        del buf717
        buf719 = buf718[0]
        assert_size_stride(buf719, (32, 20, 256, 64), (327680, 64, 1280, 1))
        del buf718
        buf724 = buf715; del buf715  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_377], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf719, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg443_1, (1280, 1280), (1, 1280), 0), out=buf724)
        del arg443_1
        buf725 = reinterpret_tensor(buf724, (32, 256, 1280), (327680, 1280, 1), 0); del buf724  # reuse
        buf729 = reinterpret_tensor(buf719, (32, 256, 1280), (327680, 1280, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_366, hidden_states_372, hidden_states_373, hidden_states_379, hidden_states_380, norm_hidden_states_29], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_add_div_native_layer_norm_55.run(buf725, arg444_1, buf710, arg437_1, buf696, arg430_1, arg445_1, arg446_1, buf729, 8192, 1280, stream=stream2)
        del arg430_1
        del arg437_1
        del arg444_1
        del arg445_1
        del arg446_1
        del buf696
        del buf710
        buf730 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_381], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf729, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg447_1, (1280, 10240), (1, 1280), 0), out=buf730)
        del arg447_1
        buf731 = buf669; del buf669  # reuse
        # Topologically Sorted Source Nodes: [gelu_9, hidden_states_383], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_56.run(buf730, arg448_1, buf731, 41943040, stream=stream2)
        del arg448_1
        buf732 = reinterpret_tensor(buf729, (8192, 1280), (1280, 1), 0); del buf729  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_385], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf731, (8192, 5120), (5120, 1), 0), reinterpret_tensor(arg449_1, (5120, 1280), (1, 5120), 0), out=buf732)
        del arg449_1
        buf733 = reinterpret_tensor(buf732, (32, 256, 1280), (327680, 1280, 1), 0); del buf732  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_386], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_57.run(buf733, arg450_1, buf725, 10485760, stream=stream2)
        del arg450_1
        buf734 = reinterpret_tensor(buf725, (8192, 1280), (1280, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_387], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf733, (8192, 1280), (1280, 1), 0), reinterpret_tensor(arg451_1, (1280, 1280), (1, 1280), 0), out=buf734)
        del arg451_1
        del buf733
        buf736 = reinterpret_tensor(buf731, (32, 1280, 32, 32), (1310720, 1, 40960, 1280), 0); del buf731  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_7, hidden_states_361, hidden_states_363, add_67, output_tensor_15, hidden_states_388, output_9, hidden_states_389], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten._to_copy, aten._unsafe_index]
        stream2 = get_raw_stream(2)
        triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_95.run(buf734, arg452_1, buf687, arg426_1, buf691, arg424_1, buf736, 40960, 1024, stream=stream2)
        del arg424_1
        del arg426_1
        del arg452_1
        del buf687
        del buf691
        del buf734
        buf737 = buf690; del buf690  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_389, hidden_states_390], Original ATen: [aten._to_copy, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_50.run(arg453_1, buf737, 1638400, 9, stream=stream2)
        del arg453_1
        # Topologically Sorted Source Nodes: [hidden_states_389, hidden_states_390], Original ATen: [aten._to_copy, aten.convolution]
        buf738 = extern_kernels.convolution(buf736, buf737, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf738, (32, 1280, 32, 32), (1310720, 1, 40960, 1280))
        del buf737
        buf739 = empty_strided_cuda((32, 1920, 32, 32), (1966080, 1024, 32, 1), torch.float16)
        buf740 = buf693; del buf693  # reuse
        buf741 = buf692; del buf692  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_391, hidden_states_392], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_96.run(buf738, arg454_1, buf252, buf739, buf740, buf741, 1024, 61440, stream=stream2)
        del arg454_1
        buf744 = empty_strided_cuda((32, 1920, 32, 32), (1966080, 1, 61440, 1920), torch.float16)
        buf752 = empty_strided_cuda((32, 1920, 32, 32), (1966080, 1, 61440, 1920), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_392, hidden_states_393, input_tensor_8], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_97.run(buf739, buf740, buf741, arg455_1, arg456_1, buf744, buf752, 61440, 1024, stream=stream2)
        del arg455_1
        del arg456_1
        del buf739
        buf745 = empty_strided_cuda((640, 1920, 3, 3), (17280, 1, 5760, 1920), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_393, hidden_states_394], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_98.run(arg457_1, buf745, 1228800, 9, stream=stream2)
        del arg457_1
        # Topologically Sorted Source Nodes: [hidden_states_393, hidden_states_394], Original ATen: [aten.silu, aten.convolution]
        buf746 = extern_kernels.convolution(buf744, buf745, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf746, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del buf744
        del buf745
        buf747 = buf682; del buf682  # reuse
        buf809 = buf681; del buf681  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_32, temb_34], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_8.run(buf14, arg5_1, buf747, buf809, 40960, stream=stream2)
        buf748 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_32, linear_138], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf747, reinterpret_tensor(arg459_1, (1280, 640), (1, 1280), 0), out=buf748)
        del arg459_1
        buf749 = buf741; del buf741  # reuse
        buf750 = buf740; del buf740  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_396], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_28.run(buf746, arg458_1, buf748, arg460_1, buf749, buf750, 1024, 20480, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_8], Original ATen: [aten.convolution]
        buf753 = extern_kernels.convolution(buf752, arg465_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf753, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del arg465_1
        del buf752
        buf755 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_396, hidden_states_397], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_29.run(buf746, arg458_1, buf748, arg460_1, buf749, buf750, arg461_1, arg462_1, buf755, 20971520, stream=stream2)
        del arg458_1
        del arg460_1
        del arg461_1
        del arg462_1
        buf756 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_397, hidden_states_399], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg463_1, buf756, 409600, 9, stream=stream2)
        del arg463_1
        # Topologically Sorted Source Nodes: [hidden_states_397, hidden_states_399], Original ATen: [aten.silu, aten.convolution]
        buf757 = extern_kernels.convolution(buf755, buf756, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf757, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf758 = buf750; del buf750  # reuse
        buf759 = buf749; del buf749  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_400], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_31.run(buf753, arg466_1, buf757, arg464_1, buf758, buf759, 1024, 20480, stream=stream2)
        buf761 = reinterpret_tensor(buf755, (32, 1024, 640), (655360, 640, 1), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_402], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_32.run(buf753, arg466_1, buf757, arg464_1, buf758, buf759, arg467_1, arg468_1, buf761, 20971520, stream=stream2)
        del arg467_1
        del arg468_1
        buf762 = reinterpret_tensor(buf746, (32768, 640), (640, 1), 0); del buf746  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_402], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (32768, 640), (640, 1), 0), reinterpret_tensor(arg469_1, (640, 640), (1, 640), 0), out=buf762)
        del arg469_1
        buf766 = buf761; del buf761  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_402, norm_hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_33.run(buf762, arg470_1, arg471_1, arg472_1, buf766, 32768, 640, stream=stream2)
        del arg471_1
        del arg472_1
        buf767 = reinterpret_tensor(buf624, (32768, 640), (640, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [query_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf766, (32768, 640), (640, 1), 0), reinterpret_tensor(arg473_1, (640, 640), (1, 640), 0), out=buf767)
        del arg473_1
        buf768 = reinterpret_tensor(buf616, (32768, 640), (640, 1), 0); del buf616  # reuse
        # Topologically Sorted Source Nodes: [key_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf766, (32768, 640), (640, 1), 0), reinterpret_tensor(arg474_1, (640, 640), (1, 640), 0), out=buf768)
        del arg474_1
        buf769 = reinterpret_tensor(buf611, (32768, 640), (640, 1), 0); del buf611  # reuse
        # Topologically Sorted Source Nodes: [value_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf766, (32768, 640), (640, 1), 0), reinterpret_tensor(arg475_1, (640, 640), (1, 640), 0), out=buf769)
        del arg475_1
        del buf766
        # Topologically Sorted Source Nodes: [hidden_states_403], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf770 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf767, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf768, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf769, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), scale=0.125)
        del buf767
        buf771 = buf770[0]
        assert_size_stride(buf771, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf770
        buf776 = buf769; del buf769  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_406], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf771, (32768, 640), (640, 1), 0), reinterpret_tensor(arg476_1, (640, 640), (1, 640), 0), out=buf776)
        del arg476_1
        buf780 = reinterpret_tensor(buf771, (32, 1024, 640), (655360, 640, 1), 0); del buf771  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_402, hidden_states_408, hidden_states_409, norm_hidden_states_31], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_34.run(buf776, arg477_1, buf762, arg470_1, arg478_1, arg479_1, buf780, 32768, 640, stream=stream2)
        del arg478_1
        del arg479_1
        buf781 = buf768; del buf768  # reuse
        # Topologically Sorted Source Nodes: [query_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf780, (32768, 640), (640, 1), 0), reinterpret_tensor(arg480_1, (640, 640), (1, 640), 0), out=buf781)
        del arg480_1
        del buf780
        buf782 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [key_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg481_1, (1024, 640), (1, 1024), 0), out=buf782)
        del arg481_1
        buf783 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [value_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg482_1, (1024, 640), (1, 1024), 0), out=buf783)
        del arg482_1
        # Topologically Sorted Source Nodes: [hidden_states_410], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf784 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf781, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf782, (32, 10, 77, 64), (49280, 64, 640, 1), 0), reinterpret_tensor(buf783, (32, 10, 77, 64), (49280, 64, 640, 1), 0), scale=0.125)
        buf785 = buf784[0]
        assert_size_stride(buf785, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf784
        buf790 = buf781; del buf781  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_413], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf785, (32768, 640), (640, 1), 0), reinterpret_tensor(arg483_1, (640, 640), (1, 640), 0), out=buf790)
        del arg483_1
        buf791 = reinterpret_tensor(buf790, (32, 1024, 640), (655360, 640, 1), 0); del buf790  # reuse
        buf795 = reinterpret_tensor(buf785, (32, 1024, 640), (655360, 640, 1), 0); del buf785  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_402, hidden_states_408, hidden_states_409, hidden_states_415, hidden_states_416, norm_hidden_states_32], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_35.run(buf791, arg484_1, buf776, arg477_1, buf762, arg470_1, arg485_1, arg486_1, buf795, 32768, 640, stream=stream2)
        del arg470_1
        del arg477_1
        del arg484_1
        del arg485_1
        del arg486_1
        del buf762
        del buf776
        buf796 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_417], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf795, (32768, 640), (640, 1), 0), reinterpret_tensor(arg487_1, (640, 5120), (1, 640), 0), out=buf796)
        del arg487_1
        buf797 = reinterpret_tensor(buf730, (32, 1024, 2560), (2621440, 2560, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [gelu_10, hidden_states_419], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_36.run(buf796, arg488_1, buf797, 83886080, stream=stream2)
        del arg488_1
        buf798 = reinterpret_tensor(buf795, (32768, 640), (640, 1), 0); del buf795  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_421], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf797, (32768, 2560), (2560, 1), 0), reinterpret_tensor(arg489_1, (2560, 640), (1, 2560), 0), out=buf798)
        del arg489_1
        buf799 = reinterpret_tensor(buf798, (32, 1024, 640), (655360, 640, 1), 0); del buf798  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_422], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_37.run(buf799, arg490_1, buf791, 20971520, stream=stream2)
        del arg490_1
        buf800 = reinterpret_tensor(buf791, (32768, 640), (640, 1), 0); del buf791  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_423], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf799, (32768, 640), (640, 1), 0), reinterpret_tensor(arg491_1, (640, 640), (1, 640), 0), out=buf800)
        del arg491_1
        del buf799
        buf801 = reinterpret_tensor(buf738, (32, 1280, 32, 32), (1310720, 1024, 32, 1), 0); del buf738  # reuse
        buf802 = buf759; del buf759  # reuse
        buf803 = buf758; del buf758  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_425, hidden_states_426], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_99.run(buf800, arg492_1, buf753, arg466_1, buf757, arg464_1, buf192, buf801, buf802, buf803, 1024, 40960, stream=stream2)
        del arg464_1
        del arg466_1
        del arg492_1
        buf806 = buf736; del buf736  # reuse
        buf814 = reinterpret_tensor(buf125, (32, 1280, 32, 32), (1310720, 1, 40960, 1280), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_426, hidden_states_427, input_tensor_9], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_100.run(buf801, buf802, buf803, arg493_1, arg494_1, buf806, buf814, 40960, 1024, stream=stream2)
        del arg493_1
        del arg494_1
        buf807 = reinterpret_tensor(buf261, (640, 1280, 3, 3), (11520, 1, 3840, 1280), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_427, hidden_states_428], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_101.run(arg495_1, buf807, 819200, 9, stream=stream2)
        del arg495_1
        # Topologically Sorted Source Nodes: [hidden_states_427, hidden_states_428], Original ATen: [aten.silu, aten.convolution]
        buf808 = extern_kernels.convolution(buf806, buf807, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf808, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del buf807
        buf810 = buf748; del buf748  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_34, linear_151], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf809, reinterpret_tensor(arg497_1, (1280, 640), (1, 1280), 0), out=buf810)
        del arg497_1
        buf811 = buf803; del buf803  # reuse
        buf812 = buf802; del buf802  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_430], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_28.run(buf808, arg496_1, buf810, arg498_1, buf811, buf812, 1024, 20480, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_9], Original ATen: [aten.convolution]
        buf815 = extern_kernels.convolution(buf814, arg503_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf815, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del arg503_1
        buf817 = reinterpret_tensor(buf800, (32, 640, 32, 32), (655360, 1, 20480, 640), 0); del buf800  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_430, hidden_states_431], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_29.run(buf808, arg496_1, buf810, arg498_1, buf811, buf812, arg499_1, arg500_1, buf817, 20971520, stream=stream2)
        del arg496_1
        del arg498_1
        del arg499_1
        del arg500_1
        buf818 = buf756; del buf756  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_431, hidden_states_433], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg501_1, buf818, 409600, 9, stream=stream2)
        del arg501_1
        # Topologically Sorted Source Nodes: [hidden_states_431, hidden_states_433], Original ATen: [aten.silu, aten.convolution]
        buf819 = extern_kernels.convolution(buf817, buf818, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf819, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf820 = buf812; del buf812  # reuse
        buf821 = buf811; del buf811  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_434], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_31.run(buf815, arg504_1, buf819, arg502_1, buf820, buf821, 1024, 20480, stream=stream2)
        buf823 = reinterpret_tensor(buf817, (32, 1024, 640), (655360, 640, 1), 0); del buf817  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_436], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_32.run(buf815, arg504_1, buf819, arg502_1, buf820, buf821, arg505_1, arg506_1, buf823, 20971520, stream=stream2)
        del arg505_1
        del arg506_1
        buf824 = reinterpret_tensor(buf808, (32768, 640), (640, 1), 0); del buf808  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_436], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf823, (32768, 640), (640, 1), 0), reinterpret_tensor(arg507_1, (640, 640), (1, 640), 0), out=buf824)
        del arg507_1
        buf828 = buf823; del buf823  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_436, norm_hidden_states_33], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_33.run(buf824, arg508_1, arg509_1, arg510_1, buf828, 32768, 640, stream=stream2)
        del arg509_1
        del arg510_1
        buf829 = reinterpret_tensor(buf757, (32768, 640), (640, 1), 0); del buf757  # reuse
        # Topologically Sorted Source Nodes: [query_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf828, (32768, 640), (640, 1), 0), reinterpret_tensor(arg511_1, (640, 640), (1, 640), 0), out=buf829)
        del arg511_1
        buf830 = reinterpret_tensor(buf753, (32768, 640), (640, 1), 0); del buf753  # reuse
        # Topologically Sorted Source Nodes: [key_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf828, (32768, 640), (640, 1), 0), reinterpret_tensor(arg512_1, (640, 640), (1, 640), 0), out=buf830)
        del arg512_1
        buf831 = reinterpret_tensor(buf192, (32768, 640), (640, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [value_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf828, (32768, 640), (640, 1), 0), reinterpret_tensor(arg513_1, (640, 640), (1, 640), 0), out=buf831)
        del arg513_1
        del buf828
        # Topologically Sorted Source Nodes: [hidden_states_437], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf832 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf829, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf830, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf831, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), scale=0.125)
        del buf829
        buf833 = buf832[0]
        assert_size_stride(buf833, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf832
        buf838 = buf831; del buf831  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_440], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf833, (32768, 640), (640, 1), 0), reinterpret_tensor(arg514_1, (640, 640), (1, 640), 0), out=buf838)
        del arg514_1
        buf842 = reinterpret_tensor(buf833, (32, 1024, 640), (655360, 640, 1), 0); del buf833  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_436, hidden_states_442, hidden_states_443, norm_hidden_states_34], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_34.run(buf838, arg515_1, buf824, arg508_1, arg516_1, arg517_1, buf842, 32768, 640, stream=stream2)
        del arg516_1
        del arg517_1
        buf843 = buf830; del buf830  # reuse
        # Topologically Sorted Source Nodes: [query_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (32768, 640), (640, 1), 0), reinterpret_tensor(arg518_1, (640, 640), (1, 640), 0), out=buf843)
        del arg518_1
        del buf842
        buf844 = buf783; del buf783  # reuse
        # Topologically Sorted Source Nodes: [key_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg519_1, (1024, 640), (1, 1024), 0), out=buf844)
        del arg519_1
        buf845 = buf782; del buf782  # reuse
        # Topologically Sorted Source Nodes: [value_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg520_1, (1024, 640), (1, 1024), 0), out=buf845)
        del arg520_1
        # Topologically Sorted Source Nodes: [hidden_states_444], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf846 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf843, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf844, (32, 10, 77, 64), (49280, 64, 640, 1), 0), reinterpret_tensor(buf845, (32, 10, 77, 64), (49280, 64, 640, 1), 0), scale=0.125)
        buf847 = buf846[0]
        assert_size_stride(buf847, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf846
        buf852 = buf843; del buf843  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_447], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf847, (32768, 640), (640, 1), 0), reinterpret_tensor(arg521_1, (640, 640), (1, 640), 0), out=buf852)
        del arg521_1
        buf853 = reinterpret_tensor(buf852, (32, 1024, 640), (655360, 640, 1), 0); del buf852  # reuse
        buf857 = reinterpret_tensor(buf847, (32, 1024, 640), (655360, 640, 1), 0); del buf847  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_436, hidden_states_442, hidden_states_443, hidden_states_449, hidden_states_450, norm_hidden_states_35], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_35.run(buf853, arg522_1, buf838, arg515_1, buf824, arg508_1, arg523_1, arg524_1, buf857, 32768, 640, stream=stream2)
        del arg508_1
        del arg515_1
        del arg522_1
        del arg523_1
        del arg524_1
        del buf824
        del buf838
        buf858 = buf796; del buf796  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_451], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf857, (32768, 640), (640, 1), 0), reinterpret_tensor(arg525_1, (640, 5120), (1, 640), 0), out=buf858)
        del arg525_1
        buf859 = buf797; del buf797  # reuse
        # Topologically Sorted Source Nodes: [gelu_11, hidden_states_453], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_36.run(buf858, arg526_1, buf859, 83886080, stream=stream2)
        del arg526_1
        buf860 = reinterpret_tensor(buf857, (32768, 640), (640, 1), 0); del buf857  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_455], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf859, (32768, 2560), (2560, 1), 0), reinterpret_tensor(arg527_1, (2560, 640), (1, 2560), 0), out=buf860)
        del arg527_1
        buf861 = reinterpret_tensor(buf860, (32, 1024, 640), (655360, 640, 1), 0); del buf860  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_456], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_37.run(buf861, arg528_1, buf853, 20971520, stream=stream2)
        del arg528_1
        buf862 = reinterpret_tensor(buf853, (32768, 640), (640, 1), 0); del buf853  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_457], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf861, (32768, 640), (640, 1), 0), reinterpret_tensor(arg529_1, (640, 640), (1, 640), 0), out=buf862)
        del arg529_1
        buf863 = empty_strided_cuda((32, 960, 32, 32), (983040, 1024, 32, 1), torch.float16)
        buf864 = buf821; del buf821  # reuse
        buf865 = buf820; del buf820  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_459, hidden_states_460], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_102.run(buf862, arg530_1, buf815, arg504_1, buf819, arg502_1, buf130, buf863, buf864, buf865, 1024, 30720, stream=stream2)
        del arg502_1
        del arg504_1
        del arg530_1
        del buf130
        buf868 = empty_strided_cuda((32, 960, 32, 32), (983040, 1, 30720, 960), torch.float16)
        buf876 = empty_strided_cuda((32, 960, 32, 32), (983040, 1, 30720, 960), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_460, hidden_states_461, input_tensor_10], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_103.run(buf863, buf864, buf865, arg531_1, arg532_1, buf868, buf876, 30720, 1024, stream=stream2)
        del arg531_1
        del arg532_1
        del buf863
        buf869 = empty_strided_cuda((640, 960, 3, 3), (8640, 1, 2880, 960), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_461, hidden_states_462], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_104.run(arg533_1, buf869, 614400, 9, stream=stream2)
        del arg533_1
        # Topologically Sorted Source Nodes: [hidden_states_461, hidden_states_462], Original ATen: [aten.silu, aten.convolution]
        buf870 = extern_kernels.convolution(buf868, buf869, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf870, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del buf868
        del buf869
        buf871 = buf809; del buf809  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_36], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_94.run(buf14, arg5_1, buf871, 40960, stream=stream2)
        buf872 = buf810; del buf810  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_36, linear_164], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf871, reinterpret_tensor(arg535_1, (1280, 640), (1, 1280), 0), out=buf872)
        del arg535_1
        buf873 = buf865; del buf865  # reuse
        buf874 = buf864; del buf864  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_464], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_28.run(buf870, arg534_1, buf872, arg536_1, buf873, buf874, 1024, 20480, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_10], Original ATen: [aten.convolution]
        buf877 = extern_kernels.convolution(buf876, arg541_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf877, (32, 640, 32, 32), (655360, 1, 20480, 640))
        del arg541_1
        del buf876
        buf879 = reinterpret_tensor(buf862, (32, 640, 32, 32), (655360, 1, 20480, 640), 0); del buf862  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_464, hidden_states_465], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_29.run(buf870, arg534_1, buf872, arg536_1, buf873, buf874, arg537_1, arg538_1, buf879, 20971520, stream=stream2)
        del arg534_1
        del arg536_1
        del arg537_1
        del arg538_1
        del buf872
        buf880 = buf818; del buf818  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_465, hidden_states_467], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg539_1, buf880, 409600, 9, stream=stream2)
        del arg539_1
        # Topologically Sorted Source Nodes: [hidden_states_465, hidden_states_467], Original ATen: [aten.silu, aten.convolution]
        buf881 = extern_kernels.convolution(buf879, buf880, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf881, (32, 640, 32, 32), (655360, 1, 20480, 640))
        buf882 = buf874; del buf874  # reuse
        buf883 = buf873; del buf873  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_468], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_31.run(buf877, arg542_1, buf881, arg540_1, buf882, buf883, 1024, 20480, stream=stream2)
        buf885 = reinterpret_tensor(buf879, (32, 1024, 640), (655360, 640, 1), 0); del buf879  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_470], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_32.run(buf877, arg542_1, buf881, arg540_1, buf882, buf883, arg543_1, arg544_1, buf885, 20971520, stream=stream2)
        del arg543_1
        del arg544_1
        buf886 = reinterpret_tensor(buf870, (32768, 640), (640, 1), 0); del buf870  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_470], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (32768, 640), (640, 1), 0), reinterpret_tensor(arg545_1, (640, 640), (1, 640), 0), out=buf886)
        del arg545_1
        buf890 = buf885; del buf885  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_470, norm_hidden_states_36], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_33.run(buf886, arg546_1, arg547_1, arg548_1, buf890, 32768, 640, stream=stream2)
        del arg547_1
        del arg548_1
        buf891 = reinterpret_tensor(buf819, (32768, 640), (640, 1), 0); del buf819  # reuse
        # Topologically Sorted Source Nodes: [query_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (32768, 640), (640, 1), 0), reinterpret_tensor(arg549_1, (640, 640), (1, 640), 0), out=buf891)
        del arg549_1
        buf892 = reinterpret_tensor(buf815, (32768, 640), (640, 1), 0); del buf815  # reuse
        # Topologically Sorted Source Nodes: [key_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (32768, 640), (640, 1), 0), reinterpret_tensor(arg550_1, (640, 640), (1, 640), 0), out=buf892)
        del arg550_1
        buf893 = reinterpret_tensor(buf861, (32768, 640), (640, 1), 0); del buf861  # reuse
        # Topologically Sorted Source Nodes: [value_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (32768, 640), (640, 1), 0), reinterpret_tensor(arg551_1, (640, 640), (1, 640), 0), out=buf893)
        del arg551_1
        del buf890
        # Topologically Sorted Source Nodes: [hidden_states_471], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf894 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf891, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf892, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf893, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), scale=0.125)
        del buf891
        buf895 = buf894[0]
        assert_size_stride(buf895, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf894
        buf900 = buf893; del buf893  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_474], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf895, (32768, 640), (640, 1), 0), reinterpret_tensor(arg552_1, (640, 640), (1, 640), 0), out=buf900)
        del arg552_1
        buf904 = reinterpret_tensor(buf895, (32, 1024, 640), (655360, 640, 1), 0); del buf895  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_470, hidden_states_476, hidden_states_477, norm_hidden_states_37], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_34.run(buf900, arg553_1, buf886, arg546_1, arg554_1, arg555_1, buf904, 32768, 640, stream=stream2)
        del arg554_1
        del arg555_1
        buf905 = buf892; del buf892  # reuse
        # Topologically Sorted Source Nodes: [query_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (32768, 640), (640, 1), 0), reinterpret_tensor(arg556_1, (640, 640), (1, 640), 0), out=buf905)
        del arg556_1
        del buf904
        buf906 = buf845; del buf845  # reuse
        # Topologically Sorted Source Nodes: [key_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg557_1, (1024, 640), (1, 1024), 0), out=buf906)
        del arg557_1
        buf907 = buf844; del buf844  # reuse
        # Topologically Sorted Source Nodes: [value_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg558_1, (1024, 640), (1, 1024), 0), out=buf907)
        del arg558_1
        # Topologically Sorted Source Nodes: [hidden_states_478], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf908 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf905, (32, 10, 1024, 64), (655360, 64, 640, 1), 0), reinterpret_tensor(buf906, (32, 10, 77, 64), (49280, 64, 640, 1), 0), reinterpret_tensor(buf907, (32, 10, 77, 64), (49280, 64, 640, 1), 0), scale=0.125)
        del buf906
        del buf907
        buf909 = buf908[0]
        assert_size_stride(buf909, (32, 10, 1024, 64), (655360, 64, 640, 1))
        del buf908
        buf914 = buf905; del buf905  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_481], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf909, (32768, 640), (640, 1), 0), reinterpret_tensor(arg559_1, (640, 640), (1, 640), 0), out=buf914)
        del arg559_1
        buf915 = reinterpret_tensor(buf914, (32, 1024, 640), (655360, 640, 1), 0); del buf914  # reuse
        buf919 = reinterpret_tensor(buf909, (32, 1024, 640), (655360, 640, 1), 0); del buf909  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_470, hidden_states_476, hidden_states_477, hidden_states_483, hidden_states_484, norm_hidden_states_38], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_35.run(buf915, arg560_1, buf900, arg553_1, buf886, arg546_1, arg561_1, arg562_1, buf919, 32768, 640, stream=stream2)
        del arg546_1
        del arg553_1
        del arg560_1
        del arg561_1
        del arg562_1
        del buf886
        del buf900
        buf920 = buf858; del buf858  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_485], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf919, (32768, 640), (640, 1), 0), reinterpret_tensor(arg563_1, (640, 5120), (1, 640), 0), out=buf920)
        del arg563_1
        buf921 = buf859; del buf859  # reuse
        # Topologically Sorted Source Nodes: [gelu_12, hidden_states_487], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_36.run(buf920, arg564_1, buf921, 83886080, stream=stream2)
        del arg564_1
        buf922 = reinterpret_tensor(buf919, (32768, 640), (640, 1), 0); del buf919  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_489], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf921, (32768, 2560), (2560, 1), 0), reinterpret_tensor(arg565_1, (2560, 640), (1, 2560), 0), out=buf922)
        del arg565_1
        buf923 = reinterpret_tensor(buf922, (32, 1024, 640), (655360, 640, 1), 0); del buf922  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_490], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_37.run(buf923, arg566_1, buf915, 20971520, stream=stream2)
        del arg566_1
        buf924 = reinterpret_tensor(buf915, (32768, 640), (640, 1), 0); del buf915  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_491], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf923, (32768, 640), (640, 1), 0), reinterpret_tensor(arg567_1, (640, 640), (1, 640), 0), out=buf924)
        del arg567_1
        del buf923
        buf926 = reinterpret_tensor(buf921, (32, 640, 64, 64), (2621440, 1, 40960, 640), 0); del buf921  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_10, hidden_states_465, hidden_states_467, add_85, output_tensor_18, hidden_states_492, output_12, hidden_states_493], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten._to_copy, aten._unsafe_index]
        stream2 = get_raw_stream(2)
        triton_poi_fused__to_copy__unsafe_index_add_clone_convolution_div_silu_105.run(buf924, arg568_1, buf877, arg542_1, buf881, arg540_1, buf926, 20480, 4096, stream=stream2)
        del arg540_1
        del arg542_1
        del arg568_1
        del buf877
        del buf881
        del buf924
        buf927 = buf880; del buf880  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_493, hidden_states_494], Original ATen: [aten._to_copy, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_30.run(arg569_1, buf927, 409600, 9, stream=stream2)
        del arg569_1
        # Topologically Sorted Source Nodes: [hidden_states_493, hidden_states_494], Original ATen: [aten._to_copy, aten.convolution]
        buf928 = extern_kernels.convolution(buf926, buf927, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf928, (32, 640, 64, 64), (2621440, 1, 40960, 640))
        del buf927
        buf929 = empty_strided_cuda((32, 960, 64, 64), (3932160, 4096, 64, 1), torch.float16)
        buf930 = buf883; del buf883  # reuse
        buf931 = buf882; del buf882  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_495, hidden_states_496], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_106.run(buf928, arg570_1, buf127, buf929, buf930, buf931, 1024, 122880, stream=stream2)
        del arg570_1
        buf934 = empty_strided_cuda((32, 960, 64, 64), (3932160, 1, 61440, 960), torch.float16)
        buf942 = empty_strided_cuda((32, 960, 64, 64), (3932160, 1, 61440, 960), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_496, hidden_states_497, input_tensor_11], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_107.run(buf929, buf930, buf931, arg571_1, arg572_1, buf934, buf942, 30720, 4096, stream=stream2)
        del arg571_1
        del arg572_1
        del buf929
        buf935 = empty_strided_cuda((320, 960, 3, 3), (8640, 1, 2880, 960), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_497, hidden_states_498], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_108.run(arg573_1, buf935, 307200, 9, stream=stream2)
        del arg573_1
        # Topologically Sorted Source Nodes: [hidden_states_497, hidden_states_498], Original ATen: [aten.silu, aten.convolution]
        buf936 = extern_kernels.convolution(buf934, buf935, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf936, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        del buf934
        del buf935
        buf937 = buf871; del buf871  # reuse
        buf999 = buf747; del buf747  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_38, temb_40], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_8.run(buf14, arg5_1, buf937, buf999, 40960, stream=stream2)
        buf938 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_38, linear_177], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf937, reinterpret_tensor(arg575_1, (1280, 320), (1, 1280), 0), out=buf938)
        del arg575_1
        del buf937
        buf939 = buf931; del buf931  # reuse
        buf940 = buf930; del buf930  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_500], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_9.run(buf936, arg574_1, buf938, arg576_1, buf939, buf940, 1024, 40960, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_11], Original ATen: [aten.convolution]
        buf943 = extern_kernels.convolution(buf942, arg581_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf943, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        del arg581_1
        del buf942
        buf945 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_500, hidden_states_501], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_10.run(buf936, arg574_1, buf938, arg576_1, buf939, buf940, arg577_1, arg578_1, buf945, 41943040, stream=stream2)
        del arg574_1
        del arg576_1
        del arg577_1
        del arg578_1
        buf946 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_501, hidden_states_503], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg579_1, buf946, 102400, 9, stream=stream2)
        del arg579_1
        # Topologically Sorted Source Nodes: [hidden_states_501, hidden_states_503], Original ATen: [aten.silu, aten.convolution]
        buf947 = extern_kernels.convolution(buf945, buf946, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf947, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf948 = buf940; del buf940  # reuse
        buf949 = buf939; del buf939  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_504], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_11.run(buf943, arg582_1, buf947, arg580_1, buf948, buf949, 1024, 40960, stream=stream2)
        buf951 = reinterpret_tensor(buf945, (32, 4096, 320), (1310720, 320, 1), 0); del buf945  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_506], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_12.run(buf943, arg582_1, buf947, arg580_1, buf948, buf949, arg583_1, arg584_1, buf951, 41943040, stream=stream2)
        del arg583_1
        del arg584_1
        buf952 = reinterpret_tensor(buf936, (131072, 320), (320, 1), 0); del buf936  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_506], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf951, (131072, 320), (320, 1), 0), reinterpret_tensor(arg585_1, (320, 320), (1, 320), 0), out=buf952)
        del arg585_1
        buf956 = buf951; del buf951  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_506, norm_hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_13.run(buf952, arg586_1, arg587_1, arg588_1, buf956, 131072, 320, stream=stream2)
        del arg587_1
        del arg588_1
        buf957 = reinterpret_tensor(buf814, (131072, 320), (320, 1), 0); del buf814  # reuse
        # Topologically Sorted Source Nodes: [query_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (131072, 320), (320, 1), 0), reinterpret_tensor(arg589_1, (320, 320), (1, 320), 0), out=buf957)
        del arg589_1
        buf958 = reinterpret_tensor(buf806, (131072, 320), (320, 1), 0); del buf806  # reuse
        # Topologically Sorted Source Nodes: [key_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (131072, 320), (320, 1), 0), reinterpret_tensor(arg590_1, (320, 320), (1, 320), 0), out=buf958)
        del arg590_1
        buf959 = reinterpret_tensor(buf801, (131072, 320), (320, 1), 0); del buf801  # reuse
        # Topologically Sorted Source Nodes: [value_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (131072, 320), (320, 1), 0), reinterpret_tensor(arg591_1, (320, 320), (1, 320), 0), out=buf959)
        del arg591_1
        del buf956
        # Topologically Sorted Source Nodes: [hidden_states_507], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf960 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf957, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf958, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf959, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), scale=0.125)
        del buf957
        buf961 = buf960[0]
        assert_size_stride(buf961, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf960
        buf966 = buf959; del buf959  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_510], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf961, (131072, 320), (320, 1), 0), reinterpret_tensor(arg592_1, (320, 320), (1, 320), 0), out=buf966)
        del arg592_1
        buf970 = reinterpret_tensor(buf961, (32, 4096, 320), (1310720, 320, 1), 0); del buf961  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_506, hidden_states_512, hidden_states_513, norm_hidden_states_40], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_14.run(buf966, arg593_1, buf952, arg586_1, arg594_1, arg595_1, buf970, 131072, 320, stream=stream2)
        del arg594_1
        del arg595_1
        buf971 = buf958; del buf958  # reuse
        # Topologically Sorted Source Nodes: [query_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf970, (131072, 320), (320, 1), 0), reinterpret_tensor(arg596_1, (320, 320), (1, 320), 0), out=buf971)
        del arg596_1
        del buf970
        buf972 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [key_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg597_1, (1024, 320), (1, 1024), 0), out=buf972)
        del arg597_1
        buf973 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [value_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg598_1, (1024, 320), (1, 1024), 0), out=buf973)
        del arg598_1
        # Topologically Sorted Source Nodes: [hidden_states_514], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf974 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf971, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf972, (32, 5, 77, 64), (24640, 64, 320, 1), 0), reinterpret_tensor(buf973, (32, 5, 77, 64), (24640, 64, 320, 1), 0), scale=0.125)
        buf975 = buf974[0]
        assert_size_stride(buf975, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf974
        buf980 = buf971; del buf971  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_517], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf975, (131072, 320), (320, 1), 0), reinterpret_tensor(arg599_1, (320, 320), (1, 320), 0), out=buf980)
        del arg599_1
        buf981 = reinterpret_tensor(buf980, (32, 4096, 320), (1310720, 320, 1), 0); del buf980  # reuse
        buf985 = reinterpret_tensor(buf975, (32, 4096, 320), (1310720, 320, 1), 0); del buf975  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_506, hidden_states_512, hidden_states_513, hidden_states_519, hidden_states_520, norm_hidden_states_41], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_15.run(buf981, arg600_1, buf966, arg593_1, buf952, arg586_1, arg601_1, arg602_1, buf985, 131072, 320, stream=stream2)
        del arg586_1
        del arg593_1
        del arg600_1
        del arg601_1
        del arg602_1
        del buf952
        del buf966
        buf986 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_521], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf985, (131072, 320), (320, 1), 0), reinterpret_tensor(arg603_1, (320, 2560), (1, 320), 0), out=buf986)
        del arg603_1
        buf987 = reinterpret_tensor(buf920, (32, 4096, 1280), (5242880, 1280, 1), 0); del buf920  # reuse
        # Topologically Sorted Source Nodes: [gelu_13, hidden_states_523], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_16.run(buf986, arg604_1, buf987, 167772160, stream=stream2)
        del arg604_1
        buf988 = reinterpret_tensor(buf985, (131072, 320), (320, 1), 0); del buf985  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_525], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf987, (131072, 1280), (1280, 1), 0), reinterpret_tensor(arg605_1, (1280, 320), (1, 1280), 0), out=buf988)
        del arg605_1
        buf989 = reinterpret_tensor(buf988, (32, 4096, 320), (1310720, 320, 1), 0); del buf988  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_526], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_17.run(buf989, arg606_1, buf981, 41943040, stream=stream2)
        del arg606_1
        buf990 = reinterpret_tensor(buf981, (131072, 320), (320, 1), 0); del buf981  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_527], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf989, (131072, 320), (320, 1), 0), reinterpret_tensor(arg607_1, (320, 320), (1, 320), 0), out=buf990)
        del arg607_1
        del buf989
        buf991 = reinterpret_tensor(buf928, (32, 640, 64, 64), (2621440, 4096, 64, 1), 0); del buf928  # reuse
        buf992 = buf949; del buf949  # reuse
        buf993 = buf948; del buf948  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_529, hidden_states_530], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_109.run(buf990, arg608_1, buf943, arg582_1, buf947, arg580_1, buf67, buf991, buf992, buf993, 1024, 81920, stream=stream2)
        del arg580_1
        del arg582_1
        del arg608_1
        buf996 = buf926; del buf926  # reuse
        buf1004 = empty_strided_cuda((32, 640, 64, 64), (2621440, 1, 40960, 640), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_530, hidden_states_531, input_tensor_12], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_110.run(buf991, buf992, buf993, arg609_1, arg610_1, buf996, buf1004, 20480, 4096, stream=stream2)
        del arg609_1
        del arg610_1
        buf997 = reinterpret_tensor(buf136, (320, 640, 3, 3), (5760, 1, 1920, 640), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_531, hidden_states_532], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_111.run(arg611_1, buf997, 204800, 9, stream=stream2)
        del arg611_1
        # Topologically Sorted Source Nodes: [hidden_states_531, hidden_states_532], Original ATen: [aten.silu, aten.convolution]
        buf998 = extern_kernels.convolution(buf996, buf997, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf998, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf1000 = buf938; del buf938  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_40, linear_190], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf999, reinterpret_tensor(arg613_1, (1280, 320), (1, 1280), 0), out=buf1000)
        del arg613_1
        del buf999
        buf1001 = buf993; del buf993  # reuse
        buf1002 = buf992; del buf992  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_534], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_9.run(buf998, arg612_1, buf1000, arg614_1, buf1001, buf1002, 1024, 40960, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_12], Original ATen: [aten.convolution]
        buf1005 = extern_kernels.convolution(buf1004, arg619_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1005, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        del arg619_1
        buf1007 = reinterpret_tensor(buf990, (32, 320, 64, 64), (1310720, 1, 20480, 320), 0); del buf990  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_534, hidden_states_535], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_10.run(buf998, arg612_1, buf1000, arg614_1, buf1001, buf1002, arg615_1, arg616_1, buf1007, 41943040, stream=stream2)
        del arg612_1
        del arg614_1
        del arg615_1
        del arg616_1
        buf1008 = buf946; del buf946  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_535, hidden_states_537], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg617_1, buf1008, 102400, 9, stream=stream2)
        del arg617_1
        # Topologically Sorted Source Nodes: [hidden_states_535, hidden_states_537], Original ATen: [aten.silu, aten.convolution]
        buf1009 = extern_kernels.convolution(buf1007, buf1008, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1009, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        buf1010 = buf1002; del buf1002  # reuse
        buf1011 = buf1001; del buf1001  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_538], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_11.run(buf1005, arg620_1, buf1009, arg618_1, buf1010, buf1011, 1024, 40960, stream=stream2)
        buf1013 = reinterpret_tensor(buf1007, (32, 4096, 320), (1310720, 320, 1), 0); del buf1007  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_540], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_12.run(buf1005, arg620_1, buf1009, arg618_1, buf1010, buf1011, arg621_1, arg622_1, buf1013, 41943040, stream=stream2)
        del arg621_1
        del arg622_1
        buf1014 = reinterpret_tensor(buf998, (131072, 320), (320, 1), 0); del buf998  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_540], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1013, (131072, 320), (320, 1), 0), reinterpret_tensor(arg623_1, (320, 320), (1, 320), 0), out=buf1014)
        del arg623_1
        buf1018 = buf1013; del buf1013  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_540, norm_hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_13.run(buf1014, arg624_1, arg625_1, arg626_1, buf1018, 131072, 320, stream=stream2)
        del arg625_1
        del arg626_1
        buf1019 = reinterpret_tensor(buf947, (131072, 320), (320, 1), 0); del buf947  # reuse
        # Topologically Sorted Source Nodes: [query_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1018, (131072, 320), (320, 1), 0), reinterpret_tensor(arg627_1, (320, 320), (1, 320), 0), out=buf1019)
        del arg627_1
        buf1020 = reinterpret_tensor(buf943, (131072, 320), (320, 1), 0); del buf943  # reuse
        # Topologically Sorted Source Nodes: [key_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1018, (131072, 320), (320, 1), 0), reinterpret_tensor(arg628_1, (320, 320), (1, 320), 0), out=buf1020)
        del arg628_1
        buf1021 = reinterpret_tensor(buf67, (131072, 320), (320, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [value_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1018, (131072, 320), (320, 1), 0), reinterpret_tensor(arg629_1, (320, 320), (1, 320), 0), out=buf1021)
        del arg629_1
        del buf1018
        # Topologically Sorted Source Nodes: [hidden_states_541], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1022 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf1019, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf1020, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf1021, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), scale=0.125)
        del buf1019
        buf1023 = buf1022[0]
        assert_size_stride(buf1023, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf1022
        buf1028 = buf1021; del buf1021  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_544], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1023, (131072, 320), (320, 1), 0), reinterpret_tensor(arg630_1, (320, 320), (1, 320), 0), out=buf1028)
        del arg630_1
        buf1032 = reinterpret_tensor(buf1023, (32, 4096, 320), (1310720, 320, 1), 0); del buf1023  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_540, hidden_states_546, hidden_states_547, norm_hidden_states_43], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_14.run(buf1028, arg631_1, buf1014, arg624_1, arg632_1, arg633_1, buf1032, 131072, 320, stream=stream2)
        del arg632_1
        del arg633_1
        buf1033 = buf1020; del buf1020  # reuse
        # Topologically Sorted Source Nodes: [query_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1032, (131072, 320), (320, 1), 0), reinterpret_tensor(arg634_1, (320, 320), (1, 320), 0), out=buf1033)
        del arg634_1
        del buf1032
        buf1034 = buf973; del buf973  # reuse
        # Topologically Sorted Source Nodes: [key_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg635_1, (1024, 320), (1, 1024), 0), out=buf1034)
        del arg635_1
        buf1035 = buf972; del buf972  # reuse
        # Topologically Sorted Source Nodes: [value_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg636_1, (1024, 320), (1, 1024), 0), out=buf1035)
        del arg636_1
        # Topologically Sorted Source Nodes: [hidden_states_548], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1036 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf1033, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf1034, (32, 5, 77, 64), (24640, 64, 320, 1), 0), reinterpret_tensor(buf1035, (32, 5, 77, 64), (24640, 64, 320, 1), 0), scale=0.125)
        buf1037 = buf1036[0]
        assert_size_stride(buf1037, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf1036
        buf1042 = buf1033; del buf1033  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_551], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1037, (131072, 320), (320, 1), 0), reinterpret_tensor(arg637_1, (320, 320), (1, 320), 0), out=buf1042)
        del arg637_1
        buf1043 = reinterpret_tensor(buf1042, (32, 4096, 320), (1310720, 320, 1), 0); del buf1042  # reuse
        buf1047 = reinterpret_tensor(buf1037, (32, 4096, 320), (1310720, 320, 1), 0); del buf1037  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_540, hidden_states_546, hidden_states_547, hidden_states_553, hidden_states_554, norm_hidden_states_44], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_15.run(buf1043, arg638_1, buf1028, arg631_1, buf1014, arg624_1, arg639_1, arg640_1, buf1047, 131072, 320, stream=stream2)
        del arg624_1
        del arg631_1
        del arg638_1
        del arg639_1
        del arg640_1
        del buf1014
        del buf1028
        buf1048 = buf986; del buf986  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_555], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1047, (131072, 320), (320, 1), 0), reinterpret_tensor(arg641_1, (320, 2560), (1, 320), 0), out=buf1048)
        del arg641_1
        buf1049 = buf987; del buf987  # reuse
        # Topologically Sorted Source Nodes: [gelu_14, hidden_states_557], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_16.run(buf1048, arg642_1, buf1049, 167772160, stream=stream2)
        del arg642_1
        buf1050 = reinterpret_tensor(buf1047, (131072, 320), (320, 1), 0); del buf1047  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_559], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1049, (131072, 1280), (1280, 1), 0), reinterpret_tensor(arg643_1, (1280, 320), (1, 1280), 0), out=buf1050)
        del arg643_1
        buf1051 = reinterpret_tensor(buf1050, (32, 4096, 320), (1310720, 320, 1), 0); del buf1050  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_560], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_17.run(buf1051, arg644_1, buf1043, 41943040, stream=stream2)
        del arg644_1
        buf1052 = reinterpret_tensor(buf1043, (131072, 320), (320, 1), 0); del buf1043  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_561], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1051, (131072, 320), (320, 1), 0), reinterpret_tensor(arg645_1, (320, 320), (1, 320), 0), out=buf1052)
        del arg645_1
        del buf1051
        buf1053 = reinterpret_tensor(buf1004, (32, 640, 64, 64), (2621440, 4096, 64, 1), 0); del buf1004  # reuse
        buf1054 = buf1011; del buf1011  # reuse
        buf1055 = buf1010; del buf1010  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_563, hidden_states_564], Original ATen: [aten.cat, aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_cat_native_group_norm_112.run(buf1052, arg646_1, buf1005, arg620_1, buf1009, arg618_1, buf2, arg8_1, buf1053, buf1054, buf1055, 1024, 81920, stream=stream2)
        del arg618_1
        del arg620_1
        del arg646_1
        del arg8_1
        buf1058 = buf996; del buf996  # reuse
        buf1066 = reinterpret_tensor(buf991, (32, 640, 64, 64), (2621440, 1, 40960, 640), 0); del buf991  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_564, hidden_states_565, input_tensor_13], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_native_group_norm_silu_110.run(buf1053, buf1054, buf1055, arg647_1, arg648_1, buf1058, buf1066, 20480, 4096, stream=stream2)
        del arg647_1
        del arg648_1
        del buf1053
        buf1059 = buf997; del buf997  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_565, hidden_states_566], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_111.run(arg649_1, buf1059, 204800, 9, stream=stream2)
        del arg649_1
        # Topologically Sorted Source Nodes: [hidden_states_565, hidden_states_566], Original ATen: [aten.silu, aten.convolution]
        buf1060 = extern_kernels.convolution(buf1058, buf1059, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1060, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        del buf1058
        del buf1059
        buf1061 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_42], Original ATen: [aten.addmm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_addmm_silu_7.run(buf1061, arg5_1, 40960, stream=stream2)
        del arg5_1
        buf1062 = buf1000; del buf1000  # reuse
        # Topologically Sorted Source Nodes: [sample_2, temb_42, linear_203], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.mm(buf1061, reinterpret_tensor(arg651_1, (1280, 320), (1, 1280), 0), out=buf1062)
        del arg651_1
        del buf1061
        buf1063 = buf1055; del buf1055  # reuse
        buf1064 = buf1054; del buf1054  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_568], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_9.run(buf1060, arg650_1, buf1062, arg652_1, buf1063, buf1064, 1024, 40960, stream=stream2)
        # Topologically Sorted Source Nodes: [input_tensor_13], Original ATen: [aten.convolution]
        buf1067 = extern_kernels.convolution(buf1066, arg657_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1067, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        del arg657_1
        del buf1066
        buf1069 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_568, hidden_states_569], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_10.run(buf1060, arg650_1, buf1062, arg652_1, buf1063, buf1064, arg653_1, arg654_1, buf1069, 41943040, stream=stream2)
        del arg650_1
        del arg652_1
        del arg653_1
        del arg654_1
        del buf1062
        buf1070 = buf1008; del buf1008  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_569, hidden_states_571], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_4.run(arg655_1, buf1070, 102400, 9, stream=stream2)
        del arg655_1
        # Topologically Sorted Source Nodes: [hidden_states_569, hidden_states_571], Original ATen: [aten.silu, aten.convolution]
        buf1071 = extern_kernels.convolution(buf1069, buf1070, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1071, (32, 320, 64, 64), (1310720, 1, 20480, 320))
        del buf1070
        buf1072 = buf1064; del buf1064  # reuse
        buf1073 = buf1063; del buf1063  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_572], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_11.run(buf1067, arg658_1, buf1071, arg656_1, buf1072, buf1073, 1024, 40960, stream=stream2)
        buf1075 = reinterpret_tensor(buf1069, (32, 4096, 320), (1310720, 320, 1), 0); del buf1069  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_574], Original ATen: [aten.clone]
        stream2 = get_raw_stream(2)
        triton_poi_fused_clone_12.run(buf1067, arg658_1, buf1071, arg656_1, buf1072, buf1073, arg659_1, arg660_1, buf1075, 41943040, stream=stream2)
        del arg659_1
        del arg660_1
        buf1076 = reinterpret_tensor(buf1060, (131072, 320), (320, 1), 0); del buf1060  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_574], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1075, (131072, 320), (320, 1), 0), reinterpret_tensor(arg661_1, (320, 320), (1, 320), 0), out=buf1076)
        del arg661_1
        buf1080 = buf1075; del buf1075  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_574, norm_hidden_states_45], Original ATen: [aten.add, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_native_layer_norm_13.run(buf1076, arg662_1, arg663_1, arg664_1, buf1080, 131072, 320, stream=stream2)
        del arg663_1
        del arg664_1
        buf1081 = buf1052; del buf1052  # reuse
        # Topologically Sorted Source Nodes: [query_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (131072, 320), (320, 1), 0), reinterpret_tensor(arg665_1, (320, 320), (1, 320), 0), out=buf1081)
        del arg665_1
        buf1082 = reinterpret_tensor(buf1009, (131072, 320), (320, 1), 0); del buf1009  # reuse
        # Topologically Sorted Source Nodes: [key_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (131072, 320), (320, 1), 0), reinterpret_tensor(arg666_1, (320, 320), (1, 320), 0), out=buf1082)
        del arg666_1
        buf1083 = reinterpret_tensor(buf1005, (131072, 320), (320, 1), 0); del buf1005  # reuse
        # Topologically Sorted Source Nodes: [value_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (131072, 320), (320, 1), 0), reinterpret_tensor(arg667_1, (320, 320), (1, 320), 0), out=buf1083)
        del arg667_1
        del buf1080
        # Topologically Sorted Source Nodes: [hidden_states_575], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1084 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf1081, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf1082, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf1083, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), scale=0.125)
        del buf1081
        buf1085 = buf1084[0]
        assert_size_stride(buf1085, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf1084
        buf1090 = buf1083; del buf1083  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_578], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1085, (131072, 320), (320, 1), 0), reinterpret_tensor(arg668_1, (320, 320), (1, 320), 0), out=buf1090)
        del arg668_1
        buf1094 = reinterpret_tensor(buf1085, (32, 4096, 320), (1310720, 320, 1), 0); del buf1085  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_574, hidden_states_580, hidden_states_581, norm_hidden_states_46], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_14.run(buf1090, arg669_1, buf1076, arg662_1, arg670_1, arg671_1, buf1094, 131072, 320, stream=stream2)
        del arg670_1
        del arg671_1
        buf1095 = buf1082; del buf1082  # reuse
        # Topologically Sorted Source Nodes: [query_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (131072, 320), (320, 1), 0), reinterpret_tensor(arg672_1, (320, 320), (1, 320), 0), out=buf1095)
        del arg672_1
        del buf1094
        buf1096 = buf1035; del buf1035  # reuse
        # Topologically Sorted Source Nodes: [key_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg673_1, (1024, 320), (1, 1024), 0), out=buf1096)
        del arg673_1
        buf1097 = buf1034; del buf1034  # reuse
        # Topologically Sorted Source Nodes: [value_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg6_1, (2464, 1024), (1024, 1), 0), reinterpret_tensor(arg674_1, (1024, 320), (1, 1024), 0), out=buf1097)
        del arg674_1
        del arg6_1
        # Topologically Sorted Source Nodes: [hidden_states_582], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1098 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf1095, (32, 5, 4096, 64), (1310720, 64, 320, 1), 0), reinterpret_tensor(buf1096, (32, 5, 77, 64), (24640, 64, 320, 1), 0), reinterpret_tensor(buf1097, (32, 5, 77, 64), (24640, 64, 320, 1), 0), scale=0.125)
        del buf1096
        del buf1097
        buf1099 = buf1098[0]
        assert_size_stride(buf1099, (32, 5, 4096, 64), (1310720, 64, 320, 1))
        del buf1098
        buf1104 = buf1095; del buf1095  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_585], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1099, (131072, 320), (320, 1), 0), reinterpret_tensor(arg675_1, (320, 320), (1, 320), 0), out=buf1104)
        del arg675_1
        buf1105 = reinterpret_tensor(buf1104, (32, 4096, 320), (1310720, 320, 1), 0); del buf1104  # reuse
        buf1109 = reinterpret_tensor(buf1099, (32, 4096, 320), (1310720, 320, 1), 0); del buf1099  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_574, hidden_states_580, hidden_states_581, hidden_states_587, hidden_states_588, norm_hidden_states_47], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        stream2 = get_raw_stream(2)
        triton_per_fused_add_div_native_layer_norm_15.run(buf1105, arg676_1, buf1090, arg669_1, buf1076, arg662_1, arg677_1, arg678_1, buf1109, 131072, 320, stream=stream2)
        del arg662_1
        del arg669_1
        del arg676_1
        del arg677_1
        del arg678_1
        del buf1076
        del buf1090
        buf1110 = buf1048; del buf1048  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_589], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1109, (131072, 320), (320, 1), 0), reinterpret_tensor(arg679_1, (320, 2560), (1, 320), 0), out=buf1110)
        del arg679_1
        buf1111 = buf1049; del buf1049  # reuse
        # Topologically Sorted Source Nodes: [gelu_15, hidden_states_591], Original ATen: [aten.gelu, aten.mul]
        stream2 = get_raw_stream(2)
        triton_poi_fused_gelu_mul_16.run(buf1110, arg680_1, buf1111, 167772160, stream=stream2)
        del arg680_1
        del buf1110
        buf1112 = reinterpret_tensor(buf1109, (131072, 320), (320, 1), 0); del buf1109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_593], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1111, (131072, 1280), (1280, 1), 0), reinterpret_tensor(arg681_1, (1280, 320), (1, 1280), 0), out=buf1112)
        del arg681_1
        del buf1111
        buf1113 = reinterpret_tensor(buf1112, (32, 4096, 320), (1310720, 320, 1), 0); del buf1112  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_594], Original ATen: [aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_17.run(buf1113, arg682_1, buf1105, 41943040, stream=stream2)
        del arg682_1
        buf1114 = reinterpret_tensor(buf1105, (131072, 320), (320, 1), 0); del buf1105  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_595], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1113, (131072, 320), (320, 1), 0), reinterpret_tensor(arg683_1, (320, 320), (1, 320), 0), out=buf1114)
        del arg683_1
        del buf1113
        buf1115 = empty_strided_cuda((32, 32, 10, 4096), (1310720, 10, 1, 320), torch.float32)
        # Topologically Sorted Source Nodes: [sample_4], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_113.run(buf1114, arg684_1, buf1067, arg658_1, buf1071, arg656_1, buf1115, 41943040, stream=stream2)
        del arg656_1
        del arg658_1
        del arg684_1
        del buf1067
        del buf1071
        buf1116 = buf1073; del buf1073  # reuse
        buf1117 = buf1072; del buf1072  # reuse
        # Topologically Sorted Source Nodes: [sample_4], Original ATen: [aten.native_group_norm]
        stream2 = get_raw_stream(2)
        triton_red_fused_native_group_norm_114.run(buf1115, buf1116, buf1117, 1024, 40960, stream=stream2)
        buf1119 = reinterpret_tensor(buf1115, (32, 320, 64, 64), (1310720, 1, 20480, 320), 0); del buf1115  # reuse
        buf1120 = reinterpret_tensor(buf1114, (32, 320, 64, 64), (1310720, 1, 20480, 320), 0); del buf1114  # reuse
        # Topologically Sorted Source Nodes: [sample_4, sample_5], Original ATen: [aten.native_group_norm, aten.silu]
        stream2 = get_raw_stream(2)
        triton_poi_fused_native_group_norm_silu_115.run(buf1119, buf1116, buf1117, arg685_1, arg686_1, buf1120, 41943040, stream=stream2)
        del arg685_1
        del arg686_1
        del buf1116
        del buf1117
        del buf1119
        buf1121 = reinterpret_tensor(buf1, (4, 320, 3, 3), (2880, 1, 960, 320), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [sample_5, sample_6], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_116.run(arg687_1, buf1121, 1280, 9, stream=stream2)
        del arg687_1
        # Topologically Sorted Source Nodes: [sample_5, sample_6], Original ATen: [aten.silu, aten.convolution]
        buf1122 = extern_kernels.convolution(buf1120, buf1121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1122, (32, 4, 64, 64), (16384, 1, 256, 4))
        del buf1120
        del buf1121
        buf1123 = reinterpret_tensor(buf0, (32, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [sample_5, sample_6], Original ATen: [aten.silu, aten.convolution]
        stream2 = get_raw_stream(2)
        triton_poi_fused_convolution_silu_117.run(buf1122, arg688_1, buf1123, 128, 4096, stream=stream2)
        del arg688_1
        del buf1122
    return (buf1123, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:2', dtype=torch.float16)
    arg1_1 = rand_strided((), (), device='cuda:2', dtype=torch.int64)
    arg2_1 = rand_strided((1280, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg3_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg4_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg5_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg6_1 = rand_strided((32, 77, 1024), (78848, 1024, 1), device='cuda:2', dtype=torch.float16)
    arg7_1 = rand_strided((320, 4, 3, 3), (36, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg8_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg9_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg10_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg11_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg12_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg13_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg14_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg15_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg16_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg17_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg18_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg19_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg20_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg21_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg22_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg23_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg24_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg25_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg26_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg27_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg28_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg29_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg30_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg31_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg32_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg33_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg34_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg35_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg36_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg37_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg38_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg39_1 = rand_strided((2560, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg40_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg41_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg42_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg43_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg44_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg45_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg46_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg47_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg48_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg49_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg50_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg51_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg52_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg53_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg54_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg55_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg56_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg57_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg58_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg59_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg60_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg61_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg62_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg63_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg64_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg65_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg66_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg67_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg68_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg69_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg70_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg71_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg72_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg73_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg74_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg75_1 = rand_strided((2560, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg76_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg77_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg78_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg79_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg80_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg81_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg82_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg83_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg84_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg85_1 = rand_strided((640, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg86_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg87_1 = rand_strided((640, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg88_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg89_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg90_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg91_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg92_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg93_1 = rand_strided((640, 320, 1, 1), (320, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg94_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg95_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg96_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg97_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg98_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg99_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg100_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg101_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg102_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg103_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg104_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg105_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg106_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg107_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg108_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg109_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg110_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg111_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg112_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg113_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg114_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg115_1 = rand_strided((5120, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg116_1 = rand_strided((5120, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg117_1 = rand_strided((640, 2560), (2560, 1), device='cuda:2', dtype=torch.float16)
    arg118_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg119_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg120_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg121_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg122_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg123_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg124_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg125_1 = rand_strided((640, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg126_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg127_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg128_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg129_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg130_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg131_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg132_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg133_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg134_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg135_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg136_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg137_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg138_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg139_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg140_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg141_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg142_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg143_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg144_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg145_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg146_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg147_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg148_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg149_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg150_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg151_1 = rand_strided((5120, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg152_1 = rand_strided((5120, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg153_1 = rand_strided((640, 2560), (2560, 1), device='cuda:2', dtype=torch.float16)
    arg154_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg155_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg156_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg157_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg158_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg159_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg160_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg161_1 = rand_strided((1280, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg162_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg163_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg164_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg165_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg166_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg167_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg168_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg169_1 = rand_strided((1280, 640, 1, 1), (640, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg170_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg171_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg172_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg173_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg174_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg175_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg176_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg177_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg178_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg179_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg180_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg181_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg182_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg183_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg184_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg185_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg186_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg187_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg188_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg189_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg190_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg191_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg192_1 = rand_strided((10240, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg193_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:2', dtype=torch.float16)
    arg194_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg195_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg196_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg197_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg198_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg199_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg200_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg201_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg202_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg203_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg204_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg205_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg206_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg207_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg208_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg209_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg210_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg211_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg212_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg213_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg214_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg215_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg216_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg217_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg218_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg219_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg220_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg221_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg222_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg223_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg224_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg225_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg226_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg227_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg228_1 = rand_strided((10240, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg229_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:2', dtype=torch.float16)
    arg230_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg231_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg232_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg233_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg234_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg235_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg236_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg237_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg238_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg239_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg240_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg241_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg242_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg243_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg244_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg245_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg246_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg247_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg248_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg249_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg250_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg251_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg252_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg253_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg254_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg255_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg256_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg257_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg258_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg259_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg260_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg261_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg262_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg263_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg264_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg265_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg266_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg267_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg268_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg269_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg270_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg271_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg272_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg273_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg274_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg275_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg276_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg277_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg278_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg279_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg280_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg281_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg282_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg283_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg284_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg285_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg286_1 = rand_strided((10240, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg287_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:2', dtype=torch.float16)
    arg288_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg289_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg290_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg291_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg292_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg293_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg294_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg295_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg296_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg297_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg298_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg299_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg300_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg301_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg302_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg303_1 = rand_strided((1280, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg304_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg305_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg306_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg307_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg308_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg309_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg310_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg311_1 = rand_strided((1280, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg312_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg313_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg314_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg315_1 = rand_strided((1280, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg316_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg317_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg318_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg319_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg320_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg321_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg322_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg323_1 = rand_strided((1280, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg324_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg325_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg326_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg327_1 = rand_strided((1280, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg328_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg329_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg330_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg331_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg332_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg333_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg334_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg335_1 = rand_strided((1280, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg336_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg337_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg338_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg339_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg340_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg341_1 = rand_strided((1280, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg342_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg343_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg344_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg345_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg346_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg347_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg348_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg349_1 = rand_strided((1280, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg350_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg351_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg352_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg353_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg354_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg355_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg356_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg357_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg358_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg359_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg360_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg361_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg362_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg363_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg364_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg365_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg366_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg367_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg368_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg369_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg370_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg371_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg372_1 = rand_strided((10240, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg373_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:2', dtype=torch.float16)
    arg374_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg375_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg376_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg377_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg378_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg379_1 = rand_strided((1280, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg380_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg381_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg382_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg383_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg384_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg385_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg386_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg387_1 = rand_strided((1280, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg388_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg389_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg390_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg391_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg392_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg393_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg394_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg395_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg396_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg397_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg398_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg399_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg400_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg401_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg402_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg403_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg404_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg405_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg406_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg407_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg408_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg409_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg410_1 = rand_strided((10240, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg411_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:2', dtype=torch.float16)
    arg412_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg413_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg414_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg415_1 = rand_strided((1920, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg416_1 = rand_strided((1920, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg417_1 = rand_strided((1280, 1920, 3, 3), (17280, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg418_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg419_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg420_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg421_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg422_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg423_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg424_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg425_1 = rand_strided((1280, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg426_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg427_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg428_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg429_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg430_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg431_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg432_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg433_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg434_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg435_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg436_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg437_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg438_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg439_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg440_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg441_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg442_1 = rand_strided((1280, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg443_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg444_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg445_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg446_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg447_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg448_1 = rand_strided((10240, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg449_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:2', dtype=torch.float16)
    arg450_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg451_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg452_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg453_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg454_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg455_1 = rand_strided((1920, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg456_1 = rand_strided((1920, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg457_1 = rand_strided((640, 1920, 3, 3), (17280, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg458_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg459_1 = rand_strided((640, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg460_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg461_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg462_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg463_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg464_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg465_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg466_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg467_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg468_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg469_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg470_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg471_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg472_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg473_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg474_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg475_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg476_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg477_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg478_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg479_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg480_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg481_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg482_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg483_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg484_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg485_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg486_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg487_1 = rand_strided((5120, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg488_1 = rand_strided((5120, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg489_1 = rand_strided((640, 2560), (2560, 1), device='cuda:2', dtype=torch.float16)
    arg490_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg491_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg492_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg493_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg494_1 = rand_strided((1280, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg495_1 = rand_strided((640, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg496_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg497_1 = rand_strided((640, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg498_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg499_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg500_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg501_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg502_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg503_1 = rand_strided((640, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg504_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg505_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg506_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg507_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg508_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg509_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg510_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg511_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg512_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg513_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg514_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg515_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg516_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg517_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg518_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg519_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg520_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg521_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg522_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg523_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg524_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg525_1 = rand_strided((5120, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg526_1 = rand_strided((5120, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg527_1 = rand_strided((640, 2560), (2560, 1), device='cuda:2', dtype=torch.float16)
    arg528_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg529_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg530_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg531_1 = rand_strided((960, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg532_1 = rand_strided((960, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg533_1 = rand_strided((640, 960, 3, 3), (8640, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg534_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg535_1 = rand_strided((640, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg536_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg537_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg538_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg539_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg540_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg541_1 = rand_strided((640, 960, 1, 1), (960, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg542_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg543_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg544_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg545_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg546_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg547_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg548_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg549_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg550_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg551_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg552_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg553_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg554_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg555_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg556_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg557_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg558_1 = rand_strided((640, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg559_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg560_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg561_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg562_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg563_1 = rand_strided((5120, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg564_1 = rand_strided((5120, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg565_1 = rand_strided((640, 2560), (2560, 1), device='cuda:2', dtype=torch.float16)
    arg566_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg567_1 = rand_strided((640, 640), (640, 1), device='cuda:2', dtype=torch.float16)
    arg568_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg569_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg570_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg571_1 = rand_strided((960, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg572_1 = rand_strided((960, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg573_1 = rand_strided((320, 960, 3, 3), (8640, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg574_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg575_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg576_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg577_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg578_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg579_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg580_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg581_1 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg582_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg583_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg584_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg585_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg586_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg587_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg588_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg589_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg590_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg591_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg592_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg593_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg594_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg595_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg596_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg597_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg598_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg599_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg600_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg601_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg602_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg603_1 = rand_strided((2560, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg604_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg605_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg606_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg607_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg608_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg609_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg610_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg611_1 = rand_strided((320, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg612_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg613_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg614_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg615_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg616_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg617_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg618_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg619_1 = rand_strided((320, 640, 1, 1), (640, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg620_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg621_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg622_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg623_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg624_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg625_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg626_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg627_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg628_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg629_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg630_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg631_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg632_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg633_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg634_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg635_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg636_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg637_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg638_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg639_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg640_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg641_1 = rand_strided((2560, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg642_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg643_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg644_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg645_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg646_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg647_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg648_1 = rand_strided((640, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg649_1 = rand_strided((320, 640, 3, 3), (5760, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg650_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg651_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg652_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg653_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg654_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg655_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg656_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg657_1 = rand_strided((320, 640, 1, 1), (640, 1, 1, 1), device='cuda:2', dtype=torch.float16)
    arg658_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg659_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg660_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg661_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg662_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg663_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg664_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg665_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg666_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg667_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg668_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg669_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg670_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg671_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg672_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg673_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg674_1 = rand_strided((320, 1024), (1024, 1), device='cuda:2', dtype=torch.float16)
    arg675_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg676_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg677_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg678_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg679_1 = rand_strided((2560, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg680_1 = rand_strided((2560, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg681_1 = rand_strided((320, 1280), (1280, 1), device='cuda:2', dtype=torch.float16)
    arg682_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg683_1 = rand_strided((320, 320), (320, 1), device='cuda:2', dtype=torch.float16)
    arg684_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg685_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg686_1 = rand_strided((320, ), (1, ), device='cuda:2', dtype=torch.float16)
    arg687_1 = rand_strided((4, 320, 3, 3), (2880, 9, 3, 1), device='cuda:2', dtype=torch.float16)
    arg688_1 = rand_strided((4, ), (1, ), device='cuda:2', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
