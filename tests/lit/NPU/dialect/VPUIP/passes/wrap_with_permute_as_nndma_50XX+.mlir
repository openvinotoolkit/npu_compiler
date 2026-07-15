//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW num-of-dpu-groups=2" --wrap-with-permute-as-nndma %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

!ParentDDRType = memref<1x16x2304x128x!qElemType, @DDR>
!StridedDDRType = memref<1x16x768x128x!qElemType, {order = #NCHW, strides = [4718592, 294912, 128, 1]}, @DDR>
!CompactDDRType = memref<1x16x768x128x!qElemType, @DDR>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x768x128x!qElemType, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 192, 0], [0, 0, 384, 0], [0, 0, 576, 0]],
    memory_shapes = [[1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 192, 0], [0, 0, 384, 0], [0, 0, 576, 0]]}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x768x16x128x!qElemType, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
    memory_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

  !OutputDistributedTypeAligned = !VPUIP.DistributedBuffer<
    1x768x16x128x!qElemType, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
    memory_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

  !DqInputDistributedType = !VPUIP.DistributedBuffer<
    1x384x16x128x!qElemType, {order = #NCHW, strides = [1572864, 2048, 128, 1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 8, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]],
    memory_shapes = [[1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]]}>

  !ScaleDDRType = memref<1x768x16x1xf16, @DDR>
  !ScaleDistributedType = !VPUIP.DistributedBuffer<
    1x768x16x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
    memory_shapes = [[1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

  !ScaleSubviewType = !VPUIP.DistributedBuffer<
    1x384x16x1xf16, {order = #NCHW, strides = [12288, 16, 1, 1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 8, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 96, 16, 1], [1, 96, 16, 1], [1, 96, 16, 1], [1, 96, 16, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]],
    memory_shapes = [[1, 96, 16, 1], [1, 96, 16, 1], [1, 96, 16, 1], [1, 96, 16, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]]}>

  !DequantOutputType = !VPUIP.DistributedBuffer<
    1x384x16x128xf16, {order = #NCHW, strides = [1572864, 2048, 128, 1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 8, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]],
    memory_shapes = [[1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]]}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func nested @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @FuseDDRCopyWithMemPermuteIfConsumerIsDynDQ
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x2304x128x!qElemType, @DDR>, [[SCALE:%.+]]: memref<1x768x16x1xf16, @DDR>)
func.func @FuseDDRCopyWithMemPermuteIfConsumerIsDynDQ(%arg0: !ParentDDRType, %arg1: !ScaleDDRType) -> !DequantOutputType {
    %sub = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 768, 128] : !ParentDDRType to !StridedDDRType

    %alloc = memref.alloc() : !CompactDDRType
    %copy0 = VPUIP.Copy inputs(%sub : !StridedDDRType) outputs(%alloc : !CompactDDRType) -> !CompactDDRType

    %distAlloc = VPURT.AllocDistributed -> !InputDistributedType
    %copy1 = VPUIP.Copy inputs(%copy0 : !CompactDDRType) outputs(%distAlloc : !InputDistributedType) -> !InputDistributedType

    %outAlloc = VPURT.AllocDistributed -> !OutputDistributedType
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_MemPermute
      inputs(%copy1 as %permuteInput: !InputDistributedType)
      outputs(%outAlloc as %permuteOutput: !OutputDistributedType)
        on tile 0 -> !OutputDistributedType {
      VPUIP.SW.Kernel.run {attrs = [[0, 2, 1, 3]]}(%permuteInput, %permuteOutput) : !InputDistributedType, !OutputDistributedType
    }

    %casted = VPUIP.DistributedCast inputs(%results : !OutputDistributedType) -> !OutputDistributedTypeAligned
    %permutedSubView = VPUIP.SubView %casted [0, 0, 0, 0] [1, 384, 16, 128] {explicit_output_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]], explicit_output_shapes = [[1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128], [1, 96, 16, 128]]} : !OutputDistributedTypeAligned to !DqInputDistributedType

    %scaleAlloc = VPURT.AllocDistributed -> !ScaleDistributedType
    %scaleCopy = VPUIP.Copy inputs(%arg1 : !ScaleDDRType) outputs(%scaleAlloc : !ScaleDistributedType) -> !ScaleDistributedType
    %scaleSubView = VPUIP.SubView %scaleCopy [0, 0, 0, 0] [1, 384, 16, 1] {explicit_output_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0]], explicit_output_shapes = [[1, 96, 16, 1], [1, 96, 16, 1], [1, 96, 16, 1], [1, 96, 16, 1]]} : !ScaleDistributedType to !ScaleSubviewType

    %dequantAlloc = VPURT.AllocDistributed -> !DequantOutputType
    %dequantized = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
      inputs(%permutedSubView as %dqInput: !DqInputDistributedType, %scaleSubView as %dqScale: !ScaleSubviewType)
      outputs(%dequantAlloc as %dqOutput: !DequantOutputType)
      on tile 0 -> !DequantOutputType {
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%dqInput, %dqScale, %dqOutput) : !DqInputDistributedType, !ScaleSubviewType, !DequantOutputType
    }

    return %dequantized : !DequantOutputType

    // CHECK:       [[SUB:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 16, 768, 128]
    // CHECK-SAME:    memref<1x16x2304x128x!qElemType, @DDR>
    // CHECK-SAME:    to memref<1x16x768x128x!qElemType, {order = #NCHW, strides = [4718592, 294912, 128, 1]}, @DDR>
    // CHECK:       [[OUT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x768x16x128x!qElemType
    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteDMA <{mem_perm = #NHCW}>
    // CHECK-SAME:      inputs([[SUB]] : memref<1x16x768x128x!qElemType, {order = #NCHW, strides = [4718592, 294912, 128, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[OUT_ALLOC]]
    // CHECK-NOT:   @VPU.SW::@builtin_MemPermute
    // CHECK:       [[CAST:%.+]] = VPUIP.DistributedCast inputs([[PERMUTE]]
    // CHECK:       [[PERMUTE_SUBVIEW:%.+]] = VPUIP.SubView [[CAST]] [0, 0, 0, 0] [1, 384, 16, 128]
    // CHECK:       [[DQ:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK:       return [[DQ]]
}
