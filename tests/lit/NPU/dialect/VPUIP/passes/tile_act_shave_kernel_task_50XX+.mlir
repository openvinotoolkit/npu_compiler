//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

!PermuteIn = !VPUIP.DistributedBuffer<
  1x16x768x128x!qElemType, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
  compute_shapes = [[1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 192, 0], [0, 0, 384, 0], [0, 0, 576, 0]],
  memory_shapes = [[1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128], [1, 16, 192, 128]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 192, 0], [0, 0, 384, 0], [0, 0, 576, 0]]}>

!PermuteOut = !VPUIP.DistributedBuffer<
  1x768x16x128x!qElemType, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
  compute_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
  compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
  memory_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
  memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

!PermuteOutAligned = !VPUIP.DistributedBuffer<
  1x768x16x128x!qElemType, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
  compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
  memory_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
  memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

!ScaleDist = !VPUIP.DistributedBuffer<
  1x768x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
  memory_shapes = [[1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1], [1, 192, 16, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

!DequantOut = !VPUIP.DistributedBuffer<
  1x768x16x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
  compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]],
  memory_shapes = [[1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128], [1, 192, 16, 128]],
  memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]}>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func nested @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
    func.func nested @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @NotTileMemPermuteIfConsumerIsDynDQ
func.func @NotTileMemPermuteIfConsumerIsDynDQ(%arg0: !PermuteIn, %arg1: !ScaleDist) -> !DequantOut {
    %permuteOutBuf = VPURT.AllocDistributed -> !PermuteOut
    %permuted = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
                inputs(%arg0 as %arg9: !PermuteIn)
                outputs(%permuteOutBuf as %arg10: !PermuteOut) on tile 0
                -> !PermuteOut {
        VPUIP.SW.Kernel.run {attrs = [[0, 2, 1, 3]]}(%arg9, %arg10) : !PermuteIn, !PermuteOut
    }

    %casted = VPUIP.DistributedCast inputs(%permuted : !PermuteOut) -> !PermuteOutAligned

    %dequantOutBuf = VPURT.AllocDistributed -> !DequantOut
    %dequantized = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
                inputs(%casted as %arg9: !PermuteOutAligned, %arg1 as %arg10: !ScaleDist)
                outputs(%dequantOutBuf as %arg11: !DequantOut) on tile 0
                -> !DequantOut {
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%arg9, %arg10, %arg11) : !PermuteOutAligned, !ScaleDist, !DequantOut
    }

    return %dequantized : !DequantOut

    // CHECK:       [[PERMUTED:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK:           VPUIP.SW.Kernel.run
    // CHECK-NOT:       VPUIP.SW.Kernel.run

    // CHECK:       [[DQ:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[DQ]]#0, [[DQ]]#1
    // CHECK:       return [[CONCAT]]
}
