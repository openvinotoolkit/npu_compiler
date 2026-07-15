//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @EmptyOpRanges {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @MultiOpRanges {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  module @VPU.SW {
    func.func nested @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 0 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 0 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 1 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 1 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 2 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 2 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 3 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 3 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 4 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 4 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 5 : i64}
      } PPE : {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0], cluster_id = 5 : i64}
      } PPE : {
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 0 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 0 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 1 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 1 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 2 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 2 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 3 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 3 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 4 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 4 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 5 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_softmax
          inputs(%arg0 as %x: memref<1x2x3x4xf16, @DDR>)
          outputs(%arg1 as %y: memref<1x2x3x4xf16, @DDR>)
          on tile 5 -> memref<1x2x3x4xf16, @DDR> {
              VPUIP.SW.Kernel.run {attrs = [0]}(%x, %y) : memref<1x2x3x4xf16, @DDR>, memref<1x2x3x4xf16, @DDR>
      }
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([
  // CHECK-DAG: #VPURegMapped.task_type<DMA>
  // CHECK-DAG: #VPURegMapped.task_type<DMA>
  // CHECK-DAG: #VPURegMapped.task_type<DMA>
  // CHECK-DAG: #VPURegMapped.task_type<DMA>
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
  // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation
  // CHECK-SAME: ])
  // CHECK-SAME: begins(
  // CHECK-SAME: !VPURegMapped.Index<0:0:0>
  // CHECK-SAME: !VPURegMapped.Index<0:1:0>
  // CHECK-SAME: !VPURegMapped.Index<1:0:0>
  // CHECK-SAME: !VPURegMapped.Index<1:1:0>
  // CHECK-SAME: !VPURegMapped.Index<0:0:0>
  // CHECK-SAME: !VPURegMapped.Index<0:0:0>
  // CHECK-SAME: !VPURegMapped.Index<1:0:0>
  // CHECK-SAME: !VPURegMapped.Index<1:0:0>
  // CHECK-SAME: !VPURegMapped.Index<2:0:0>
  // CHECK-SAME: !VPURegMapped.Index<2:0:0>
  // CHECK-SAME: !VPURegMapped.Index<3:0:0>
  // CHECK-SAME: !VPURegMapped.Index<3:0:0>
  // CHECK-SAME: !VPURegMapped.Index<4:0:0>
  // CHECK-SAME: !VPURegMapped.Index<4:0:0>
  // CHECK-SAME: !VPURegMapped.Index<5:0:0>
  // CHECK-SAME: !VPURegMapped.Index<5:0:0>
  // CHECK-SAME: !VPURegMapped.Index<0:0:0>
  // CHECK-SAME: !VPURegMapped.Index<0:0:0>
  // CHECK-SAME: !VPURegMapped.Index<1:0:0>
  // CHECK-SAME: !VPURegMapped.Index<1:0:0>
  // CHECK-SAME: !VPURegMapped.Index<2:0:0>
  // CHECK-SAME: !VPURegMapped.Index<2:0:0>
  // CHECK-SAME: !VPURegMapped.Index<3:0:0>
  // CHECK-SAME: !VPURegMapped.Index<3:0:0>
  // CHECK-SAME: !VPURegMapped.Index<4:0:0>
  // CHECK-SAME: !VPURegMapped.Index<4:0:0>
  // CHECK-SAME: !VPURegMapped.Index<5:0:0>
  // CHECK-SAME: !VPURegMapped.Index<5:0:0>
  // CHECK-SAME: )
  // CHECK-SAME: ends(
  // CHECK-SAME: !VPURegMapped.Index<0:0:1>
  // CHECK-SAME: !VPURegMapped.Index<0:1:1>
  // CHECK-SAME: !VPURegMapped.Index<1:0:1>
  // CHECK-SAME: !VPURegMapped.Index<1:1:1>
  // CHECK-SAME: !VPURegMapped.Index<0:0:1>
  // CHECK-SAME: !VPURegMapped.Index<0:0:1>
  // CHECK-SAME: !VPURegMapped.Index<1:0:1>
  // CHECK-SAME: !VPURegMapped.Index<1:0:1>
  // CHECK-SAME: !VPURegMapped.Index<2:0:1>
  // CHECK-SAME: !VPURegMapped.Index<2:0:1>
  // CHECK-SAME: !VPURegMapped.Index<3:0:1>
  // CHECK-SAME: !VPURegMapped.Index<3:0:1>
  // CHECK-SAME: !VPURegMapped.Index<4:0:1>
  // CHECK-SAME: !VPURegMapped.Index<4:0:1>
  // CHECK-SAME: !VPURegMapped.Index<5:0:1>
  // CHECK-SAME: !VPURegMapped.Index<5:0:1>
  // CHECK-SAME: !VPURegMapped.Index<0:0:1>
  // CHECK-SAME: !VPURegMapped.Index<0:0:1>
  // CHECK-SAME: !VPURegMapped.Index<1:0:1>
  // CHECK-SAME: !VPURegMapped.Index<1:0:1>
  // CHECK-SAME: !VPURegMapped.Index<2:0:1>
  // CHECK-SAME: !VPURegMapped.Index<2:0:1>
  // CHECK-SAME: !VPURegMapped.Index<3:0:1>
  // CHECK-SAME: !VPURegMapped.Index<3:0:1>
  // CHECK-SAME: !VPURegMapped.Index<4:0:1>
  // CHECK-SAME: !VPURegMapped.Index<4:0:1>
  // CHECK-SAME: !VPURegMapped.Index<5:0:1>
  // CHECK-SAME: !VPURegMapped.Index<5:0:1>
  // CHECK-SAME: )
}
