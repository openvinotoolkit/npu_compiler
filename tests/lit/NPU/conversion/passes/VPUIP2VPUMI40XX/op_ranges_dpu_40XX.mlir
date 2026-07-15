//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleDPUTile0 {

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "input_0" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x2x3x4xf16>
}

func.func nested @main() {
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
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  return

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-SAME: ])
  // CHECK-SAME: begins([[VAL0:%[0-9]+]], [[VAL1:%[0-9]+]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
  // CHECK-SAME: ends([[VAL0]], [[VAL1]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleDPUTile0 {

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "input_0" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x2x3x4xf16>
}

func.func nested @main() {
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
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  return

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-SAME: ])
  // CHECK-SAME: begins({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
  // CHECK-SAME: ends({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:2>)
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleDPUTile2 {

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "input_0" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x2x3x4xf16>
}

func.func nested @main() {
  %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  %6 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [2] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [2] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 2]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0, 2] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0, 2] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  %9 = VPURT.DeclareBuffer <CMX_NN> [2] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>

  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 2]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  return

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-SAME: ])
  // CHECK-SAME: begins([[VAL0:%[0-9]+]], [[VAL1:%[0-9]+]] : !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>)
  // CHECK-SAME: ends([[VAL0]], [[VAL1]] : !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>)
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleDPUTile2 {

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "input_0" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x2x3x4xf16>
}

func.func nested @main() {
  %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  %6 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [2] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [2] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 2]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0, 2] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0, 2] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  %9 = VPURT.DeclareBuffer <CMX_NN> [2] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>

  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 2]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 2]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 2]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 2]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  return

  // CHECK-NOT: return
  // CHECK: VPUMI40XX.OpRanges
  // CHECK-SAME: types([
  // CHECK-DAG: #VPURegMapped.task_type<DPUInvariant>
  // CHECK-DAG: #VPURegMapped.task_type<DPUVariant>
  // CHECK-SAME: ])
  // CHECK-SAME: begins({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>)
  // CHECK-SAME: ends({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<2:0:2>, !VPURegMapped.Index<2:0:2>)
}
}
