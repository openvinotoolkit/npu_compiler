//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @OneDMAWithoutAttributes {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<1x2x3x4xf16, @DDR>
  // CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<1x2x3x4xf16, @DDR>
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       [[VAL0:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[ARG_0]] : memref<1x2x3x4xf16, @DDR>) outputs([[ARG_1]] : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return [[ARG_1]] : memref<1x2x3x4xf16, @DDR>
  }
}

// -----

module @OneDMAWithAttributes {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<1x2x3x4xf16, @DDR>
  // CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<1x2x3x4xf16, @DDR>
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{is_out_of_order, is_critical, port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       [[VAL0:%.+]] = VPUMI37XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64}> inputs([[ARG_0]] : memref<1x2x3x4xf16, @DDR>) outputs([[ARG_1]] : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return [[ARG_1]] : memref<1x2x3x4xf16, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneDMA {
func.func @UnrollDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<64x32x1x1xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG: [[CST:%.+]] = const.Declare memref<64x32x1x1xf16, {order = #NHWC}, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK: [[WEIGHTS:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<64x32x1x1xf16, {order = #NHWC}, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }
  // CHECK: [[BUFF_TILE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK: [[BUFF_TILE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA0:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[CST]] : memref<64x32x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUFF_TILE_0]], [[BUFF_TILE_1]] : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}
