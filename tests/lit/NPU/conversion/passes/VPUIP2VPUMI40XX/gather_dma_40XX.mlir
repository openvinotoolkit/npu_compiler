//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @GatherDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x1x16x256xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x1x16x256xf16>
  }
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0>-> memref<1x1x8x256xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <DDR> <0>-> memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
    %indices_input = VPURT.DeclareBuffer  <CMX_NN> [0] <0> -> memref<1x1x8x1xi64, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.GatherDMA {block_size = 2 : i64} <{addressingMode = 1 : i64,
        port = 0 : i64, elementSize = 16, padding = 0}>
        inputs(%0 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>)
        indices(%indices_input :  memref<1x1x8x1xi64, {order = #NHWC}, [@CMX_NN, 0]>)
        outputs(%1 : memref<1x1x8x256xf16, {order = #NHWC}, [@CMX_NN, 0]>)  -> memref<1x1x8x256xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.GatherDMA
    // CHECK:     VPUMI40XX.NNDMA <{addressingMode = 1 : i64, allow_different_in_out_shapes, port = 0 : i64}> inputs({{%.+}} : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) outputs({{%.+}} : memref<1x1x8x256xf16, {order = #NHWC}, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}indices({{%.+}} : memref<1x1x8x1xi64, {order = #NHWC}, [@CMX_NN, 0]>) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
  }
}
