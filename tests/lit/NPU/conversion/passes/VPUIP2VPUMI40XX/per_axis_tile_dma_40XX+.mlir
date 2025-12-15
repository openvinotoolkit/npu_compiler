//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @perAxisTileDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1xf16>  // Dummy value
  } outputsInfo : {
    DataInfo "output_0" : tensor<1xf16> // Dummy value
  }

  // Func simply returns arg1 without copying any PerAxisTileDMA results to it beforehand
  func.func @main(%arg0: memref<1xf16, @DDR>, %arg1: memref<1xf16, @DDR>) -> memref<1xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x122x120xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x122x240xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task {
      %2 = VPUIP.PerAxisTileDMA {axis = 3 : i64, port = 0 : i64, tiles = 2 : i64} inputs(%0 : memref<1x4x122x120xf16, #NHWC, @DDR>) outputs(%1 : memref<1x4x122x240xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x122x240xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %arg1 : memref<1xf16, @DDR>

    // CHECK: [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> [[INPUT_TYPE_0:.+]]

    // CHECK: [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_0:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 0
    // CHECK-SAME:      inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PerAxisTileDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_0]]
    // CHECK-SAME:              outputType = [[OUTPUT_TYPE_0]]
    // CHECK-SAME:              axis = 3
    // CHECK-SAME:              tiles = 2
    // CHECK-SAME:  -> !VPURegMapped.Index<0:0:0>
  }
}
