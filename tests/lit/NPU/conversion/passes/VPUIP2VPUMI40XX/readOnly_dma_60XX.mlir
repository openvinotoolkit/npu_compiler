//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU6010

module @ReadOnlyDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16>) ->  memref<1x2x3x4xf16> {
    %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task {
      %3 = VPUIP.ReadOnlyDMA <{port = 0 : i64}> inputs(%1 : memref<1x2x3x4xf16, @DDR>)}

    // CHECK-NOT: VPUIP.ReadOnlyDMA
    // CHECK-DAG: [[BUF_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK: [[VAL0:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    return %arg0 : memref<1x2x3x4xf16>

  }
}
