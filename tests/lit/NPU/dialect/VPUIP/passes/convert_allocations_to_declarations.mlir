//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-allocations-to-declarations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @StaticAlloc
func.func @StaticAlloc() -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x2048xf16, [@CMX_NN, 0]>) {
    %0 = VPUIP.StaticAlloc<0> -> memref<1x1x1x1000xf16, @DDR>
    %1 = VPUIP.StaticAlloc<2000> -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>

    return %0, %1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x2048xf16, [@CMX_NN, 0]>

    // CHECK:       [[VAR0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[VAR1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    // CHECK: return [[VAR0]], [[VAR1]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x2048xf16, [@CMX_NN, 0]>
}
