//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --barrier-legalization %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DmaSequenceDoNotOptimizeBarriers
func.func @DmaSequenceDoNotOptimizeBarriers(%arg0: memref<1x16x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {

    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            {port = 0 : i64}
            inputs(%arg0: memref<1x16x32x32xf16, #NHWC, @DDR>)
            outputs(%buf0: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            {port = 0 : i64}
            inputs(%buf0: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%arg1: memref<1x16x32x32xf16, #NHWC, @DDR>)
            -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }

    return %arg1 : memref<1x16x32x32xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NNDMA
}
