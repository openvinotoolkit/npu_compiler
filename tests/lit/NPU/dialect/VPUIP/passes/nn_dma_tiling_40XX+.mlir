//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --nn-dma-tiling %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = memref<1x4x360x216xf16, {order = #NHWC, strides = [6220800, 1, 8640, 8]}, @DDR>
!Output_CMX = memref<1x4x360x216xf16, #NHWC, [@CMX_NN, 0]>

func.func @DoNotSplitNNDMA(%input: !Input_DDR, %output: !Output_CMX) -> !Output_CMX {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <18662408> -> !Input_DDR
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <622080> -> !Output_CMX

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : memref<1x4x360x216xf16, {order = #NHWC, strides = [6220800, 1, 8640, 8]}, @DDR>) outputs(%1 : memref<1x4x360x216xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x360x216xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %1: !Output_CMX

    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    [[INPUT_DDR_BUF0:%.*]] = VPURT.DeclareBuffer <DDR> <18662408> -> memref<1x4x360x216xf16, {order = #NHWC, strides = [6220800, 1, 8640, 8]}, @DDR>
    // CHECK:    [[OUTPUT_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <622080> -> memref<1x4x360x216xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:      VPURT.Task
    // CHECK:      VPUIP.NNDMA
    // CHECK-SAME    inputs([[INPUT_DDR_BUF0]] : memref<1x4x360x216xf16, {order = #NHWC, strides = [6220800, 1, 8640, 8]}, @DDR>)
    // CHECK-SAME    outputs([[OUTPUT_BUF]] : memref<1x4x360x216xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:      return [[OUTPUT_BUF]] : memref<1x4x360x216xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!Input_DDR = memref<1x32x64x64x1xf16, {order = #NCDHW, strides = [2097152, 65536, 512, 2, 1]}, @DDR>
!Output_CMX = memref<1x32x64x64x1xf16, #NCDHW, [@CMX_NN, 0]>

func.func @DoNotSplit5DNNDMAWithLevel3Striding(%input: !Input_DDR, %output: !Output_CMX) -> !Output_CMX {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <18662408> -> !Input_DDR
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <622080> -> !Output_CMX

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : memref<1x32x64x64x1xf16, {order = #NCDHW, strides = [2097152, 65536, 512, 2, 1]}, @DDR>) outputs(%1 : memref<1x32x64x64x1xf16, #NCDHW, [@CMX_NN, 0]>) -> memref<1x32x64x64x1xf16, #NCDHW, [@CMX_NN, 0]>
    }

    return %1: !Output_CMX

    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    [[INPUT_DDR_BUF0:%.*]] = VPURT.DeclareBuffer <DDR> <18662408> -> memref<1x32x64x64x1xf16, {order = #NCDHW, strides = [2097152, 65536, 512, 2, 1]}, @DDR>
    // CHECK:    [[OUTPUT_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <622080> -> memref<1x32x64x64x1xf16, [@CMX_NN, 0]>

    // CHECK:      VPURT.Task
    // CHECK:      VPUIP.NNDMA
    // CHECK-SAME    inputs([[INPUT_DDR_BUF0]] : memref<1x32x64x64x1xf16, {order = #NCDHW, strides = [2097152, 65536, 512, 2, 1]}, @DDR>)
    // CHECK-SAME    outputs([[OUTPUT_BUF]] : memref<1x32x64x64x1xf16, [@CMX_NN, 0]>)

    // CHECK:      return [[OUTPUT_BUF]] : memref<1x32x64x64x1xf16, [@CMX_NN, 0]>
}
