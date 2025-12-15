//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-segmented-dma  %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @FuseSimpleConstant
func.func @FuseSimpleConstant(%arg0: !DummyT) -> !DummyT {
    %cst0 = const.Declare memref<320x1x1x4xsi32> = dense<0> : tensor<320x1x1x4xsi32>
    %cst1 = const.Declare memref<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>
    %cst2 = const.Declare memref<320x1x1x4xsi32> = dense<2> : tensor<320x1x1x4xsi32>
    // CHECK:       [[CST_012:%.+]] = const.Declare memref<3x320x1x1x4xsi32> = dense<0> : tensor<320x1x1x4xsi32>, [#const.Fuse<tensor<3x320x1x1x4xsi32>, 
    // CHECK-SAME:   constants = <[dense<0> : tensor<320x1x1x4xsi32>, 
    // CHECK-SAME:   dense<1> : tensor<320x1x1x4xsi32>,
    // CHECK-SAME:   dense<2> : tensor<320x1x1x4xsi32>]>>]
    

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 0]>
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 1]>
    %cmx2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_012:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3x320x1x1x4xsi32, {order = #NCDHW, strides = [524288, 4, 4, 4, 1]}, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%cst0 : memref<320x1x1x4xsi32>) outputs(%cmx0 : memref<320x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%cst1 : memref<320x1x1x4xsi32>) outputs(%cmx1 : memref<320x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%cst2 : memref<320x1x1x4xsi32>) outputs(%cmx2 : memref<320x1x1x4xsi32, [@CMX_NN, 2]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 2]>
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[CST_012]] : memref<3x320x1x1x4xsi32>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_012]] :  memref<3x320x1x1x4xsi32, {order = #NCDHW, strides = [524288, 4, 4, 4, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->  memref<3x320x1x1x4xsi32, {order = #NCDHW, strides = [524288, 4, 4, 4, 1]}, [@CMX_NN, 0]>

    return %arg0 : !DummyT
}

//
// -----
//

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @FuseCompactBuffer2BufferDma
func.func @FuseCompactBuffer2BufferDma(%arg0: !DummyT) -> !DummyT {
    %ddr0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x48x18x56xf16, @DDR>
    %ddr1 = VPURT.DeclareBuffer <DDR> <96768> -> memref<1x48x18x56xf16, @DDR>
    %ddr2 = VPURT.DeclareBuffer <DDR> <193536> -> memref<1x48x18x56xf16, @DDR>
    // CHECK:       [[BUFFER_DDR_012:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<3x1x48x18x56xf16, @DDR>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x48x18x56xf16, [@CMX_NN, 0]>
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x48x18x56xf16, [@CMX_NN, 1]>
    %cmx2 = VPURT.DeclareBuffer <CMX_NN> [2] <512> -> memref<1x48x18x56xf16, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_012:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<3x1x48x18x56xf16, {order = #NCDHW, strides = [1048576, 48384, 1008, 56, 1]}, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%ddr0 : memref<1x48x18x56xf16, @DDR>) outputs(%cmx0 : memref<1x48x18x56xf16, [@CMX_NN, 0]>) -> memref<1x48x18x56xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%ddr1 : memref<1x48x18x56xf16, @DDR>) outputs(%cmx1 : memref<1x48x18x56xf16, [@CMX_NN, 1]>) -> memref<1x48x18x56xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%ddr2 : memref<1x48x18x56xf16, @DDR>) outputs(%cmx2 : memref<1x48x18x56xf16, [@CMX_NN, 2]>) -> memref<1x48x18x56xf16, [@CMX_NN, 2]>
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[BUFFER_DDR_012]] : memref<3x1x48x18x56xf16, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_012]] :  memref<3x1x48x18x56xf16, {order = #NCDHW, strides = [1048576, 48384, 1008, 56, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->  memref<3x1x48x18x56xf16, {order = #NCDHW, strides = [1048576, 48384, 1008, 56, 1]}, [@CMX_NN, 0]>
    return %arg0 : !DummyT
}
