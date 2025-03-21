//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-segmented-dma  %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @FuseSimpleConstant
func.func @FuseSimpleConstant(%arg0: !DummyT) -> !DummyT {
    %cst0 = const.Declare memref<320x1x1x4xsi32> = dense<0> : tensor<320x1x1x4xsi32>
    %cst1 = const.Declare memref<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>
    %cst2 = const.Declare memref<320x1x1x4xsi32> = dense<2> : tensor<320x1x1x4xsi32>
    %cst3 = const.Declare memref<320x1x1x4xsi32> = dense<3> : tensor<320x1x1x4xsi32>
    %cst4 = const.Declare memref<320x1x1x4xsi32> = dense<4> : tensor<320x1x1x4xsi32>
    %cst5 = const.Declare memref<320x1x1x4xsi32> = dense<5> : tensor<320x1x1x4xsi32>
    // CHECK:       [[CST_01:%.+]] = const.Declare memref<2x320x1x1x4xsi32> = dense<0> : tensor<320x1x1x4xsi32>, [#const.Fuse<tensor<2x320x1x1x4xsi32>, constants = <[dense<0> : tensor<320x1x1x4xsi32>, dense<1> : tensor<320x1x1x4xsi32>]>>]
    // CHECK:       [[CST_23:%.+]] = const.Declare memref<2x320x1x1x4xsi32> = dense<2> : tensor<320x1x1x4xsi32>, [#const.Fuse<tensor<2x320x1x1x4xsi32>, constants = <[dense<2> : tensor<320x1x1x4xsi32>, dense<3> : tensor<320x1x1x4xsi32>]>>]
    // CHECK:       [[CST_45:%.+]] = const.Declare memref<2x320x1x1x4xsi32> = dense<4> : tensor<320x1x1x4xsi32>, [#const.Fuse<tensor<2x320x1x1x4xsi32>, constants = <[dense<4> : tensor<320x1x1x4xsi32>, dense<5> : tensor<320x1x1x4xsi32>]>>]

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 0]>
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 1]>
    %cmx2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 2]>
    %cmx3 = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 3]>
    %cmx4 = VPURT.DeclareBuffer <CMX_NN> [4] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 4]>
    %cmx5 = VPURT.DeclareBuffer <CMX_NN> [5] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 5]>
    // CHECK:       [[BUFFER_CMX_01:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [524288, 4, 4, 4, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[BUFFER_CMX_23:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [2097152, 4, 4, 4, 1]}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_45:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <0> -> memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [8388608, 4, 4, 4, 1]}, [@CMX_NN, 4]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%cst0 : memref<320x1x1x4xsi32>) outputs(%cmx0 : memref<320x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%cst1 : memref<320x1x1x4xsi32>) outputs(%cmx1 : memref<320x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 1 : i64, port = 0 : i64} inputs(%cst2 : memref<320x1x1x4xsi32>) outputs(%cmx2 : memref<320x1x1x4xsi32, [@CMX_NN, 2]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 2]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 1 : i64, port = 1 : i64} inputs(%cst3 : memref<320x1x1x4xsi32>) outputs(%cmx3 : memref<320x1x1x4xsi32, [@CMX_NN, 3]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 3]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 2 : i64, port = 0 : i64} inputs(%cst4 : memref<320x1x1x4xsi32>) outputs(%cmx4 : memref<320x1x1x4xsi32, [@CMX_NN, 4]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 4]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 2 : i64, port = 1 : i64} inputs(%cst5 : memref<320x1x1x4xsi32>) outputs(%cmx5 : memref<320x1x1x4xsi32, [@CMX_NN, 5]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 5]>
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[CST_01]] : memref<2x320x1x1x4xsi32>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_01]] :  memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [524288, 4, 4, 4, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->  memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [524288, 4, 4, 4, 1]}, [@CMX_NN, 0]>

    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 1 : i64} inputs([[CST_23]] : memref<2x320x1x1x4xsi32>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_23]] :  memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [2097152, 4, 4, 4, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          ->  memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [2097152, 4, 4, 4, 1]}, [@CMX_NN, 2]>

    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[CST_45]] : memref<2x320x1x1x4xsi32>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_45]] :  memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [8388608, 4, 4, 4, 1]}, [@CMX_NN, 4]>)
    // CHECK-SAME:          ->  memref<2x320x1x1x4xsi32, {order = #NCDHW, strides = [8388608, 4, 4, 4, 1]}, [@CMX_NN, 4]>

    return %arg0 : !DummyT
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Weights = memref<128x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
!WeightsCmx0 = memref<128x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
!WeightsCmx1 = memref<128x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 1]>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @FuseSwizzledConstant
func.func @FuseSwizzledConstant(%arg0: !DummyT) -> !DummyT {
    %cst0 = const.Declare !Weights = dense<1.0> : tensor<384x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [128, 64, 1, 1]>, #const.SwizzleConstant<5 : i64, 4 : i64>]
    %cst1 = const.Declare !Weights = dense<2.0> : tensor<384x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [128, 64, 1, 1]>, #const.SwizzleConstant<5 : i64, 4 : i64>]
    // CHECK:       [[CST_01:%.+]] = const.Declare memref<2x128x64x1x1xf16, {order = #GNHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}> = 
    // CHECK-SAME:    dense<1.000000e+00> : tensor<384x64x1x1xf32>, [
    // CHECK-SAME:    #const.Fuse<tensor<2x128x64x1x1xf16, {order = #GNHWC}>, 
    // CHECK-SAME:      constants = <[dense<1.000000e+00> : tensor<384x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [128, 64, 1, 1]>, #const.SwizzleConstant<5 : i64, 4 : i64>], 
    // CHECK-SAME:      dense<2.000000e+00> : tensor<384x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [128, 64, 1, 1]>, #const.SwizzleConstant<5 : i64, 4 : i64>]]>>]
     
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> !WeightsCmx0
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> !WeightsCmx1
    // CHECK:       [[BUFFER_CMX_01:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<2x128x64x1x1xf16, {order = #GNHWC, strides = [1048576, 64, 1, 64, 64], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%cst0 : !Weights) outputs(%cmx0 : !WeightsCmx0) -> !WeightsCmx0
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%cst1 : !Weights) outputs(%cmx1 : !WeightsCmx1) -> !WeightsCmx1
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[CST_01]] : memref<2x128x64x1x1xf16, {order = #GNHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_01]] :  memref<2x128x64x1x1xf16, {order = #GNHWC, strides = [1048576, 64, 1, 64, 64], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->  memref<2x128x64x1x1xf16, {order = #GNHWC, strides = [1048576, 64, 1, 64, 64], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
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
    // CHECK:       [[BUFFER_DDR_01:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<2x1x48x18x56xf16, @DDR>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x48x18x56xf16, [@CMX_NN, 0]>
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x48x18x56xf16, [@CMX_NN, 1]>
    // CHECK:       [[BUFFER_CMX_01:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<2x1x48x18x56xf16, {order = #NCDHW, strides = [1048576, 48384, 1008, 56, 1]}, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%ddr0 : memref<1x48x18x56xf16, @DDR>) outputs(%cmx0 : memref<1x48x18x56xf16, [@CMX_NN, 0]>) -> memref<1x48x18x56xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%ddr1 : memref<1x48x18x56xf16, @DDR>) outputs(%cmx1 : memref<1x48x18x56xf16, [@CMX_NN, 1]>) -> memref<1x48x18x56xf16, [@CMX_NN, 1]>
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[BUFFER_DDR_01]] : memref<2x1x48x18x56xf16, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_01]] :  memref<2x1x48x18x56xf16, {order = #NCDHW, strides = [1048576, 48384, 1008, 56, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->  memref<2x1x48x18x56xf16, {order = #NCDHW, strides = [1048576, 48384, 1008, 56, 1]}, [@CMX_NN, 0]>
    return %arg0 : !DummyT
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


!DummyT = memref<1x3x224x224xf16, @DDR>
!DdrT = memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
!CMX0T = memref<1x3x56x224xf16, [@CMX_NN, 0]>
!CMX1T = memref<1x3x56x224xf16, [@CMX_NN, 1]>

// CHECK-LABEL: @FuseStridedBuffer2BufferDma
func.func @FuseStridedBuffer2BufferDma(%arg0: !DummyT) -> !DummyT {
    %ddr0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !DdrT
    %ddr1 = VPURT.DeclareBuffer <NetworkInput> [0] <25088> -> !DdrT
    // CHECK:       [[BUFFER_DDR_01:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<2x1x3x56x224xf16, 
    // CHECK-SAME:    {order = #NCDHW,  strides = [12544, 150528, 50176, 224, 1]}, @DDR>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> !CMX0T
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> !CMX1T
    // CHECK:       [[BUFFER_CMX_01:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<2x1x3x56x224xf16, 
    // CHECK-SAME:    {order = #NCDHW, strides = [1048576, 37632, 12544, 224, 1]}, [@CMX_NN, 0]>
    

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%ddr0 : !DdrT) outputs(%cmx0 : !CMX0T) -> !CMX0T
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%ddr1 : !DdrT) outputs(%cmx1 : !CMX1T) -> !CMX1T
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[BUFFER_DDR_01]] : memref<2x1x3x56x224xf16, {order = #NCDHW, strides = [12544, 150528, 50176, 224, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_01]] :   memref<2x1x3x56x224xf16, {order = #NCDHW, strides = [1048576, 37632, 12544, 224, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->   memref<2x1x3x56x224xf16, {order = #NCDHW, strides = [1048576, 37632, 12544, 224, 1]}, [@CMX_NN, 0]>

    return %arg0 : !DummyT
}

//
// -----
//

!DummyT = memref<1x3x224x224xf16, @DDR>
!qElemType = !quant.uniform<u8<0:254>:f16:1, {1.0:127,2.0:127}>

// CHECK: [[Q_ELEM_TYPE:!.+]] = !quant.uniform<u8<0:254>:f16:2, {1.000000e+00:127,2.000000e+00:127}>

// CHECK: @FusePerAxisQuantizedCompactBuffer2BufferDma
func.func @FusePerAxisQuantizedCompactBuffer2BufferDma(%arg0: !DummyT) -> !DummyT {
    %ddr0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x18x56x!qElemType, @DDR>
    %ddr1 = VPURT.DeclareBuffer <DDR> <2016> -> memref<1x2x18x56x!qElemType, @DDR>
    // CHECK:       [[BUFFER_DDR_01:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<2x1x2x18x56x[[Q_ELEM_TYPE]], @DDR>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x2x18x56x!qElemType, [@CMX_NN, 0]>
    %cmx1 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x2x18x56x!qElemType, [@CMX_NN, 1]>
    // CHECK:       [[BUFFER_CMX_01:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> ->  memref<2x1x2x18x56x[[Q_ELEM_TYPE]], {order = #NCDHW, strides = [2097152, 2016, 1008, 56, 1]}, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 0 : i64} inputs(%ddr0 : memref<1x2x18x56x!qElemType, @DDR>) outputs(%cmx0 : memref<1x2x18x56x!qElemType, [@CMX_NN, 0]>) -> memref<1x2x18x56x!qElemType, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {fusionId = 0 : i64, port = 1 : i64} inputs(%ddr1 : memref<1x2x18x56x!qElemType, @DDR>) outputs(%cmx1 : memref<1x2x18x56x!qElemType, [@CMX_NN, 1]>) -> memref<1x2x18x56x!qElemType, [@CMX_NN, 1]>
    }
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:       VPUIP.NNDMA {port = 0 : i64} inputs([[BUFFER_DDR_01]] : memref<2x1x2x18x56x[[Q_ELEM_TYPE]], @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_01]] :  memref<2x1x2x18x56x[[Q_ELEM_TYPE]], {order = #NCDHW, strides = [2097152, 2016, 1008, 56, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          ->  memref<2x1x2x18x56x[[Q_ELEM_TYPE]], {order = #NCDHW, strides = [2097152, 2016, 1008, 56, 1]}, [@CMX_NN, 0]>
    return %arg0 : !DummyT
}
