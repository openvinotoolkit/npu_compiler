//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="vpu-arch=%arch%" --split-dma-to-balance-load  %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitBuffer2BufferDma
func.func @SplitBuffer2BufferDma(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x48x18x56xf16, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [2] <100000> -> memref<1x48x18x56xf16, [@CMX_NN, 2]>
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%2 : memref<1x48x18x56xf16, @DDR>) outputs(%3 : memref<1x48x18x56xf16, [@CMX_NN, 2]>) -> memref<1x48x18x56xf16, [@CMX_NN, 2]>
    }
    // CHECK:       [[BUFFER_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, @DDR>
    // CHECK:       [[BUFFER_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <48384> -> memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, @DDR>

    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <100000> -> memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <148384> -> memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[BUFFER_DDR_0]] : memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[BUFFER_DDR_1]] : memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x24x18x56xf16, {order = #NCHW, strides = [48384, 1008, 56, 1]}, [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Weights = memref<1x1x1x168960xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
!WeightsCmx = memref<1x1x1x168960xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitFusedSwizzledConstant
func.func @SplitFusedSwizzledConstant(%arg0: !DummyT) -> !DummyT {
    // The below content shape is not the same one in IR as you cann't find its final shape in const declare op IR
    // This final shape 168960x1x1x1 is catched from contentType.getShape() in the code
    // To make the case clear, I use the final shape for it directly
    %cst = const.Declare !Weights = dense<1.0> : tensor<168960x1x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<168960x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 84480]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<168960x1x1x1xf32>, [#const.SubView<[0, 0, 0, 84480], [1, 1, 1, 84480]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !WeightsCmx
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <168960> -> memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : !Weights) outputs(%2 : !WeightsCmx) -> !WeightsCmx
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x1x1x84480xf16, {order = #NHWC, strides = [168960, 1, 168960, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>


    return %arg0 : !DummyT
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitStridedBuffer2BufferDma
func.func @SplitStridedBuffer2BufferDma(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <DDR> <100000> -> memref<1x512x5x80xf16, {order = #NHWC, strides = [29491200, 1, 40960, 512]}, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x512x5x80xf16, #NHWC, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <100000> -> memref<1x512x2x80xf16, {order = #NHWC, strides = [29491200, 1, 40960, 512]}, @DDR>
    // CHECK:       [[BUFFER_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <263840> -> memref<1x512x3x80xf16, {order = #NHWC, strides = [29491200, 1, 40960, 512]}, @DDR>

    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x512x2x80xf16, {order = #NHWC, strides = [204800, 1, 40960, 512]}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <163840> -> memref<1x512x3x80xf16, {order = #NHWC, strides = [204800, 1, 40960, 512]}, [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%2 : memref<1x512x5x80xf16, {order = #NHWC, strides = [29491200, 1, 40960, 512]}, @DDR>) outputs(%3 : memref<1x512x5x80xf16, #NHWC, [@CMX_NN, 2]>) -> memref<1x512x5x80xf16, #NHWC, [@CMX_NN, 2]>
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[BUFFER_DDR_0]] : memref<1x512x2x80xf16, {order = #NHWC, strides = [29491200, 1, 40960, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<1x512x2x80xf16, {order = #NHWC, strides = [204800, 1, 40960, 512]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x512x2x80xf16, {order = #NHWC, strides = [204800, 1, 40960, 512]}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[BUFFER_DDR_1]] : memref<1x512x3x80xf16, {order = #NHWC, strides = [29491200, 1, 40960, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<1x512x3x80xf16, {order = #NHWC, strides = [204800, 1, 40960, 512]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x512x3x80xf16, {order = #NHWC, strides = [204800, 1, 40960, 512]}, [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitSimpleConstant
func.func @SplitSimpleConstant(%arg0: !DummyT) -> !DummyT {
    %cst = const.Declare memref<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<160x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [160, 1, 1, 4]>]
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<160x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>, [#const.SubView<[160, 0, 0, 0], [160, 1, 1, 4]>]

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<320x1x1x4xsi32, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<160x1x1x4xsi32, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <2560> -> memref<160x1x1x4xsi32, [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : memref<320x1x1x4xsi32>) outputs(%2 : memref<320x1x1x4xsi32, [@CMX_NN, 2]>) -> memref<320x1x1x4xsi32, [@CMX_NN, 2]>
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<160x1x1x4xsi32>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<160x1x1x4xsi32, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<160x1x1x4xsi32, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<160x1x1x4xsi32>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<160x1x1x4xsi32, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<160x1x1x4xsi32, [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitFused5DWeights
func.func @SplitFused5DWeights(%arg0: !DummyT) -> !DummyT {
    %cst = const.Declare memref<1x1x1x270336xf16> = dense<1> : tensor<2x512x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x270336xf16>,
            weightsTable = <dense<1> : tensor<2x512x1x1x4xsi32>>,
            weights = <dense<1.000000e+00> : tensor<2x512x256x1x1xf16, {order = #GNHWC}>>>]
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}> = dense<1> : tensor<2x512x1x1x4xsi32>, [#const.FuseWeights<tensor<1x1x1x135168xf16>, weightsTable = <dense<1> : tensor<2x512x1x1x4xsi32>>, weights = <dense<1.000000e+00> : tensor<2x512x256x1x1xf16, {order = #GNHWC}>, [#const.SubView<[0, 0, 0, 0, 0], [1, 496, 256, 1, 1]>]>>]
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}> = dense<1> : tensor<2x512x1x1x4xsi32>, [#const.FuseWeights<tensor<1x1x1x135168xf16>, weights = <dense<1.000000e+00> : tensor<2x512x256x1x1xf16, {order = #GNHWC}>, [#const.Reshape<[1, 1, 1, 1, 262144]>, #const.SubView<[0, 0, 0, 0, 126976], [1, 1, 1, 1, 135168]>]>>]
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x1x1x270336xf16, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <270336> -> memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}, [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : memref<1x1x1x270336xf16>) outputs(%2 : memref<1x1x1x270336xf16, [@CMX_NN, 2]>) -> memref<1x1x1x270336xf16, [@CMX_NN, 2]>
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x1x1x135168xf16, {order = #NCHW, strides = [270336, 270336, 270336, 1]}, [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

// Quantization scale is index of channel for simplicity
!qElemType = !quant.uniform<u8<0:254>:f16:0, {1.0:127,2.0:127,3.0:127,4.0:127,5.0:127,6.0:127,7.0:127,8.0:127,9.0:127,10.0:127,11.0:127,12.0:127,13.0:127,14.0:127,15.0:127,16.0:127,17.0:127,18.0:127,19.0:127,20.0:127,21.0:127,22.0:127,23.0:127,24.0:127,25.0:127,26.0:127,27.0:127,28.0:127,29.0:127,30.0:127,31.0:127,32.0:127,33.0:127,34.0:127,35.0:127,36.0:127,37.0:127,38.0:127,39.0:127,40.0:127,41.0:127,42.0:127,43.0:127,44.0:127,45.0:127,46.0:127,47.0:127,48.0:127,49.0:127,50.0:127,51.0:127,52.0:127,53.0:127,54.0:127,55.0:127,56.0:127,57.0:127,58.0:127,59.0:127,60.0:127,61.0:127,62.0:127,63.0:127,64.0:127,65.0:127,66.0:127,67.0:127,68.0:127,69.0:127,70.0:127,71.0:127,72.0:127,73.0:127,74.0:127,75.0:127,76.0:127,77.0:127,78.0:127,79.0:127,80.0:127,81.0:127,82.0:127,83.0:127,84.0:127,85.0:127,86.0:127,87.0:127,88.0:127,89.0:127,90.0:127,91.0:127,92.0:127,93.0:127,94.0:127,95.0:127,96.0:127,97.0:127,98.0:127,99.0:127,100.0:127,101.0:127,102.0:127,103.0:127,104.0:127,105.0:127,106.0:127,107.0:127,108.0:127,109.0:127,110.0:127,111.0:127,112.0:127,113.0:127,114.0:127,115.0:127,116.0:127,117.0:127,118.0:127,119.0:127,120.0:127,121.0:127,122.0:127,123.0:127,124.0:127,125.0:127,126.0:127,127.0:127,128.0:127,129.0:127,130.0:127,131.0:127,132.0:127,133.0:127,134.0:127,135.0:127,136.0:127,137.0:127,138.0:127,139.0:127,140.0:127,141.0:127,142.0:127,143.0:127,144.0:127,145.0:127,146.0:127,147.0:127,148.0:127,149.0:127,150.0:127,151.0:127,152.0:127,153.0:127,154.0:127,155.0:127,156.0:127,157.0:127,158.0:127,159.0:127,160.0:127,161.0:127,162.0:127,163.0:127,164.0:127,165.0:127,166.0:127,167.0:127,168.0:127,169.0:127,170.0:127,171.0:127,172.0:127,173.0:127,174.0:127,175.0:127,176.0:127,177.0:127,178.0:127,179.0:127,180.0:127,181.0:127,182.0:127,183.0:127,184.0:127,185.0:127,186.0:127,187.0:127,188.0:127,189.0:127,190.0:127,191.0:127,192.0:127,193.0:127,194.0:127,195.0:127,196.0:127,197.0:127,198.0:127,199.0:127,200.0:127,201.0:127,202.0:127,203.0:127,204.0:127,205.0:127,206.0:127,207.0:127,208.0:127,209.0:127,210.0:127,211.0:127,212.0:127,213.0:127,214.0:127,215.0:127,216.0:127,217.0:127,218.0:127,219.0:127,220.0:127,221.0:127,222.0:127,223.0:127,224.0:127,225.0:127,226.0:127,227.0:127,228.0:127,229.0:127,230.0:127,231.0:127,232.0:127,233.0:127,234.0:127,235.0:127,236.0:127,237.0:127,238.0:127,239.0:127,240.0:127,241.0:127,242.0:127,243.0:127,244.0:127,245.0:127,246.0:127,247.0:127,248.0:127,249.0:127,250.0:127,251.0:127,252.0:127,253.0:127,254.0:127,255.0:127,256.0:127,257.0:127,258.0:127,259.0:127,260.0:127,261.0:127,262.0:127,263.0:127,264.0:127,265.0:127,266.0:127,267.0:127,268.0:127,269.0:127,270.0:127,271.0:127,272.0:127,273.0:127,274.0:127,275.0:127,276.0:127,277.0:127,278.0:127,279.0:127,280.0:127,281.0:127,282.0:127,283.0:127,284.0:127,285.0:127,286.0:127,287.0:127,288.0:127,289.0:127,290.0:127,291.0:127,292.0:127,293.0:127,294.0:127,295.0:127,296.0:127,297.0:127,298.0:127,299.0:127,300.0:127,301.0:127,302.0:127,303.0:127,304.0:127,305.0:127,306.0:127,307.0:127,308.0:127,309.0:127,310.0:127,311.0:127,312.0:127,313.0:127,314.0:127,315.0:127,316.0:127,317.0:127,318.0:127,319.0:127,320.0:127}>

!Weights = memref<320x960x1x1x!qElemType>
!WeightsCmx = memref<320x960x1x1x!qElemType, [@CMX_NN, 2]>
!DummyT = memref<1x3x224x224xf16, @DDR>

// Verify that quant type was splitted properly.
// CHECK: [[Q_ELEM_TYPE_0:!.+]] = !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127
// CHECK: [[Q_ELEM_TYPE_1:!.+]] = !quant.uniform<u8<0:254>:f16:0, {1.610000e+02:127,1.620000e+02:127,1.630000e+02:127,1.640000e+02:127

// CHECK: @SplitPerAxisQuantized
func.func @SplitPerAxisQuantized(%arg0: !DummyT) -> !DummyT {
    %cst = const.Declare !Weights = dense<1> : tensor<320x960x1x1xui8>, [#const.CastElemType<!qElemType>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare memref<160x960x1x1x[[Q_ELEM_TYPE_0]]> = dense<1> : tensor<320x960x1x1xui8>, [#const.SubView<[0, 0, 0, 0], [160, 960, 1, 1]>, #const.CastElemType<[[Q_ELEM_TYPE_0]]>]
    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare memref<160x960x1x1x[[Q_ELEM_TYPE_1]]> = dense<1> : tensor<320x960x1x1xui8>, [#const.SubView<[160, 0, 0, 0], [160, 960, 1, 1]>, #const.CastElemType<[[Q_ELEM_TYPE_1]]>]

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !WeightsCmx
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<160x960x1x1x[[Q_ELEM_TYPE_0]], [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <153600> -> memref<160x960x1x1x[[Q_ELEM_TYPE_1]], [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : !Weights) outputs(%2 : !WeightsCmx) -> !WeightsCmx
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<160x960x1x1x[[Q_ELEM_TYPE_0]]>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<160x960x1x1x[[Q_ELEM_TYPE_0]], [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<160x960x1x1x[[Q_ELEM_TYPE_0]], [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<160x960x1x1x[[Q_ELEM_TYPE_1]]>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<160x960x1x1x[[Q_ELEM_TYPE_1]], [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<160x960x1x1x[[Q_ELEM_TYPE_1]], [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Weights = memref<128x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
!WeightsCmx = memref<128x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitSwizzledConstant
func.func @SplitSwizzledConstant(%arg0: !DummyT) -> !DummyT {
    %cst = const.Declare !Weights = dense<1.0> : tensor<384x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [128, 64, 1, 1]>, #const.SwizzleConstant<5 : i64, 4 : i64>]
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<4096x1x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<4096x1x1x1xf16, {order = #NHWC}>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !WeightsCmx
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <8192> -> memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : !Weights) outputs(%2 : !WeightsCmx) -> !WeightsCmx
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>


    return %arg0 : !DummyT
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Weights = memref<1x1x12x1280xi32, #NHWC, @DDR>
!WeightsCmx = memref<1x1x12x1280xi32, #NHWC, [@CMX_NN, 2]>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitThirdAxis
func.func @SplitThirdAxis(%arg0: !DummyT) -> !DummyT {
    %cst = const.Declare !Weights = dense<1> : tensor<1x1x12x1280xi32, {order = #NHWC}>
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, @DDR> = dense<1> : tensor<1x1x12x1280xi32, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [1, 1, 6, 1280]>]
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, @DDR> = dense<1> : tensor<1x1x12x1280xi32, {order = #NHWC}>, [#const.SubView<[0, 0, 6, 0], [1, 1, 6, 1280]>]

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !WeightsCmx
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <30720> -> memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, [@CMX_NN, 2]>


    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : !Weights) outputs(%2 : !WeightsCmx) -> !WeightsCmx
    }
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<1x1x6x1280xi32, {order = #NHWC, strides = [15360, 1, 1280, 1]}, [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparsityMap = memref<512x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
!SparsityMapCmx = memref<512x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>
!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitSwizzledSubbyteConstant
func.func @SplitSwizzledSubbyteConstant(%arg0: !DummyT) -> !DummyT {
    %cst = const.Declare !SparsityMap = dense<1> : tensor<512x512x1x1xui8>, [#const.CastElemType<i1>, #const.Reorder<#NHWC>, #const.SwizzleConstant<5 : i64, 4 : i64>]
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}> = dense<true> : tensor<131072x1x1x1xi1, {order = #NHWC}>
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}> = dense<true> : tensor<131072x1x1x1xi1, {order = #NHWC}>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !SparsityMapCmx
    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>
    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <16384> -> memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs(%cst : !SparsityMap) outputs(%2 : !SparsityMapCmx) -> !SparsityMapCmx
    }

    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CST_0]] : memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_0]] : memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[CST_1]] : memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
    // CHECK-SAME:         outputs([[BUFFER_CMX_1]] : memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>)
    // CHECK-SAME:          -> memref<256x512x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 2]>

    return %arg0 : !DummyT
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPUIP.DistributedBuffer<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitConstantSourceDuplicatedNNDMA
func.func @SplitConstantSourceDuplicatedNNDMA(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = const.Declare memref<1x1x1x368768xf16> = dense<1.0> : tensor<1x1x1x368768xf16>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <164352> -> !DistributedType
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64, split_candidate}
          inputs(%2 : memref<1x1x1x368768xf16>)
          outputs(%3 : !DistributedType) -> !DistributedType
    }
    // CHECK:       [[CST_0:%.+]] = const.Declare memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}> = dense<1.000000e+00> : tensor<1x1x1x368768xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 184384]>]
    // CHECK:       [[CST_1:%.+]] = const.Declare memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}> = dense<1.000000e+00> : tensor<1x1x1x368768xf16>, [#const.SubView<[0, 0, 0, 184384], [1, 1, 1, 184384]>]

    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <164352>
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <533120>
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME{LITERAL}:   {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64}
    // CHECK-SAME:        inputs([[CST_0]] : memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}>)
    // CHECK-SAME:        outputs([[BUFFER_CMX_0]] : !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64}
    // CHECK-SAME:        inputs([[CST_1]] : memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}>)
    // CHECK-SAME:        outputs([[BUFFER_CMX_1]] : !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    return %arg0 : !DummyT
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPUIP.DistributedBuffer<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @SplitBufferSourceDuplicatedNNDMA
func.func @SplitBufferSourceDuplicatedNNDMA(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x368768xf16, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <164352> -> !DistributedType
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64, split_candidate}
          inputs(%2 : memref<1x1x1x368768xf16, @DDR>)
          outputs(%3 : !DistributedType) -> !DistributedType
    }
    // CHECK:       [[BUFFER_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>
    // CHECK:       [[BUFFER_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <368768> -> memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>

    // CHECK:       [[BUFFER_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <164352>
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[BUFFER_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <533120>
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME{LITERAL}:   {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64}
    // CHECK-SAME:        inputs([[BUFFER_DDR_0]] : memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>)
    // CHECK-SAME:        outputs([[BUFFER_CMX_0]] : !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64}
    // CHECK-SAME:        inputs([[BUFFER_DDR_1]] : memref<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>)
    // CHECK-SAME:        outputs([[BUFFER_CMX_1]] : !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x1x1x184384xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN,
    // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384], [1, 1, 1, 184384]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    return %arg0 : !DummyT
}
