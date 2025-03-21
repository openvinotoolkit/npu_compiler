//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --canonicalize --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @Eliminate
func.func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [4, 4] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.AffineReshape
    // CHECK:     return %arg0
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<16xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0, 1]], shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16xf32>, [#const.Reshape<[4, 4]>]
    // CHECK-NOT:   IE.AffineReshape
    // CHECK:       return [[VAL0]]
}

// -----

!qElemType = !quant.uniform<i8<-127:127>:f16:1, {0.002174197219488189,0.0013370063361220473}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.002174197219488189,0.0013370063361220473}>
// CHECK-LABEL: @ConstFoldQuant
func.func @ConstFoldQuant(%arg0 : tensor<1x48x27x27xf16> ) -> tensor<2x48x5x5xf16> {
    %cst = const.Declare tensor<1x2x48x25x!qElemType> = dense<1> : tensor<1x2x48x25xsi8>, [#const.CastElemType<!qElemType>]
    %1 = IE.AffineReshape(%cst) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [2, 48, 5, 5]} : tensor<1x2x48x25x!qElemType> -> tensor<2x48x5x5x!qElemType1>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<2x48x5x5x!qElemType1> -> tensor<2x48x5x5xf16>
    return %2 : tensor<2x48x5x5xf16>

    // CHECK-NOT: IE.AffineReshape
    // CHECK:     [[CST:%.+]] = const.Declare tensor<2x48x5x5x!qElemType> = dense<1> : tensor<1x2x48x25xsi8>
    // CHECK-SAME:      #const.CastElemType<!qElemType1>, #const.ChangeShapeAndElemType<[2, 48, 5, 5], !qElemType>
    // CHECK:     [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<2x48x5x5x!qElemType> -> tensor<2x48x5x5xf16>
    // CHECK:     return [[DEQUANT]]
}

// -----

// CHECK-LABEL: @FuseWithReshape
func.func @FuseWithReshape(%arg0: tensor<15x4xf32>) -> tensor<20x3x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [20, 3] } : tensor<15x4xf32> -> tensor<20x3xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0, 1], [2]], shape_value = [20, 3, 1] } : tensor<20x3xf32> -> tensor<20x3x1xf32>

    return %1 : tensor<20x3x1xf32>

    // CHECK:     [[RESHAPE_0:%.+]] = IE.Reshape(%arg0) {shape_value = [20, 3, 1]} : tensor<15x4xf32> -> tensor<20x3x1xf32>
    // CHECK: return [[RESHAPE_0]] : tensor<20x3x1xf32>
}

// -----

// CHECK-LABEL: @ReshapeChangedToAffineReshape
func.func @ReshapeChangedToAffineReshape(%arg0: tensor<15x2xf32>) -> tensor<10x3x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [30, 1] } : tensor<15x2xf32> -> tensor<30x1xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0, 1], [2]], shape_value = [10, 3, 1] } : tensor<30x1xf32> -> tensor<10x3x1xf32>

    return %1 : tensor<10x3x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0, 1]], shape_value = [30, 1]} : tensor<15x2xf32> -> tensor<30x1xf32>
    // CHECK: [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2]], shape_value = [10, 3, 1]} : tensor<30x1xf32> -> tensor<10x3x1xf32>

    // CHECK: return [[VAL1]] : tensor<10x3x1xf32>
}

// -----

// CHECK-LABEL: @FuseWithSqueeze
func.func @FuseWithSqueeze(%arg0: tensor<15x2x1xf32>) -> tensor<30xf32> {
    %0 = IE.Squeeze(%arg0) { axes_value = [2] } : tensor<15x2x1xf32> -> tensor<15x2xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf32> -> tensor<30xf32>

    return %1 : tensor<30xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0]], shape_value = [30]} : tensor<15x2x1xf32> -> tensor<30xf32>
    // CHECK: return [[VAL0]] : tensor<30xf32>
}

// -----

// CHECK-LABEL: @FuseWithUnsqueeze
func.func @FuseWithUnsqueeze(%arg0: tensor<15x2xf32>) -> tensor<1x30xf32> {
    %0 = IE.Unsqueeze(%arg0) { axes_value = [0, 1] } : tensor<15x2xf32> -> tensor<1x1x15x2xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0], [0], [1], [1]], shape_value = [1, 30] } : tensor<1x1x15x2xf32> -> tensor<1x30xf32>

    return %1 : tensor<1x30xf32>

    // CHECK: [[VAL0:%.+]] = IE.Reshape(%arg0) {shape_value = [1, 30]} : tensor<15x2xf32> -> tensor<1x30xf32>
    // CHECK: return [[VAL0]] : tensor<1x30xf32>
}

// -----

// CHECK-LABEL: @FuseWithAffineReshape
func.func @FuseWithAffineReshape(%arg0: tensor<1x512x64x64xf16>) -> tensor<1x1x512x4096xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 512, 4096]} : tensor<1x512x64x64xf16> -> tensor<1x512x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x4096xf16> -> tensor<1x1x512x4096xf16>
    return %1 : tensor<1x1x512x4096xf16>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x64x64xf16> -> tensor<1x1x512x4096xf16>
    // CHECK: return [[VAL0]] : tensor<1x1x512x4096xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @NotFuseWithAffineReshapeWithNonNCHW
func.func @NotFuseWithAffineReshapeWithNonNCHW(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x1x512x4096xf16, {order = #NCWH}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 512, 4096]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096xf16, {order = #map}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x4096xf16, {order = #map}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>

    return %1 : tensor<1x1x512x4096xf16, {order = #NCWH}>
    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 512, 4096]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096xf16, {order = #map}>
    // CHECK: [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x4096xf16, {order = #map}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>
    // CHECK: return [[VAL1]] : tensor<1x1x512x4096xf16, {order = #NCWH}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @FuseAffineReshape
func.func @FuseAffineReshape(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 512, 4096, 1]} : tensor<1x1x512x4096xf16, {order = #NCWH}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    return %1 : tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 4096, 1]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK: return [[VAL0]] : tensor<1x512x4096x1xf16, {order = #NHWC}>

}

// -----

// CHECK-LABEL: @ReshapeNotConvertAffineReshape
func.func @ReshapeNotConvertAffineReshape(%arg0: tensor<2x3x15x4xf32>) -> tensor<2x3x30x2xf32> {
    %0 = IE.Reshape(%arg0) {shape_value = [2, 3, 30, 2]} : tensor<2x3x15x4xf32> -> tensor<2x3x30x2xf32>
    return %0 : tensor<2x3x30x2xf32>

    // CHECK: [[VAL0:%.+]] = IE.Reshape(%arg0) {shape_value = [2, 3, 30, 2]} : tensor<2x3x15x4xf32> -> tensor<2x3x30x2xf32>
    // CHECK: return [[VAL0]] : tensor<2x3x30x2xf32>
}

// CHECK-LABEL: @ReshapeConvertAffineReshapeWithOutputLastDim1
func.func @ReshapeConvertAffineReshapeWithOutputLastDim1(%arg0: tensor<2x3x15x4xf32>) -> tensor<2x3x60x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [2,3,60,1] } : tensor<2x3x15x4xf32> -> tensor<2x3x60x1xf32>

    return %0 : tensor<2x3x60x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [2, 3, 60, 1]} : tensor<2x3x15x4xf32> -> tensor<2x3x60x1xf32>

    // CHECK: return [[VAL0]] : tensor<2x3x60x1xf32>
}

// -----

// CHECK-LABEL: @ReshapeConvertAffineReshapeWithOutputLast2Dim1
func.func @ReshapeConvertAffineReshapeWithOutputLast2Dim1(%arg0: tensor<2x3x15x4xf32>) -> tensor<2x180x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [2,180,1,1] } : tensor<2x3x15x4xf32> -> tensor<2x180x1x1xf32>

    return %0 : tensor<2x180x1x1xf32>

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1, 2, 3]], shape_value = [2, 180, 1, 1]} : tensor<2x3x15x4xf32> -> tensor<2x180x1x1xf32>

    // CHECK: return [[VAL0]] : tensor<2x180x1x1xf32>
}

// CHECK-LABEL: @ReshapeConvertAffineReshapeWithOutputLast3Dim1
func.func @ReshapeConvertAffineReshapeWithOutputLast3Dim1(%arg0: tensor<2x3x15x4xf32>) -> tensor<360x1x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [360,1,1,1] } : tensor<2x3x15x4xf32> -> tensor<360x1x1x1xf32>

    return %0 : tensor<360x1x1x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [360, 1, 1, 1]} : tensor<2x3x15x4xf32> -> tensor<360x1x1x1xf32>

    // CHECK: return [[VAL0]] : tensor<360x1x1x1xf32>
}

// -----

// CHECK-LABEL: @ReshapeConvertAffineReshapeWithOutputLast3Dim1InputSize2
func.func @ReshapeConvertAffineReshapeWithOutputLast3Dim1InputSize2(%arg0: tensor<15x2xf32>) -> tensor<30x1x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [30,1,1,1] } : tensor<15x2xf32> -> tensor<30x1x1x1xf32>

    return %0 : tensor<30x1x1x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0, 1, 2, 3]], shape_value = [30, 1, 1, 1]} : tensor<15x2xf32> -> tensor<30x1x1x1xf32>

    // CHECK: return [[VAL0]] : tensor<30x1x1x1xf32>
}

// -----

// CHECK-LABEL: @ReshapeConvertAffineReshapeWithOutputLast3Dim1InputSize3
func.func @ReshapeConvertAffineReshapeWithOutputLast3Dim1InputSize3(%arg0: tensor<3x5x2xf32>) -> tensor<30x1x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [30,1,1,1] } : tensor<3x5x2xf32> -> tensor<30x1x1x1xf32>

    return %0 : tensor<30x1x1x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0, 1, 2, 3]], shape_value = [30, 1, 1, 1]} : tensor<3x5x2xf32> -> tensor<30x1x1x1xf32>

    // CHECK: return [[VAL0]] : tensor<30x1x1x1xf32>
}
