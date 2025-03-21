//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --adjust-for-optimized-layers %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForSoftmaxAxisZeroOptNCHW
func.func @AdjustForSoftmaxAxisZeroOptNCHW(%arg0: tensor<1x64x16x1xf16>) -> tensor<1x64x16x1xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x1xf16> -> tensor<1x64x16x1xf16>
    return %0 : tensor<1x64x16x1xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 64, 1, 16]} inputs(%arg0 : tensor<1x64x16x1xf16>) -> tensor<1x64x1x16xf16>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x64x1x16xf16> -> tensor<1x64x1x16xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 64, 16, 1]} inputs([[SOFTMAX]] : tensor<1x64x1x16xf16>) -> tensor<1x64x16x1xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxAxisZeroOptNCHW
func.func @NotAdjustForSoftmaxAxisZeroOptNCHW(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    return %0 : tensor<1x64x16x16xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @AdjustForSoftmaxAxisZeroOptNHWC
func.func @AdjustForSoftmaxAxisZeroOptNHWC(%arg0: tensor<1x1x16x64xf16, {order = #NHWC}>) -> tensor<1x1x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x1x16x64xf16, {order = #NHWC}> -> tensor<1x1x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x1x16x64xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 64, 16, 1]} inputs(%arg0 : tensor<1x1x16x64xf16, {order = #NHWC}>) -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x64x16x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 16, 64]} inputs([[SOFTMAX]] : tensor<1x64x16x1xf16, {order = #NHWC}>) -> tensor<1x1x16x64xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxAxisZeroOptNHWC
func.func @NotAdjustForSoftmaxAxisZeroOptNHWC(%arg0: tensor<1x2x16x64xf16, {order = #NHWC}>) -> tensor<1x2x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNCHW
func.func @NotAdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    return %0 : tensor<1x2x16x16xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNHWC
func.func @NotAdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x2x16x64xf16, {order = #NHWC}>) -> tensor<1x2x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// -----

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL:   @AdjustForGeluMultiShaveOptNCHW
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x2x2xf16>)
func.func @AdjustForGeluMultiShaveOptNCHW(%arg0: tensor<1x2x2x2xf16>) -> tensor<1x2x2x2xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 8, 1, 1]} inputs([[INPUT]] : tensor<1x2x2x2xf16>) -> tensor<1x8x1x1xf16>
    // CHECK:        [[GELU:%.+]] = VPU.Gelu([[SHAPECAST_IN]]) : tensor<1x8x1x1xf16> -> tensor<1x8x1x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 2, 2, 2]} inputs([[GELU]] : tensor<1x8x1x1xf16>) -> tensor<1x2x2x2xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL:   @NotAdjustForGeluMultiShaveOptNCHWWithSuitableDimH
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x16x32xf16>)
func.func @NotAdjustForGeluMultiShaveOptNCHWWithSuitableDimH(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[GELU:%.+]] = VPU.Gelu([[INPUT]]) : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    // CHECK:        return [[GELU]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL:   @AdjustForGeluMultiShaveOptNHWC
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x2x2xf16, {order = #NHWC}>)
func.func @AdjustForGeluMultiShaveOptNHWC(%arg0: tensor<1x2x2x2xf16, {order = #NHWC}>) -> tensor<1x2x2x2xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) : tensor<1x2x2x2xf16, {order = #NHWC}> -> tensor<1x2x2x2xf16, {order = #NHWC}>
    return %0 : tensor<1x2x2x2xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 1, 8, 1]} inputs([[INPUT]] : tensor<1x2x2x2xf16, {order = #NHWC}>) -> tensor<1x1x8x1xf16, {order = #NHWC}>
    // CHECK:        [[GELU:%.+]] = VPU.Gelu([[SHAPECAST_IN]]) : tensor<1x1x8x1xf16, {order = #NHWC}> -> tensor<1x1x8x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 2, 2, 2]} inputs([[GELU]] : tensor<1x1x8x1xf16, {order = #NHWC}>) -> tensor<1x2x2x2xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForGeluMultiShaveOptNHWCWithSuitableDimW
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x2x16xf16, {order = #NHWC}>)
func.func @NotAdjustForGeluMultiShaveOptNHWCWithSuitableDimW(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        [[GELU:%.+]] = VPU.Gelu([[INPUT]]) : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[GELU]]
}

// -----

// CHECK-LABEL:   @AdjustForGeluMultiShaveOptForBatchSize
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1024x2560x1x1xf16>)
func.func @AdjustForGeluMultiShaveOptForBatchSize(%arg0: tensor<1024x2560x1x1xf16>) -> tensor<1024x2560x1x1xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1024x2560x1x1xf16> -> tensor<1024x2560x1x1xf16>
    return %0 : tensor<1024x2560x1x1xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 2621440, 1, 1]} inputs([[INPUT]] : tensor<1024x2560x1x1xf16>) -> tensor<1x2621440x1x1xf16>
    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[SHAPECAST_IN]]) : tensor<1x2621440x1x1xf16> -> tensor<1x2621440x1x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1024, 2560, 1, 1]} inputs([[GELU]] : tensor<1x2621440x1x1xf16>) -> tensor<1024x2560x1x1xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @NotAdjustForGeluMultiShaveOptNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16>)
func.func @NotAdjustForGeluMultiShaveOptNCHW(%arg0: tensor<1x1x1024x1xf16>) -> tensor<1x1x1024x1xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x1x1024x1xf16> -> tensor<1x1x1024x1xf16>
    return %0 : tensor<1x1x1024x1xf16>

    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[INPUT]]) : tensor<1x1x1024x1xf16> -> tensor<1x1x1024x1xf16>
    // CHECK:        return [[GELU]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForGeluMultiShaveOptNHWC
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForGeluMultiShaveOptNHWC(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[INPUT]]) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    // CHECK:        return [[GELU]]
}

// -----

// CHECK-LABEL:   @AdjustForMultiplyMultiShaveOptNCHW
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x2x16x32xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x2x16x32xf16>)
func.func @AdjustForMultiplyMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>, %arg1: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN_1:%.*]] = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs([[INPUT0]] : tensor<1x2x16x32xf16>) -> tensor<1x1024x1x1xf16>
    // CHECK:        [[SHAPECAST_IN_2:%.*]] = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs([[INPUT1]] : tensor<1x2x16x32xf16>) -> tensor<1x1024x1x1xf16>
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[SHAPECAST_IN_1]], [[SHAPECAST_IN_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<1x1024x1x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[MULTIPLY]] : tensor<1x1024x1x1xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForMultiplyMultiShaveOptNHWC
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x32x2x16xf16, {order = #NHWC}>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x32x2x16xf16, {order = #NHWC}>)
func.func @AdjustForMultiplyMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>, %arg1: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x32x2x16xf16, {order = #NHWC}>, tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>

    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN_1:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT0]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_IN_2:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT1]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[SHAPECAST_IN_1]], [[SHAPECAST_IN_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1xf16, {order = #NHWC}>, tensor<1x1x1024x1xf16, {order = #NHWC}> -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[MULTIPLY]] : tensor<1x1x1024x1xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @NotAdjustForMultiplyMultiShaveOptForBroadcast
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x3x384x2xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x3x1x2xf16>)
func.func @NotAdjustForMultiplyMultiShaveOptForBroadcast(%arg0: tensor<1x3x384x2xf16>, %arg1: tensor<1x3x1x2xf16>) -> tensor<1x3x384x2xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x3x384x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x384x2xf16>

    return %0 : tensor<1x3x384x2xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x384x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x384x2xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[MULTIPLY]]
}

// -----

// CHECK-LABEL:   @NotAdjustForMultiplyMultiShaveOptForConstInput
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x128x32xf16>)
func.func @NotAdjustForMultiplyMultiShaveOptForConstInput(%arg0: tensor<1x1x128x32xf16>) -> tensor<1x1x128x32xf16> {
    %cst = const.Declare tensor<1x1x128x32xf16> = dense<1.0> : tensor<1x128x32xf16>, [#const.Reshape<[1, 1, 128, 32]>]
    %0 = VPU.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x128x32xf16>, tensor<1x1x128x32xf16> -> tensor<1x1x128x32xf16>

    return %0 : tensor<1x1x128x32xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[CST:%.*]] = const.Declare tensor<1x1x128x32xf16> = dense<1.000000e+00> : tensor<1x128x32xf16>, [#const.Reshape<[1, 1, 128, 32]>]
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x128x32xf16>, tensor<1x1x128x32xf16> -> tensor<1x1x128x32xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[MULTIPLY]]
}

// -----

// CHECK-LABEL:   @AdjustForMVNWithBatchAndAcrossChannels
// CHECK-SAME:    [[INPUT:%.*]]: tensor<8x16x2x8xf16>
func.func @AdjustForMVNWithBatchAndAcrossChannels(%arg0: tensor<8x16x2x8xf16>) -> tensor<8x16x2x8xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = true, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<8x16x2x8xf16> -> tensor<8x16x2x8xf16>

    return %0 : tensor<8x16x2x8xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 8, 256, 1]} inputs([[INPUT]] : tensor<8x16x2x8xf16>) -> tensor<1x8x256x1xf16>
    // CHECK:        [[MVN:%.*]] = VPU.MVN([[SHAPECAST_IN]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x8x256x1xf16> -> tensor<1x8x256x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [8, 16, 2, 8]} inputs([[MVN]] : tensor<1x8x256x1xf16>) -> tensor<8x16x2x8xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @AdjustForMVNWithBatchAndNotAcrossChannels
// CHECK-SAME:    [[INPUT:%.*]]: tensor<8x16x2x8xf16>
func.func @AdjustForMVNWithBatchAndNotAcrossChannels(%arg0: tensor<8x16x2x8xf16>) -> tensor<8x16x2x8xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<8x16x2x8xf16> -> tensor<8x16x2x8xf16>

    return %0 : tensor<8x16x2x8xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 128, 2, 8]} inputs([[INPUT]] : tensor<8x16x2x8xf16>) -> tensor<1x128x2x8xf16>
    // CHECK:        [[MVN:%.*]] = VPU.MVN([[SHAPECAST_IN]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x2x8xf16> -> tensor<1x128x2x8xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [8, 16, 2, 8]} inputs([[MVN]] : tensor<1x128x2x8xf16>) -> tensor<8x16x2x8xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForNCEPermute
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1023x1x3584xf16>
func.func @AdjustForNCEPermute(%arg0: tensor<1x1023x1x3584xf16>) -> tensor<1x1024x1x3584xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 1024 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x1024x1x3584xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x1x3584xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 1023, 224, 16]} inputs([[INPUT]] : tensor<1x1023x1x3584xf16>) -> tensor<1x1023x224x16xf16>
    // CHECK:        [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[SHAPECAST_IN]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 1024 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x1024x224x16xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 1024, 1, 3584]} inputs([[NCE_PERMUTE]] : tensor<1x1024x224x16xf16, {order = #NHWC}>) -> tensor<1x1024x1x3584xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForNCEPermuteWithEnoughDimSizeForSplit
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x24x64xf16>
func.func @NotAdjustForNCEPermuteWithEnoughDimSizeForSplit(%arg0: tensor<1x1024x24x64xf16>) -> tensor<1x1024x24x64xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 1024 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x1024x24x64xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x64xf16, {order = #NHWC}>

    // CHECK:        [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 1024 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x1024x24x64xf16, {order = #NHWC}>
    // CHECK:        return [[NCE_PERMUTE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForNCEPermuteWithNonDivisibleSpatialShape
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x1x16xf16>
func.func @NotAdjustForNCEPermuteWithNonDivisibleSpatialShape(%arg0: tensor<1x1024x1x16xf16>) -> tensor<1x1024x1x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 1024 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x1024x1x16xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x1x16xf16, {order = #NHWC}>

    // CHECK:        [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 1024 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x1024x1x16xf16, {order = #NHWC}>
    // CHECK:        return [[NCE_PERMUTE]]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL:   @LowerMatMulToNCEAndBalanceOddHW
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<12x1x80x77x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
func.func @LowerMatMulToNCEAndBalanceOddHW(%arg0: tensor<12x1x80x77x1xf16, {order = #GNHWC}>, %arg1: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
                                ) -> tensor<12x1x64x77x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<12x64x1x1x4xsi32> = dense<1> : tensor<12x64x1x1x4xsi32>
    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [12, 64, 80, 1, 1], strides = [1, 1]
            } -> tensor<12x1x64x77x1xf16, {order = #GNHWC}>

    return %0 : tensor<12x1x64x77x1xf16, {order = #GNHWC}>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<12x64x1x1x4xsi32>

    // CHECK:       [[AFFINE_RESHAPE_IN:%.+]] = VPU.AffineReshape([[INPUT0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [12, 1, 80, 11, 7]
    // CHECK-SAME:      tensor<12x1x80x77x1xf16, {order = #GNHWC}> -> tensor<12x1x80x11x7xf16, {order = #GNHWC}>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[AFFINE_RESHAPE_IN]], [[INPUT1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [12, 64, 80, 1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<12x1x64x11x7xf16, {order = #GNHWC}>

    // CHECK:       [[AFFINE_RESHAPE_OUT:%.+]] = VPU.AffineReshape([[MATMUL]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [12, 1, 64, 77, 1]
    // CHECK-SAME:      tensor<12x1x64x11x7xf16, {order = #GNHWC}> -> tensor<12x1x64x77x1xf16, {order = #GNHWC}>

    // CHECK:       return [[AFFINE_RESHAPE_OUT]] : tensor<12x1x64x77x1xf16, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL:   @LowerMatMulToNCEAndBalanceEvenHW
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<12x1x80x32x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
func.func @LowerMatMulToNCEAndBalanceEvenHW(%arg0: tensor<12x1x80x32x1xf16, {order = #GNHWC}>, %arg1: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
                                ) -> tensor<12x1x64x32x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<12x64x1x1x4xsi32> = dense<1> : tensor<12x64x1x1x4xsi32>
    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [12, 64, 80, 1, 1], strides = [1, 1]
            } -> tensor<12x1x64x32x1xf16, {order = #GNHWC}>

    return %0 : tensor<12x1x64x32x1xf16, {order = #GNHWC}>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<12x64x1x1x4xsi32>

    // CHECK:       [[AFFINE_RESHAPE_IN:%.+]] = VPU.AffineReshape([[INPUT0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [12, 1, 80, 8, 4]
    // CHECK-SAME:      tensor<12x1x80x32x1xf16, {order = #GNHWC}> -> tensor<12x1x80x8x4xf16, {order = #GNHWC}>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[AFFINE_RESHAPE_IN]], [[INPUT1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [12, 64, 80, 1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<12x1x64x8x4xf16, {order = #GNHWC}>

    // CHECK:       [[AFFINE_RESHAPE_OUT:%.+]] = VPU.AffineReshape([[MATMUL]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [12, 1, 64, 32, 1]
    // CHECK-SAME:      tensor<12x1x64x8x4xf16, {order = #GNHWC}> -> tensor<12x1x64x32x1xf16, {order = #GNHWC}>

    // CHECK:       return [[AFFINE_RESHAPE_OUT]] : tensor<12x1x64x32x1xf16, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL:   @LowerMatMulToNCEAndCannotBalanceHW
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<12x1x80x17x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
func.func @LowerMatMulToNCEAndCannotBalanceHW(%arg0: tensor<12x1x80x17x1xf16, {order = #GNHWC}>, %arg1: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
                                ) -> tensor<12x1x64x17x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<12x64x1x1x4xsi32> = dense<1> : tensor<12x64x1x1x4xsi32>
    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [12, 64, 80, 1, 1], strides = [1, 1]
            } -> tensor<12x1x64x17x1xf16, {order = #GNHWC}>

    return %0 : tensor<12x1x64x17x1xf16, {order = #GNHWC}>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<12x64x1x1x4xsi32>

    // CHECK-NOT:   VPU.AffineReshape

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[INPUT0]], [[INPUT1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [12, 64, 80, 1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<12x1x64x17x1xf16, {order = #GNHWC}>

    // CHECK:       return [[MATMUL]] : tensor<12x1x64x17x1xf16, {order = #GNHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForEltwiseAvgPool
// CHECK-SAME:    [[INPUT:%.+]]: tensor<80x512x1x1xf16, {order = #NHWC}>
func.func @AdjustForEltwiseAvgPool(%arg0: tensor<80x512x1x1xf16, {order = #NHWC}>) -> tensor<80x512x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
        quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
        strides = [1, 1]
    } -> tensor<80x512x1x1xf16, {order = #NHWC}>

    return %0 : tensor<80x512x1x1xf16, {order = #NHWC}>

    // CHECK:           [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 16, 64, 40]} inputs([[INPUT]] : tensor<80x512x1x1xf16, {order = #NHWC}>) -> tensor<1x16x64x40xf16, {order = #NHWC}>
    // CHECK:           [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[SHAPECAST_IN]]) {
    // CHECK-SAME:              kernel_size = [1, 1],
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
    // CHECK-SAME:              quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x64x40xf16, {order = #NHWC}>
    // CHECK:           [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [80, 512, 1, 1]} inputs([[AVGPOOL]] : tensor<1x16x64x40xf16, {order = #NHWC}>) -> tensor<80x512x1x1xf16, {order = #NHWC}>

    // CHECK:           return [[SHAPECAST_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForAvgPoolWithPermute
// CHECK-SAME:    [[INPUT:%.+]]: tensor<2x16x4x4xf16, {order = #NHWC}>
func.func @NotAdjustForAvgPoolWithPermute(%arg0: tensor<2x16x4x4xf16, {order = #NHWC}>) -> tensor<2x16x4x4xf16> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
        quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
        strides = [1, 1]
    } -> tensor<2x16x4x4xf16>

    return %0 : tensor<2x16x4x4xf16>

    // CHECK:           [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:              kernel_size = [1, 1],
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
    // CHECK-SAME:              quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:      } -> tensor<2x16x4x4xf16>

    // CHECK:           return [[AVGPOOL]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128,
                                       0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForPerAxisQuantAvgPool
// CHECK-SAME:    [[INPUT:%.+]]: tensor<2x32x4x4x!qElemType, {order = #NHWC}>
func.func @NotAdjustForPerAxisQuantAvgPool(%arg0: tensor<2x32x4x4x!qElemType, {order = #NHWC}>) -> tensor<2x32x4x4x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
        quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
        strides = [1, 1]
    } -> tensor<2x32x4x4x!qElemType, {order = #NHWC}>

    return %0 : tensor<2x32x4x4x!qElemType, {order = #NHWC}>

    // CHECK:           [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:              kernel_size = [1, 1],
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
    // CHECK-SAME:              quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:      } -> tensor<2x32x4x4x!qElemType, {order = #NHWC}>

    // CHECK:           return [[AVGPOOL]]
}
