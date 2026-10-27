//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --use-group-quant-correction %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#transposeWeights = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionForGroupMajorTranspose
// CHECK-NOT: IE.GroupQuantCorrection
func.func @DoNotUseGroupQuantCorrectionForGroupMajorTranspose(%activation: tensor<1x1x512xf16>,
                                                               %weights: tensor<8x96x64xui2>,
                                                               %scale: tensor<8x96x1xf16>,
                                                               %zeroPoint: tensor<8x96x1xui2>)
        -> tensor<1x1x96xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui2> -> tensor<8x96x64xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui2> -> tensor<8x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
        : tensor<8x96x64xf16> -> tensor<8x64x96xf16>
    %weightsReshape = IE.AffineReshape(%weightsTranspose) {
        dim_mapping = [[0], [0], [1]], shape_value = [512, 96]
    } : tensor<8x64x96xf16> -> tensor<512x96xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x512xf16>, tensor<512x96xf16> -> tensor<1x1x96xf16>
    return %matmul : tensor<1x1x96xf16>

}

// -----

#transposeWeights = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK-LABEL: @UseGroupQuantCorrectionWithF32Scale
// CHECK-SAME:      [[ACTIVATION:%[^:]+]]: tensor<1x1x512xf32>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<8x96x64xui2>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<8x96xf16>
// CHECK-SAME:      [[ZERO_POINT:%[^:]+]]: tensor<8x96x1xui2>
func.func @UseGroupQuantCorrectionWithF32Scale(%activation: tensor<1x1x512xf32>, %weights: tensor<8x96x64xui2>,
                                               %scale: tensor<8x96xf16>, %zeroPoint: tensor<8x96x1xui2>)
        -> tensor<1x1x96xf32> {
    %weightsF32 = IE.Convert(%weights) {dstElemType = f32} : tensor<8x96x64xui2> -> tensor<8x96x64xf32>
    %zeroPointF32 = IE.Convert(%zeroPoint) {dstElemType = f32} : tensor<8x96x1xui2> -> tensor<8x96x1xf32>
    %weightsSub = IE.Subtract(%weightsF32, %zeroPointF32) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf32>, tensor<8x96x1xf32> -> tensor<8x96x64xf32>
    %scaleReshape = IE.AffineReshape(%scale) {dim_mapping = [[0], [1, 2]], shape_value = [8, 96, 1]}
        : tensor<8x96xf16> -> tensor<8x96x1xf16>
    %scaleF32 = IE.Convert(%scaleReshape) {dstElemType = f32} : tensor<8x96x1xf16> -> tensor<8x96x1xf32>
    %weightsDequant = IE.Multiply(%weightsSub, %scaleF32) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf32>, tensor<8x96x1xf32> -> tensor<8x96x64xf32>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
        : tensor<8x96x64xf32> -> tensor<96x8x64xf32>
    %weightsReshape = IE.AffineReshape(%weightsTranspose) {
        dim_mapping = [[0], [1], [1]], shape_value = [96, 512]
    } : tensor<96x8x64xf32> -> tensor<96x512xf32>
    %matmul = IE.MatMul(%activation, %weightsReshape) {transpose_b}
        : tensor<1x1x512xf32>, tensor<96x512xf32> -> tensor<1x1x96xf32>
    return %matmul : tensor<1x1x96xf32>

    // CHECK:       [[SCALE_RESHAPE:%.+]] = IE.AffineReshape([[SCALE]])
    // CHECK:       [[SCALE_F32:%.+]] = IE.Convert([[SCALE_RESHAPE]]) {dstElemType = f32}
    // CHECK:       [[SCALES_RESHAPE:%.+]] = IE.Reshape([[SCALE_F32]]) {shape_value = [8, 96, 1, 1]}
    // CHECK:       [[GROUP_QUANT_CORRECTION:%.+]] = IE.GroupQuantCorrection
    // CHECK-SAME:      [[SCALES_RESHAPE]]
}

// -----

#transposeWeights = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionForUnsupportedGroupSize
// CHECK-SAME:      [[ACTIVATION:%[^:]+]]: tensor<1x1x256xf16>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<8x96x32xui2>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<8x96x1xf16>
// CHECK-SAME:      [[ZERO_POINT:%[^:]+]]: tensor<8x96x1xui2>
func.func @DoNotUseGroupQuantCorrectionForUnsupportedGroupSize(%activation: tensor<1x1x256xf16>,
                                                               %weights: tensor<8x96x32xui2>,
                                                               %scale: tensor<8x96x1xf16>,
                                                               %zeroPoint: tensor<8x96x1xui2>)
        -> tensor<1x1x96xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x32xui2> -> tensor<8x96x32xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui2> -> tensor<8x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x32xf16>, tensor<8x96x1xf16> -> tensor<8x96x32xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x32xf16>, tensor<8x96x1xf16> -> tensor<8x96x32xf16>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
        : tensor<8x96x32xf16> -> tensor<8x32x96xf16>
    %weightsReshape = IE.AffineReshape(%weightsTranspose) {
        dim_mapping = [[0], [0], [1]], shape_value = [256, 96]
    } : tensor<8x32x96xf16> -> tensor<256x96xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x256xf16>, tensor<256x96xf16> -> tensor<1x1x96xf16>
    return %matmul : tensor<1x1x96xf16>

    // CHECK:       [[WEIGHTS_F16:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16}
    // CHECK:       [[ZERO_POINT_F16:%.+]] = IE.Convert([[ZERO_POINT]]) {dstElemType = f16}
    // CHECK:       [[WEIGHTS_SUB:%.+]] = IE.Subtract([[WEIGHTS_F16]], [[ZERO_POINT_F16]])
    // CHECK:       [[WEIGHTS_DEQUANT:%.+]] = IE.Multiply([[WEIGHTS_SUB]], [[SCALE]])
    // CHECK:       [[WEIGHTS_TRANSPOSE:%.+]] = IE.Transpose([[WEIGHTS_DEQUANT]])
    // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_TRANSPOSE]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ACTIVATION]], [[WEIGHTS_RESHAPE]])
    // CHECK-NOT:   IE.GroupQuantCorrection
    // CHECK:       return [[MATMUL]] : tensor<1x1x96xf16>
}

// -----

#transposeWeights = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionForBroadcastParameters
func.func @DoNotUseGroupQuantCorrectionForBroadcastParameters(
        %activation: tensor<1x1x512xf16>, %weights: tensor<8x96x64xui2>, %scale: tensor<1x96x1xf16>,
        %zeroPoint: tensor<1x96x1xui2>) -> tensor<1x1x96xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui2> -> tensor<8x96x64xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<1x96x1xui2> -> tensor<1x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<1x96x1xf16> -> tensor<8x96x64xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<1x96x1xf16> -> tensor<8x96x64xf16>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
        : tensor<8x96x64xf16> -> tensor<8x64x96xf16>
    %weightsReshape = IE.AffineReshape(%weightsTranspose) {
        dim_mapping = [[0], [0], [1]], shape_value = [512, 96]
    } : tensor<8x64x96xf16> -> tensor<512x96xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x512xf16>, tensor<512x96xf16> -> tensor<1x1x96xf16>
    return %matmul : tensor<1x1x96xf16>

    // CHECK-NOT:   IE.GroupQuantCorrection
    // CHECK:       return
}

// -----

#unsupportedTransposeWeights = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionForUnsupportedTranspose
func.func @DoNotUseGroupQuantCorrectionForUnsupportedTranspose(
        %activation: tensor<1x1x512xf16>, %weights: tensor<8x96x64xui2>, %scale: tensor<8x96x1xf16>,
        %zeroPoint: tensor<8x96x1xui2>) -> tensor<1x1x96xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui2> -> tensor<8x96x64xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui2> -> tensor<8x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #unsupportedTransposeWeights}
        : tensor<8x96x64xf16> -> tensor<64x96x8xf16>
    %weightsReshape = IE.AffineReshape(%weightsTranspose) {
        dim_mapping = [[0], [0], [1]], shape_value = [512, 96]
    } : tensor<64x96x8xf16> -> tensor<512x96xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x512xf16>, tensor<512x96xf16> -> tensor<1x1x96xf16>
    return %matmul : tensor<1x1x96xf16>

    // CHECK-NOT:   IE.GroupQuantCorrection
    // CHECK:       return
}

// -----

#transposeWeights = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionForNonUi2Weights
// CHECK-SAME:      [[ACTIVATION:%[^:]+]]: tensor<1x1x512xf16>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<8x96x64xui4>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<8x96x1xf16>
// CHECK-SAME:      [[ZERO_POINT:%[^:]+]]: tensor<8x96x1xui4>
func.func @DoNotUseGroupQuantCorrectionForNonUi2Weights(%activation: tensor<1x1x512xf16>,
                                                         %weights: tensor<8x96x64xui4>,
                                                         %scale: tensor<8x96x1xf16>,
                                                         %zeroPoint: tensor<8x96x1xui4>)
        -> tensor<1x1x96xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui4> -> tensor<8x96x64xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui4> -> tensor<8x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
        : tensor<8x96x64xf16> -> tensor<8x64x96xf16>
    %weightsReshape = IE.AffineReshape(%weightsTranspose) {
        dim_mapping = [[0], [0], [1]], shape_value = [512, 96]
    } : tensor<8x64x96xf16> -> tensor<512x96xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x512xf16>, tensor<512x96xf16> -> tensor<1x1x96xf16>
    return %matmul : tensor<1x1x96xf16>

    // CHECK:       [[WEIGHTS_F16:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16}
    // CHECK:       [[ZERO_POINT_F16:%.+]] = IE.Convert([[ZERO_POINT]]) {dstElemType = f16}
    // CHECK:       [[WEIGHTS_SUB:%.+]] = IE.Subtract([[WEIGHTS_F16]], [[ZERO_POINT_F16]])
    // CHECK:       [[WEIGHTS_DEQUANT:%.+]] = IE.Multiply([[WEIGHTS_SUB]], [[SCALE]])
    // CHECK:       [[WEIGHTS_TRANSPOSE:%.+]] = IE.Transpose([[WEIGHTS_DEQUANT]])
    // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_TRANSPOSE]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ACTIVATION]], [[WEIGHTS_RESHAPE]])
    // CHECK-NOT:   IE.GroupQuantCorrection
    // CHECK:       return [[MATMUL]] : tensor<1x1x96xf16>
}

// -----

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionWithoutTranspose
// CHECK-SAME:      [[ACTIVATION:%[^:]+]]: tensor<1x1x768xf16>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<8x96x64xui2>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<8x96x1xf16>
// CHECK-SAME:      [[ZERO_POINT:%[^:]+]]: tensor<8x96x1xui2>
func.func @DoNotUseGroupQuantCorrectionWithoutTranspose(%activation: tensor<1x1x768xf16>,
                                                         %weights: tensor<8x96x64xui2>,
                                                         %scale: tensor<8x96x1xf16>,
                                                         %zeroPoint: tensor<8x96x1xui2>)
        -> tensor<1x1x64xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui2> -> tensor<8x96x64xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui2> -> tensor<8x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsReshape = IE.AffineReshape(%weightsDequant) {
        dim_mapping = [[0], [0], [1]], shape_value = [768, 64]
    } : tensor<8x96x64xf16> -> tensor<768x64xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x768xf16>, tensor<768x64xf16> -> tensor<1x1x64xf16>
    return %matmul : tensor<1x1x64xf16>

    // CHECK:       [[WEIGHTS_F16:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16}
    // CHECK:       [[ZERO_POINT_F16:%.+]] = IE.Convert([[ZERO_POINT]]) {dstElemType = f16}
    // CHECK:       [[WEIGHTS_SUB:%.+]] = IE.Subtract([[WEIGHTS_F16]], [[ZERO_POINT_F16]])
    // CHECK:       [[WEIGHTS_DEQUANT:%.+]] = IE.Multiply([[WEIGHTS_SUB]], [[SCALE]])
    // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_DEQUANT]])
    // CHECK-NOT:   IE.Transpose
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ACTIVATION]], [[WEIGHTS_RESHAPE]])
    // CHECK-NOT:   IE.GroupQuantCorrection
    // CHECK:       return [[MATMUL]] : tensor<1x1x64xf16>
}

// -----

#transposeWeights = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @DoNotUseGroupQuantCorrectionWithoutAffineReshape
// CHECK-SAME:      [[ACTIVATION:%[^:]+]]: tensor<1x1x512xf16>
// CHECK-SAME:      [[WEIGHTS:%[^:]+]]: tensor<8x96x64xui2>
// CHECK-SAME:      [[SCALE:%[^:]+]]: tensor<8x96x1xf16>
// CHECK-SAME:      [[ZERO_POINT:%[^:]+]]: tensor<8x96x1xui2>
func.func @DoNotUseGroupQuantCorrectionWithoutAffineReshape(%activation: tensor<1x1x512xf16>,
                                                             %weights: tensor<8x96x64xui2>,
                                                             %scale: tensor<8x96x1xf16>,
                                                             %zeroPoint: tensor<8x96x1xui2>)
        -> tensor<1x1x96xf16> {
    %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui2> -> tensor<8x96x64xf16>
    %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui2> -> tensor<8x96x1xf16>
    %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
    %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
        : tensor<8x96x64xf16> -> tensor<8x64x96xf16>
    %weightsReshape = IE.Reshape(%weightsTranspose) {shape_value = [512, 96]}
        : tensor<8x64x96xf16> -> tensor<512x96xf16>
    %matmul = IE.MatMul(%activation, %weightsReshape)
        : tensor<1x1x512xf16>, tensor<512x96xf16> -> tensor<1x1x96xf16>
    return %matmul : tensor<1x1x96xf16>

    // CHECK:       [[WEIGHTS_F16:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16}
    // CHECK:       [[ZERO_POINT_F16:%.+]] = IE.Convert([[ZERO_POINT]]) {dstElemType = f16}
    // CHECK:       [[WEIGHTS_SUB:%.+]] = IE.Subtract([[WEIGHTS_F16]], [[ZERO_POINT_F16]])
    // CHECK:       [[WEIGHTS_DEQUANT:%.+]] = IE.Multiply([[WEIGHTS_SUB]], [[SCALE]])
    // CHECK:       [[WEIGHTS_TRANSPOSE:%.+]] = IE.Transpose([[WEIGHTS_DEQUANT]])
    // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = IE.Reshape([[WEIGHTS_TRANSPOSE]]) {shape_value = [512, 96]}
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ACTIVATION]], [[WEIGHTS_RESHAPE]])
    // CHECK-NOT:   IE.GroupQuantCorrection
    // CHECK:       return [[MATMUL]] : tensor<1x1x96xf16>
}
