//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --reshape-matmul-inputs="enable-grouped-matmul=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----

// CHECK-LABEL: @MatMulBatchedLhs1dRhs
func.func @MatMulBatchedLhs1dRhs(%LHS: tensor<4x16x32xf16>, %RHS: tensor<32xf16>) -> tensor<4x16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<4x16x32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [64, 32]
    // CHECK-SAME:  } : tensor<4x16x32xf16> -> tensor<64x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<4x16x32xf16>, tensor<32xf16> -> tensor<4x16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<64x32xf16>, tensor<1x32xf16> -> tensor<64x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[FC]]) {
    // CHECK-SAME:      shape_value = [4, 16]
    // CHECK-SAME:  } : tensor<64x1xf16> -> tensor<4x16xf16>

    return %GEMM : tensor<4x16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<4x16xf16>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @MatMul3dArgs
func.func @MatMul3dArgs(%LHS: tensor<4x8x16xf16>, %RHS: tensor<4x16x64xf16>) -> tensor<4x8x64xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<4x8x16xf16>, [[RHS:%.+]]: tensor<4x16x64xf16>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RHS]]) {
    // CHECK-SAME:      order_value = #map
    // CHECL-SAMEL  } : tensor<4x16x64xf16> -> tensor<4x64x16xf16>

    // CHECK:   [[LHS_TO_4D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 4, 8, 16]
    // CHECK-SAME:  } : tensor<4x8x16xf16> -> tensor<1x4x8x16xf16>

    // CHECK:   [[RHS_TO_4D:%.+]] = IE.AffineReshape([[TRANSPOSE_RHS]]) {
    // CHECK-SAME:      shape_value = [1, 4, 64, 16]
    // CHECK-SAME:  } : tensor<4x64x16xf16> -> tensor<1x4x64x16xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<4x8x16xf16>, tensor<4x16x64xf16> -> tensor<4x8x64xf16>
    // CHECK:   [[GEMM:%.+]] = IE.MatMul([[LHS_TO_4D]], [[RHS_TO_4D]]) {transpose_b} :
    // CHECK-SAME:  tensor<1x4x8x16xf16>, tensor<1x4x64x16xf16> -> tensor<1x4x8x64xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[GEMM]]) {
    // CHECK-SAME:      shape_value = [4, 8, 64]
    // CHECK-SAME:  } : tensor<1x4x8x64xf16> -> tensor<4x8x64xf16>

    return %GEMM : tensor<4x8x64xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<4x8x64xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul4dLhs2dRhs
func.func @MatMul4dLhs2dRhs(%LHS: tensor<2x3x4x16xf16>, %RHS: tensor<16x32xf16>) -> tensor<2x3x4x32xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<2x3x4x16xf16>, [[RHS:%.+]]: tensor<16x32xf16>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RHS]]) {
    // CHECK-SAME:      order_value = #CN
    // CHECL-SAMEL  } : tensor<16x32xf16> -> tensor<32x16xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [24, 16]
    // CHECK-SAME:  } : tensor<2x3x4x16xf16> -> tensor<24x16xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<2x3x4x16xf16>, tensor<16x32xf16> -> tensor<2x3x4x32xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[TRANSPOSE_RHS]]) :
    // CHECK-SAME:  tensor<24x16xf16>, tensor<32x16xf16> -> tensor<24x32xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 32]
    // CHECK-SAME:  } : tensor<24x32xf16> -> tensor<2x3x4x32xf16>

    return %GEMM : tensor<2x3x4x32xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<2x3x4x32xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MatMul4dArgs
func.func @MatMul4dArgs(%LHS: tensor<2x3x4x16xf16>, %RHS: tensor<2x3x16x32xf16>) -> tensor<2x3x4x32xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<2x3x4x16xf16>, [[RHS:%.+]]: tensor<2x3x16x32xf16>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RHS]]) {
    // CHECK-SAME:      order_value = #NCWH
    // CHECL-SAMEL  } : tensor<2x3x16x32xf16> -> tensor<2x3x32x16xf16>

    // CHECK:   [[LHS_TO_4D:%.+]] = IE.Reshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 6, 4, 16]
    // CHECK-SAME:  } : tensor<2x3x4x16xf16> -> tensor<1x6x4x16xf16>

    // CHECK:   [[RHS_TO_4D:%.+]] = IE.Reshape([[TRANSPOSE_RHS]]) {
    // CHECK-SAME:      shape_value = [1, 6, 32, 16]
    // CHECK-SAME:  } : tensor<2x3x32x16xf16> -> tensor<1x6x32x16xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<2x3x4x16xf16>, tensor<2x3x16x32xf16> -> tensor<2x3x4x32xf16>
    // CHECK:   [[GEMM:%.+]] = IE.MatMul([[LHS_TO_4D]], [[RHS_TO_4D]]) {transpose_b} :
    // CHECK-SAME:  tensor<1x6x4x16xf16>, tensor<1x6x32x16xf16> -> tensor<1x6x4x32xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[GEMM]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 32]
    // CHECK-SAME:  } : tensor<1x6x4x32xf16> -> tensor<2x3x4x32xf16>

    return %GEMM : tensor<2x3x4x32xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<2x3x4x32xf16>
}

// -----

// CHECK-LABEL: @MatMul1dArgs
func.func @MatMul1dArgs(%LHS: tensor<32xf16>, %RHS: tensor<32xf16>) -> tensor<f16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<32xf16>, tensor<32xf16> -> tensor<f16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[FC]]) {
    // CHECK-SAME:      shape_value = []
    // CHECK-SAME:  } : tensor<1x1xf16> -> tensor<f16>

    return %GEMM : tensor<f16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<f16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul1dLhs2dRhs
func.func @MatMul1dLhs2dRhs(%LHS: tensor<32xf16>, %RHS: tensor<32x16xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32x16xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RHS]]) {
    // CHECK-SAME:      order_value = #CN
    // CHECK-SAME:  } : tensor<32x16xf16> -> tensor<16x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<32xf16>, tensor<32x16xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[TRANSPOSE_RHS]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<16x32xf16> -> tensor<1x16xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<1x16xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

// CHECK-LABEL: @MatMul2dLhs1dRhs
func.func @MatMul2dLhs1dRhs(%LHS: tensor<16x32xf16>, %RHS: tensor<32xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<16x32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<16x32xf16>, tensor<32xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<16x32xf16>, tensor<1x32xf16> -> tensor<16x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<16x1xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

// CHECK-LABEL: @MatMul1dArgsTransposeA
func.func @MatMul1dArgsTransposeA(%LHS: tensor<32xf16>, %RHS: tensor<32xf16>) -> tensor<f16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_a} : tensor<32xf16>, tensor<32xf16> -> tensor<f16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[FC]]) {
    // CHECK-SAME:      shape_value = []
    // CHECK-SAME:  } : tensor<1x1xf16> -> tensor<f16>

    return %GEMM : tensor<f16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<f16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul1dLhs2dRhsTransposeA
func.func @MatMul1dLhs2dRhsTransposeA(%LHS: tensor<32xf16>, %RHS: tensor<32x16xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32x16xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RHS]]) {
    // CHECK-SAME:      order_value = #CN
    // CHECK-SAME:  } : tensor<32x16xf16> -> tensor<16x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_a} : tensor<32xf16>, tensor<32x16xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[TRANSPOSE_RHS]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<16x32xf16> -> tensor<1x16xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<1x16xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul2dLhs1dRhsTransposeA
func.func @MatMul2dLhs1dRhsTransposeA(%LHS: tensor<32x16xf16>, %RHS: tensor<32xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<32x16xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[TRANSPOSE_LHS:%.+]] = IE.Transpose([[LHS]]) {
    // CHECK-SAME:      order_value = #CN
    // CHECK-SAME:  } : tensor<32x16xf16> -> tensor<16x32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_a} : tensor<32x16xf16>, tensor<32xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[TRANSPOSE_LHS]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<16x32xf16>, tensor<1x32xf16> -> tensor<16x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<16x1xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

// CHECK-LABEL: @MatMul1dArgsTransposeB
func.func @MatMul1dArgsTransposeB(%LHS: tensor<32xf16>, %RHS: tensor<32xf16>) -> tensor<f16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_b} : tensor<32xf16>, tensor<32xf16> -> tensor<f16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[FC]]) {
    // CHECK-SAME:      shape_value = []
    // CHECK-SAME:  } : tensor<1x1xf16> -> tensor<f16>

    return %GEMM : tensor<f16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<f16>
}

// -----

// CHECK-LABEL: @MatMul1dLhs2dRhsTransposeB
func.func @MatMul1dLhs2dRhsTransposeB(%LHS: tensor<32xf16>, %RHS: tensor<16x32xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<16x32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_b} : tensor<32xf16>, tensor<16x32xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<16x32xf16> -> tensor<1x16xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<1x16xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

// CHECK-LABEL: @MatMul2dLhs1dRhsTransposeB
func.func @MatMul2dLhs1dRhsTransposeB(%LHS: tensor<16x32xf16>, %RHS: tensor<32xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<16x32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_b} : tensor<16x32xf16>, tensor<32xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<16x32xf16>, tensor<1x32xf16> -> tensor<16x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<16x1xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

// CHECK-LABEL: @MatMul1dArgsTransposeBoth
func.func @MatMul1dArgsTransposeBoth(%LHS: tensor<32xf16>, %RHS: tensor<32xf16>) -> tensor<f16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {
        transpose_a,
        transpose_b
    } : tensor<32xf16>, tensor<32xf16> -> tensor<f16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[FC]]) {
    // CHECK-SAME:      shape_value = []
    // CHECK-SAME:  } : tensor<1x1xf16> -> tensor<f16>

    return %GEMM : tensor<f16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<f16>
}

// -----

// CHECK-LABEL: @MatMul1dLhs2dRhsTransposeBoth
func.func @MatMul1dLhs2dRhsTransposeBoth(%LHS: tensor<32xf16>, %RHS: tensor<16x32xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<16x32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {
        transpose_a,
        transpose_b
    } : tensor<32xf16>, tensor<16x32xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS]]) :
    // CHECK-SAME:  tensor<1x32xf16>, tensor<16x32xf16> -> tensor<1x16xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<1x16xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul2dLhs1dRhsTransposeBoth
func.func @MatMul2dLhs1dRhsTransposeBoth(%LHS: tensor<32x16xf16>, %RHS: tensor<32xf16>) -> tensor<16xf16> {
    // CHECK:   [[LHS:%.+]]: tensor<32x16xf16>, [[RHS:%.+]]: tensor<32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[TRANSPOSE_LHS:%.+]] = IE.Transpose([[LHS]]) {
    // CHECK-SAME:      order_value = #CN
    // CHECK-SAME:  } : tensor<32x16xf16> -> tensor<16x32xf16>

    %GEMM = IE.MatMul(%LHS, %RHS) {
        transpose_a,
        transpose_b
    } : tensor<32x16xf16>, tensor<32xf16> -> tensor<16xf16>
    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[TRANSPOSE_LHS]], [[RHS_TO_2D]]) :
    // CHECK-SAME:  tensor<16x32xf16>, tensor<1x32xf16> -> tensor<16x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]]) {
    // CHECK-SAME:      shape_value = [16]
    // CHECK-SAME:  } : tensor<16x1xf16> -> tensor<16xf16>

    return %GEMM : tensor<16xf16>
    // CHECK:   return [[RESHAPE_OUT]] : tensor<16xf16>
}

// -----

// CHECK-LABEL: @Softmax4dArgs(
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x6x4x16xf16>, [[INPUT_1:%.+]]: tensor<1x6x32x16xf16>) -> tensor<2x3x4x32xf16>
func.func @Softmax4dArgs(%arg0: tensor<1x6x4x16xf16>, %arg1: tensor<1x6x32x16xf16>) -> tensor<2x3x4x32xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x6x4x16xf16>, tensor<1x6x32x16xf16> -> tensor<1x6x4x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [2, 3, 4, 32]} : tensor<1x6x4x32xf16> -> tensor<2x3x4x32xf16>
    %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<2x3x4x32xf16> -> tensor<2x3x4x32xf16>

    return %2 : tensor<2x3x4x32xf16>
    // CHECK:    [[MATMUL:%.+]] = IE.MatMul([[INPUT_0]], [[INPUT_1]]) {transpose_b} :
    // CHECK-SAME:  tensor<1x6x4x16xf16>, tensor<1x6x32x16xf16> -> tensor<1x6x4x32xf16>
    // CHECK:    [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL]]) {axisInd = 3 : i64} : tensor<1x6x4x32xf16> -> tensor<1x6x4x32xf16>
    // CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 32]
    // CHECK-SAME:  } : tensor<1x6x4x32xf16> -> tensor<2x3x4x32xf16>
    // CHECK:    return [[RESHAPE]] : tensor<2x3x4x32xf16>
}

// -----

// CHECK-LABEL: @Softmax4dFQ(
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x6x4x16xf16>, [[INPUT_1:%.+]]: tensor<1x6x32x16xf16>) -> tensor<2x3x4x32xf16>
func.func @Softmax4dFQ(%arg0: tensor<1x6x4x16xf16>, %arg1: tensor<1x6x32x16xf16>) -> tensor<2x3x4x32xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<205.0> : tensor<1x1x1x1xf32>
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x6x4x16xf16>, tensor<1x6x32x16xf16> -> tensor<1x6x4x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [2, 3, 4, 32]} : tensor<1x6x4x32xf16> -> tensor<2x3x4x32xf16>
    %2 = IE.FakeQuantize(%1, %cst, %cst_0, %cst_1, %cst_2)
      { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
      tensor<2x3x4x32xf16>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x32xf16>

    %3 = IE.SoftMax(%2) {axisInd = 3 : i64} : tensor<2x3x4x32xf16> -> tensor<2x3x4x32xf16>

    return %3 : tensor<2x3x4x32xf16>
    // CHECK:    [[MATMUL:%.+]] = IE.MatMul([[INPUT_0]], [[INPUT_1]]) {transpose_b} :
    // CHECK-SAME:  tensor<1x6x4x16xf16>, tensor<1x6x32x16xf16> -> tensor<1x6x4x32xf16>
    // CHECK:    [[RESHAPE0:%.+]] = IE.Reshape([[MATMUL]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 32]
    // CHECK-SAME:  } : tensor<1x6x4x32xf16> -> tensor<2x3x4x32xf16>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE0]]
    // CHECK:    [[RESHAPE1:%.+]] = IE.Reshape([[FQ]]) {
    // CHECK-SAME:      shape_value = [1, 6, 4, 32]
    // CHECK-SAME:  } : tensor<2x3x4x32xf16> -> tensor<1x6x4x32xf16>
    // CHECK:    [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE1]]) {axisInd = 3 : i64} : tensor<1x6x4x32xf16> -> tensor<1x6x4x32xf16>
    // CHECK:    [[RESHAPE2:%.+]] = IE.Reshape([[SOFTMAX]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 32]
    // CHECK-SAME:  } : tensor<1x6x4x32xf16> -> tensor<2x3x4x32xf16>
    // CHECK:    return [[RESHAPE2]] : tensor<2x3x4x32xf16>
}
