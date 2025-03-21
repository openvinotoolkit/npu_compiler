//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-fft-to-conv  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: @DFT
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8x2xf16>
func.func @DFT(%arg0: tensor<4x8x2xf16>) -> tensor<4x8x2xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %0 = IE.DFT(%arg0, %cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<4x8x2xf16>, tensor<2xsi64> -> tensor<4x8x2xf16>
  return %0 : tensor<4x8x2xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose([[INPUT]]) {order_value = #HCW} : tensor<4x8x2xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 8]} : tensor<8x4x2xf16> -> tensor<1x8x8xf16>
    // CHECK: [[CST_0:%.*]] = const.Declare tensor<8x8xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x8xf16>, tensor<8x8xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.*]] = const.Declare tensor<16x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<16x16xf16> -> tensor<1x4x16xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 8, 2]} : tensor<1x4x16xf16> -> tensor<4x8x2xf16>
    // CHECK: return [[VAL7]] : tensor<4x8x2xf16>
}

// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: @IDFT
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8x2xf16>
func.func @IDFT(%arg0: tensor<4x8x2xf16>) -> tensor<4x8x2xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %0 = IE.IDFT(%arg0, %cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<4x8x2xf16>, tensor<2xsi64> -> tensor<4x8x2xf16>
  return %0 : tensor<4x8x2xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose([[INPUT]]) {order_value = #HCW} : tensor<4x8x2xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 8]} : tensor<8x4x2xf16> -> tensor<1x8x8xf16>
    // CHECK: [[CST_0:%.*]] = const.Declare tensor<8x8xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x8xf16>, tensor<8x8xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.*]] = const.Declare tensor<16x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<16x16xf16> -> tensor<1x4x16xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 8, 2]} : tensor<1x4x16xf16> -> tensor<4x8x2xf16>
    // CHECK: return [[VAL7]] : tensor<4x8x2xf16>
}

// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#CN = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @RDFT
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8xf16>
func.func @RDFT(%arg0: tensor<4x8xf16>) -> tensor<4x5x2xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %0 = IE.RDFT(%arg0, %cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<4x8xf16>, tensor<2xsi64> -> tensor<4x5x2xf16>
  return %0 : tensor<4x5x2xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose([[INPUT]]) {order_value = #CN} : tensor<4x8xf16> -> tensor<8x4xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 4]} : tensor<8x4xf16> -> tensor<1x8x4xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<8x4xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x4xf16>, tensor<8x4xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<10x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<10x16xf16> -> tensor<1x4x10xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 5, 2]} : tensor<1x4x10xf16> -> tensor<4x5x2xf16>
    // CHECK: return [[VAL7]] : tensor<4x5x2xf16>
}

// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: @IRDFT
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8x2xf16>
func.func @IRDFT(%arg0: tensor<4x8x2xf16>) -> tensor<4x14xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %0 = IE.IRDFT(%arg0, %cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<4x8x2xf16>, tensor<2xsi64> -> tensor<4x14xf16>
  return %0 : tensor<4x14xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose(%arg0) {order_value = #HCW} : tensor<4x8x2xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 8]} : tensor<8x4x2xf16> -> tensor<1x8x8xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<8x8xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x8xf16>, tensor<8x8xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<14x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<14x16xf16> -> tensor<1x4x14xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 14]} : tensor<1x4x14xf16> -> tensor<4x14xf16>
    // CHECK: return [[VAL7]] : tensor<4x14xf16>
}

// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: @DFTTransformWithSignalSize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8x2xf16>
func.func @DFTTransformWithSignalSize(%arg0: tensor<4x8x2xf16>) -> tensor<4x8x2xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %cst_1 = const.Declare tensor<2xsi64> = dense<[4, 8]> : tensor<2xsi64>
  %0 = IE.DFT(%arg0, %cst, %cst_1) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<4x8x2xf16>, tensor<2xsi64>, tensor<2xsi64> -> tensor<4x8x2xf16>
  return %0 : tensor<4x8x2xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[CST1:%.*]] = const.Declare tensor<2xsi64> = dense<[4, 8]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose([[INPUT]]) {order_value = #HCW} : tensor<4x8x2xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 8]} : tensor<8x4x2xf16> -> tensor<1x8x8xf16>
    // CHECK: [[CST_0:%.*]] = const.Declare tensor<8x8xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x8xf16>, tensor<8x8xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.*]] = const.Declare tensor<16x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<16x16xf16> -> tensor<1x4x16xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 8, 2]} : tensor<1x4x16xf16> -> tensor<4x8x2xf16>
    // CHECK: return [[VAL7]] : tensor<4x8x2xf16>
}

// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#CN = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @RDFTTransformWithSignalSize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8xf16>
func.func @RDFTTransformWithSignalSize(%arg0: tensor<4x8xf16>) -> tensor<4x5x2xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %cst_1 = const.Declare tensor<2xsi64> = dense<[4, 8]> : tensor<2xsi64>
  %0 = IE.RDFT(%arg0, %cst, %cst_1) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<4x8xf16>, tensor<2xsi64>, tensor<2xsi64> -> tensor<4x5x2xf16>
  return %0 : tensor<4x5x2xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[CST1:%.*]] = const.Declare tensor<2xsi64> = dense<[4, 8]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose([[INPUT]]) {order_value = #CN} : tensor<4x8xf16> -> tensor<8x4xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 4]} : tensor<8x4xf16> -> tensor<1x8x4xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<8x4xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x4xf16>, tensor<8x4xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<10x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<10x16xf16> -> tensor<1x4x10xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 5, 2]} : tensor<1x4x10xf16> -> tensor<4x5x2xf16>
    // CHECK: return [[VAL7]] : tensor<4x5x2xf16>
}

// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: @IRDFTTransformWithSignalSize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<4x8x2xf16>
func.func @IRDFTTransformWithSignalSize(%arg0: tensor<4x8x2xf16>) -> tensor<4x14xf16> {
  %cst = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
  %cst_1 = const.Declare tensor<2xsi64> = dense<[4, 14]> : tensor<2xsi64>
  %0 = IE.IRDFT(%arg0, %cst, %cst_1) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<4x8x2xf16>, tensor<2xsi64>, tensor<2xsi64> -> tensor<4x14xf16>
  return %0 : tensor<4x14xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    // CHECK: [[CST1:%.*]] = const.Declare tensor<2xsi64> = dense<[4, 14]> : tensor<2xsi64>
    // CHECK: [[VAL0:%.+]] = IE.Transpose(%arg0) {order_value = #HCW} : tensor<4x8x2xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 8, 8]} : tensor<8x4x2xf16> -> tensor<1x8x8xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<8x8xf16> = dense<
    // CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[CST_0]]) {transpose_b} : tensor<1x8x8xf16>, tensor<8x8xf16> -> tensor<1x8x8xf16>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [8, 4, 2]} : tensor<1x8x8xf16> -> tensor<8x4x2xf16>
    // CHECK: [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #HCW} : tensor<8x4x2xf16> -> tensor<4x8x2xf16>
    // CHECK: [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {shape_value = [1, 4, 16]} : tensor<4x8x2xf16> -> tensor<1x4x16xf16>
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<14x16xf16> = dense<
    // CHECK: [[VAL6:%.+]] = IE.MatMul([[VAL5]], [[CST_1]]) {transpose_b} : tensor<1x4x16xf16>, tensor<14x16xf16> -> tensor<1x4x14xf16>
    // CHECK: [[VAL7:%.+]] = IE.Reshape([[VAL6]]) {shape_value = [4, 14]} : tensor<1x4x14xf16> -> tensor<4x14xf16>
    // CHECK: return [[VAL7]] : tensor<4x14xf16>
}
