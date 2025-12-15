//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-gather-to-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertGatherToSliceAxis0
func.func @ConvertGatherToSliceAxis0(%arg0: tensor<18x8x72x64xf16>) -> tensor<8x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<9> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<8x72x64xf16>

    return %0 : tensor<8x72x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<si32> = dense<9> : tensor<si32>
    // CHECK:       [[SLICE:%.*]] = IE.Slice %arg0 [9, 0, 0, 0] [1, 8, 72, 64] : tensor<18x8x72x64xf16> to tensor<1x8x72x64xf16>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[SLICE]]) {shape_value = [8, 72, 64]} : tensor<1x8x72x64xf16> -> tensor<8x72x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @ConvertGatherToSliceAxis1
func.func @ConvertGatherToSliceAxis1(%arg0: tensor<18x8x72x64xf16>) -> tensor<18x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<3> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<18x72x64xf16>

    return %0 : tensor<18x72x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<si32> = dense<3> : tensor<si32>
    // CHECK:       [[SLICE:%.*]] = IE.Slice %arg0 [0, 3, 0, 0] [18, 1, 72, 64] : tensor<18x8x72x64xf16> to tensor<18x1x72x64xf16>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[SLICE]]) {shape_value = [18, 72, 64]} : tensor<18x1x72x64xf16> -> tensor<18x72x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @ConvertGatherToSlicewith3DShape
func.func @ConvertGatherToSlicewith3DShape(%arg0: tensor<8x72x64xf16>) -> tensor<8x64xf16> {
    %cst = const.Declare tensor<si32> = dense<8> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<8x72x64xf16>, tensor<si32> -> tensor<8x64xf16>

    return %0 : tensor<8x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<si32> = dense<8> : tensor<si32>
    // CHECK:       [[SLICE:%.*]] = IE.Slice %arg0 [0, 8, 0] [8, 1, 64] : tensor<8x72x64xf16> to tensor<8x1x64xf16>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[SLICE]]) {shape_value = [8, 64]} : tensor<8x1x64xf16> -> tensor<8x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @CannotConvertGatherToSlice
func.func @CannotConvertGatherToSlice(%arg0: tensor<1xf32>, %arg1: tensor<1x8x16x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = si32} : tensor<1xf32> -> tensor<1xsi32>
    %1 = IE.Gather(%arg1, %0) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x8x16x16xf16>, tensor<1xsi32> -> tensor<1x1x16x16xf16>

    return %1 : tensor<1x1x16x16xf16>

    // CHECK:       %0 = IE.Convert(%arg0) {dstElemType = si32} : tensor<1xf32> -> tensor<1xsi32>
    // CHECK:       %1 = IE.Gather(%arg1, %0) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x8x16x16xf16>, tensor<1xsi32> -> tensor<1x1x16x16xf16>
    // CHECK:       return %1 : tensor<1x1x16x16xf16>
}
