//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --swap-mvn-with-transpose %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SwapTransposeWithMVN
func.func @SwapTransposeWithMVN(%arg0: tensor<128x32x64xf16>) -> tensor<1x128x32x64xf16> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 128, 32, 64] } : tensor<128x32x64xf16> -> tensor<1x128x32x64xf16>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x128x32x64xf16> -> tensor<1x128x64x32xf16>
    %2 = IE.MVN(%1) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x64x32xf16> -> tensor<1x128x64x32xf16>
    %3 = IE.Transpose(%2) {order_value = #NCWH} : tensor<1x128x64x32xf16> -> tensor<1x128x32x64xf16>

    return %3 : tensor<1x128x32x64xf16>

    // CHECK:   %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 128, 32, 64]} : tensor<128x32x64xf16> -> tensor<1x128x32x64xf16>
    // CHECK:   %[[MVN:.*]] = IE.MVN(%[[RESHAPE]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>
    // CHECK:   %[[TRANSPOSE1:.*]] = IE.Transpose(%[[MVN]]) {order_value = #NCWH} : tensor<1x128x32x64xf16> -> tensor<1x128x64x32xf16>
    // CHECK:   %[[TRANSPOSE2:.*]] = IE.Transpose(%[[TRANSPOSE1]]) {order_value = #NCWH} : tensor<1x128x64x32xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   return %[[TRANSPOSE2]] : tensor<1x128x32x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotSwapForBlockArgumentInput
func.func @NotSwapForBlockArgumentInput(%arg0: tensor<1x128x32x64xf16>) -> tensor<1x128x32x64xf16> {
    %1 = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x128x32x64xf16> -> tensor<1x128x64x32xf16>
    %2 = IE.MVN(%1) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x64x32xf16> -> tensor<1x128x64x32xf16>
    %3 = IE.Transpose(%2) {order_value = #NCWH} : tensor<1x128x64x32xf16> -> tensor<1x128x32x64xf16>

    return %3 : tensor<1x128x32x64xf16>

    // CHECK:   %[[TRANSPOSE1:.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x128x32x64xf16> -> tensor<1x128x64x32xf16>
    // CHECK:   %[[MVN:.*]] = IE.MVN(%[[TRANSPOSE1]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x64x32xf16> -> tensor<1x128x64x32xf16>
    // CHECK:   %[[TRANSPOSE2:.*]] = IE.Transpose(%[[MVN]]) {order_value = #NCWH} : tensor<1x128x64x32xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   return %[[TRANSPOSE2]] : tensor<1x128x32x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotSwapTransposeWithCrossChannelMVN
func.func @NotSwapTransposeWithCrossChannelMVN(%arg0: tensor<128x32x64xf16>) -> tensor<1x128x32x64xf16> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 128, 32, 64] } : tensor<128x32x64xf16> -> tensor<1x128x32x64xf16>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x128x32x64xf16> -> tensor<1x128x64x32xf16>
    %2 = IE.MVN(%1) {across_channels = true, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x64x32xf16> -> tensor<1x128x64x32xf16>
    %3 = IE.Transpose(%2) {order_value = #NCWH} : tensor<1x128x64x32xf16> -> tensor<1x128x32x64xf16>

    return %3 : tensor<1x128x32x64xf16>

    // CHECK:   %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 128, 32, 64]} : tensor<128x32x64xf16> -> tensor<1x128x32x64xf16>
    // CHECK:   %[[TRANSPOSE1:.*]] = IE.Transpose(%[[RESHAPE]]) {order_value = #NCWH} : tensor<1x128x32x64xf16> -> tensor<1x128x64x32xf16>
    // CHECK:   %[[MVN:.*]] = IE.MVN(%[[TRANSPOSE1]]) {across_channels = true, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x64x32xf16> -> tensor<1x128x64x32xf16>
    // CHECK:   %[[TRANSPOSE2:.*]] = IE.Transpose(%[[MVN]]) {order_value = #NCWH} : tensor<1x128x64x32xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   return %[[TRANSPOSE2]] : tensor<1x128x32x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotSwapMVNWithChannelSwappingTranspose
func.func @NotSwapMVNWithChannelSwappingTranspose(%arg0: tensor<128x32x64xf16>) -> tensor<1x128x32x64xf16> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 128, 32, 64] } : tensor<128x32x64xf16> -> tensor<1x128x32x64xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x128x32x64xf16> -> tensor<1x32x64x128xf16>
    %2 = IE.MVN(%1) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x32x64x128xf16> -> tensor<1x32x64x128xf16>
    %3 = IE.Transpose(%2) {order_value = #NWCH} : tensor<1x32x64x128xf16> -> tensor<1x128x32x64xf16>

    return %3 : tensor<1x128x32x64xf16>

    // CHECK:   %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 128, 32, 64]} : tensor<128x32x64xf16> -> tensor<1x128x32x64xf16>
    // CHECK:   %[[TRANSPOSE1:.*]] = IE.Transpose(%[[RESHAPE]]) {order_value = #NHWC} : tensor<1x128x32x64xf16> -> tensor<1x32x64x128xf16>
    // CHECK:   %[[MVN:.*]] = IE.MVN(%[[TRANSPOSE1]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x32x64x128xf16> -> tensor<1x32x64x128xf16>
    // CHECK:   %[[TRANSPOSE2:.*]] = IE.Transpose(%[[MVN]]) {order_value = #NWCH} : tensor<1x32x64x128xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   return %[[TRANSPOSE2]] : tensor<1x128x32x64xf16>
}
