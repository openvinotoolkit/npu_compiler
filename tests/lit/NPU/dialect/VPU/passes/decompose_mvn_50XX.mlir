//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-mvn %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @DecomposeMVNAcrossChannelFalseNHWC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x515971xf16, {order = #NHWC}>
func.func @DecomposeMVNAcrossChannelFalseNHWC(%arg0: tensor<1x1x1x515971xf16, {order = #NHWC}>) -> (tensor<1x1x1x515971xf16, {order = #NHWC}>) {
      %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x1x1x515971xf16, {order = #NHWC}> -> tensor<1x1x1x515971xf16, {order = #NHWC}>
      return %0 : tensor<1x1x1x515971xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_1:%.+]] = VPU.ShapeCast {shape = [1, 1, 515971, 1]} inputs([[INPUT]] : tensor<1x1x1x515971xf16, {order = #NHWC}>) -> tensor<1x1x515971x1xf16, {order = #NHWC}>
    // CHECK:        [[MVN1SUM:%.+]] = VPU.MVN1SumOp([[SHAPECAST_1]]) {across_channels = false, normalize_variance = true, output_height = 3 : i64}
    // CHECK-SAME:       : tensor<1x1x515971x1xf16, {order = #NHWC}> -> tensor<1x1x3x2xf32, {order = #NHWC}>
    // CHECK:        [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[MVN1SUM]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true, orig_shape = [1, 1, 1, 515971], output_type = f16}
    // CHECK-SAME:       : tensor<1x1x3x2xf32, {order = #NHWC}> -> tensor<1x1x1x2xf16, {order = #NHWC}>
    // CHECK:        [[MVN1NORMALIZE:%.+]] = VPU.MVN1Normalize([[SHAPECAST_1]], [[MVN1MEANVAR]]) {across_channels = false, normalize_variance = true}
    // CHECK-SAME:       : tensor<1x1x515971x1xf16, {order = #NHWC}>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x515971x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_2:%.+]] = VPU.ShapeCast {shape = [1, 1, 1, 515971]} inputs([[MVN1NORMALIZE]] : tensor<1x1x515971x1xf16, {order = #NHWC}>) -> tensor<1x1x1x515971xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_2]] : tensor<1x1x1x515971xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @DecomposeMVNAcrossChannelTrueNHWC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x6x85995x1xf16, {order = #NHWC}>
func.func @DecomposeMVNAcrossChannelTrueNHWC(%arg0: tensor<1x6x85995x1xf16, {order = #NHWC}>) -> (tensor<1x6x85995x1xf16, {order = #NHWC}>) {
      %0 = VPU.MVN(%arg0) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x6x85995x1xf16, {order = #NHWC}> -> tensor<1x6x85995x1xf16, {order = #NHWC}>
      return %0 : tensor<1x6x85995x1xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_1:%.+]] = VPU.ShapeCast {shape = [1, 1, 515970, 1]} inputs([[INPUT]] : tensor<1x6x85995x1xf16, {order = #NHWC}>) -> tensor<1x1x515970x1xf16, {order = #NHWC}>
    // CHECK:        [[MVN1SUM:%.+]] = VPU.MVN1SumOp([[SHAPECAST_1]]) {across_channels = true, normalize_variance = true, output_height = 3 : i64}
    // CHECK-SAME:       : tensor<1x1x515970x1xf16, {order = #NHWC}> -> tensor<1x1x3x2xf32, {order = #NHWC}>
    // CHECK:        [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[MVN1SUM]]) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true, orig_shape = [1, 6, 85995, 1], output_type = f16}
    // CHECK-SAME:       : tensor<1x1x3x2xf32, {order = #NHWC}> -> tensor<1x1x1x2xf16, {order = #NHWC}>
    // CHECK:        [[MVN1NORMALIZE:%.+]] = VPU.MVN1Normalize([[SHAPECAST_1]], [[MVN1MEANVAR]]) {across_channels = true, normalize_variance = true}
    // CHECK-SAME:       : tensor<1x1x515970x1xf16, {order = #NHWC}>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x515970x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_2:%.+]] = VPU.ShapeCast {shape = [1, 6, 85995, 1]} inputs([[MVN1NORMALIZE]] : tensor<1x1x515970x1xf16, {order = #NHWC}>) -> tensor<1x6x85995x1xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_2]] : tensor<1x6x85995x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @DecomposeMVNAcrossChannelTrueNCHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x5x1x115971xf16>
func.func @DecomposeMVNAcrossChannelTrueNCHW(%arg0: tensor<1x5x1x115971xf16>) -> (tensor<1x5x1x115971xf16>) {
      %0 = VPU.MVN(%arg0) {across_channels = true, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x5x1x115971xf16> -> tensor<1x5x1x115971xf16>
      return %0 : tensor<1x5x1x115971xf16>

    // CHECK:        [[SHAPECAST_1:%.+]] = VPU.ShapeCast {shape = [1, 1, 579855, 1]} inputs([[INPUT]] : tensor<1x5x1x115971xf16>) -> tensor<1x1x579855x1xf16>
    // CHECK:        [[MVN1SUM:%.+]] = VPU.MVN1SumOp([[SHAPECAST_1]]) {across_channels = true, normalize_variance = true, output_height = 3 : i64}
    // CHECK-SAME:       : tensor<1x1x579855x1xf16> -> tensor<1x1x3x2xf32, {order = #NHWC}>
    // CHECK:        [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[MVN1SUM]]) {across_channels = true, eps = 6.0892105102539063E-4 : f64, normalize_variance = true, orig_shape = [1, 5, 1, 115971], output_type = f16}
    // CHECK-SAME:       : tensor<1x1x3x2xf32, {order = #NHWC}> -> tensor<1x1x1x2xf16, {order = #NHWC}>
    // CHECK:        [[MVN1NORMALIZE:%.+]] = VPU.MVN1Normalize([[SHAPECAST_1]], [[MVN1MEANVAR]]) {across_channels = true, normalize_variance = true}
    // CHECK-SAME:       : tensor<1x1x579855x1xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x579855x1xf16>
    // CHECK:        [[SHAPECAST_2:%.+]] = VPU.ShapeCast {shape = [1, 5, 1, 115971]} inputs([[MVN1NORMALIZE]] : tensor<1x1x579855x1xf16>) -> tensor<1x5x1x115971xf16>
    // CHECK:        return [[SHAPECAST_2]] : tensor<1x5x1x115971xf16>
}
