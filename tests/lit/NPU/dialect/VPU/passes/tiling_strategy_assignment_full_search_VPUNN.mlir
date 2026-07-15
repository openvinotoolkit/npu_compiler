//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --tiling-strategy-assignment="enable-tiling-full-search-space=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FullSearchAssignsTilingAttributes
// CHECK-SAME: [[INPUT:%arg[0-9]]]: tensor<1x1024x64x4xf16, {order = #NHWC}>
func.func @FullSearchAssignsTilingAttributes(%arg0: tensor<1x1024x64x4xf16, {order = #NHWC}>) -> tensor<1x1024x64x4xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %weights = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [1024, 1024, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } : tensor<1x1024x64x4xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x64x4xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x64x4xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<1024x1x1x4xsi32> = dense<1>
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>

    // COM: Check the strategy cost and tiling scenario and strategy are added as attributes
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) rawFilterShape [1024, 1024, 1, 1]
    // CHECK-SAME: strategyCost = {{[0-9]+}}
    // CHECK-SAME: tilingScenario = "{{ISOLATED|PREFETCH|PIPELINE}}"
    // CHECK-SAME: tilingStrategy = [{{.*[2-9].*}}]

    // CHECK: return [[CONV]] : tensor<1x1024x64x4xf16, {order = #NHWC}>
}
