//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-extra-static-shape-ops=false" --dynamic-shape-transformations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#WNCH = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
!BoundedInType = tensor<1x512x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
!BoundedOutType = tensor<1x16x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
!BoundedTransposeType = tensor<?x1x16x4xf32, {bounds = #const.OpaqueI64Elements<[320, 1, 16, 4]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @DynamicConvAddTranpose
// CHECK-SAME: [[IN:%.+]]: tensor<1x512x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicConvAddTranpose(%arg0: !BoundedInType) -> !BoundedTransposeType {
    %weights = const.Declare tensor<16x512x1x1xf32> = dense<1.000000e+00> : tensor<16x512x1x1xf32>
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>

    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : !BoundedInType, tensor<16x512x1x1xf32> -> !BoundedOutType

    %add = IE.Add(%conv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : !BoundedOutType, tensor<1x16x1x1xf32> -> !BoundedOutType

    %transpose = IE.Transpose(%add) {order_value = #WNCH}
        : !BoundedOutType -> !BoundedTransposeType
    return %transpose : !BoundedTransposeType

    // CHECK-DAG:   [[DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK-DAG:   [[DIM_4:%.+]] = const.Declare tensor<1xsi64> = dense<4> : tensor<1xsi64>
    // CHECK-DAG:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:       [[DYN_EXPAND:%.+]] = IE.DynamicExpand([[IN]])
    // CHECK-SAME:       : tensor<1x512x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x512x4x320xf32>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[DYN_EXPAND]], {{%.+}})
    // CHECK-SAME:       : tensor<1x512x4x320xf32>, tensor<16x512x1x1xf32> -> tensor<1x16x4x320xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONV]], {{%.+}})
    // CHECK-SAME:       : tensor<1x16x4x320xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x4x320xf32>

    // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])
    // CHECK-SAME:      -> tensor<4xsi64>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]
    // CHECK-SAME:      to tensor<1xsi64>
    // CHECK:       [[NEW_SHAPE:%.+]] = IE.Concat([[DIM_1]], [[DIM_16]], [[DIM_4]], [[SLICE]])
    // CHECK-SAME:      -> tensor<4xsi64>
    // CHECK:       [[STRIDED_SLICE:%.+]] = IE.StridedSlice([[ADD]], [[NEW_SHAPE]])
    // CHECK-SAME:      -> tensor<?x?x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[STRIDED_SLICE]], [[NEW_SHAPE]])
    // CHECK-SAME:    {output_bounds = [1, 16, 4, 320], output_shape = [1, 16, 4, -9223372036854775808]}
    // CHECK-SAME:    : tensor<?x?x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 320]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:    -> tensor<1x16x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[DYN_RESHAPE]]) {order_value = #map}
    // CHECK-SAME:    : tensor<1x16x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:    -> tensor<?x1x16x4xf32, {bounds = #const.OpaqueI64Elements<[320, 1, 16, 4]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[TRANSPOSE]]
}
