//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --apply-dynamic-boundary-correction %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// -----

// CHECK-LABEL: @ClearDynGarbageAfterDynamicAdd
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>
func.func @ClearDynGarbageAfterDynamicAdd(%arg0: tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>) -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}> {
    %cst_0 = const.Declare tensor<1x3x1x1xf32> = dense<10.000000e+00> : tensor<1x3x1x1xf32> isSplat
    %cst_1 = const.Declare tensor<3x3x3x3xf32> = dense<3.000000e+00> : tensor<3x3x3x3xf32>
    %1 = IE.Add(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<1x3x1x1xf32> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<1.000000e+01> : tensor<1x3x1x1xf32>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<3x3x3x3xf32> = dense<3.000000e+00> : tensor<3x3x3x3xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[ARG0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<1x3x1x1xf32> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>

    // CHECK: [[C3:%.+]] = arith.constant 3 : index
    // CHECK: [[DIM:%.+]] = tensor.dim [[ADD]], [[C3]] : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK: [[CST_3:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK: [[IDX_CAST:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[IDX_CAST]] : tensor<1xi64>
    // CHECK: [[BITCAST:%.+]] = tensor.bitcast [[FROM_ELEMENTS]] : tensor<1xi64> to tensor<1xsi64>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_1]], [[CST_2]], [[CST_3]], [[BITCAST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK: [[MASK:%.+]] = IE.DynamicDataMask([[CONCAT]]) {outputTensorType = tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>} : tensor<4xsi64> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ADD]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}

    %2 = IE.Convolution(%1, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<3x3x3x3xf32> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[MULTIPLY]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<3x3x3x3xf32> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}

    return %2 : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>
    // CHECK: return [[CONV]] : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}

}

// -----

// CHECK-LABEL: @ClearDynGarbageIsNotNeeded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>
func.func @ClearDynGarbageIsNotNeeded(%arg0: tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>) -> tensor<1x3x18x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 18, 34]> : tensor<4xsi64>}> {
    %cst_0 = const.Declare tensor<1x3x1x1xf32> = dense<10.000000e+00> : tensor<1x3x1x1xf32> isSplat
    %cst_1 = const.Declare tensor<3x3x1x1xf32> = dense<3.000000e+00> : tensor<3x3x1x1xf32>

    %1 = IE.Add(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<1x3x1x1xf32> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>
    %2 = IE.Convolution(%1, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<3x3x1x1xf32> -> tensor<1x3x18x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 18, 34]> : tensor<4xsi64>}>

    return %2 : tensor<1x3x18x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 18, 34]> : tensor<4xsi64>}>
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<1.000000e+01> : tensor<1x3x1x1xf32>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<3x3x1x1xf32> = dense<3.000000e+00> : tensor<3x3x1x1xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[ARG0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<1x3x1x1xf32> -> tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}

    // CHECK-NOT: IE.DynamicDataMask

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ADD]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>}>, tensor<3x3x1x1xf32> -> tensor<1x3x18x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 18, 34]> : tensor<4xsi64>}
    // CHECK: return [[CONV]] : tensor<1x3x18x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 18, 34]> : tensor<4xsi64>}
}
