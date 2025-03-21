//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --populate-dynamic-dimensions-hw %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: ConvertReLU
func.func @ConvertReLU(
    %IN: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf32

    %RELU = IE.ReLU(%IN) :
        tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK: [[RELU:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-SAME:  tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[DYN_DIM_IDX:%.+]] = arith.constant 3 : index
    // CHECK:   [[DYN_DIM_VALUE:%.+]] = tensor.dim [[RELU]], [[DYN_DIM_IDX]]
    // CHECK:   [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[STATIC_DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK:   [[DYN_DIM_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE]] : index to i64
    // CHECK:   [[I64_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I64]] : tensor<1xi64>
    // CHECK:   [[DYN_DIM_SI64:%.+]] = tensor.bitcast [[I64_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[CONCAT_DIMS:%.+]] = IE.Concat([[STATIC_DIM_1]], [[STATIC_DIM_3]], [[STATIC_DIM_16]], [[DYN_DIM_SI64]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[SLICE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [1, 3, 16, 32],
    // CHECK-SAME:      output_shape = [1, 3, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    return %RELU : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   return [[RESHAPE]] : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: ConvertReLUTwoDynamicDims
func.func @ConvertReLUTwoDynamicDims(
    %IN: tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x?x16x?xf32

    %RELU = IE.ReLU(%IN) :
        tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK: [[RELU:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-SAME:  tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[DYN_DIM_IDX_3:%.+]] = arith.constant 1 : index
    // CHECK:   [[DYN_DIM_3:%.+]] = tensor.dim [[RELU]], [[DYN_DIM_IDX_3]]
    // CHECK:   [[DYN_DIM_IDX_32:%.+]] = arith.constant 3 : index
    // CHECK:   [[DYN_DIM_32:%.+]] = tensor.dim [[RELU]], [[DYN_DIM_IDX_32]]
    // CHECK:   [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DYN_DIM_3_I64:%.+]] = arith.index_cast [[DYN_DIM_3]] : index to i64
    // CHECK:   [[DYN_DIM_3_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_3_I64]] : tensor<1xi64>
    // CHECK:   [[DYN_DIM_3_SI64:%.+]] = tensor.bitcast [[DYN_DIM_3_TENSOR]] : tensor<1xi64> to tensor<1xsi64>
    // CHECK:   [[STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK:   [[DYN_DIM_32_I64:%.+]] = arith.index_cast [[DYN_DIM_32]] : index to i64
    // CHECK:   [[DYN_DIM_32_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_32_I64]] : tensor<1xi64>
    // CHECK:   [[DYN_DIM_32_SI64:%.+]] = tensor.bitcast [[DYN_DIM_32_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[CONCAT_DIMS:%.+]] = IE.Concat([[STATIC_DIM_1]], [[DYN_DIM_3_SI64]], [[STATIC_DIM_16]], [[DYN_DIM_32_SI64]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:  } : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[SLICE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [1, 3, 16, 32],
    // CHECK-SAME:      output_shape = [1, -9223372036854775808, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    return %RELU : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   return [[RESHAPE]] : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: Convert3dReLU
func.func @Convert3dReLU(
    %IN: tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
) -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}> {
    // CHECK:   [[IN:%.+]]: tensor<3x16x?xf32

    %RELU = IE.ReLU(%IN) :
        tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
        -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
    // CHECK: [[RELU:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-SAME:  tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
    // CHECK-SAME:  -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>

    // CHECK:   [[DYN_DIM_IDX:%.+]] = arith.constant 2 : index
    // CHECK:   [[DYN_DIM_VALUE:%.+]] = tensor.dim [[RELU]], [[DYN_DIM_IDX]]
    // CHECK:   [[STATIC_DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK:   [[DYN_DIM_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE]] : index to i64
    // CHECK:   [[I64_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I64]] : tensor<1xi64>
    // CHECK:   [[DYN_DIM_SI64:%.+]] = tensor.bitcast [[I64_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[CONCAT_DIMS:%.+]] = IE.Concat([[STATIC_DIM_3]], [[STATIC_DIM_16]], [[DYN_DIM_SI64]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>

    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1]
    // CHECK-SAME:  } : tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>, tensor<3xsi64> -> tensor<?x?x?xf32, {bounds = [3, 16, 32], order = #CHW}>

    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[SLICE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [3, 16, 32],
    // CHECK-SAME:      output_shape = [3, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<?x?x?xf32, {bounds = [3, 16, 32], order = #CHW}>, tensor<3xsi64> -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>

    return %RELU : tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
    // CHECK:   return [[RESHAPE]] : tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: ReturnWithTwoOperands
func.func @ReturnWithTwoOperands(
    %IN: tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> (tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>) {
    // CHECK:   [[IN:%.+]]: tensor<1x?x16x?xf32

    %RELU_0 = IE.ReLU(%IN) :
        tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK: [[RELU_0:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-SAME:  tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    %RELU_1 = IE.ReLU(%IN) :
        tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK: [[RELU_1:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-SAME:  tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RELU_0_DYN_DIM_IDX_3:%.+]] = arith.constant 1 : index
    // CHECK:   [[RELU_0_DYN_DIM_3:%.+]] = tensor.dim [[RELU_0]], [[RELU_0_DYN_DIM_IDX_3]]
    // CHECK:   [[RELU_0_DYN_DIM_IDX_32:%.+]] = arith.constant 3 : index
    // CHECK:   [[RELU_0_DYN_DIM_32:%.+]] = tensor.dim [[RELU_0]], [[RELU_0_DYN_DIM_IDX_32]]
    // CHECK:   [[RELU_0_STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[RELU_0_DYN_DIM_3_I64:%.+]] = arith.index_cast [[RELU_0_DYN_DIM_3]] : index to i64
    // CHECK:   [[RELU_0_DYN_DIM_3_TENSOR:%.+]] = tensor.from_elements [[RELU_0_DYN_DIM_3_I64]] : tensor<1xi64>
    // CHECK:   [[RELU_0_DYN_DIM_3_SI64:%.+]] = tensor.bitcast [[RELU_0_DYN_DIM_3_TENSOR]] : tensor<1xi64> to tensor<1xsi64>
    // CHECK:   [[RELU_0_STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK:   [[RELU_0_DYN_DIM_32_I64:%.+]] = arith.index_cast [[RELU_0_DYN_DIM_32]] : index to i64
    // CHECK:   [[RELU_0_DYN_DIM_32_TENSOR:%.+]] = tensor.from_elements [[RELU_0_DYN_DIM_32_I64]] : tensor<1xi64>
    // CHECK:   [[RELU_0_DYN_DIM_32_SI64:%.+]] = tensor.bitcast [[RELU_0_DYN_DIM_32_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[RELU_0_CONCAT_DIMS:%.+]] = IE.Concat([[RELU_0_STATIC_DIM_1]], [[RELU_0_DYN_DIM_3_SI64]], [[RELU_0_STATIC_DIM_16]], [[RELU_0_DYN_DIM_32_SI64]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[SLICE_0:%.+]] = IE.StridedSlice([[RELU_0]], [[RELU_0_CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:  } : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RESHAPE_0:%.+]] = IE.DynamicReshape([[SLICE_0]], [[RELU_0_CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [1, 3, 16, 32],
    // CHECK-SAME:      output_shape = [1, -9223372036854775808, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RELU_1_DYN_DIM_IDX_3:%.+]] = arith.constant 1 : index
    // CHECK:   [[RELU_1_DYN_DIM_3:%.+]] = tensor.dim [[RELU_1]], [[RELU_1_DYN_DIM_IDX_3]]
    // CHECK:   [[RELU_1_DYN_DIM_IDX_32:%.+]] = arith.constant 3 : index
    // CHECK:   [[RELU_1_DYN_DIM_32:%.+]] = tensor.dim [[RELU_1]], [[RELU_1_DYN_DIM_IDX_32]]
    // CHECK:   [[RELU_1_STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[RELU_1_DYN_DIM_3_I64:%.+]] = arith.index_cast [[RELU_1_DYN_DIM_3]] : index to i64
    // CHECK:   [[RELU_1_DYN_DIM_3_TENSOR:%.+]] = tensor.from_elements [[RELU_1_DYN_DIM_3_I64]] : tensor<1xi64>
    // CHECK:   [[RELU_1_DYN_DIM_3_SI64:%.+]] = tensor.bitcast [[RELU_1_DYN_DIM_3_TENSOR]] : tensor<1xi64> to tensor<1xsi64>
    // CHECK:   [[RELU_1_STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK:   [[RELU_1_DYN_DIM_32_I64:%.+]] = arith.index_cast [[RELU_1_DYN_DIM_32]] : index to i64
    // CHECK:   [[RELU_1_DYN_DIM_32_TENSOR:%.+]] = tensor.from_elements [[RELU_1_DYN_DIM_32_I64]] : tensor<1xi64>
    // CHECK:   [[RELU_1_DYN_DIM_32_SI64:%.+]] = tensor.bitcast [[RELU_1_DYN_DIM_32_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[RELU_1_CONCAT_DIMS:%.+]] = IE.Concat([[RELU_1_STATIC_DIM_1]], [[RELU_1_DYN_DIM_3_SI64]], [[RELU_1_STATIC_DIM_16]], [[RELU_1_DYN_DIM_32_SI64]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[SLICE_1:%.+]] = IE.StridedSlice([[RELU_1]], [[RELU_1_CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:  } : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RESHAPE_1:%.+]] = IE.DynamicReshape([[SLICE_1]], [[RELU_1_CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [1, 3, 16, 32],
    // CHECK-SAME:      output_shape = [1, -9223372036854775808, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    return %RELU_0, %RELU_1 : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   return [[RESHAPE_0]], [[RESHAPE_1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: SkipEmptyNets
func.func @SkipEmptyNets(
    %IN: tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x?x16x?xf32

    return %IN : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   return [[IN]] : tensor<1x?x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: SkipOperationsWithoutReifyInterface
func.func @SkipOperationsWithoutReifyInterface(
    %IN: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>,
    %DIMS: tensor<4xsi64>
) -> tensor<1x16x3x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}> {
    // CHECK:       [[IN:%.+]]: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, [[DIMS:%.+]]: tensor<4xsi64>

    %RESHAPE = IE.DynamicReshape(%IN, %DIMS) {
        output_bounds = [1, 3, 16, 32],
        output_shape = [1, 16, 3, -9223372036854775808]
    } : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x16x3x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[IN]], [[DIMS]]) {

    return %RESHAPE : tensor<1x16x3x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   return [[RESHAPE]] : tensor<1x16x3x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
}

// -----

// CHECK-LABEL: SkipReLUWithStaticShape
func.func @SkipReLUWithStaticShape(%IN: tensor<1x3x16x32xf32>) -> tensor<1x3x16x32xf32> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x32xf32>

    %RELU = IE.ReLU(%IN) : tensor<1x3x16x32xf32> -> tensor<1x3x16x32xf32>
    // CHECK: [[RELU:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-NOT: tensor.dim

    return %RELU : tensor<1x3x16x32xf32>
    // CHECK:   return [[RELU]] : tensor<1x3x16x32xf32>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: ConvertReLUWithReshape
func.func @ConvertReLUWithReshape(
    %IN: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf32

    %OUT_SHAPE = const.Declare tensor<3xsi64> = dense<[3, 16, -1]> : tensor<3xsi64>
    // CHECK:   [[OUT_SHAPE:%.+]] = const.Declare tensor<3xsi64> = dense<[3, 16, -1]> : tensor<3xsi64>

    %RELU = IE.ReLU(%IN) :
        tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK: [[RELU:%.+]] = IE.ReLU([[IN]]) :
    // CHECK-SAME:  tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[DYN_DIM_IDX:%.+]] = arith.constant 3 : index
    // CHECK:   [[DYN_DIM_VALUE:%.+]] = tensor.dim [[RELU]], [[DYN_DIM_IDX]]
    // CHECK:   [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[STATIC_DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>
    // CHECK:   [[DYN_DIM_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE]] : index to i64
    // CHECK:   [[I64_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I64]] : tensor<1xi64>
    // CHECK:   [[DYN_DIM_SI64:%.+]] = tensor.bitcast [[I64_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[CONCAT_DIMS:%.+]] = IE.Concat([[STATIC_DIM_1]], [[STATIC_DIM_3]], [[STATIC_DIM_16]], [[DYN_DIM_SI64]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[SLICE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [1, 3, 16, 32],
    // CHECK-SAME:      output_shape = [1, 3, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<?x?x?x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64> -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    %OUT_RESHAPE = IE.DynamicReshape(%RELU, %OUT_SHAPE) {
        output_bounds = [3, 16, 32],
        output_shape = [3, 16, -9223372036854775808]
    } : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<3xsi64>
        -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>

    // CHECK:   [[OUT_RESHAPE:%.+]] = IE.DynamicReshape([[RESHAPE]], [[OUT_SHAPE]]) {
    // CHECK-SAME:      output_bounds = [3, 16, 32],
    // CHECK-SAME:      output_shape = [3, 16, -9223372036854775808]
    // CHECK-SAME:  } : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<3xsi64>
    // CHECK-SAME:      -> tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>

    return %OUT_RESHAPE : tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
    // CHECK:   return [[OUT_RESHAPE]] : tensor<3x16x?xf32, {bounds = [3, 16, 32], order = #CHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#WNHC = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>

// CHECK-LABEL: ConvertTransposeWithDynamicReshape
func.func @ConvertTransposeWithDynamicReshape(
    %IN: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf32

    %TRANSPOSE = IE.Transpose(%IN) {order_value = #WNHC} :
        tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])
    // CHECK-SAME:  : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}>

    // CHECK-DAG:   [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG:   [[STATIC_DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK-DAG:   [[STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>

    // CHECK-DAG:   [[DYN_DIM_IDX:%.+]] = arith.constant 0 : index
    // CHECK-DAG:   [[DYN_DIM_VALUE:%.+]] = tensor.dim [[TRANSPOSE]], [[DYN_DIM_IDX]]
    // CHECK-DAG:   [[DYN_DIM_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE]] : index to i64
    // CHECK-DAG:   [[I64_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I64]] : tensor<1xi64>
    // CHECK-DAG:   [[DYN_DIM_SI64:%.+]] = tensor.bitcast [[I64_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[CONCAT_DIMS:%.+]] = IE.Concat([[DYN_DIM_SI64]], [[STATIC_DIM_1]], [[STATIC_DIM_16]], [[STATIC_DIM_3]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[TRANSPOSE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [32, 1, 16, 3],
    // CHECK-SAME:      output_shape = [-9223372036854775808, 1, 16, 3]
    // CHECK-SAME:  } : tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:    -> tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}>

    return %TRANSPOSE : tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}>
    // CHECK:   return [[RESHAPE]] : tensor<?x1x16x3xf32, {bounds = [32, 1, 16, 3], order = #NCHW}>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CWNH = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: ConvertTransposeWithStridedSlice
func.func @ConvertTransposeWithStridedSlice(
    %IN: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf32

    %TRANSPOSE = IE.Transpose(%IN) {order_value = #CWNH} :
        tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])
    // CHECK-SAME:  : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  -> tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>

    // CHECK-DAG:   [[DYN_DIM_IDX:%.+]] = arith.constant 1 : index
    // CHECK-DAG:   [[DYN_DIM_VALUE:%.+]] = tensor.dim [[TRANSPOSE]], [[DYN_DIM_IDX]]
    // CHECK-DAG:   [[DYN_DIM_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE]] : index to i64
    // CHECK-DAG:   [[I64_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I64]] : tensor<1xi64>
    // CHECK-DAG:   [[DYN_DIM_SI64:%.+]] = tensor.bitcast [[I64_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG:   [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG:   [[STATIC_DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK-DAG:   [[STATIC_DIM_16:%.+]] = const.Declare tensor<1xsi64> = dense<16> : tensor<1xsi64>

    // CHECK:   [[CONCAT_DIMS:%.+]] = IE.Concat([[STATIC_DIM_3]], [[DYN_DIM_SI64]], [[STATIC_DIM_1]], [[STATIC_DIM_16]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[TRANSPOSE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      begin_mask = [],
    // CHECK-SAME:      begins_attr = [0, 0, 0, 0],
    // CHECK-SAME:      ellipsis_mask = [],
    // CHECK-SAME:      end_mask = [],
    // CHECK-SAME:      new_axis_mask = [],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>,
    // CHECK-SAME:      shrink_axis_mask = [],
    // CHECK-SAME:      strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:  } : tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:    -> tensor<?x?x?x?xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>

    // CHECK:   [[RESHAPE:%.+]] = IE.DynamicReshape([[SLICE]], [[CONCAT_DIMS]]) {
    // CHECK-SAME:      output_bounds = [3, 32, 1, 16],
    // CHECK-SAME:      output_shape = [3,  -9223372036854775808, 1, 16]
    // CHECK-SAME:  } : tensor<?x?x?x?xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:    -> tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>

    return %TRANSPOSE : tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>
    // CHECK:   return [[RESHAPE]] : tensor<3x?x1x16xf32, {bounds = [3, 32, 1, 16], order = #NCHW}>
}
