//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --pad-dynamic-inputs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaxPool
func.func @MaxPool(%IN: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>)
    -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>

    %DIVISOR = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    // CHECK:   [[DIVISOR:%.+]] = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x16x32xf16>

    %POOL = IE.MaxPool(%IN) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[EXPAND]]) {
    // CHECK-SAME:  } : tensor<1x3x16x32xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])

    %DYN_DIM_16 = IE.Slice %SHAPE_OF [3] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[DYN_DIM_16:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]

    %DIV_DIM = IE.Divide(%DYN_DIM_16, %DIVISOR) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xsi64>
    // CHECK:   [[DIV_DIM:%.+]] = IE.Divide([[DYN_DIM_16]], [[DIVISOR]])

    %CONCAT = IE.Concat(%DIM_1, %DIM_3, %DIM_8, %DIV_DIM) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DIM_3]], [[DIM_8]], [[DIV_DIM]])

    %SLICE_OUT = IE.StridedSlice(%POOL, %CONCAT) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[POOL]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReLU
func.func @ReLU(%IN: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>)
    -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %RELU = IE.ReLU(%IN) : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[RELU:%.+]] = IE.ReLU([[EXPAND]]) : tensor<1x3x8x16xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])

    %DYN_DIM_16 = IE.Slice %SHAPE_OF [3] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[DYN_DIM_16:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]

    %CONCAT = IE.Concat(%DIM_1, %DIM_3, %DIM_8, %DYN_DIM_16) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DIM_3]], [[DIM_8]], [[DYN_DIM_16]])

    %SLICE_OUT = IE.StridedSlice(%RELU, %CONCAT) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Add
func.func @Add(
    %IN: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>,
    %BIAS: tensor<1x3x1x1xf16>
) -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, [[BIAS:%.+]]: tensor<1x3x1x1xf16>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %ADD = IE.Add(%IN, %BIAS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    }   : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<1x3x1x1xf16>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[ADD:%.+]] = IE.Add([[EXPAND]], [[BIAS]])
    // CHECK-SAME:  tensor<1x3x8x16xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])

    %DYN_DIM_16 = IE.Slice %SHAPE_OF [3] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[DYN_DIM_16:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]

    %CONCAT = IE.Concat(%DIM_1, %DIM_3, %DIM_8, %DYN_DIM_16) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DIM_3]], [[DIM_8]], [[DYN_DIM_16]])

    %SLICE_OUT = IE.StridedSlice(%ADD, %CONCAT) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[ADD]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Convolution
func.func @Convolution(
    %IN: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>,
    %KERNEL: tensor<3x3x1x1xf16>
) -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, [[KERNEL:%.+]]: tensor<3x3x1x1xf16>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %CONV = IE.Convolution(%IN, %KERNEL) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<3x3x1x1xf16>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[KERNEL]])
    // CHECK-SAME:  tensor<1x3x8x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])

    %DYN_DIM_16 = IE.Slice %SHAPE_OF [3] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[DYN_DIM_16:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]

    %CONCAT = IE.Concat(%DIM_1, %DIM_3, %DIM_8, %DYN_DIM_16) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DIM_3]], [[DIM_8]], [[DYN_DIM_16]])

    %SLICE_OUT = IE.StridedSlice(%CONV, %CONCAT) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[CONV]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaxPoolReLU
func.func @MaxPoolReLU(%IN: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>)
    -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>

    %DIVISOR = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    // CHECK:   [[DIVISOR:%.+]] = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x16x32xf16>

    %POOL = IE.MaxPool(%IN) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[EXPAND]]) {
    // CHECK-SAME:  } : tensor<1x3x16x32xf16> -> tensor<1x3x8x16xf16>

    %RELU = IE.ReLU(%POOL) : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[RELU:%.+]] = IE.ReLU([[POOL]]) : tensor<1x3x8x16xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])

    %DYN_DIM_16 = IE.Slice %SHAPE_OF [3] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[DYN_DIM_16:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]

    %DIV_DIM = IE.Divide(%DYN_DIM_16, %DIVISOR) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xsi64>
    // CHECK:   [[DIV_DIM:%.+]] = IE.Divide([[DYN_DIM_16]], [[DIVISOR]])

    %CONCAT = IE.Concat(%DIM_1, %DIM_3, %DIM_8, %DIV_DIM) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DIM_3]], [[DIM_8]], [[DIV_DIM]])

    %SLICE_OUT = IE.StridedSlice(%RELU, %CONCAT) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipSingleReshape
func.func @SkipSingleReshape(%IN: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>)
    -> tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
    %DIMS = const.Declare tensor<4xsi64> = dense<[1, 16, 3, 0]> : tensor<4xsi64>
    // CHECK:   [[DIMS:%.+]] = const.Declare tensor<4xsi64> = dense<[1, 16, 3, 0]> : tensor<4xsi64>

    %RESHAPE_OUT = IE.DynamicReshape(%IN, %DIMS) {
        output_bounds = [1, 16, 3, 32],
        output_shape = [1, 16, 3, -9223372036854775808]
    } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[IN]], [[DIMS]]) {
    // CHECK-SAME:  } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipEmptySubgraph
func.func @SkipEmptySubgraph(
    %IN: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>,
    %DIMS: tensor<4xsi64>
)
    -> tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK-SAME:  [[DIMS:%.+]]: tensor<4xsi64>

    %SLICE_OUT = IE.StridedSlice(%IN, %DIMS) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[IN]], [[DIMS]]) {
    // CHECK-SAME:  } : tensor<1x3x16x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %DIMS) {
        output_bounds = [1, 16, 3, 32],
        output_shape = [1, 16, 3, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[DIMS]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = [1, 3, 16, 32], order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x16x3x?xf16, {bounds = [1, 16, 3, 32], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoStridedSliceAfterStaticSubgraph
func.func @NoStridedSliceAfterStaticSubgraph(
    %IN: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>,
    %KERNEL: tensor<3x3x1x1xf16>
) -> tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, [[KERNEL:%.+]]: tensor<3x3x1x1xf16>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %CONV = IE.Convolution(%IN, %KERNEL) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>, tensor<3x3x1x1xf16>
        -> tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[KERNEL]])
    // CHECK-SAME:  tensor<1x3x8x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x8x16xf16>

    %TRANSPOSE = IE.Transpose(%CONV) {order_value = #NWCH}
        : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}>
        -> tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}>

    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[CONV]])
    // CHECK-SAME:  tensor<1x3x8x16xf16> -> tensor<1x16x3x8xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = [1, 3, 8, 16], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])

    %DYN_DIM_16 = IE.Slice %SHAPE_OF [3] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[DYN_DIM_16:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]

    %CONCAT = IE.Concat(%DIM_1, %DYN_DIM_16, %DIM_3, %DIM_8) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DYN_DIM_16]], [[DIM_3]], [[DIM_8]])

    %RESHAPE_OUT = IE.DynamicReshape(%TRANSPOSE, %CONCAT) {
        output_bounds = [1, 16, 3, 8],
        output_shape = [1, -9223372036854775808, 3, 8]
    } : tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[TRANSPOSE]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x16x3x8xf16>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x?x3x8xf16, {bounds = [1, 16, 3, 8], order = #NCHW}>
}
