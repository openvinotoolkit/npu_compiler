//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --pad-dynamic-inputs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaxPool
func.func @MaxPool(%IN: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>

    %DIVISOR = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    // CHECK:   [[DIVISOR:%.+]] = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x16x32xf16>

    %POOL = IE.MaxPool(%IN) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[EXPAND]]) {
    // CHECK-SAME:  } : tensor<1x3x16x32xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
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
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[POOL]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DynamicPermuteQuantize
func.func @DynamicPermuteQuantize(%IN: tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK: [[IN:%.+]]: tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    %DIM_32 = const.Declare tensor<1xsi64> = dense<32> : tensor<1xsi64>
    // CHECK:  [[DIM_32:%.+]] = const.Declare tensor<1xsi64> = dense<32> : tensor<1xsi64>

    %DIM_12 = const.Declare tensor<1xsi64> = dense<12> : tensor<1xsi64>
    // CHECK:  [[DIM_12:%.+]] = const.Declare tensor<1xsi64> = dense<12> : tensor<1xsi64>

    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:  [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK: [[EXPAND:%.+]] = IE.DynamicExpand([[IN]])
    // CHECK-SAME:   : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:   -> tensor<1x12x4x32xf16>

    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%IN) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
    : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>
    -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[PERMUTE:%.+]] = IE.PermuteQuantize([[EXPAND]])
    // CHECK-SAME:   {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x12x4x32xf16>
    // CHECK-SAME:   -> tensor<1x12x4x32xf16, {order = #NHWC}>

    %SHAPE_OF = IE.ShapeOf(%IN) {dstElemType = si64} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])
    // CHECK-SAME:   {dstElemType = si64} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:   -> tensor<4xsi64>

    %SLICE = IE.Slice %SHAPE_OF [2] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [2] [1] : tensor<4xsi64> to tensor<1xsi64>

    %CONCAT = IE.Concat(%DIM_1, %DIM_12, %SLICE, %DIM_32) {per_axis = #IE.Concat<axis = 0 : i64>}
    : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>
    -> tensor<4xsi64>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DIM_1]], [[DIM_12]], [[SLICE]], [[DIM_32]])
    // CHECK-SAME:   {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>
    // CHECK-SAME:   -> tensor<4xsi64>

    %RESHAPE_OUT = IE.DynamicReshape(%PERMUTE_QUANTIZE, %CONCAT) {output_bounds = [1, 12, 4, 32], output_shape = [1, 12, -9223372036854775808, 32]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>, tensor<4xsi64> -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[PERMUTE]], [[CONCAT]])
    //CHECK-SAME:   {output_bounds = [1, 12, 4, 32], output_shape = [1, 12, -9223372036854775808, 32]} : tensor<1x12x4x32xf16, {order = #NHWC}>, tensor<4xsi64>
    //CHECK-SAME:   -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:    [[RESHAPE_OUT]] : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReLU
func.func @ReLU(%IN: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %RELU = IE.ReLU(%IN) : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[RELU:%.+]] = IE.ReLU([[EXPAND]]) : tensor<1x3x8x16xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
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
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Add
func.func @Add(
    %IN: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>,
    %BIAS: tensor<1x3x1x1xf16>
) -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, [[BIAS:%.+]]: tensor<1x3x1x1xf16>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %ADD = IE.Add(%IN, %BIAS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    }   : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x3x1x1xf16>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[ADD:%.+]] = IE.Add([[EXPAND]], [[BIAS]])
    // CHECK-SAME:  tensor<1x3x8x16xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
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
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[ADD]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Convolution
func.func @Convolution(
    %IN: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>,
    %KERNEL: tensor<3x3x1x1xf16>
) -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, [[KERNEL:%.+]]: tensor<3x3x1x1xf16>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %CONV = IE.Convolution(%IN, %KERNEL) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<3x3x1x1xf16>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[KERNEL]])
    // CHECK-SAME:  tensor<1x3x8x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
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
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[CONV]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MaxPoolReLU
func.func @MaxPoolReLU(%IN: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>

    %DIVISOR = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    // CHECK:   [[DIVISOR:%.+]] = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x16x32xf16>

    %POOL = IE.MaxPool(%IN) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[EXPAND]]) {
    // CHECK-SAME:  } : tensor<1x3x16x32xf16> -> tensor<1x3x8x16xf16>

    %RELU = IE.ReLU(%POOL) : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[RELU:%.+]] = IE.ReLU([[POOL]]) : tensor<1x3x8x16xf16> -> tensor<1x3x8x16xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
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
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[RELU]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x3x8x16xf16>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %CONCAT) {
        output_bounds = [1, 3, 8, 16],
        output_shape = [1, 3, 8, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipSingleReshape
func.func @SkipSingleReshape(%IN: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
    %DIMS = const.Declare tensor<4xsi64> = dense<[1, 16, 3, 0]> : tensor<4xsi64>
    // CHECK:   [[DIMS:%.+]] = const.Declare tensor<4xsi64> = dense<[1, 16, 3, 0]> : tensor<4xsi64>

    %RESHAPE_OUT = IE.DynamicReshape(%IN, %DIMS) {
        output_bounds = [1, 16, 3, 32],
        output_shape = [1, 16, 3, -9223372036854775808]
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[IN]], [[DIMS]]) {
    // CHECK-SAME:  } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipEmptySubgraph
func.func @SkipEmptySubgraph(
    %IN: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>,
    %DIMS: tensor<4xsi64>
)
    -> tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   ([[IN:%.+]]: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:  [[DIMS:%.+]]: tensor<4xsi64>)

    %SLICE_OUT = IE.StridedSlice(%IN, %DIMS) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE_OUT:%.+]] = IE.StridedSlice([[IN]], [[DIMS]]) {
    // CHECK-SAME:  } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>

    %RESHAPE_OUT = IE.DynamicReshape(%SLICE_OUT, %DIMS) {
        output_bounds = [1, 16, 3, 32],
        output_shape = [1, 16, 3, -9223372036854775808]
    } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[SLICE_OUT]], [[DIMS]]) {
    // CHECK-SAME:  } : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x16x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 32]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoStridedSliceAfterStaticSubgraph
func.func @NoStridedSliceAfterStaticSubgraph(
    %IN: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>,
    %KERNEL: tensor<3x3x1x1xf16>
) -> tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, [[KERNEL:%.+]]: tensor<3x3x1x1xf16>

    %DIM_8 = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    // CHECK:   [[DIM_8:%.+]] = const.Declare tensor<1xsi64> = dense<8> : tensor<1xsi64>
    %DIM_3 = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    // CHECK:   [[DIM_3:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
    %DIM_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK:   [[EXPAND:%.+]] = IE.DynamicExpand([[IN]]) :
    // CHECK-SAME:  tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:      -> tensor<1x3x8x16xf16>

    %CONV = IE.Convolution(%IN, %KERNEL) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<3x3x1x1xf16>
        -> tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[KERNEL]])
    // CHECK-SAME:  tensor<1x3x8x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x8x16xf16>

    %TRANSPOSE = IE.Transpose(%CONV) {order_value = #NWCH}
        : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[CONV]])
    // CHECK-SAME:  tensor<1x3x8x16xf16> -> tensor<1x16x3x8xf16>

    %SHAPE_OF = IE.ShapeOf(%IN) {
        dstElemType = si64
    } : tensor<1x3x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 8, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
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
    } : tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[TRANSPOSE]], [[CONCAT]]) {
    // CHECK-SAME:  } : tensor<1x16x3x8xf16>, tensor<4xsi64>
    // CHECK-SAME:      -> tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RESHAPE_OUT]] : tensor<1x?x3x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3, 8]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

{-#
  dialect_resources: {
    builtin: {
      vpux_ow_1: "0x1000000000000000",
      vpux_ow_2: "0x10000000000000AB"
    }
  }
#-}

// CHECK-LABEL: @PadTwoDynamicInputsSubgraph
func.func @PadTwoDynamicInputsSubgraph(
    %IN: tensor<1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}>
) -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>  {
  %cst = const.Declare tensor<1xsi64> = dense<128> : tensor<1xsi64>
  %cst_0 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  %cst_1 = const.Declare tensor<2x128xf32> = dense_resource<vpux_ow_2> : tensor<2x128xf32>
  %cst_2 = const.Declare tensor<8x128xf32> = dense_resource<vpux_ow_1> : tensor<8x128xf32>
  %0 = IE.Gather(%cst_2, %IN) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<8x128xf32>, tensor<1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>
  %1 = IE.Gather(%cst_1, %IN) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<2x128xf32>, tensor<1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>
  %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>, tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>
  %3 = IE.ShapeOf(%0) {dstElemType = si64} : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }> -> tensor<3xsi64>
  %4 = IE.Slice %3 [1] [1] : tensor<3xsi64> to tensor<1xsi64>
  %5 = IE.Concat(%cst_0, %4, %cst) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
  %6 = IE.DynamicReshape(%2, %5) {output_bounds = [1, 64, 128], output_shape = [1, -9223372036854775808, 128]} : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>, tensor<3xsi64> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>
  return %6 : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW }>

  // CHECK:   [[IN:%.+]]: tensor<1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}>
  // CHECK:   [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<128> : tensor<1xsi64>
  // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  // CHECK:   [[CST_1:%.+]] = const.Declare tensor<2x128xf32> = dense_resource<vpux_ow_2> : tensor<2x128xf32>
  // CHECK:   [[CST_2:%.+]] = const.Declare tensor<8x128xf32> = dense_resource<vpux_ow_1> : tensor<8x128xf32>
  // CHECK:   [[GATHER_0:%.+]] = IE.Gather([[CST_2]], [[IN]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<8x128xf32>, tensor<1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}>
  // CHECK:   [[GATHER_1:%.+]] = IE.Gather([[CST_1]], [[IN]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<2x128xf32>, tensor<1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}>
  // CHECK:   [[EXPAND_0:%.+]] = IE.DynamicExpand([[GATHER_0]]) : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}> -> tensor<1x64x128xf32>
  // CHECK:   [[EXPAND_1:%.+]] = IE.DynamicExpand([[GATHER_1]]) : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}> -> tensor<1x64x128xf32>
  // CHECK:   [[ADD:%.+]] = IE.Add([[EXPAND_0]], [[EXPAND_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128xf32>, tensor<1x64x128xf32> -> tensor<1x64x128xf32>
  // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[GATHER_0]]) {dstElemType = si64} : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}> -> tensor<3xsi64>
  // CHECK:   [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [1] [1] : tensor<3xsi64> to tensor<1xsi64>
  // CHECK:   [[CONCAT:%.+]] = IE.Concat([[CST_0]], [[SLICE]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
  // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[ADD]], [[CONCAT]]) {output_bounds = [1, 64, 128], output_shape = [1, -9223372036854775808, 128]} : tensor<1x64x128xf32>, tensor<3xsi64> -> tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}>
  // CHECK:   return [[RESHAPE_OUT]] : tensor<1x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128]> : tensor<3xsi64>, order = #CHW}>
}

// CHECK-LABEL: @SecondDynamicInputsSubgraph
func.func @SecondDynamicInputsSubgraph(%IN1: tensor<1x1x768xf32>, %IN2: tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
  %cst = const.Declare tensor<1xsi64> = dense<768> : tensor<1xsi64>
  %cst_0 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  %0 = IE.Add(%IN1, %IN2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x768xf32>, tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
  %1 = IE.ShapeOf(%IN2) {dstElemType = si64} : tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<3xsi64>
  %2 = IE.Slice %1 [1] [1] : tensor<3xsi64> to tensor<1xsi64>
  %3 = IE.Concat(%cst_0, %2, %cst) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
  %4 = IE.DynamicReshape(%0, %3) {output_bounds = [1, 10, 768], output_shape = [1, -9223372036854775808, 768]} : tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>, tensor<3xsi64> -> tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
  return %4 : tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>

  // CHECK:   [[IN1:%.+]]: tensor<1x1x768xf32>
  // CHECK:   [[IN2:%.+]]: tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = #CHW}>
  // CHECK:   [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<768> : tensor<1xsi64>
  // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
  // CHECK:   [[EXPAND_0:%.+]] = IE.DynamicExpand([[IN2]]) : tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = #CHW}> -> tensor<1x10x768xf32>
  // CHECK:   [[ADD:%.+]] = IE.Add([[IN1]], [[EXPAND_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x768xf32>, tensor<1x10x768xf32> -> tensor<1x10x768xf32>
  // CHECK:   [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN2]]) {dstElemType = si64} : tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = #CHW}> -> tensor<3xsi64>
  // CHECK:   [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [1] [1] : tensor<3xsi64> to tensor<1xsi64>
  // CHECK:   [[CONCAT:%.+]] = IE.Concat([[CST_0]], [[SLICE]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
  // CHECK:   [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[ADD]], [[CONCAT]]) {output_bounds = [1, 10, 768], output_shape = [1, -9223372036854775808, 768]} : tensor<1x10x768xf32>, tensor<3xsi64> -> tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = #CHW}>
  // CHECK:   return [[RESHAPE_OUT]] : tensor<1x?x768xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 768]> : tensor<3xsi64>, order = #CHW}>
}
