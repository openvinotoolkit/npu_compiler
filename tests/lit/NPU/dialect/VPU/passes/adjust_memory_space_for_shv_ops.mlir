//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --adjust-memory-space-for-shv-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Case A: Not all input and output tensors fit in CMX. Try to work as much as possible with CMX.
// The input tensor is smaller so it will be placed in DDR, while the output will be placed in CMX.

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseA
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x128x50x50xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseA(%input: tensor<1x128x50x50xf16>) -> tensor<1x128x75x75xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 50, 50], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 75, 75], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.50000e+00, 1.50000e+00], sizes_attr = [1, 128, 75, 75], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x50x50xf16> -> tensor<1x128x75x75xf16>
    return %output : tensor<1x128x75x75xf16>

    // CHECK:      [[OUTPUT_CMX:%.+]] = VPU.Interpolate([[INPUT_DDR]])
    // CHECK-SAME:    : tensor<1x128x50x50xf16>
    // CHECK-SAME:   -> tensor<1x128x75x75xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[COPY_DDR:%.+]] = VPU.Copy([[OUTPUT_CMX]])
    // CHECK-SAME:   -> tensor<1x128x75x75xf16>
    // CHECK:      return [[COPY_DDR]]
}

// -----

// Case B: Not all input and output tensors fit in CMX. Try to work as much as possible with CMX.
// The input tensor is larger so it will be placed in CMX, while the output will be placed in DDR.

// CHECK-LABEL:  @SingleSWLayerTooLargeForCMXCaseB
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x128x75x75xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseB(%input: tensor<1x128x75x75xf16>) -> tensor<1x128x50x50xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 75, 75], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 50, 50], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 0.666666e+00, 0.666666e+00], sizes_attr = [1, 128, 50, 50], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x75x75xf16> -> tensor<1x128x50x50xf16>
    return %output : tensor<1x128x50x50xf16>

    // CHECK:      [[COPY_CMX:%.+]] = VPU.Copy([[INPUT_DDR]])
    // CHECK-SAME:   -> tensor<1x128x75x75xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[OUTPUT_DDR:%.+]] = VPU.Interpolate([[COPY_CMX]])
    // CHECK-SAME:    : tensor<1x128x75x75xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK-SAME:   -> tensor<1x128x50x50xf16>
    // CHECK:      return [[OUTPUT_DDR]]
}

// -----

// Case C: Neither the input or output tensors fit in CMX.
// Both tensors will be placed in DDR.

// CHECK-LABEL:  @SingleSWLayerTooLargeForCMXCaseC
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x1x1x1000000xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseC(%input: tensor<1x1x1x1000000xf16>) -> tensor<1x1x1x1000000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x1000000xf16> -> tensor<1x1x1x1000000xf16>
    return %output: tensor<1x1x1x1000000xf16>

    // CHECK:      [[OUTPUT_DDR:%.+]] = VPU.SoftMax([[INPUT_DDR]])
    // CHECK-SAME:    : tensor<1x1x1x1000000xf16>
    // CHECK-SAME:   -> tensor<1x1x1x1000000xf16>
    // CHECK:      return [[OUTPUT_DDR]]

}

// -----

// Case D: Not all input and output tensors fit in CMX. Try to work as much as possible with CMX.
// Both inputs can fit together in CMX and together are larger than the output,
// therefore the inputs will be placed in CMX and the output will be placed in DDR.

// CHECK-LABEL:  SingleSWLayerTooLargeForCMXCaseD
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x16x210x210xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseD(%input: tensor<1x16x210x210xf16>) -> tensor<1x8x208x208xf16> {
    %cst = const.Declare tensor<8x8x3x3xf16> = dense<2.0> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
    %output = VPU.GroupConvolution(%input, %cst) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x210x210xf16>, tensor<8x8x3x3xf16> -> tensor<1x8x208x208xf16>
    return %output : tensor<1x8x208x208xf16>

    // CHECK:      [[CST_DDR:%.+]] = const.Declare tensor<8x8x3x3xf16> = dense<2.000000e+00> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
    // CHECK:      [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_DDR]])
    // CHECK-SAME:   -> tensor<1x16x210x210xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[CST_CMX:%.+]] = VPU.Copy([[CST_DDR]])
    // CHECK-SAME:   -> tensor<8x8x3x3xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[OUTPUT_DDR:%.+]] = VPU.GroupConvolution([[INPUT_CMX]], [[CST_CMX]])
    // CHECK-SAME:   -> tensor<1x8x208x208xf16>
    // CHECK:      return [[OUTPUT_DDR]]
}

// -----

// Only Concat operations working with dynamic data are lowered to SHAVE

// CHECK-LABEL:  @SkipNonSHAVEConcat
// CHECK-SAME:      ([[INPUT0_DDR:%.+]]: tensor<1x2x3x8xf16>,
// CHECK-SAME:       [[INPUT1_DDR:%.+]]: tensor<1x2x3x8xf16>)
func.func @SkipNonSHAVEConcat(%input0: tensor<1x2x3x8xf16>,
                              %input1: tensor<1x2x3x8xf16>)
        -> tensor<1x4x3x8xf16> {
    %concat = VPU.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]}
        : tensor<1x2x3x8xf16>,
          tensor<1x2x3x8xf16>
        -> tensor<1x4x3x8xf16>
    return %concat : tensor<1x4x3x8xf16>
    // CHECK:       [[CONCAT_DDR:%.+]] = VPU.Concat([[INPUT0_DDR]], [[INPUT1_DDR]])
    // CHECK-SAME:    -> tensor<1x4x3x8xf16>
    // CHECK:       return [[CONCAT_DDR]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @SHAVEConcat
// CHECK-SAME:      ([[INPUT0_DDR:%.+]]: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
// CHECK-SAME:       [[INPUT1_DDR:%.+]]: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>)
func.func @SHAVEConcat(%input0: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
                       %input1: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]>: tensor<4xsi64>, order = #NCHW}> {
    %concat = VPU.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]}
        : tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
    return %concat : tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[INPUT0_CMX:%.+]] = VPU.Copy([[INPUT0_DDR]])
    // CHECK-SAME:    -> tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[INPUT1_CMX:%.+]] = VPU.Copy([[INPUT1_DDR]])
    // CHECK-SAME:    -> tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INPUT0_CMX]], [[INPUT1_CMX]])
    // CHECK-SAME:    -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[OUTPUT_DDR:%.+]] = VPU.Copy([[CONCAT]])
    // CHECK-SAME:    -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[OUTPUT_DDR]]
}

// -----

// Only StridedSlice operations working with dynamic data are lowered to SHAVE

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @SkipNonSHAVEStridedSlice
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x16x64x128xf16>)
func.func @SkipNonSHAVEStridedSlice(%input: tensor<1x16x64x128xf16>)
        -> tensor<1x16x64x128xf16> {
    %strided_slice = VPU.StridedSlice(%input) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        end_mask = [0, 0, 0, 0],
        ends_attr = [1, 16, 64, 128],
        new_axis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x16x64x128xf16> -> tensor<1x16x64x128xf16>
    return %strided_slice : tensor<1x16x64x128xf16>

    // CHECK:       [[STRIDED_SLICE:%.+]] = VPU.StridedSlice([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x16x64x128xf16>
    // CHECK:       return [[STRIDED_SLICE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @SHAVEStridedSlice
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x16x64x128xf16>,
// CHECK-SAME:       [[ENDS_DDR:%.+]]: tensor<4xsi32>)
func.func @SHAVEStridedSlice(%input: tensor<1x16x64x128xf16>,
                             %ends: tensor<4xsi32>)
        -> tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}> {
    %strided_slice = VPU.StridedSlice(%input, %ends) {
        bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>,
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x16x64x128xf16>, tensor<4xsi32>
      -> tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}>
    return %strided_slice : tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x16x64x128xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[ENDS_CMX:%.+]] = VPU.Copy([[ENDS_DDR]])
    // CHECK-SAME:    -> tensor<4xsi32, {mem_space = [@CMX_NN, 0], order = #C}>
    // CHECK:       [[STRIDED_SLICE:%.+]] = VPU.StridedSlice([[INPUT_CMX]], [[ENDS_CMX]])
    // CHECK-SAME:    -> tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[OUTPUT_DDR:%.+]] = VPU.Copy([[STRIDED_SLICE]])
    // CHECK-SAME:    -> tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[OUTPUT_DDR]]
}

// -----

// Only PermuteCast operations working with dynamic data are lowered to SHAVE

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @SkipNonSHAVEPermuteCast
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x32x32x16xf16>)
func.func @SkipNonSHAVEPermuteCast(%input: tensor<1x32x32x16xf16>)
        -> tensor<1x16x32x32xf16, {order = #NHWC}> {
    %permute_cast = VPU.PermuteCast(%input) {
        dst_order = #NHWC,
        mem_perm = #NCHW
    } : tensor<1x32x32x16xf16> -> tensor<1x16x32x32xf16, {order = #NHWC}>
    return %permute_cast : tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTE_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @SHAVEPermuteCast
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x32x64x16xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 1, 0]> : tensor<4xsi64>, order = #NCHW}>)
func.func @SHAVEPermuteCast(%input: tensor<1x32x64x16xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 1, 0]>: tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NHWC}> {
    %permute_cast = VPU.PermuteCast(%input) {
        dst_order = #NHWC,
        mem_perm = #NCHW
    } : tensor<1x32x64x16xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 1, 0]>: tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NHWC}>
    return %permute_cast : tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]>: tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x32x64x16xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 1, 0]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[INPUT_CMX]])
    // CHECK-SAME:    -> tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[OUTPUT_DDR:%.+]] = VPU.Copy([[PERMUTE_CAST]])
    // CHECK-SAME:    -> tensor<1x16x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[OUTPUT_DDR]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @SHAVEConvert
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x3x4x4xf16>)
func.func @SHAVEConvert(%input: tensor<1x3x4x4xf16>)
        -> tensor<1x3x4x4xf32> {
    %convert = VPU.Convert(%input) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %convert : tensor<1x3x4x4xf32>

    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x3x4x4xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.Convert([[INPUT_CMX]])
    // CHECK-SAME:    -> tensor<1x3x4x4xf32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[OUTPUT_DDR:%.+]] = VPU.Copy([[PERMUTE_CAST]])
    // CHECK-SAME:    -> tensor<1x3x4x4xf32>
    // CHECK:       return [[OUTPUT_DDR]]
}

// -----

// CHECK-LABEL: @TopKWithAuxBuffer
// CHECK-SAME:  ([[INPUT_DDR:%.+]]: tensor<1x1x1x1001xf16>)
func.func @TopKWithAuxBuffer(%input: tensor<1x1x1x1001xf16>) -> (tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>) {
    %aux = VPU.Empty : tensor<1x1x1x16016xui8>
    %output_values, %target_shape = VPU.TopK(%input, %aux) {
        axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>
    } : tensor<1x1x1x1001xf16>, tensor<1x1x1x16016xui8> -> tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
    return %output_values, %target_shape : tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>

    // CHECK:       [[AUX_CMX:%.+]] = VPU.Empty : tensor<1x1x1x16016xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x1x1x1001xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[OUTPUT:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK([[INPUT_CMX]], [[AUX_CMX]])
    // CHECK-SAME:    -> tensor<1x1x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>, tensor<1x1x1x1xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[TENSOR_SHAPE_DDR:%.+]] = VPU.Copy([[TARGET_SHAPE]])
    // CHECK-SAME:    -> tensor<1x1x1x1xsi32>
    // CHECK:       [[OUTPUT_DDR:%.+]] = VPU.Copy([[OUTPUT]])
    // CHECK-SAME:    -> tensor<1x1x1x1xf16>
    // CHECK:       return [[OUTPUT_DDR]], [[TENSOR_SHAPE_DDR]]
}

// -----

module {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
}

// CHECK-LABEL:   @DetectionOutputSortWithConstantAuxBuffers
// CHECK-SAME:  [[INPUT_DDR:%.+]]: tensor<1x1x11x76725xf16>)
func.func @DetectionOutputSortWithConstantAuxBuffers(%confidence: tensor<1x1x11x76725xf16>) -> (tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>) {
    %aux_indices = const.Declare tensor<1x1x11x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>
    %aux_sorting = const.Declare tensor<1x1x32x256xsi32> = dense<0> : tensor<1x1x32x256xsi32>
    %out_confidence, %out_indices, %out_sizes = VPU.DetectionOutputSort(%confidence, %aux_indices, %aux_sorting) {
        confidence_threshold = 0.20000000298023224 : f64, top_k = 100 : i64
    } : tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x32x256xsi32> -> tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>
    return %out_confidence, %out_indices, %out_sizes : tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>

    // CHECK:       [[AUX_INDICES:%.+]] = const.Declare tensor<1x1x11x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>
    // CHECK:       [[AUX_SORTING:%.+]] = const.Declare tensor<1x1x32x256xsi32> = dense<0> : tensor<1x1x32x256xsi32>
    // CHECK:       [[AUX_SORTING_CMX:%.+]] = VPU.Copy([[AUX_SORTING]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:    -> tensor<1x1x32x256xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[OUT_CONFIDENCE:%.+]], [[OUT_INDICES:%.+]], [[OUT_SIZES:%.+]] = VPU.DetectionOutputSort([[INPUT_DDR]], [[AUX_INDICES]], [[AUX_SORTING_CMX]])
    // CHECK-SAME:    -> tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[OUT_SIZES_DDR:%.+]] = VPU.Copy([[OUT_SIZES]])
    // CHECK-SAME:    -> tensor<1x1x11x1xsi32>
    // CHECK:       return [[OUT_CONFIDENCE]], [[OUT_INDICES]], [[OUT_SIZES_DDR]] : tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>
}

}
