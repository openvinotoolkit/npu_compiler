//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" -convert-deformable-conv-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @Convert_DeformableConv_Kernel1
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x128x19x19xf16>, [[OFFSET:%.+]]: tensor<1x2x19x19xf16>, [[KERNEL:%.+]]: tensor<128x128x1x1xf16>, [[MASK:%.+]]: tensor<1x1x19x19xf16>) -> tensor<1x128x19x19xf16>
func.func @Convert_DeformableConv_Kernel1(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x2x19x19xf16>, %arg2: tensor<128x128x1x1xf16>, %arg3: tensor<1x1x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %0 = IE.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x2x19x19xf16>, tensor<128x128x1x1xf16>, tensor<1x1x19x19xf16> -> tensor<1x128x19x19xf16>
    return %0 : tensor<1x128x19x19xf16>

    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x2xf16> = dense<1.110840e-01> : tensor<1x1x1x2xf16>, [#const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x19x19x2xf16>
    // CHECK-SAME:  tensor<1x19x19x2xf16>, [#const.CastElemType<f16>]

    // CHECK:    [[RESHAPED_OFFSET:%.+]] = IE.Reshape([[OFFSET]]) {shape_value = [1, 2, 1, 1, 19, 19]} : tensor<1x2x19x19xf16> -> tensor<1x2x1x1x19x19xf16>
    // CHECK:    [[MEMPERMUTE1:%.+]] = IE.MemPermute([[RESHAPED_OFFSET]]) {dst_order = #map, mem_perm = #map1} : tensor<1x2x1x1x19x19xf16> -> tensor<1x1x19x1x19x2xf16>
    // CHECK:    [[RESHAPED1:%.+]] = IE.Reshape([[MEMPERMUTE1]]) {shape_value = [1, 19, 19, 2]} : tensor<1x1x19x1x19x2xf16> -> tensor<1x19x19x2xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPED1]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x19x19x2xf16>, tensor<1x19x19x2xf16> -> tensor<1x19x19x2xf16>
    // CHECK:    [[MULTIPLY1:%.+]] = IE.Multiply([[ADD]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x19x19x2xf16>, tensor<1x1x1x2xf16> -> tensor<1x19x19x2xf16>

    // CHECK:    [[GRIDSAMPLE:%.+]] = IE.GridSample([[INPUT]], [[MULTIPLY1]]) {mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<ZEROS>} : tensor<1x128x19x19xf16>, tensor<1x19x19x2xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    [[RESHAPED_MASK:%.+]] = IE.Reshape([[MASK]]) {shape_value = [1, 1, 1, 1, 19, 19]} : tensor<1x1x19x19xf16> -> tensor<1x1x1x1x19x19xf16>
    // CHECK:    [[MEMPERMUTE2:%.+]] = IE.MemPermute([[RESHAPED_MASK]]) {dst_order = #map, mem_perm = #map2} : tensor<1x1x1x1x19x19xf16> -> tensor<1x1x1x19x1x19xf16>
    // CHECK:    [[RESHAPED2:%.+]] = IE.Reshape([[MEMPERMUTE2]]) {shape_value = [1, 1, 19, 19]} : tensor<1x1x1x19x1x19xf16> -> tensor<1x1x19x19xf16>

    // CHECK:    [[MULTIPLY2:%.+]] = IE.Multiply([[RESHAPED2]], [[GRIDSAMPLE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x19x19xf16>, tensor<1x128x19x19xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[MULTIPLY2]], [[KERNEL]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    return [[CONV]] : tensor<1x128x19x19xf16>
}

// CHECK-LABEL: @Convert_DeformableConv_Kernel1_no_biliniar_pad
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x128x19x19xf16>, [[OFFSET:%.+]]: tensor<1x2x19x19xf16>, [[KERNEL:%.+]]: tensor<128x128x1x1xf16>, [[MASK:%.+]]: tensor<1x1x19x19xf16>) -> tensor<1x128x19x19xf16>
func.func @Convert_DeformableConv_Kernel1_no_biliniar_pad(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x2x19x19xf16>, %arg2: tensor<128x128x1x1xf16>, %arg3: tensor<1x1x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %0 = IE.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x2x19x19xf16>, tensor<128x128x1x1xf16>, tensor<1x1x19x19xf16> -> tensor<1x128x19x19xf16>
    return %0 : tensor<1x128x19x19xf16>

    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x2xf16> = dense<1.110840e-01> : tensor<1x1x1x2xf16>, [#const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x19x19x2xf16>
    // CHECK-SAME:  tensor<1x19x19x2xf16>, [#const.CastElemType<f16>]

    // CHECK:    [[RESHAPED_OFFSET:%.+]] = IE.Reshape([[OFFSET]]) {shape_value = [1, 2, 1, 1, 19, 19]} : tensor<1x2x19x19xf16> -> tensor<1x2x1x1x19x19xf16>
    // CHECK:    [[MEMPERMUTE1:%.+]] = IE.MemPermute([[RESHAPED_OFFSET]]) {dst_order = #map, mem_perm = #map1} : tensor<1x2x1x1x19x19xf16> -> tensor<1x1x19x1x19x2xf16>
    // CHECK:    [[RESHAPED1:%.+]] = IE.Reshape([[MEMPERMUTE1]]) {shape_value = [1, 19, 19, 2]} : tensor<1x1x19x1x19x2xf16> -> tensor<1x19x19x2xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPED1]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x19x19x2xf16>, tensor<1x19x19x2xf16> -> tensor<1x19x19x2xf16>
    // CHECK:    [[MULTIPLY1:%.+]] = IE.Multiply([[ADD]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x19x19x2xf16>, tensor<1x1x1x2xf16> -> tensor<1x19x19x2xf16>

    // CHECK:    [[GRIDSAMPLE:%.+]] = IE.GridSample([[INPUT]], [[MULTIPLY1]]) {mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x128x19x19xf16>, tensor<1x19x19x2xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    [[RESHAPED_MASK:%.+]] = IE.Reshape([[MASK]]) {shape_value = [1, 1, 1, 1, 19, 19]} : tensor<1x1x19x19xf16> -> tensor<1x1x1x1x19x19xf16>
    // CHECK:    [[MEMPERMUTE2:%.+]] = IE.MemPermute([[RESHAPED_MASK]]) {dst_order = #map, mem_perm = #map2} : tensor<1x1x1x1x19x19xf16> -> tensor<1x1x1x19x1x19xf16>
    // CHECK:    [[RESHAPED2:%.+]] = IE.Reshape([[MEMPERMUTE2]]) {shape_value = [1, 1, 19, 19]} : tensor<1x1x1x19x1x19xf16> -> tensor<1x1x19x19xf16>

    // CHECK:    [[MULTIPLY2:%.+]] = IE.Multiply([[RESHAPED2]], [[GRIDSAMPLE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x19x19xf16>, tensor<1x128x19x19xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[MULTIPLY2]], [[KERNEL]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    return [[CONV]] : tensor<1x128x19x19xf16>
}

// CHECK-LABEL: @Convert_DeformableConv_Kernel3
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x128x19x19xf16>, [[OFFSET:%.+]]: tensor<1x18x19x19xf16>, [[KERNEL:%.+]]: tensor<128x128x3x3xf16>, [[MASK:%.+]]: tensor<1x9x19x19xf16>) -> tensor<1x128x19x19xf16>
func.func @Convert_DeformableConv_Kernel3(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x18x19x19xf16>, %arg2: tensor<128x128x3x3xf16>, %arg3: tensor<1x9x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %0 = IE.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x18x19x19xf16>, tensor<128x128x3x3xf16>, tensor<1x9x19x19xf16> -> tensor<1x128x19x19xf16>
    return %0 : tensor<1x128x19x19xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x2xf16> = dense<1.110840e-01> : tensor<1x1x1x2xf16>, [#const.CastElemType<f16>]
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<1x57x57x2xf16>
    // CHECK-SAME:  tensor<1x57x57x2xf16>, [#const.CastElemType<f16>]

    // CHECK:    [[RESHAPED_OFFSET:%.+]] = IE.Reshape([[OFFSET]]) {shape_value = [1, 2, 3, 3, 19, 19]} : tensor<1x18x19x19xf16> -> tensor<1x2x3x3x19x19xf16>
    // CHECK:    [[MEMPERMUTE1:%.+]] = IE.MemPermute([[RESHAPED_OFFSET]]) {dst_order = #map, mem_perm = #map1} : tensor<1x2x3x3x19x19xf16> -> tensor<1x3x19x3x19x2xf16>
    // CHECK:    [[RESHAPED1:%.+]] = IE.Reshape([[MEMPERMUTE1]]) {shape_value = [1, 57, 57, 2]} : tensor<1x3x19x3x19x2xf16> -> tensor<1x57x57x2xf16>
    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPED1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x57x57x2xf16>, tensor<1x57x57x2xf16> -> tensor<1x57x57x2xf16>
    // CHECK:    [[MULTIPLY1:%.+]] = IE.Multiply([[ADD]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x57x57x2xf16>, tensor<1x1x1x2xf16> -> tensor<1x57x57x2xf16>
    // CHECK:    [[GRIDSAMPLE:%.+]] = IE.GridSample([[INPUT]], [[MULTIPLY1]]) {mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<ZEROS>} : tensor<1x128x19x19xf16>, tensor<1x57x57x2xf16> -> tensor<1x128x57x57xf16>

    // CHECK:    [[RESHAPED_MASK:%.+]] = IE.Reshape([[MASK]]) {shape_value = [1, 1, 3, 3, 19, 19]} : tensor<1x9x19x19xf16> -> tensor<1x1x3x3x19x19xf16>
    // CHECK:    [[MEMPERMUTE2:%.+]] = IE.MemPermute([[RESHAPED_MASK]]) {dst_order = #map, mem_perm = #map2} : tensor<1x1x3x3x19x19xf16> -> tensor<1x1x3x19x3x19xf16>
    // CHECK:    [[RESHAPED2:%.+]] = IE.Reshape([[MEMPERMUTE2]]) {shape_value = [1, 1, 57, 57]} : tensor<1x1x3x19x3x19xf16> -> tensor<1x1x57x57xf16>
    // CHECK:    [[MULTIPLY2:%.+]] = IE.Multiply([[RESHAPED2]], [[GRIDSAMPLE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x57x57xf16>, tensor<1x128x57x57xf16> -> tensor<1x128x57x57xf16>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[MULTIPLY2]], [[KERNEL]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 3]} : tensor<1x128x57x57xf16>, tensor<128x128x3x3xf16> -> tensor<1x128x19x19xf16>

    // CHECK:    return [[CONV]] : tensor<1x128x19x19xf16>
}

// CHECK-LABEL: @Not_Convert_DeformableConv_LargeKernel
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x128x19x19xf16>, [[OFFSET:%.+]]: tensor<1x162x19x19xf16>, [[KERNEL:%.+]]: tensor<128x128x9x9xf16>, [[MASK:%.+]]: tensor<1x81x19x19xf16>) -> tensor<1x128x19x19xf16>
func.func @Not_Convert_DeformableConv_LargeKernel(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x162x19x19xf16>, %arg2: tensor<128x128x9x9xf16>, %arg3: tensor<1x81x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %0 = IE.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [4, 4], pads_end = [4, 4], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x162x19x19xf16>, tensor<128x128x9x9xf16>, tensor<1x81x19x19xf16> -> tensor<1x128x19x19xf16>
    return %0 : tensor<1x128x19x19xf16>

    // CHECK:    [[OUTPUT:%.+]] = IE.DeformableConvolution([[INPUT]], [[OFFSET]], [[KERNEL]], [[MASK]]) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [4, 4], pads_end = [4, 4], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x162x19x19xf16>, tensor<128x128x9x9xf16>, tensor<1x81x19x19xf16> -> tensor<1x128x19x19xf16>
    // CHECK:    return [[OUTPUT]] : tensor<1x128x19x19xf16>
}
