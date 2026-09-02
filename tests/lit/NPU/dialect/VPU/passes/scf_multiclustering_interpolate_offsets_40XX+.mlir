//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-multiclustering --canonicalize --cse %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Tests that SCF tiling of VPU.Interpolate correctly propagates tile offsets
// (both input and output) to the generated tiled operations, rather than
// using zero defaults. When tiling splits the op along a spatial axis,
// each tile's offsets must reflect its position in the original tensor.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test SOH (SplitOverHeight) with LINEAR_ONNX mode.
// The H offsets must be dynamic (kDynamic sentinel in the attr) because
// the tile's position along H depends on the loop iteration variable.

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @InterpolateSOH_LinearOnnx_Offsets
func.func @InterpolateSOH_LinearOnnx_Offsets(%arg0: tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 32]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>,
            nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 16, 64, 32],
        initial_output_dims_attr = [1, 16, 128, 32],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        scales_attr = [2.0, 1.0],
        sizes_attr = [128, 32],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 32]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 32]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 32]> : tensor<4xsi64>, order = #NHWC}>

    // Verify tiled op has dynamic H offset (kDynamic sentinel = -9223372036854775808)
    // and that the dynamic offset operands originate from tensor.from_elements with
    // arith.index_cast of the computed tile positions.
    // CHECK-DAG:   [[C0_I64:%.+]] = arith.constant 0 : i64
    // CHECK:       scf.forall ([[ITER:%.+]]) =
    // CHECK:           [[IN_OFF_CAST:%.+]] = arith.index_cast %{{.+}} : index to i64
    // CHECK:           [[OUT_OFF_CAST:%.+]] = arith.index_cast [[ITER]] : index to i64
    // CHECK:           [[IN_OFF_TENSOR:%.+]] = tensor.from_elements [[C0_I64]], [[C0_I64]], [[IN_OFF_CAST]], [[C0_I64]] : tensor<4xi64>
    // CHECK:           [[OUT_OFF_TENSOR:%.+]] = tensor.from_elements [[C0_I64]], [[C0_I64]], [[OUT_OFF_CAST]], [[C0_I64]] : tensor<4xi64>
    // CHECK:           VPU.Interpolate(%{{.+}}, [[IN_OFF_TENSOR]], [[OUT_OFF_TENSOR]])
    // CHECK-SAME:          initial_input_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK-SAME:          initial_output_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 1>
}
}

// -----

// Test SOH with NEAREST mode.
// Verifies that nearest-neighbor interpolation also gets correct offsets.

#NHWC2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @InterpolateSOH_Nearest_Offsets
func.func @InterpolateSOH_Nearest_Offsets(%arg0: tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 16]> : tensor<4xsi64>, order = #NHWC2}>) -> tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 96, 16]> : tensor<4xsi64>, order = #NHWC2}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>,
            nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 32, 48, 16],
        initial_output_dims_attr = [1, 32, 96, 16],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        scales_attr = [2.0, 1.0],
        sizes_attr = [96, 16],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 16]> : tensor<4xsi64>, order = #NHWC2}> -> tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 96, 16]> : tensor<4xsi64>, order = #NHWC2}>
    return %0 : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 96, 16]> : tensor<4xsi64>, order = #NHWC2}>

    // Verify H offsets are dynamic, W offsets remain 0 (no W tiling).
    // Dynamic offset operands are tensor.from_elements packing [0, 0, <dynamic_H>, 0].
    // CHECK-DAG:   [[C0_I64_2:%.+]] = arith.constant 0 : i64
    // CHECK:       scf.forall ([[ITER2:%.+]]) =
    // CHECK:           [[IN_OFF_CAST2:%.+]] = arith.index_cast %{{.+}} : index to i64
    // CHECK:           [[OUT_OFF_CAST2:%.+]] = arith.index_cast [[ITER2]] : index to i64
    // CHECK:           [[IN_OFF_TENSOR2:%.+]] = tensor.from_elements [[C0_I64_2]], [[C0_I64_2]], [[IN_OFF_CAST2]], [[C0_I64_2]] : tensor<4xi64>
    // CHECK:           [[OUT_OFF_TENSOR2:%.+]] = tensor.from_elements [[C0_I64_2]], [[C0_I64_2]], [[OUT_OFF_CAST2]], [[C0_I64_2]] : tensor<4xi64>
    // CHECK:           VPU.Interpolate(%{{.+}}, [[IN_OFF_TENSOR2]], [[OUT_OFF_TENSOR2]])
    // CHECK-SAME:          initial_input_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK-SAME:          initial_output_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 1>
}
}

// -----

// Test SOH with ALIGN_CORNERS coordinate mode.
// ALIGN_CORNERS computes coordinates differently, but offsets should still
// reflect tile position.

#NHWC3 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @InterpolateSOH_AlignCorners_Offsets
func.func @InterpolateSOH_AlignCorners_Offsets(%arg0: tensor<1x8x?x20xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 20]> : tensor<4xsi64>, order = #NHWC3}>) -> tensor<1x8x?x20xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 64, 20]> : tensor<4xsi64>, order = #NHWC3}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>,
            nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 8, 32, 20],
        initial_output_dims_attr = [1, 8, 64, 20],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        scales_attr = [2.0, 1.0],
        sizes_attr = [64, 20],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x8x?x20xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 20]> : tensor<4xsi64>, order = #NHWC3}> -> tensor<1x8x?x20xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 64, 20]> : tensor<4xsi64>, order = #NHWC3}>
    return %0 : tensor<1x8x?x20xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 64, 20]> : tensor<4xsi64>, order = #NHWC3}>

    // ALIGN_CORNERS mode: H offsets must still be dynamic
    // CHECK:       scf.forall
    // CHECK:           VPU.Interpolate
    // CHECK:               initial_input_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK:               initial_output_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK:               operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 1>
}
}

// -----

// Test SOH with CUBIC mode.
// Cubic interpolation requires wider input halo (2px each side instead of 1).
// Offsets must still correctly reflect tile position.

#NHWC5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @InterpolateSOH_Cubic_Offsets
func.func @InterpolateSOH_Cubic_Offsets(%arg0: tensor<1x4x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 40, 16]> : tensor<4xsi64>, order = #NHWC5}>) -> tensor<1x4x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 80, 16]> : tensor<4xsi64>, order = #NHWC5}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <CUBIC>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>,
            nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 4, 40, 16],
        initial_output_dims_attr = [1, 4, 80, 16],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        scales_attr = [2.0, 1.0],
        sizes_attr = [80, 16],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x4x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 40, 16]> : tensor<4xsi64>, order = #NHWC5}> -> tensor<1x4x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 80, 16]> : tensor<4xsi64>, order = #NHWC5}>
    return %0 : tensor<1x4x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 80, 16]> : tensor<4xsi64>, order = #NHWC5}>

    // Cubic mode: H offsets must be dynamic despite wider halo requirements
    // CHECK:       scf.forall
    // CHECK:           VPU.Interpolate
    // CHECK:               initial_input_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK:               initial_output_offset_attr = [0, 0, -9223372036854775808, 0]
    // CHECK:               operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 1>
}
}

// -----

// Test SOK (SplitOverKernel) with LINEAR_ONNX mode.
// SOK splits the output along the C (channel) dimension. The C offsets must be
// dynamic because the tile's position along C depends on the loop iteration variable.
// Spatial offsets remain 0 (no spatial tiling).

#NHWC6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @InterpolateSOK_LinearOnnx_Offsets
func.func @InterpolateSOK_LinearOnnx_Offsets(%arg0: tensor<1x96x32x32xf16, {order = #NHWC6}>) -> tensor<1x96x64x64xf16, {order = #NHWC6}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>,
            nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 96, 32, 32],
        initial_output_dims_attr = [1, 96, 64, 64],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        scales_attr = [2.0, 2.0],
        sizes_attr = [64, 64],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x96x32x32xf16, {order = #NHWC6}> -> tensor<1x96x64x64xf16, {order = #NHWC6}>
    return %0 : tensor<1x96x64x64xf16, {order = #NHWC6}>

    // SOK: C offset is dynamic, spatial offsets remain 0
    // CHECK:       scf.forall
    // CHECK:           VPU.Interpolate
    // CHECK:               initial_input_offset_attr = [0, -9223372036854775808, 0, 0]
    // CHECK:               initial_output_offset_attr = [0, -9223372036854775808, 0, 0]
    // CHECK:               operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 1>
}
}
