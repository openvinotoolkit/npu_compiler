//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// An Interpolate that was SCF-tiled with a non-constant (loop-dependent) tile position carries a
// kDynamic sentinel in initial_input_offset_attr and a runtime offsets tensor. The SW-kernel
// Interpolate ABI encodes offsets as static attributes and has no operand for runtime offsets, so
// SW-kernel lowering must fail early with an actionable diagnostic instead of emitting a kernel
// with invalid offsets. The non-constant offset element is modeled by a function argument.

// CHECK: Interpolate SW-kernel lowering requires a compile-time-constant 'dynamic_input_offsets'
// CHECK-SAME: The SW-kernel ABI does not support runtime offsets

func.func @InterpolateDynamicOffsetNonConst(%input: tensor<1x2x540x960xf16>, %off: i64) -> tensor<1x2x256x256xf16> {
    %c0_i64 = arith.constant 0 : i64
    %offsets = tensor.from_elements %c0_i64, %c0_i64, %c0_i64, %off : tensor<4xi64>
    %output = VPU.Interpolate(%input, %offsets) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 0, 0, -9223372036854775808], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 0, 0, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 256], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x2x540x960xf16>, tensor<4xi64> -> tensor<1x2x256x256xf16>
    return %output : tensor<1x2x256x256xf16>
}
