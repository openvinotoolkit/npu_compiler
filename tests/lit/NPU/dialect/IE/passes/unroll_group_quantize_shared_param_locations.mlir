//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --unroll-group-quantize --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Regression test for a location-uniqueness violation in UnrollGroupQuantize.
//
// One quantization scale (%SCALE) is the dequantization scale of TWO IE.DynamicDequantize ops - as happens
// in grouped-INT4 attention projections, where the per-group scale feeds both the weight dequant and the
// activation/matmul dequant. UnrollGroupQuantize unrolls each consumer and slices the shared scale in both.
// Before the fix every slice was located after the shared scale value (".../slice_d{axis}_{idx}"), so the two
// consumers produced byte-identical slice locations and StopLocationVerifierPass aborted the compile with
// "Found N duplicated names after full verification". The fix locates each slice after its CONSUMING op plus
// an operand tag, so the slices stay unique.
//
// Note: this is a minimal reduction - both consumers here unroll the shared scale along the same axis,
// whereas the field case additionally unrolls them along different axes. The violation (slices of a shared
// value named after that value) is identical either way; the axis count is incidental. The consumer ops
// carry metadata-bearing fused locations to match importer-produced IR and exercise appendLoc's
// preserve-existing-FusedLoc branch.

!qElemType = !quant.uniform<u4:f16, 1.000000e+00>

// CHECK-LABEL: @SharedScaleTwoConsumers
func.func @SharedScaleTwoConsumers(%data0: tensor<2x1536x128x!qElemType>, %data1: tensor<2x1536x128x!qElemType>,
                                   %scale: tensor<2x1536x1xf16>, %act0: tensor<1x256xf16>, %act1: tensor<1x256xf16>)
        -> (tensor<1x1536xf16>, tensor<1x1536xf16>) {
    %0 = IE.DynamicDequantize(%data0, %scale) {dstElemType = f16} : tensor<2x1536x128x!qElemType>, tensor<2x1536x1xf16> -> tensor<2x1536x128xf16> loc(fused<{name = "weight_dq", type = "DynamicDequantize"}>["weight_dq"])
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<2x1536x128xf16> -> tensor<1536x2x128xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [1536, 256]} : tensor<1536x2x128xf16> -> tensor<1536x256xf16>
    %3 = IE.FullyConnected(%act0, %2) : tensor<1x256xf16>, tensor<1536x256xf16> -> tensor<1x1536xf16>

    %4 = IE.DynamicDequantize(%data1, %scale) {dstElemType = f16} : tensor<2x1536x128x!qElemType>, tensor<2x1536x1xf16> -> tensor<2x1536x128xf16> loc(fused<{name = "matmul_dq", type = "DynamicDequantize"}>["matmul_dq"])
    %5 = IE.Transpose(%4) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<2x1536x128xf16> -> tensor<1536x2x128xf16>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [1]], shape_value = [1536, 256]} : tensor<1536x2x128xf16> -> tensor<1536x256xf16>
    %7 = IE.FullyConnected(%act1, %6) : tensor<1x256xf16>, tensor<1536x256xf16> -> tensor<1x1536xf16>

    return %3, %7 : tensor<1x1536xf16>, tensor<1x1536xf16>

    // Bind each shared-scale (and data) slice to its IE.Slice op, then assert the captured location resolves
    // to a fused loc rooted at the CONSUMING op + an operand tag. The two consumers' "scale_slice_*" (and
    // "input_slice_*") therefore resolve to distinct locations (weight_dq vs matmul_dq roots), which is the
    // invariant the fix restores. Pre-fix the suffix is the un-tagged "slice_d{axis}_0" rooted at the shared
    // scale, so the "scale_slice_*"/"input_slice_*" alias definitions below are absent and the test fails.

    // Consumer A ("weight_dq"): its data slice then its shared-scale slice.
    // CHECK:     IE.Slice %arg0 [0, 0, 0] [1, 1536, 128] {{.*}} loc([[A_INPUT0:#.+]])
    // CHECK:     IE.Slice %arg2 [0, 0, 0] [1, 1536, 1] {{.*}} loc([[A_SCALE0:#.+]])
    // Consumer B ("matmul_dq"): its data slice then its shared-scale slice.
    // CHECK:     IE.Slice %arg1 [0, 0, 0] [1, 1536, 128] {{.*}} loc([[B_INPUT0:#.+]])
    // CHECK:     IE.Slice %arg2 [0, 0, 0] [1, 1536, 1] {{.*}} loc([[B_SCALE0:#.+]])

    // CHECK-DAG: [[WEIGHT_DQ:#.+]] = loc("weight_dq")
    // CHECK-DAG: [[MATMUL_DQ:#.+]] = loc("matmul_dq")
    // CHECK-DAG: [[INPUT_SLICE0:#.+]] = loc("input_slice_d{{[0-9]+}}_0")
    // CHECK-DAG: [[SCALE_SLICE0:#.+]] = loc("scale_slice_d{{[0-9]+}}_0")

    // CHECK-DAG: [[A_INPUT0]] = loc(fused<{{.*}}>{{\[}}[[WEIGHT_DQ]], [[INPUT_SLICE0]]])
    // CHECK-DAG: [[A_SCALE0]] = loc(fused<{{.*}}>{{\[}}[[WEIGHT_DQ]], [[SCALE_SLICE0]]])
    // CHECK-DAG: [[B_INPUT0]] = loc(fused<{{.*}}>{{\[}}[[MATMUL_DQ]], [[INPUT_SLICE0]]])
    // CHECK-DAG: [[B_SCALE0]] = loc(fused<{{.*}}>{{\[}}[[MATMUL_DQ]], [[SCALE_SLICE0]]])
}
