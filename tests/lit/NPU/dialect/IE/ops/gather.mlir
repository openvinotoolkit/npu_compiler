//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FoldGatherWithConstInputs
func.func @FoldGatherWithConstInputs() -> tensor<256x256x64xf16> {
    %cst_input = const.Declare tensor<32x64xf16> = dense<1.000000e+00> : tensor<32x64xf16>
    %cst_indices = const.Declare tensor<256x256xsi32> = dense<0> : tensor<256x256xsi32>
    %0 = IE.Gather(%cst_input, %cst_indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<32x64xf16>, tensor<256x256xsi32> -> tensor<256x256x64xf16>

    return %0 : tensor<256x256x64xf16>

    // CHECK: [[RESULT:%.+]] = const.Declare tensor<256x256x64xf16> = dense<1.000000e+00> : tensor<256x256x64xf16>
    // CHECK: return [[RESULT]] : tensor<256x256x64xf16>
}

// -----

// CHECK-LABEL: @BypassSignednessConvertIndices
// CHECK-SAME: ([[DATA:%.+]]: tensor<40x16xf16>, [[INDICES:%.+]]: tensor<32xsi32>)
func.func @BypassSignednessConvertIndices(%data: tensor<40x16xf16>, %indices: tensor<32xsi32>) -> tensor<32x16xf16> {
    %converted = IE.Convert(%indices) {dstElemType = ui32} : tensor<32xsi32> -> tensor<32xui32>
    %0 = IE.Gather(%data, %converted) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<40x16xf16>, tensor<32xui32> -> tensor<32x16xf16>

    return %0 : tensor<32x16xf16>

    // The same-width signed->unsigned Convert on the indices is a no-op for Gather and must be bypassed.
    // CHECK-NOT: IE.Convert
    // CHECK: [[GATHER:%.+]] = IE.Gather([[DATA]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<40x16xf16>, tensor<32xsi32> -> tensor<32x16xf16>
    // CHECK: return [[GATHER]] : tensor<32x16xf16>
}
