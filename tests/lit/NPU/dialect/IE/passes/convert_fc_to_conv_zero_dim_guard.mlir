//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// Regression test: ConvertFCToConv must not crash (SIGABRT) when an
// IE.FullyConnected has a zero-sized channel dimension.  This can occur
// when per-group INT4 quantization decomposition produces intermediate
// FC operations with degenerate shapes.
//
// Before the fix, vpux-opt would hit:
//   LLVM ERROR: Failed to infer result type(s).
// because IE.Convolution type-inference receives a tensor<Nx0x1x1> input.
//
// After the fix, the zero-dim FC op is dynamically marked as legal and
// survives the pass unchanged — no crash, no "failed to legalize".

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-fc-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @PreserveZeroDimFC
// CHECK: IE.FullyConnected
func.func @PreserveZeroDimFC(%arg0: tensor<1x0xf16>) -> tensor<1x64xf16> {
    %weights = const.Declare tensor<64x0xf16> = dense<0.0> : tensor<64x0xf16>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<1x0xf16>, tensor<64x0xf16> -> tensor<1x64xf16>
    return %0 : tensor<1x64xf16>
}
