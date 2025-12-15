//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

func.func @ConstantFolding() -> tensor<1x5x28x31xf16> {
    %cst = const.Declare tensor<1x5x10x11xf16> = dense<1.0> : tensor<1x5x10x11xf16>
    %0 = IE.ExpandDilated(%cst) {dilations = [3, 3]} : tensor<1x5x10x11xf16> -> tensor<1x5x28x31xf16>
    return %0 : tensor<1x5x28x31xf16>

    // CHECK-DAG:       %[[CST:.*]] = const.Declare tensor<1x5x28x31xf16> = dense<1.000000e+00> : tensor<1x5x10x11xf16>, [#const.ExpandDilated<[3, 3]>]
    // CHECK:       return %[[CST]] : tensor<1x5x28x31xf16>
}
