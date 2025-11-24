//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-divide-to-multiply --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @NotConvertForSmallDivideOutputRatio
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x151x1x768xf16>, [[ARG1:%.+]]: tensor<1x151x1x1xf16>) -> tensor<1x151x1x768xf16>
func.func @NotConvertForSmallDivideOutputRatio(%arg0: tensor<1x151x1x768xf16>, %arg1: tensor<1x151x1x1xf16>) -> tensor<1x151x1x768xf16> {
    %0 = IE.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x151x1x768xf16>, tensor<1x151x1x1xf16> -> tensor<1x151x1x768xf16>

    return %0 : tensor<1x151x1x768xf16>

    // CHECK: [[DIVIDE:%.+]] = IE.Divide
    // CHECK:   return   [[DIVIDE]]
}
