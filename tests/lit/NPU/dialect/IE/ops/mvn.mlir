//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @LegalizeEps
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x100x512x1xf32>)
func.func @LegalizeEps(%arg0 : tensor<1x100x512x1xf32>) -> tensor<1x100x512x1xf32> {
    %0 = IE.MVN(%arg0) {across_channels = false, eps = 9.999999960041972E-13 : f64, normalize_variance = true} : tensor<1x100x512x1xf32> -> tensor<1x100x512x1xf32>
    return %0 : tensor<1x100x512x1xf32>

    // CHECK:    [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 1.1920928955078125E-7 : f64, normalize_variance = true} : tensor<1x100x512x1xf32> -> tensor<1x100x512x1xf32>
    // CHECK:    return [[MVN]]
}
