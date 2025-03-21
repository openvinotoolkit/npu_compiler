//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-grn-to-normalizel2  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

// CHECK-LABEL: @ConvertGRNToNormalizeL2
func.func @ConvertGRNToNormalizeL2(%arg0: tensor<1x8x24x64xf16>) -> tensor<1x8x24x64xf16> {
    %0 = IE.GRN(%arg0) {bias = 0.33000001311302185 : f64} : tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>
    return %0 : tensor<1x8x24x64xf16>

    // CHECK:       [[NORMALIZEL2:%.*]] = IE.NormalizeL2(%arg0)
    // CHECK-SAME:       {axes_value = [1 : si64], eps = 0.33000001311302185 : f64, eps_mode = #IE.eps_mode<ADD>}
    // CHECK-SAME:        : tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>

    // CHECK:       return [[NORMALIZEL2]]
}
