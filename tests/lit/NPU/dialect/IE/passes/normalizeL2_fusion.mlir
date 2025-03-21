//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --normalizeL2-fusion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @main(%arg0: tensor<1x192xf32>) -> tensor<1x192xf32> {
    %cst = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x192xf32>, tensor<1xsi64> -> tensor<1x1xf32>
    %1 = IE.Clamp(%0) {max = 1.7976931348623157E+308 : f64, min = 9.999999960041972E-13 : f64} : tensor<1x1xf32> -> tensor<1x1xf32>
    %2 = IE.Divide(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192xf32>, tensor<1x1xf32> -> tensor<1x192xf32>
    return %2 : tensor<1x192xf32>


    // CHECK-NOT: IE.ReduceL2
    // CHECK:   [[NORMALIZEL2:%.*]] = IE.NormalizeL2(%arg0) {axes_value = [1], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x192xf32> -> tensor<1x192xf32>
}
