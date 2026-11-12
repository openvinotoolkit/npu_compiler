//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-mvn %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: func.func @NotDecomposeMVN
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x16x32xf16>
func.func @NotDecomposeMVN(%arg0: tensor<1x3x16x32xf16>) -> (tensor<1x3x16x32xf16>) {
      %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x3x16x32xf16> -> tensor<1x3x16x32xf16>
      return %0 : tensor<1x3x16x32xf16>

    // CHECK:            [[VAL0:%.+]] = VPU.MVN([[INPUT]])
    // CHECK:            return [[VAL0]]
}
