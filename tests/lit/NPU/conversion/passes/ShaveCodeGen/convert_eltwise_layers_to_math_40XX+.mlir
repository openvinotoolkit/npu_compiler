//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @SingleCosLayer {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = math.cos [[ARG0]] : tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}
