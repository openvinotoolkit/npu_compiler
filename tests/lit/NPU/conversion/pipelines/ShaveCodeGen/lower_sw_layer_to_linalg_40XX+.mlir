//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-sw-layers-to-linalg %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @SingleCosLayer {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res : tensor<1x1x1x1000xf16>
  }
}

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
// CHECK: [[LINALG_GENERIC:%.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#NCHW, #NCHW]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins([[ARG0]] : tensor<1x1x1x1000xf16>)
// CHECK-SAME: outs([[ARG0]] : tensor<1x1x1x1000xf16>) {

// CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16)
// CHECK: [[COS_RES:%.+]] = math.cos [[IN]] : f16
// CHECK: linalg.yield [[COS_RES]] : f16
// CHEKC: } -> tensor<1x1x1x1000xf16>

// CHECK: return [[LINALG_GENERIC]] : tensor<1x1x1x1000xf16>
