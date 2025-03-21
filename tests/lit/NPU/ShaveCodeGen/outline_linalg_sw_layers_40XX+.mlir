//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outline-linalg-sw-layers %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @SingleCosLayer {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf16>) outs(%arg0 : tensor<1x1x1x1000xf16>) {
    ^bb0(%in: f16, %out: f16):
      %1 = math.cos %in : f16
      linalg.yield %1 : f16
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
  }
}

  // CHECK: module @VPU.SW
  // CHECK: func.func @generated_0
  // CHECK-SAME: [[VAR0:%.+]]: tensor<1x1x1x1000xf16>,
  // CHECK-SAME: [[VAR1:%.+]]: tensor<1x1x1x1000xf16>
  // CHECK-SAME: -> tensor<1x1x1x1000xf16>

  // CHECK: [[COMPUTATION_RESULT:%.+]] = linalg.generic
  // CHECK-SAME: ins([[VAR0]]
  // CHECK-SAME: outs([[VAR1]]
  // CHECK: [[SCALAR_COS_RES:%.+]] = math.cos
  // CHECK: linalg.yield [[SCALAR_COS_RES]]

  // CHECK: return [[COMPUTATION_RESULT]] : tensor<1x1x1x1000xf16>

  // CHECK: func.func @main
  // CHECK: [[GENERIC_SW_LAYER_RES:%.+]] = VPU.GenericSwLayer
  // CHECK-SAME: callee = @VPU.SW::@generated_0
  // CHECK-SAME: tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
  // CHECK: return [[GENERIC_SW_LAYER_RES]] : tensor<1x1x1x1000xf16>
