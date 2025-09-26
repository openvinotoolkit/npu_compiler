//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// IE.Atan

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleAtanF16Layer
module @SingleAtanF16Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Atan(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Atan
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[RES:%.+]] = math.atan [[IN]] fastmath<afn> : f16
    // CHECK-NEXT:      linalg.yield [[RES]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Tan

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleTanF16Layer
module @SingleTanF16Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Tan(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Tan
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXTS:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[SIN:%.+]] = math.sin [[EXTS]] : f32
    // CHECK-NEXT:      [[XSIN:%.+]] = arith.truncf [[SIN]] : f32 to f16
    // CHECK-NEXT:      [[EXTC:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[COS:%.+]] = math.cos [[EXTC]] : f32
    // CHECK-NEXT:      [[XCOS:%.+]] = arith.truncf [[COS]] : f32 to f16
    // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[XSIN]], [[XCOS]] : f16
    // CHECK-NEXT:      linalg.yield [[RES]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}
