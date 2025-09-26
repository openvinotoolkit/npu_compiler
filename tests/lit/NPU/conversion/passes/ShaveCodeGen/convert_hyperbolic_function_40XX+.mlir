//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX


// IE.Sinh

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SinhF16Layer
module @SinhF16Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Sinh(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Sinh
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[CST0:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-NEXT:      [[CST:%.+]] = arith.constant 5.000000e-01 : f16
    // CHECK-NEXT:      [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f16
    // CHECK-NEXT:      [[NEG:%.+]] = arith.subf [[CST0]], [[IN]] : f16
    // CHECK-NEXT:      [[EXPN:%.+]] = math.exp [[NEG]] fastmath<afn> : f16
    // CHECK-NEXT:      [[SUB:%.+]] = arith.subf [[EXP]], [[EXPN]] : f16
    // CHECK-NEXT:      [[RES:%.+]] = arith.mulf [[SUB]], [[CST]] : f16
    // CHECK-NEXT:      linalg.yield [[RES]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Cosh

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: CoshF16Layer
module @CoshF16Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Cosh(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Cosh
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[CST0:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-NEXT:      [[CST:%.+]] = arith.constant 5.000000e-01 : f16
    // CHECK-NEXT:      [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f16
    // CHECK-NEXT:      [[NEG:%.+]] = arith.subf [[CST0]], [[IN]] : f16
    // CHECK-NEXT:      [[EXPN:%.+]] = math.exp [[NEG]] fastmath<afn> : f16
    // CHECK-NEXT:      [[ADD:%.+]] = arith.addf [[EXP]], [[EXPN]] : f16
    // CHECK-NEXT:      [[RES:%.+]] = arith.mulf [[ADD]], [[CST]] : f16
    // CHECK-NEXT:      linalg.yield [[RES]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Atanh

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: AtanhF16Layer
module @AtanhF16Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Atanh(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Atanh
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[CST:%.+]] = arith.constant 1.000000e+00 : f16
    // CHECK-NEXT:      [[CST0:%.+]] = arith.constant 5.000000e-01 : f16
    // CHECK-NEXT:      [[ADD:%.+]] = arith.addf [[CST]], [[IN]] : f16
    // CHECK-NEXT:      [[SUB:%.+]] = arith.subf [[CST]], [[IN]] : f16
    // CHECK-NEXT:      [[DIV:%.+]] = arith.divf [[ADD]], [[SUB]] : f16
    // CHECK-NEXT:      [[LOG:%.+]] = math.log [[DIV]] fastmath<afn> : f16
    // CHECK-NEXT:      [[RES:%.+]] = arith.mulf [[LOG]], [[CST0]] : f16
    // CHECK-NEXT:      linalg.yield [[RES]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Tanh

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: TanhF16Layer
module @TanhF16Layer  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Tanh(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Tanh
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[ARG0]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[RES:%.+]] = math.tanh [[IN]] fastmath<afn> : f16
    // CHECK-NEXT:      linalg.yield [[RES]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: TanhF32Layer
module @TanhF32Layer  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf32>) {
      %1 = IE.Tanh(%arg1) : tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf32>
      IE.CGCYield %1 : tensor<1x1x1x1000xf32>
    } -> tensor<1x1x1x1000xf32>
    return %0 : tensor<1x1x1x1000xf32>

    // CHECK-NOT:     IE.Tanh
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf32>) outs([[ARG0]] : tensor<1x1x1x1000xf32>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f32, {{%.+}}: f32):
    // CHECK-NEXT:      [[RES:%.+]] = math.tanh [[IN]] fastmath<afn> : f32
    // CHECK-NEXT:      linalg.yield [[RES]] : f32
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf32>
    // CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf32>
  }
}
