//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX


// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @ReLU
module @ReLU {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.ReLU(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
// CHECK-NOT:     IE.ReLU
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[CMP:%.+]] = arith.cmpf ole, [[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP]], [[ZERO]], [[IN]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @LeakyReLU
module @LeakyReLU {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.LeakyRelu(%arg1) {negative_slope = 2.500000e-01 : f64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
// CHECK-NOT:     IE.LeakyRelu
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[CMP:%.+]] = arith.cmpf ole, [[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[POSITIVE:%.+]] = arith.select [[CMP]], [[ZERO]], [[IN]] : f16
// CHECK-NEXT:      [[NEGATIVE:%.+]] = arith.select [[CMP]], [[IN]], [[ZERO]] : f16
// CHECK-NEXT:      [[SLOPE:%.+]] = arith.constant 2.500000e-01 : f16
// CHECK-NEXT:      [[SLOPED_NEGATIVE:%.+]] = arith.mulf [[NEGATIVE]], [[SLOPE]] : f16
// CHECK-NEXT:      [[OP:%.+]] = arith.addf [[POSITIVE]], [[SLOPED_NEGATIVE]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @ClampU16
module @ClampU16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xui16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui16>) -> tensor<1x1x1x1000xui16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xui16>) {
      %1 = IE.Clamp(%arg1) {max = 6.500000e+00 : f64, min = 5.000000e-01 : f64} : tensor<1x1x1x1000xui16> -> tensor<1x1x1x1000xui16>
      IE.CGCYield %1 : tensor<1x1x1x1000xui16>
    } -> tensor<1x1x1x1000xui16>
    return %0 : tensor<1x1x1x1000xui16>
// CHECK-NOT:     IE.Clamp
// CHECK:         [[ARG_BC:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xui16> to tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG_BC]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: i16, {{%.+}}: i16):
// CHECK-NEXT:      [[LOW:%.+]] = arith.constant 1 : i16
// CHECK-NEXT:      [[HIGH:%.+]] = arith.constant 6 : i16
// CHECK-NEXT:      [[CMP_LOW:%.+]] = arith.cmpi ule, [[IN]], [[LOW]] : i16
// CHECK-NEXT:      [[CLAMPED_LOW:%.+]] = arith.select [[CMP_LOW]], [[LOW]], [[IN]] : i16
// CHECK-NEXT:      [[CMP_HIGH:%.+]] = arith.cmpi ule, [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP_HIGH]], [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      linalg.yield [[OP]] : i16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi16> to tensor<1x1x1x1000xui16>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @ClampU16NonRepresentable
module @ClampU16NonRepresentable {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xui16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui16>) -> tensor<1x1x1x1000xui16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xui16>) {
      %1 = IE.Clamp(%arg1) {max = 7.000050e+04 : f64, min = -5.500000e+00 : f64} : tensor<1x1x1x1000xui16> -> tensor<1x1x1x1000xui16>
      IE.CGCYield %1 : tensor<1x1x1x1000xui16>
    } -> tensor<1x1x1x1000xui16>
    return %0 : tensor<1x1x1x1000xui16>
// CHECK-NOT:     IE.Clamp
// CHECK:         [[ARG_BC:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xui16> to tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG_BC]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: i16, {{%.+}}: i16):
// CHECK-NEXT:      [[LOW:%.+]] = arith.constant 0 : i16
// CHECK-NEXT:      [[HIGH:%.+]] = arith.constant -1 : i16
// CHECK-NEXT:      [[CMP_LOW:%.+]] = arith.cmpi ule, [[IN]], [[LOW]] : i16
// CHECK-NEXT:      [[CLAMPED_LOW:%.+]] = arith.select [[CMP_LOW]], [[LOW]], [[IN]] : i16
// CHECK-NEXT:      [[CMP_HIGH:%.+]] = arith.cmpi ule, [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP_HIGH]], [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      linalg.yield [[OP]] : i16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi16> to tensor<1x1x1x1000xui16>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @ClampS16
module @ClampS16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi16>) -> tensor<1x1x1x1000xsi16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xsi16>) {
      %1 = IE.Clamp(%arg1) {max = 6.500000e+00 : f64, min = 5.000000e-01 : f64} : tensor<1x1x1x1000xsi16> -> tensor<1x1x1x1000xsi16>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi16>
    } -> tensor<1x1x1x1000xsi16>
    return %0 : tensor<1x1x1x1000xsi16>
// CHECK-NOT:     IE.Clamp
// CHECK:         [[ARG_BC:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xsi16> to tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG_BC]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: i16, {{%.+}}: i16):
// CHECK-NEXT:      [[LOW:%.+]] = arith.constant 1 : i16
// CHECK-NEXT:      [[HIGH:%.+]] = arith.constant 6 : i16
// CHECK-NEXT:      [[CMP_LOW:%.+]] = arith.cmpi sle, [[IN]], [[LOW]] : i16
// CHECK-NEXT:      [[CLAMPED_LOW:%.+]] = arith.select [[CMP_LOW]], [[LOW]], [[IN]] : i16
// CHECK-NEXT:      [[CMP_HIGH:%.+]] = arith.cmpi sle, [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP_HIGH]], [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      linalg.yield [[OP]] : i16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi16> to tensor<1x1x1x1000xsi16>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @ClampS16NonRepresentable
module @ClampS16NonRepresentable {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi16>) -> tensor<1x1x1x1000xsi16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xsi16>) {
      %1 = IE.Clamp(%arg1) {max = 7.000050e+04 : f64, min = -7.000050e+04 : f64} : tensor<1x1x1x1000xsi16> -> tensor<1x1x1x1000xsi16>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi16>
    } -> tensor<1x1x1x1000xsi16>
    return %0 : tensor<1x1x1x1000xsi16>

// CHECK-NOT:     IE.Clamp
// CHECK:         [[ARG_BC:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xsi16> to tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG_BC]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: i16, {{%.+}}: i16):
// CHECK-NEXT:      [[LOW:%.+]] = arith.constant -32768 : i16
// CHECK-NEXT:      [[HIGH:%.+]] = arith.constant 32767 : i16
// CHECK-NEXT:      [[CMP_LOW:%.+]] = arith.cmpi sle, [[IN]], [[LOW]] : i16
// CHECK-NEXT:      [[CLAMPED_LOW:%.+]] = arith.select [[CMP_LOW]], [[LOW]], [[IN]] : i16
// CHECK-NEXT:      [[CMP_HIGH:%.+]] = arith.cmpi sle, [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP_HIGH]], [[HIGH]], [[CLAMPED_LOW]] : i16
// CHECK-NEXT:      linalg.yield [[OP]] : i16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi16>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi16> to tensor<1x1x1x1000xsi16>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @ClampF16
module @ClampF16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Clamp(%arg1) {max = 6.500000e+00 : f64, min = 5.000000e-01 : f64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Clamp
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[LOW:%.+]] = arith.constant 5.000000e-01 : f16
// CHECK-NEXT:      [[HIGH:%.+]] = arith.constant 6.500000e+00 : f16
// CHECK-NEXT:      [[CMP_LOW:%.+]] = arith.cmpf ole, [[IN]], [[LOW]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[CLAMPED_LOW:%.+]] = arith.select [[CMP_LOW]], [[LOW]], [[IN]] : f16
// CHECK-NEXT:      [[CMP_HIGH:%.+]] = arith.cmpf ole, [[CLAMPED_LOW]], [[HIGH]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP_HIGH]], [[CLAMPED_LOW]], [[HIGH]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}
