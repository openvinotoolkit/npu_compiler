//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// IE.Add

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @AddFOp
module @AddFOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Add
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, [[ARG1:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.addf [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @AddIOp
module @AddIOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.Add
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.addi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @AddWithReluPostOp
module @AddWithReluPostOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Add
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:   [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ADD:%.+]] = arith.addf [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[CMP:%.+]] = arith.cmpf ole, [[ADD]], [[ZERO]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[OP:%.+]] = arith.select [[CMP]], [[ZERO]], [[ADD]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>

  }
}

// -----

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @AddWithLeakyReluPostOp
module @AddWithLeakyReluPostOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.LeakyRelu<negative_slope = 2.000000e-01 : f64>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Add
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ADD:%.+]] = arith.addf [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[CMP:%.+]] = arith.cmpf ole, [[ADD]], [[ZERO]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[POS:%.+]] = arith.select [[CMP]], [[ZERO]], [[ADD]] : f16
// CHECK-NEXT:      [[NEG:%.+]] = arith.select [[CMP]], [[ADD]], [[ZERO]] : f16
// CHECK-NEXT:      [[SLOPE:%.+]] = arith.constant 1.999510e-01 : f16
// CHECK-NEXT:      [[WSLOP:%.+]] = arith.mulf [[NEG]], [[SLOPE]] : f16
// CHECK-NEXT:      [[OP:%.+]] = arith.addf [[POS]], [[WSLOP]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Multiply

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @MulFOp
module @MulFOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Multiply
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.mulf [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @MulIOp
module @MulIOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.Multiply
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.muli [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>

  }
}

// -----
// IE.Subtract

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @SubFOp
module @SubFOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Subtract(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Subtract
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:   [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.subf [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @SubIOp
module @SubIOp {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Subtract(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.Subtract
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.subi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}
