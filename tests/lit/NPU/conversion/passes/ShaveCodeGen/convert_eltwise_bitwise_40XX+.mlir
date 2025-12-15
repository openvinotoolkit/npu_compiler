//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX


// BitwiseAnd

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: BitwiseAnd
module @BitwiseAnd {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xui32>, %arg1 as %arg3: tensor<1x1x1x1000xui32>) {
      %1 = IE.BitwiseAnd(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT:     IE.BitwiseAnd
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.andi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// BitwiseXor

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: BitwiseXorLayer
module @BitwiseXorLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xui32>, %arg1 as %arg3: tensor<1x1x1x1000xui32>) {
      %1 = IE.BitwiseXor(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT:     IE.BitwiseXor
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.xori [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// BitwiseOr

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: BitwiseOrLayer
module @BitwiseOrLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xui32>, %arg1 as %arg3: tensor<1x1x1x1000xui32>) {
      %1 = IE.BitwiseOr(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT:     IE.BitwiseOr
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.ori [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// BitwiseNot

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: BitwiseNotLayer
module @BitwiseNotLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xui32>) {
      %1 = IE.BitwiseNot(%arg1) : tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT:     IE.BitwiseNot
// CHECK:         [[ARG_BC:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG_BC]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[VAL:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[ALLONES:%.+]] = arith.constant -1 : i32
// CHECK-NEXT:      [[OP:%.+]] = arith.xori [[VAL]], [[ALLONES]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
  }
}
