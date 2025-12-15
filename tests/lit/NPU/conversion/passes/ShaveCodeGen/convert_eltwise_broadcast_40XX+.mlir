//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" \
// RUN:     --convert-eltwise-layers-to-math | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @BroadcastMax {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x16xf16>
    DataInfo "input1" : tensor<1x1x1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x1x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x16x16xf16>, %arg1 as %arg3: tensor<1x1x1x16xf16>) {
      %1 = IE.Maximum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x16x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x16x16xf16>
      IE.CGCYield %1 : tensor<1x1x16x16xf16>
    } -> tensor<1x1x16x16xf16>
    return %0 : tensor<1x1x16x16xf16>
  }
// CHECK: #[[NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
// CHECK: module @BroadcastMax

// CHECK-NOT:     IE.Maximum
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[LINALG:%.+]] = linalg.generic {indexing_maps = [#[[NCHW]], #[[MAP_RHS]], #[[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x1x16x16xf16>, tensor<1x1x1x16xf16>) outs([[EMPTY]] : tensor<1x1x16x16xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:      [[MAX:%.+]] = arith.maximumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[MAX]] : f16
// CHECK-NEXT:    } -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG]] : tensor<1x1x16x16xf16>
}

// -----

module @BroadcastDiv {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x16xf16>
    DataInfo "input1" : tensor<1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x16x16xf16>, %arg1 as %arg3: tensor<1x16xf16>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x16x16xf16>, tensor<1x16xf16> -> tensor<1x1x16x16xf16>
      IE.CGCYield %1 : tensor<1x1x16x16xf16>
    } -> tensor<1x1x16x16xf16>
    return %0 : tensor<1x1x16x16xf16>
  }
// CHECK: #[[NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (0, d3)>
// CHECK: module @BroadcastDiv

// CHECK-NOT:     IE.Divide
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[LINALG:%.+]] = linalg.generic {indexing_maps = [#[[NCHW]], #[[MAP_RHS]], #[[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x1x16x16xf16>, tensor<1x16xf16>) outs([[EMPTY]] : tensor<1x1x16x16xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:      [[DIV:%.+]] = arith.divf [[LHS]], [[RHS]] fastmath<arcp> : f16
// CHECK-NEXT:      linalg.yield [[DIV]] : f16
// CHECK-NEXT:    } -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG]] : tensor<1x1x16x16xf16>
}

// -----

module @BroadcastMin {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x16x16xf16>
    DataInfo "input1" : tensor<10x1x1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<10x1x16x16xf16>
  }

  func.func @main(%arg0: tensor<1x16x16xf16>, %arg1: tensor<10x1x1x16xf16>) -> tensor<10x1x16x16xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x16x16xf16>, %arg1 as %arg3: tensor<10x1x1x16xf16>) {
      %1 = IE.Minimum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16xf16>, tensor<10x1x1x16xf16> -> tensor<10x1x16x16xf16>
      IE.CGCYield %1 : tensor<10x1x16x16xf16>
    } -> tensor<10x1x16x16xf16>
    return %0 : tensor<10x1x16x16xf16>
  }
// CHECK: #[[NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
// CHECK: module @BroadcastMin

// CHECK-NOT:     IE.Minimum
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<10x1x16x16xf16>
// CHECK-NEXT:    [[LINALG:%.+]] = linalg.generic {indexing_maps = [#[[MAP_LHS]], #[[MAP_RHS]], #[[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x16x16xf16>, tensor<10x1x1x16xf16>) outs([[EMPTY]] : tensor<10x1x16x16xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:      [[MIN:%.+]] = arith.minimumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[MIN]] : f16
// CHECK-NEXT:    } -> tensor<10x1x16x16xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG]] : tensor<10x1x16x16xf16>
}
