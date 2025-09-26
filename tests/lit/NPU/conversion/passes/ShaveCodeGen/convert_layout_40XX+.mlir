//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: @LayoutFP
module @LayoutFP {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>, %arg1: tensor<1x3x28x29xf16, {order = #NHWC}>) -> tensor<1x3x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x28x29xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x28x29xf16, {order = #NHWC}>, tensor<1x3x28x29xf16, {order = #NHWC}> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>

// CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[LHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>)
// CHECK-NEXT:    [[CastLHS:%.+]] = IE.PermuteCast([[LHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x28x29xf16, {order = [[NHWC]]}> -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[CastRHS:%.+]] = IE.PermuteCast([[RHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x28x29xf16, {order = [[NHWC]]}> -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[CastRHS]] : tensor<1x28x29x3xf16>, tensor<1x28x29x3xf16>) outs([[CastLHS]] : tensor<1x28x29x3xf16>) {
// CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[CastOut:%.+]] = IE.PermuteCast([[DIV]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    IE.CGCYield [[CastOut]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: @LayoutInt
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @LayoutInt {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xsi32, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x29xsi32, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xsi32, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xsi32, {order = #NHWC}>, %arg1: tensor<1x3x28x29xsi32, {order = #NHWC}>) -> tensor<1x3x28x29xsi32, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x28x29xsi32, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x28x29xsi32, {order = #NHWC}>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x28x29xsi32, {order = #NHWC}>, tensor<1x3x28x29xsi32, {order = #NHWC}> -> tensor<1x3x28x29xsi32, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x3x28x29xsi32, {order = #NHWC}>
    } -> tensor<1x3x28x29xsi32, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xsi32, {order = #NHWC}>

// CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xsi32, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xsi32, {order = [[NHWC]]}>) -> tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
// CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[LHS:%.+]]: tensor<1x3x28x29xsi32, {order = [[NHWC]]}>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x29xsi32, {order = [[NHWC]]}>)
// CHECK-NEXT:    [[Cast_LHS:%.+]] = IE.PermuteCast([[LHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x28x29xsi32, {order = [[NHWC]]}> -> tensor<1x28x29x3xsi32>
// CHECK-NEXT:    [[BC_LHS:%.+]] = tensor.bitcast [[Cast_LHS]] : tensor<1x28x29x3xsi32> to tensor<1x28x29x3xi32>
// CHECK-NEXT:    [[Cast_RHS:%.+]] = IE.PermuteCast([[RHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x28x29xsi32, {order = [[NHWC]]}> -> tensor<1x28x29x3xsi32>
// CHECK-NEXT:    [[BC_RHS:%.+]] = tensor.bitcast [[Cast_RHS]] : tensor<1x28x29x3xsi32> to tensor<1x28x29x3xi32>
// CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[BC_LHS]], [[BC_RHS]] : tensor<1x28x29x3xi32>, tensor<1x28x29x3xi32>) outs([[BC_LHS]] : tensor<1x28x29x3xi32>) {
// CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: i32, [[ScalarRHS:%.+]]: i32, [[ScalarOut:%.+]]: i32):
// CHECK-NEXT:      [[RES:%.+]] = arith.divsi [[ScalarLHS]], [[ScalarRHS]] : i32
// CHECK-NEXT:      linalg.yield [[RES]] : i32
// CHECK-NEXT:    } -> tensor<1x28x29x3xi32>
// CHECK-NEXT:    [[OutBC:%.+]] = tensor.bitcast [[DIV]] : tensor<1x28x29x3xi32> to tensor<1x28x29x3xsi32>
// CHECK-NEXT:    [[OutCast:%.+]] = IE.PermuteCast([[OutBC]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x28x29x3xsi32> -> tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
// CHECK-NEXT:    IE.CGCYield [[OutCast]] : tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
// CHECK-NEXT:    } -> tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
// CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: @MixedFP
module @MixedFP {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x29xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>, %arg1: tensor<1x3x28x29xf16>) -> tensor<1x3x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x28x29xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x28x29xf16>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x28x29xf16, {order = #NHWC}>, tensor<1x3x28x29xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>

// CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[LHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x29xf16>)
// CHECK-NEXT:    [[CastLHS:%.+]] = IE.PermuteCast([[LHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x28x29xf16, {order = [[NHWC]]}> -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NWCH]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x28x29x3xf16>, tensor<1x3x28x29xf16>) outs([[CastLHS]] : tensor<1x28x29x3xf16>) {
// CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[CastOut:%.+]] = IE.PermuteCast([[DIV]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    IE.CGCYield [[CastOut]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, 0, d2)>
// CHECK: @MixedFPBroadcast1
module @MixedFPBroadcast1 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x1x29xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>, %arg1: tensor<1x3x1x29xf16>) -> tensor<1x3x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x28x29xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x1x29xf16>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x28x29xf16, {order = #NHWC}>, tensor<1x3x1x29xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>

// CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x1x29xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[LHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x1x29xf16>)
// CHECK-NEXT:    [[CastLHS:%.+]] = IE.PermuteCast([[LHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x28x29xf16, {order = [[NHWC]]}> -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[OUT:%.+]] = tensor.empty() : tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[MAP]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x28x29x3xf16>, tensor<1x3x1x29xf16>) outs([[OUT]] : tensor<1x28x29x3xf16>) {
// CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
// CHECK-NEXT:    [[CastOut:%.+]] = IE.PermuteCast([[DIV]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    IE.CGCYield [[CastOut]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
// CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: [[MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
// CHECK: @MixedFPBroadcast2
module @MixedFPBroadcast2 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x1x29xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x29xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x1x29xf16, {order = #NHWC}>, %arg1: tensor<1x3x28x29xf16>) -> tensor<1x3x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x1x29xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x28x29xf16>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x29xf16, {order = #NHWC}>, tensor<1x3x28x29xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>

  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x1x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[LHS:%.+]]: tensor<1x3x1x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x29xf16>)
  // CHECK-NEXT:    [[CastLHS:%.+]] = IE.PermuteCast([[LHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x1x29xf16, {order = [[NHWC]]}> -> tensor<1x1x29x3xf16>
  // CHECK-NEXT:    [[OUT:%.+]] = tensor.empty() : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[NWCH]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x1x29x3xf16>, tensor<1x3x28x29xf16>) outs([[OUT]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[CastOut:%.+]] = IE.PermuteCast([[DIV]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    IE.CGCYield [[CastOut]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
// CHECK: [[MAP1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, 0)>
// CHECK: @MixedFPBroadcast3
module @MixedFPBroadcast3 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x1x29xf16, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x1x29xf16, {order = #NHWC}>, %arg1: tensor<1x3x28x1xf16>) -> tensor<1x3x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x1x29xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x28x1xf16>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x29xf16, {order = #NHWC}>, tensor<1x3x28x1xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>

  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x1x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x1xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[LHS:%.+]]: tensor<1x3x1x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x1xf16>)
  // CHECK-NEXT:    [[CastLHS:%.+]] = IE.PermuteCast([[LHS]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x1x29xf16, {order = [[NHWC]]}> -> tensor<1x1x29x3xf16>
  // CHECK-NEXT:    [[OUT:%.+]] = tensor.empty() : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP1]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x1x29x3xf16>, tensor<1x3x28x1xf16>) outs([[OUT]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[CastOut:%.+]] = IE.PermuteCast([[DIV]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    IE.CGCYield [[CastOut]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  }
}
