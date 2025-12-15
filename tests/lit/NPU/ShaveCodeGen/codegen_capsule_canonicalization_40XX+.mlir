//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @BitcastCanonicalization {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x256x56xsi32>
  } outputsInfo : {
    DataInfo "output0" : tensor<1x1x256x56xsi32>
  }

  func.func @main(%arg0: tensor<1x1x256x56xsi32>) -> tensor<1x1x256x56xsi32> {
    %capsule = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x256x56xsi32>) {
      %ct = arith.constant -1 : i32
      %1 = tensor.bitcast %arg1 : tensor<1x1x256x56xsi32> to tensor<1x1x256x56xi32>
      %2 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1x256x56xi32>) outs(%1 : tensor<1x1x256x56xi32>) {
      ^bb0(%in: i32, %out: i32):
        %5 = arith.xori %in, %ct : i32
        linalg.yield %5 : i32
      } -> tensor<1x1x256x56xi32>
      %3 = tensor.bitcast %2 : tensor<1x1x256x56xi32> to tensor<1x1x256x56xsi32>
      IE.CGCYield %3 : tensor<1x1x256x56xsi32>
    } -> tensor<1x1x256x56xsi32>
    return %capsule : tensor<1x1x256x56xsi32>
  }
  // CHECK: module @BitcastCanonicalization
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x256x56xsi32>) -> tensor<1x1x256x56xsi32> {
  // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x256x56xi32>) {
  // CHECK-NOT: tensor.bitcast
  // CHECK: [[GEN_RES:%.+]] = linalg.generic
  // CHECK-SAME: ins([[CAPSULE_ARG]] : tensor<1x1x256x56xi32>)
  // CHECK-SAME: outs([[CAPSULE_ARG]] : tensor<1x1x256x56xi32>)
  // CHECK-NOT: tensor.bitcast
  // CHECK: IE.CGCYield [[GEN_RES]] : tensor<1x1x256x56xi32>
  // CHECK: } -> tensor<1x1x256x56xsi32>
  // CHECK: return [[VAR0]] : tensor<1x1x256x56xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @PermuteCastCanonicalization {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x1x1x1xf16>
    DataInfo "input2" : tensor<1x1x1x1xf16>
  } outputsInfo : {
    DataInfo "output1" : tensor<1x1x1x1xf16>
  }
  func.func @main(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16> {
    %0 = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    %1 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1xf16>, %0 as %arg3: tensor<1x1x1x1xf16, {order = #NHWC}>) {
      %3 = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
      %4 = tensor.empty() : tensor<1x1x1x1xf16>
      %5 = linalg.generic {indexing_maps = [#NWCH, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %3 : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>) outs(%4 : tensor<1x1x1x1xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %7 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %7 : f16
      } -> tensor<1x1x1x1xf16>
      %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {order = #NHWC}>
      IE.CGCYield %6 : tensor<1x1x1x1xf16, {order = #NHWC}>
    } -> tensor<1x1x1x1xf16, {order = #NHWC}>
    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
    return %2 : tensor<1x1x1x1xf16>
  }
  // CHECK: module @PermuteCastCanonicalization
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1xf16>, [[ARG1:%.+]]: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16> {
  // CHECK: [[VAR0:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {order = #NHWC}>
  // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAP_ARG0:%.+]]: tensor<1x1x1x1xf16>, [[VAR0]] as [[CAP_ARG1:%.+]]: tensor<1x1x1x1xf16>) {
  // CHECK-NOT: IE.PermuteCast
  // CHECK: [[EMPTY_TENSOR:%.+]] = tensor.empty() : tensor<1x1x1x1xf16>
  // CHECK: [[GEN_RES:%.+]] = linalg.generic
  // CHECK-SAME: ins([[CAP_ARG0]], [[CAP_ARG1]] : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>)
  // CHECK-SAME: outs([[EMPTY_TENSOR]] : tensor<1x1x1x1xf16>)
  // CHECK-NOT: IE.PermuteCast
  // CHECK: IE.CGCYield [[GEN_RES]] : tensor<1x1x1x1xf16>
  // CHECK: } -> tensor<1x1x1x1xf16, {order = #NHWC}>
  // CHECK: [[VAR2:%.+]] = IE.PermuteCast([[VAR1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
  // CHECK: return [[VAR2]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
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
      %1 = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xf16, {order = #NHWC}> -> tensor<1x28x29x3xf16>
      %2 = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xf16, {order = #NHWC}> -> tensor<1x28x29x3xf16>
      %3 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %2 : tensor<1x28x29x3xf16>, tensor<1x28x29x3xf16>) outs(%1 : tensor<1x28x29x3xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %5 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %5 : f16
      } -> tensor<1x28x29x3xf16>
      %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %4 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[CastLHS:%.+]]: tensor<1x28x29x3xf16>, [[FuncRHS:%.+]] as [[CastRHS:%.+]]: tensor<1x28x29x3xf16>)
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[CastRHS]] : tensor<1x28x29x3xf16>, tensor<1x28x29x3xf16>) outs([[CastLHS]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    IE.CGCYield [[DIV]] : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: @LayoutInt
module @LayoutInt {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xsi32, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x29xsi32, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xsi32, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xsi32, {order = #NHWC}>, %arg1: tensor<1x3x28x29xsi32, {order = #NHWC}>) -> tensor<1x3x28x29xsi32, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x3x28x29xsi32, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3x28x29xsi32, {order = #NHWC}>) {
      %1 = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xsi32, {order = #NHWC}> -> tensor<1x28x29x3xsi32>
      %2 = tensor.bitcast %1 : tensor<1x28x29x3xsi32> to tensor<1x28x29x3xi32>
      %3 = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xsi32, {order = #NHWC}> -> tensor<1x28x29x3xsi32>
      %4 = tensor.bitcast %3 : tensor<1x28x29x3xsi32> to tensor<1x28x29x3xi32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %4 : tensor<1x28x29x3xi32>, tensor<1x28x29x3xi32>) outs(%2 : tensor<1x28x29x3xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %8 = arith.divsi %in, %in_0 : i32
        linalg.yield %8 : i32
      } -> tensor<1x28x29x3xi32>
      %6 = tensor.bitcast %5 : tensor<1x28x29x3xi32> to tensor<1x28x29x3xsi32>
      %7 = IE.PermuteCast(%6) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xsi32> -> tensor<1x3x28x29xsi32, {order = #NHWC}>
      IE.CGCYield %7 : tensor<1x3x28x29xsi32, {order = #NHWC}>
    } -> tensor<1x3x28x29xsi32, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xsi32, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xsi32, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xsi32, {order = [[NHWC]]}>) -> tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[BC_LHS:%.+]]: tensor<1x28x29x3xi32>, [[FuncRHS:%.+]] as [[BC_RHS:%.+]]: tensor<1x28x29x3xi32>)
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[BC_LHS]], [[BC_RHS]] : tensor<1x28x29x3xi32>, tensor<1x28x29x3xi32>) outs([[BC_LHS]] : tensor<1x28x29x3xi32>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: i32, [[ScalarRHS:%.+]]: i32, [[ScalarOut:%.+]]: i32):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divsi [[ScalarLHS]], [[ScalarRHS]] : i32
  // CHECK-NEXT:      linalg.yield [[RES]] : i32
  // CHECK-NEXT:    } -> tensor<1x28x29x3xi32>
  // CHECK-NEXT:    IE.CGCYield [[DIV]] : tensor<1x28x29x3xi32>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xsi32, {order = [[NHWC]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: @LayoutTablegenPat
module @LayoutTablegenPat {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>) -> tensor<1x3x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xf16, {order = #NHWC}> -> tensor<1x28x29x3xf16>
      %2 = math.log %1 fastmath<afn> : tensor<1x28x29x3xf16>
      %3 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %3 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncARG:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncARG:%.+]] as [[Cast_arg:%.+]]: tensor<1x28x29x3xf16>)
  // CHECK-NEXT:    [[Res:%.+]] = math.log [[Cast_arg]] fastmath<afn> : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    IE.CGCYield [[Res]] : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
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
      %1 = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xf16, {order = #NHWC}> -> tensor<1x28x29x3xf16>
      %2 = linalg.generic {indexing_maps = [#NCHW, #NWCH, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg3 : tensor<1x28x29x3xf16>, tensor<1x3x28x29xf16>) outs(%1 : tensor<1x28x29x3xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %4 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %4 : f16
      } -> tensor<1x28x29x3xf16>
      %3 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %3 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[CastLHS:%.+]]: tensor<1x28x29x3xf16>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x29xf16>)
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NWCH]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x28x29x3xf16>, tensor<1x3x28x29xf16>) outs([[CastLHS]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    IE.CGCYield [[DIV]] : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, 0, d2)>
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
      %1 = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x28x29xf16, {order = #NHWC}> -> tensor<1x28x29x3xf16>
      %2 = tensor.empty() : tensor<1x28x29x3xf16>
      %3 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg3 : tensor<1x28x29x3xf16>, tensor<1x3x1x29xf16>) outs(%2 : tensor<1x28x29x3xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %5 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %5 : f16
      } -> tensor<1x28x29x3xf16>
      %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %4 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x28x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x1x29xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[CastLHS:%.+]]: tensor<1x28x29x3xf16>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x1x29xf16>)
  // CHECK-NEXT:    [[OUT:%.+]] = tensor.empty() : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[MAP]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x28x29x3xf16>, tensor<1x3x1x29xf16>) outs([[OUT]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    IE.CGCYield [[DIV]] : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
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
      %1 = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x1x29xf16, {order = #NHWC}> -> tensor<1x1x29x3xf16>
      %2 = tensor.empty() : tensor<1x28x29x3xf16>
      %3 = linalg.generic {indexing_maps = [#map, #NWCH, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg3 : tensor<1x1x29x3xf16>, tensor<1x3x28x29xf16>) outs(%2 : tensor<1x28x29x3xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %5 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %5 : f16
      } -> tensor<1x28x29x3xf16>
      %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %4 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x1x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x29xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[CastLHS:%.+]]: tensor<1x1x29x3xf16>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x29xf16>)
  // CHECK-NEXT:    [[OUT:%.+]] = tensor.empty() : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[NWCH]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x1x29x3xf16>, tensor<1x3x28x29xf16>) outs([[OUT]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    IE.CGCYield [[DIV]] : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, 0)>
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
      %1 = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x1x29xf16, {order = #NHWC}> -> tensor<1x1x29x3xf16>
      %2 = tensor.empty() : tensor<1x28x29x3xf16>
      %3 = linalg.generic {indexing_maps = [#map, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg3 : tensor<1x1x29x3xf16>, tensor<1x3x28x1xf16>) outs(%2 : tensor<1x28x29x3xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %5 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %5 : f16
      } -> tensor<1x28x29x3xf16>
      %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x29x3xf16> -> tensor<1x3x28x29xf16, {order = #NHWC}>
      IE.CGCYield %4 : tensor<1x3x28x29xf16, {order = #NHWC}>
    } -> tensor<1x3x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xf16, {order = #NHWC}>
  }
  // CHECK: func.func @main([[FuncLHS:%.+]]: tensor<1x3x1x29xf16, {order = [[NHWC]]}>, [[FuncRHS:%.+]]: tensor<1x3x28x1xf16>) -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK: [[CGRES:%.+]] = IE.CodeGenCapsule inputs([[FuncLHS:%.+]] as [[CastLHS:%.+]]: tensor<1x1x29x3xf16>, [[FuncRHS:%.+]] as [[RHS:%.+]]: tensor<1x3x28x1xf16>)
  // CHECK-NEXT:    [[OUT:%.+]] = tensor.empty() : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    [[DIV:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP1]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[CastLHS]], [[RHS]] : tensor<1x1x29x3xf16>, tensor<1x3x28x1xf16>) outs([[OUT]] : tensor<1x28x29x3xf16>) {
  // CHECK-NEXT:    ^bb0([[ScalarLHS:%.+]]: f16, [[ScalarRHS:%.+]]: f16, [[OUT:%.+]]: f16):
  // CHECK-NEXT:      [[RES:%.+]] = arith.divf [[ScalarLHS]], [[ScalarRHS]] fastmath<arcp> : f16
  // CHECK-NEXT:      linalg.yield [[RES]] : f16
  // CHECK-NEXT:    } -> tensor<1x28x29x3xf16>
  // CHECK-NEXT:    IE.CGCYield [[DIV]] : tensor<1x28x29x3xf16>
  // CHECK-NEXT:    } -> tensor<1x3x28x29xf16, {order = [[NHWC]]}>
  // CHECK-NEXT:    return [[CGRES]] : tensor<1x3x28x29xf16, {order = [[NHWC]]}>
}
