//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,1,1 enable-cascaded-unrolling=false" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// When all manual unroll factors are <= 1, the pass should be skipped entirely.
// The IR must remain unchanged.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @NoopAllOnesSkip inputsInfo : {
    DataInfo "input" : tensor<1x32x64x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x64xf16>
  }

  // CHECK-LABEL: func.func @NoopAllOnesSkip
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}>
  func.func @NoopAllOnesSkip(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index

    %0 = tensor.empty() : tensor<1x32x64x64xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c32 step %c2 iter_args(%arg1 = %0) -> (tensor<1x32x64x64xf16, {order = #NHWC}>) {
        %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 32, 2, 64] [1, 1, 1, 1]
            : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>

        %3 = tensor.insert_slice %2 into %arg1[0, 0, %i, 0] [1, 32, 2, 64] [1, 1, 1, 1]
            : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>

        scf.yield %3 : tensor<1x32x64x64xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x32x64x64xf16, {order = #NHWC}>
  }

  // Pass is skipped: loop step remains 2, no unrolling artifacts present.
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[C32:%.+]] = arith.constant 32 : index
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK: [[RESULT:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[C32]] step [[C2]] iter_args([[ARG:%.+]] = [[EMPTY]]) -> (tensor<1x32x64x64xf16, {order = #NHWC}>) {
  // No step doubling should occur
  // CHECK-NOT: arith.constant 4 : index
  // CHECK:   [[E:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[IV]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK:   [[I:%.+]] = tensor.insert_slice [[E]] into [[ARG]][0, 0, [[IV]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK:   scf.yield [[I]] : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return [[RESULT]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

// Test nested loops with all-ones factors. Both loops should remain unchanged.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @NoopNestedLoops inputsInfo : {
    DataInfo "input" : tensor<1x32x64x96xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x96xf16>
  }

  // CHECK-LABEL: func.func @NoopNestedLoops
  func.func @NoopNestedLoops(%arg0: tensor<1x32x64x96xf16, {order = #NHWC}>) -> tensor<1x32x64x96xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty() : tensor<1x32x64x96xf16, {order = #NHWC}>
    %1 = scf.for %h = %c0 to %c64 step %c2 iter_args(%arg1 = %0) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
        %2 = scf.for %w = %c0 to %c96 step %c3 iter_args(%arg2 = %arg1) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
            %3 = tensor.extract_slice %arg0[0, 0, %h, %w] [1, 32, 2, 3] [1, 1, 1, 1]
                : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
            %4 = tensor.insert_slice %3 into %arg2[0, 0, %h, %w] [1, 32, 2, 3] [1, 1, 1, 1]
                : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
            scf.yield %4 : tensor<1x32x64x96xf16, {order = #NHWC}>
        }
        scf.yield %2 : tensor<1x32x64x96xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x32x64x96xf16, {order = #NHWC}>
  }

  // Both loops unchanged: outer step remains 2, inner step remains 3.
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[C3:%.+]] = arith.constant 3 : index
  // CHECK: scf.for {{.+}} step [[C2]]
  // CHECK:   scf.for {{.+}} step [[C3]]
  // CHECK:     tensor.extract_slice
  // CHECK:     tensor.insert_slice
  // CHECK:     scf.yield
  // CHECK:   scf.yield
  // CHECK: return
}
