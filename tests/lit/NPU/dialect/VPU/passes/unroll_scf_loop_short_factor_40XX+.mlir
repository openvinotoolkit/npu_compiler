//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,3 enable-cascaded-unrolling=false" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Test with fewer unroll factors than tensor dimensions.
// Factor array [1,1,3] has 3 elements but loops reference dim 3 (idx 3).
// Regression test for the bounds check in processUnrolledLoops where
// accessing unrollFactor[idx] with idx >= size would crash (SmallVector OOB).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @ShortFactorNestedLoops inputsInfo : {
    DataInfo "input" : tensor<1x32x48x96xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x48x96xf16>
  }

  // CHECK-LABEL: func.func @ShortFactorNestedLoops
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x48x96xf16, {order = #NHWC}>) -> tensor<1x32x48x96xf16, {order = #NHWC}>
  func.func @ShortFactorNestedLoops(%arg0: tensor<1x32x48x96xf16, {order = #NHWC}>) -> tensor<1x32x48x96xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c48 = arith.constant 48 : index
    %c96 = arith.constant 96 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty() : tensor<1x32x48x96xf16, {order = #NHWC}>
    %1 = scf.for %h = %c0 to %c48 step %c2 iter_args(%arg1 = %0) -> (tensor<1x32x48x96xf16, {order = #NHWC}>) {
        %2 = scf.for %w = %c0 to %c96 step %c3 iter_args(%arg2 = %arg1) -> (tensor<1x32x48x96xf16, {order = #NHWC}>) {
            %3 = tensor.extract_slice %arg0[0, 0, %h, %w] [1, 32, 2, 3] [1, 1, 1, 1]
                : tensor<1x32x48x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>

            %4 = tensor.insert_slice %3 into %arg2[0, 0, %h, %w] [1, 32, 2, 3] [1, 1, 1, 1]
                : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x48x96xf16, {order = #NHWC}>

            scf.yield %4 : tensor<1x32x48x96xf16, {order = #NHWC}>
        }
        scf.yield %2 : tensor<1x32x48x96xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x32x48x96xf16, {order = #NHWC}>
  }

  // Outer H loop (dim 2) unrolled by factor 3: step 2 * 3 = 6.
  // Inner W loop (dim 3) NOT unrolled (no factor in array for idx 3): step stays 3.
  // After fusion: 3 extract/insert pairs for H offsets (H, H+2, H+4).

  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[C48:%.+]] = arith.constant 48 : index
  // CHECK: [[C96:%.+]] = arith.constant 96 : index
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[C3:%.+]] = arith.constant 3 : index
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x48x96xf16, {order = #NHWC}>
  // Outer loop step tripled: 2 * 3 = 6
  // CHECK: [[C6:%.+]] = arith.constant 6 : index
  // CHECK: [[RESULT:%.+]] = scf.for [[H:%.+]] = [[C0]] to [[C48]] step [[C6]] iter_args([[OARG:%.+]] = [[EMPTY]]) -> (tensor<1x32x48x96xf16, {order = #NHWC}>) {
  // H+2 and H+4 offset computations
  // CHECK:   {{%.+}} = arith.muli [[C2]], {{%.+}} : index
  // CHECK:   [[H2:%.+]] = arith.addi [[H]], {{%.+}} : index
  // CHECK:   {{%.+}} = arith.muli [[C2]], {{%.+}} : index
  // CHECK:   [[H3:%.+]] = arith.addi [[H]], {{%.+}} : index
  // Inner loop step unchanged at 3
  // CHECK:   [[INNER:%.+]] = scf.for [[W:%.+]] = [[C0]] to [[C96]] step [[C3]] iter_args({{.+}}) -> (tensor<1x32x48x96xf16, {order = #NHWC}>) {
  // 3 extract/insert pairs for H, H+2, H+4 (all at same W)
  // CHECK:     [[E1:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x48x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     {{%.+}} = tensor.insert_slice [[E1]] into {{%.+}}[0, 0, [[H]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1]
  // CHECK:     [[E2:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H2]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x48x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     {{%.+}} = tensor.insert_slice [[E2]] into {{%.+}}[0, 0, [[H2]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1]
  // CHECK:     [[E3:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H3]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x48x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     {{%.+}} = tensor.insert_slice [[E3]] into {{%.+}}[0, 0, [[H3]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1]
  // CHECK:     scf.yield {{%.+}} : tensor<1x32x48x96xf16, {order = #NHWC}>
  // CHECK:   }
  // CHECK:   scf.yield [[INNER]] : tensor<1x32x48x96xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return [[RESULT]] : tensor<1x32x48x96xf16, {order = #NHWC}>
}

// -----

// Test single loop with shorter factor array [1,1,3].
// Only one loop exists on dim 2. The factor array has no entry for dim 3,
// but the insert_slice has a constant 0 offset at dim 3 (no loop involvement).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @ShortFactorSingleLoop inputsInfo : {
    DataInfo "input" : tensor<1x32x48x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x48x64xf16>
  }

  // CHECK-LABEL: func.func @ShortFactorSingleLoop
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x48x64xf16, {order = #NHWC}>) -> tensor<1x32x48x64xf16, {order = #NHWC}>
  func.func @ShortFactorSingleLoop(%arg0: tensor<1x32x48x64xf16, {order = #NHWC}>) -> tensor<1x32x48x64xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c2 = arith.constant 2 : index

    %0 = tensor.empty() : tensor<1x32x48x64xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c24 step %c2 iter_args(%arg1 = %0) -> (tensor<1x32x48x64xf16, {order = #NHWC}>) {
        %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 32, 2, 64] [1, 1, 1, 1]
            : tensor<1x32x48x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>

        %3 = tensor.insert_slice %2 into %arg1[0, 0, %i, 0] [1, 32, 2, 64] [1, 1, 1, 1]
            : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x48x64xf16, {order = #NHWC}>

        scf.yield %3 : tensor<1x32x48x64xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x32x48x64xf16, {order = #NHWC}>
  }

  // Loop step tripled: 2 * 3 = 6. Body has 3 extract/insert pairs.
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[C24:%.+]] = arith.constant 24 : index
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x48x64xf16, {order = #NHWC}>
  // CHECK: [[C6:%.+]] = arith.constant 6 : index
  // CHECK: [[RESULT:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[C24]] step [[C6]] iter_args([[ARG:%.+]] = [[EMPTY]]) -> (tensor<1x32x48x64xf16, {order = #NHWC}>) {
  // CHECK:   {{%.+}} = arith.muli [[C2]], {{%.+}} : index
  // CHECK:   [[IV2:%.+]] = arith.addi [[IV]], {{%.+}} : index
  // CHECK:   {{%.+}} = arith.muli [[C2]], {{%.+}} : index
  // CHECK:   [[IV3:%.+]] = arith.addi [[IV]], {{%.+}} : index
  // CHECK:   tensor.extract_slice [[ARG_0]][0, 0, [[IV]], 0] [1, 32, 2, 64] [1, 1, 1, 1]
  // CHECK:   tensor.insert_slice
  // CHECK:   tensor.extract_slice [[ARG_0]][0, 0, [[IV2]], 0] [1, 32, 2, 64] [1, 1, 1, 1]
  // CHECK:   tensor.insert_slice
  // CHECK:   tensor.extract_slice [[ARG_0]][0, 0, [[IV3]], 0] [1, 32, 2, 64] [1, 1, 1, 1]
  // CHECK:   tensor.insert_slice
  // CHECK:   scf.yield {{%.+}} : tensor<1x32x48x64xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return [[RESULT]] : tensor<1x32x48x64xf16, {order = #NHWC}>
}
