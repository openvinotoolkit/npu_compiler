//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,2,2 enable-cascaded-unrolling=false" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,2,5 enable-cascaded-unrolling=false" %s | FileCheck %s --check-prefix=CHECK-EXTRA
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Test multi-dimension manual unrolling with factors on both H (dim 2) and W (dim 3).
// Regression test for the crash when multiple manual unroll factors were specified
// on nested loops (e.g., loop-unroll-factor=1,1,30,4 caused SmallVector OOB assertion).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @MultiDimLoopUnroll inputsInfo : {
    DataInfo "input" : tensor<1x32x64x96xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x96xf16>
  }

  // CHECK-LABEL: func.func @MultiDimLoopUnroll
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x64x96xf16, {order = #NHWC}>) -> tensor<1x32x64x96xf16, {order = #NHWC}>
  func.func @MultiDimLoopUnroll(%arg0: tensor<1x32x64x96xf16, {order = #NHWC}>) -> tensor<1x32x64x96xf16, {order = #NHWC}> {
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

  // Outer loop step doubled from 2 to 4 (factor 2 on dim 2).
  // Inner loop step doubled from 3 to 6 (factor 2 on dim 3).
  // After unrolling: 4 extract/insert pairs for all (H,H+2) x (W,W+3) combinations.

  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[C64:%.+]] = arith.constant 64 : index
  // CHECK: [[C96:%.+]] = arith.constant 96 : index
  // CHECK: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[C3:%.+]] = arith.constant 3 : index
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x64x96xf16, {order = #NHWC}>
  // Outer loop unrolled: step 2 * 2 = 4
  // CHECK: [[C4:%.+]] = arith.constant 4 : index
  // CHECK: [[RESULT:%.+]] = scf.for [[H:%.+]] = [[C0]] to [[C64]] step [[C4]] iter_args([[OARG:%.+]] = [[EMPTY]]) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
  // Inner loop unrolled: step 3 * 2 = 6
  // CHECK:   {{%.+}} = arith.constant 6 : index
  // H offset for second unrolled iteration
  // CHECK:   {{%.+}} = arith.muli [[C2]], {{%.+}} : index
  // CHECK:   [[H2:%.+]] = arith.addi [[H]], {{%.+}} : index
  // CHECK:   {{%.+}} = arith.constant 6 : index
  // CHECK:   [[INNER:%.+]] = scf.for [[W:%.+]] = [[C0]] to [[C96]] step {{%.+}} iter_args([[IARG:%.+]] = [[OARG]]) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
  // W offset computations for second unrolled inner iteration
  // CHECK:     {{%.+}} = arith.muli [[C3]], {{%.+}} : index
  // CHECK:     [[W2:%.+]] = arith.addi [[W]], {{%.+}} : index
  // CHECK:     {{%.+}} = arith.muli [[C3]], {{%.+}} : index
  // CHECK:     [[W3:%.+]] = arith.addi [[W]], {{%.+}} : index
  // 4 extract/insert pairs: (H,W), (H,W+3), (H+2,W), (H+2,W+3)
  // CHECK:     [[E1:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     [[I1:%.+]] = tensor.insert_slice [[E1]] into [[IARG]][0, 0, [[H]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     [[E2:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H]], [[W2]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     {{%.+}} = tensor.insert_slice [[E2]] into [[I1]][0, 0, [[H]], [[W2]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     [[E3:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H2]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     {{%.+}} = tensor.insert_slice [[E3]] into [[IARG]][0, 0, [[H2]], [[W]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     [[E4:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[H2]], [[W3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     {{%.+}} = tensor.insert_slice [[E4]] into {{%.+}}[0, 0, [[H2]], [[W3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     scf.yield {{%.+}} : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:   }
  // CHECK:   scf.yield [[INNER]] : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return [[RESULT]] : tensor<1x32x64x96xf16, {order = #NHWC}>
}

// -----

// Test that extra unused factors in the unroll factor array do not cause issues.
// With [1,1,2,5] on a single loop mapping to dim 2, the dim 3 factor (5) is unused.
// The loop should be unrolled only on dim 2 with factor 2.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @SingleLoopExtraFactor inputsInfo : {
    DataInfo "input" : tensor<1x32x64x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x64xf16>
  }

  // CHECK-EXTRA-LABEL: func.func @SingleLoopExtraFactor
  // CHECK-EXTRA-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}>
  func.func @SingleLoopExtraFactor(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
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

  // Only dim 2 (H) has a matching loop; dim 3 factor 5 is unused.
  // Step should be doubled from 2 to 4, with 2 extract/insert pairs.
  // CHECK-EXTRA: [[C0:%.+]] = arith.constant 0 : index
  // CHECK-EXTRA: [[C32:%.+]] = arith.constant 32 : index
  // CHECK-EXTRA: [[C2:%.+]] = arith.constant 2 : index
  // CHECK-EXTRA: [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK-EXTRA: [[C4:%.+]] = arith.constant 4 : index
  // CHECK-EXTRA: [[RESULT:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[C32]] step [[C4]] iter_args([[ARG:%.+]] = [[EMPTY]]) -> (tensor<1x32x64x64xf16, {order = #NHWC}>) {
  // CHECK-EXTRA:   {{%.+}} = arith.muli [[C2]], {{%.+}} : index
  // CHECK-EXTRA:   [[IV2:%.+]] = arith.addi [[IV]], {{%.+}} : index
  // CHECK-EXTRA:   [[E1:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[IV]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK-EXTRA:   [[I1:%.+]] = tensor.insert_slice [[E1]] into [[ARG]][0, 0, [[IV]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK-EXTRA:   [[E2:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[IV2]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK-EXTRA:   {{%.+}} = tensor.insert_slice [[E2]] into [[I1]][0, 0, [[IV2]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK-EXTRA:   scf.yield {{%.+}} : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK-EXTRA: }
  // CHECK-EXTRA: return [[RESULT]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}
