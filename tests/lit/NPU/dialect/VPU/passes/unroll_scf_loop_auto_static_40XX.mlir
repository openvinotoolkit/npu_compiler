//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=NPU4000 compilation-mode=HostCompile_Interpreter allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=outer enable-cascaded-unrolling=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// 17 static trips with 16 full source tiles. On NPU4000 (6 DPU tiles) auto-unrolling uses factor 16 / 6 = 2.
// Splitting after all 16 full tiles would leave the partial tail as a single-trip residual loop, which
// canonicalization promotes to straight-line code; the main loop's command list (no_await_all) would then
// never be submitted. One more main-loop iteration is moved into the residual loop, which runs 3 trips.
module @StaticSingleTripTail {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x34x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x34x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x34x16xf16, {order = #NHWC}>) -> tensor<1x16x34x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c33 = arith.constant 33 : index
    %0 = tensor.empty() : tensor<1x16x34x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c33 step %c2 iter_args(%acc = %0) -> (tensor<1x16x34x16xf16, {order = #NHWC}>) {
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x34x16xf16, {order = #NHWC}> to tensor<1x16x2x16xf16, {order = #NHWC}>
      %3 = tensor.insert_slice %2 into %acc[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x2x16xf16, {order = #NHWC}> into tensor<1x16x34x16xf16, {order = #NHWC}>
      scf.yield %3 : tensor<1x16x34x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x34x16xf16, {order = #NHWC}>
  }

  // CHECK-LABEL: module @StaticSingleTripTail
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[C4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[C28:%.+]] = arith.constant 28 : index
  // CHECK-DAG: [[C33:%.+]] = arith.constant 33 : index
  // CHECK: [[MAIN:%.+]] = scf.for {{%.+}} = [[C0]] to [[C28]] step [[C4]]
  // CHECK: } {no_await_all = true}
  // CHECK: [[RESIDUAL:%.+]] = scf.for {{%.+}} = [[C28]] to [[C33]] step [[C2]] iter_args({{%.+}} = [[MAIN]])
  // CHECK: } {no_reset_cmdlist = true}
  // CHECK: return [[RESIDUAL]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// 52 static trips without a partial tail. Auto-unrolling uses factor 52 / 6 = 8, leaving 4 trips. The cascaded
// factor 2 stage would cover these 4 trips exactly and leave an empty epilogue that canonicalization erases,
// so the stage is skipped and the 4 trips stay in the residual loop.
module @StaticCascadeNoRemainder {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x104x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x104x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x104x16xf16, {order = #NHWC}>) -> tensor<1x16x104x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c104 = arith.constant 104 : index
    %0 = tensor.empty() : tensor<1x16x104x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c104 step %c2 iter_args(%acc = %0) -> (tensor<1x16x104x16xf16, {order = #NHWC}>) {
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x104x16xf16, {order = #NHWC}> to tensor<1x16x2x16xf16, {order = #NHWC}>
      %3 = tensor.insert_slice %2 into %acc[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x2x16xf16, {order = #NHWC}> into tensor<1x16x104x16xf16, {order = #NHWC}>
      scf.yield %3 : tensor<1x16x104x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x104x16xf16, {order = #NHWC}>
  }

  // CHECK-LABEL: module @StaticCascadeNoRemainder
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[C16:%.+]] = arith.constant 16 : index
  // CHECK-DAG: [[C96:%.+]] = arith.constant 96 : index
  // CHECK-DAG: [[C104:%.+]] = arith.constant 104 : index
  // CHECK: [[MAIN:%.+]] = scf.for {{%.+}} = [[C0]] to [[C96]] step [[C16]]
  // CHECK: } {no_await_all = true}
  // CHECK: [[RESIDUAL:%.+]] = scf.for {{%.+}} = [[C96]] to [[C104]] step [[C2]] iter_args({{%.+}} = [[MAIN]])
  // CHECK: } {no_reset_cmdlist = true}
  // CHECK-NOT: scf.for
  // CHECK: return [[RESIDUAL]]
}
