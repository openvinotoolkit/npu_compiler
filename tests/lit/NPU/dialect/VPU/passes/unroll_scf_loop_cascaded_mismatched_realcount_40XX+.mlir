//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,1,10 enable-cascaded-unrolling=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// This dimension's real tile-position count (from its scf.index_switch case list below: MIDDLE,
// END, START all present) is large (>=3), but cascaded unrolling produces an intermediate stage
// (factor 2) whose *local* merge-group block count is only 2. The merge must still enumerate the
// MIDDLE-involving combinations for that stage, driven by the real case set - not by the local
// block count of the current cascade stage.
module @CascadedMismatchedRealCount {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_0" : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>
  } outputsInfo : {
    DataInfo "Result_0" : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>
  }

  func.func private @caseFn_MIDDLE(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    return %arg0 : tensor<1x16x1x1xf16>
  }
  func.func private @caseFn_END(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    return %arg0 : tensor<1x16x1x1xf16>
  }
  func.func private @caseFn_START(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    return %arg0 : tensor<1x16x1x1xf16>
  }

  func.func @main(%arg0: tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>) -> tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index

    %cdimW = arith.constant 3 : index
    %dimW = tensor.dim %arg0, %cdimW : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>
    %0 = tensor.empty(%dimW) : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>

    %res = scf.for %w = %c0 to %dimW step %c512 iter_args(%acc = %0) -> (tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>) {
      %wIsZero = arith.cmpi eq, %w, %c0 : index
      %wIsLast = arith.cmpi sge, %w, %c512 : index
      %wPosNotStart = arith.select %wIsLast, %c1, %c0 : index
      %wPos = arith.select %wIsZero, %c2, %wPosNotStart : index

      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %w] [1, 16, 1, 1] [1, 1, 1, 1] : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}> to tensor<1x16x1x1xf16>

      %switched = scf.index_switch %wPos -> tensor<1x16x1x1xf16>
      case 0 {
        %r = func.call @caseFn_MIDDLE(%extracted_slice) : (tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
        scf.yield %r : tensor<1x16x1x1xf16>
      }
      case 1 {
        %r = func.call @caseFn_END(%extracted_slice) : (tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
        scf.yield %r : tensor<1x16x1x1xf16>
      }
      case 2 {
        %r = func.call @caseFn_START(%extracted_slice) : (tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
        scf.yield %r : tensor<1x16x1x1xf16>
      }
      default {
        %false = arith.constant false
        cf.assert %false, "Unsupported case"
        %r = func.call @caseFn_MIDDLE(%extracted_slice) : (tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
        scf.yield %r : tensor<1x16x1x1xf16>
      }

      %inserted_slice = tensor.insert_slice %switched into %acc[0, 0, 0, %w] [1, 16, 1, 1] [1, 1, 1, 1] : tensor<1x16x1x1xf16> into tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>
      scf.yield %inserted_slice : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>
    }
    return %res : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1024]> : tensor<4xsi64>}>
  }
}

// CHECK-LABEL: func.func @main(

// CHECK: scf.for {{.*}} step %c5120
// CHECK: scf.index_switch
// CHECK-DAG: case 2 {
// CHECK-DAG: case 6 {
// CHECK-DAG: case 0 {
// CHECK-DAG: case 4 {

// CHECK: scf.for {{.*}} step %c2560
// CHECK: scf.index_switch
// CHECK-DAG: case 2 {
// CHECK-DAG: case 6 {
// CHECK-DAG: case 0 {
// CHECK-DAG: case 4 {

// The regression this test guards: this cascade stage's own merge-group block count is 2, but the
// dimension's real (pre-cascade) block count is >2 - all 4 cases, including the MIDDLE-involving
// ones, must still be present.
// CHECK: scf.for {{.*}} step %c1024
// CHECK: scf.index_switch
// CHECK-DAG: case 2 {
// CHECK-DAG: case 6 {
// CHECK-DAG: case 0 {
// CHECK-DAG: case 4 {

// CHECK: scf.for {{.*}} step %c512
// CHECK: scf.index_switch
// CHECK-DAG: case 0 {
// CHECK-DAG: case 1 {
// CHECK-DAG: case 2 {
