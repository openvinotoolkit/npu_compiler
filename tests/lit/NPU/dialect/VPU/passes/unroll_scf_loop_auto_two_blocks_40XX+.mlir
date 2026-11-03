//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=outer enable-cascaded-unrolling=false" --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (s0 - 48)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 48)>
#map2 = affine_map<(d0) -> (0, d0 - 1)>
#map3 = affine_map<(d0) -> (-d0 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
#map6 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
module @NPUModule {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func private @caseFn_END(%arg0: tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16> {
    %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 16, 48, 1280] [1, 1, 1, 1] : tensor<1x16x49x1280xf16, {order = #NHWC}> to tensor<1x16x48x1280xf16, {order = #NHWC}>
    %1 = tensor.cast %0 : tensor<1x16x48x1280xf16, {order = #NHWC}> to tensor<1x16x48x1280xf16>
    return %1 : tensor<1x16x48x1280xf16>
  }
  func.func private @caseFn_START(%arg0: tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16> {
    %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 16, 48, 1280] [1, 1, 1, 1] : tensor<1x16x49x1280xf16, {order = #NHWC}> to tensor<1x16x48x1280xf16, {order = #NHWC}>
    %1 = tensor.cast %0 : tensor<1x16x48x1280xf16, {order = #NHWC}> to tensor<1x16x48x1280xf16>
    return %1 : tensor<1x16x48x1280xf16>
  }

  func.func @main(%arg0: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index

    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %3 = scf.for %arg1 = %c0 to %dim step %c48 iter_args(%arg2 = %0) -> (tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
      %5 = arith.addi %arg1, %c48 : index
      %6 = arith.cmpi sgt, %5, %dim : index
      %7 = scf.if %6 -> (index) {
        %20 = affine.apply #map()[%dim]
        scf.yield %20 : index
      } else {
        scf.yield %arg1 : index
      }
      %9 = arith.cmpi eq, %arg1, %c0 : index
      %16 = scf.if %9 -> (index) {
        // First iteration -> START.
        scf.yield %c2 : index
      } else {
        // Any later iteration -> END. With bounds capped at 96 (= 2 * 48), a genuine middle
        // iteration can never occur, so no MIDDLE branch/case exists at all.
        scf.yield %c1 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %7, 0] [1, 16, 49, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x49x1280xf16, {order = #NHWC}>
      %17 = arith.shli %16, %c2 : index
      %18 = arith.ori %c0, %17 : index
      %19 = scf.index_switch %18 -> tensor<1x16x48x1280xf16>
      case 4 {
        %20 = func.call @caseFn_END(%extracted_slice) : (tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      case 8 {
        %20 = func.call @caseFn_START(%extracted_slice) : (tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      default {
        %false = arith.constant false
        cf.assert %false, "Unsupported case"
        %20 = func.call @caseFn_START(%extracted_slice) : (tensor<1x16x49x1280xf16, {order = #NHWC}>) -> tensor<1x16x48x1280xf16>
        scf.yield %20 : tensor<1x16x48x1280xf16>
      }
      %cast_14 = tensor.cast %19 : tensor<1x16x48x1280xf16> to tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %cast_14 into %arg2[0, 0, %7, 0] [1, 16, %c48, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %3 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 96, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }

  // The regression this test guards: the merge-group block count here is 2 (ranged case in
  // generateCombinationsRecursively), but the real position set for this dimension has only 2
  // members {START, END} - not the 3 (or 4) that a fixed-shape assumption would expect. Only the
  // (START, END) combination must be produced; the merge must not try to look up any
  // MIDDLE-involving case and must not crash.
  // CHECK-DAG: func.func private @merged_vpu_func_0(
  // CHECK-DAG: func.func private @caseFn_END(
  // CHECK-DAG: func.func private @caseFn_START(

  // CHECK-LABEL: func.func @main(

  // Merged main loop: exactly one real (START, END) combination -> one case, no MIDDLE-involving
  // case was ever looked up (which is what used to crash).
  // CHECK:      scf.for
  // CHECK:        scf.index_switch
  // CHECK-NEXT:   case
  // CHECK:          func.call @merged_vpu_func_0
  // CHECK:        default {

  // Residual loop: the two original per-iteration cases (END, START) are untouched.
  // CHECK:      scf.for
  // CHECK:        scf.index_switch
  // CHECK-DAG:      case 4 {
  // CHECK-DAG:      case 8 {
}
