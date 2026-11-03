//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=NPU5010 compilation-mode=HostCompile_Interpreter allow-custom-values=true" --adjust-block-size-for-scf-tiling --convert-dynamic-to-static-kernels --unroll-scf-loop="auto-unrolling-mode=outer enable-cascaded-unrolling=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// This test documents the pipeline contract: AdjustBlockSizeForScfTiling owns tail backtracking,
// ConvertDynamicToStaticKernels preserves it, and UnrollSCFLoop must keep using the corrected offset.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#TILE_SIZE = affine_map<(d0) -> (-d0 + 33, 5)>

module @StaticTailSafety {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x33x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x33x16xf16, {order = #NHWC}>
  }

  func.func nested @main_func0(%arg0: tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}> {
    return %arg0 : tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x33x16xf16, {order = #NHWC}>) -> tensor<1x16x33x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c33 = arith.constant 33 : index
    %0 = tensor.empty() : tensor<1x16x33x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c33 step %c5 iter_args(%acc = %0) -> (tensor<1x16x33x16xf16, {order = #NHWC}>) {
      %size = affine.min #TILE_SIZE(%i)
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, %size, 16] [1, 1, 1, 1]
          : tensor<1x16x33x16xf16, {order = #NHWC}> to tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%2) : (tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}>
      %4 = tensor.insert_slice %3 into %acc[0, 0, %i, 0] [1, 16, %size, 16] [1, 1, 1, 1]
          : tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 16]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x33x16xf16, {order = #NHWC}>
      scf.yield %4 : tensor<1x16x33x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x33x16xf16, {order = #NHWC}>
  }

  // CHECK-DAG: [[$MAIN_MAP:#.+]] = affine_map<(d0) -> (10, d0)>
  // CHECK-DAG: [[$TAIL_MAP:#.+]] = affine_map<(d0) -> (28, d0)>
  // CHECK-LABEL: module @StaticTailSafety
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C5:%.+]] = arith.constant 5 : index
  // CHECK-DAG: [[C10:%.+]] = arith.constant 10 : index
  // CHECK-DAG: [[C20:%.+]] = arith.constant 20 : index
  // CHECK-DAG: [[C33:%.+]] = arith.constant 33 : index
  // 7 static trips with 6 full tiles: factor 6 / 3 = 2. Splitting after all 6 full tiles would leave the
  // partial tail as a single-trip loop, so one main-loop iteration is moved into the residual loop.
  // The unrolled main loop processes two size-5 tiles at once over [0, 20).
  // CHECK: [[MAIN:%.+]] = scf.for [[MAIN_IV:%.+]] = [[C0]] to [[C20]] step [[C10]]
  // CHECK: [[MAIN_OFFSET:%.+]] = affine.min [[$MAIN_MAP]]([[MAIN_IV]])
  // CHECK: tensor.extract_slice {{%.+}}[0, 0, [[MAIN_OFFSET]], 0] [1, 16, 10, 16]
  // CHECK: func.call @merged_vpu_func_0
  // CHECK: tensor.insert_slice {{%.+}} into {{%.+}}[0, 0, [[MAIN_OFFSET]], 0] [1, 16, 10, 16]
  // CHECK: } {no_await_all = true}
  // The residual loop covers the full tiles at 20 and 25 and the partial tile at 30, which is backtracked
  // to 28. The resulting [28, 33) access is in bounds.
  // CHECK: [[TAIL:%.+]] = scf.for [[TAIL_IV:%.+]] = [[C20]] to [[C33]] step [[C5]] iter_args({{%.+}} = [[MAIN]])
  // CHECK: [[TAIL_OFFSET:%.+]] = affine.min [[$TAIL_MAP]]([[TAIL_IV]])
  // CHECK: tensor.extract_slice {{%.+}}[0, 0, [[TAIL_OFFSET]], 0] [1, 16, 5, 16]
  // CHECK: func.call @main_func0_static
  // CHECK: tensor.insert_slice {{%.+}} into {{%.+}}[0, 0, [[TAIL_OFFSET]], 0] [1, 16, 5, 16]
  // CHECK: } {no_reset_cmdlist = true}
  // CHECK: return [[TAIL]]
}
