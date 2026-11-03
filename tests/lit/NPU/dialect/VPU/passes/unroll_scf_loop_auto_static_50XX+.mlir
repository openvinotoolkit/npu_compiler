//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=NPU5010 compilation-mode=HostCompile_Interpreter allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=outer enable-cascaded-unrolling=true" --canonicalize %s | FileCheck %s --check-prefix=AUTO
// RUN: vpux-opt --split-input-file --init-compiler="platform=NPU5010 compilation-mode=HostCompile_Interpreter allow-custom-values=true" --symbol-dce --pack-nested-modules %s | FileCheck %s --check-prefix=PACK
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// The non-divisible upper bound produces 8 static trips, but only 7 full source tiles. On NPU5010
// (3 DPU tiles), auto-unrolling uses factor 7 / 3 = 2. Since 8 is divisible by 2, the pass must
// explicitly split after 6 full-tile trips; otherwise loopUnrollByFactor would merge the tail trip
// into the main loop. The backing tensor has 16 elements so the final source slice remains valid.
module @StaticAutoUnroll {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c15 = arith.constant 15 : index
    %0 = tensor.empty() : tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c15 step %c2 iter_args(%acc = %0) -> (tensor<1x16x16x16xf16, {order = #NHWC}>) {
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x16x16xf16, {order = #NHWC}> to tensor<1x16x2x16xf16, {order = #NHWC}>
      %3 = tensor.insert_slice %2 into %acc[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x2x16xf16, {order = #NHWC}> into tensor<1x16x16x16xf16, {order = #NHWC}>
      scf.yield %3 : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
  }

  // AUTO-DAG: [[C0:%.+]] = arith.constant 0 : index
  // AUTO-DAG: [[C2:%.+]] = arith.constant 2 : index
  // AUTO-DAG: [[C4:%.+]] = arith.constant 4 : index
  // AUTO-DAG: [[C12:%.+]] = arith.constant 12 : index
  // AUTO-DAG: [[C15:%.+]] = arith.constant 15 : index
  // AUTO: [[MAIN:%.+]] = scf.for {{%.+}} = [[C0]] to [[C12]] step [[C4]] iter_args({{%.+}} = {{%.+}}) -> (tensor<1x16x16x16xf16, {order = #NHWC}>) {
  // AUTO: scf.yield
  // AUTO: [[RESIDUAL:%.+]] = scf.for {{%.+}} = [[C12]] to [[C15]] step [[C2]] iter_args({{%.+}} = [[MAIN]]) -> (tensor<1x16x16x16xf16, {order = #NHWC}>) {
  // AUTO: scf.yield
  // AUTO: return [[RESIDUAL]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// The divisible upper bound produces 8 complete trips. Auto-unrolling uses factor 8 / 3 = 2,
// and loopUnrollByFactor can cover all trips in the main loop without creating a residual loop.
module @StaticAutoUnrollWithoutResidual {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %0 = tensor.empty() : tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c16 step %c2 iter_args(%acc = %0) -> (tensor<1x16x16x16xf16, {order = #NHWC}>) {
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x16x16xf16, {order = #NHWC}> to tensor<1x16x2x16xf16, {order = #NHWC}>
      %3 = tensor.insert_slice %2 into %acc[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x2x16xf16, {order = #NHWC}> into tensor<1x16x16x16xf16, {order = #NHWC}>
      scf.yield %3 : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
  }

  // AUTO-LABEL: module @StaticAutoUnrollWithoutResidual
  // AUTO-DAG: [[C0_NO_TAIL:%.+]] = arith.constant 0 : index
  // AUTO-DAG: [[C4_NO_TAIL:%.+]] = arith.constant 4 : index
  // AUTO-DAG: [[C16_NO_TAIL:%.+]] = arith.constant 16 : index
  // AUTO: [[MAIN_NO_TAIL:%.+]] = scf.for {{%.+}} = [[C0_NO_TAIL]] to [[C16_NO_TAIL]] step [[C4_NO_TAIL]] iter_args({{%.+}} = {{%.+}}) -> (tensor<1x16x16x16xf16, {order = #NHWC}>) {
  // AUTO: scf.yield
  // AUTO-NOT: scf.for
  // AUTO: return [[MAIN_NO_TAIL]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DropUnreferencedComputeFunction {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x2x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x2x16xf16, {order = #NHWC}>
  }

  func.func private @unused_compute(%arg0: tensor<1x16x2x16xf16, {order = #NCHW}>) -> tensor<1x16x2x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        -> tensor<1x16x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x2x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x2x16xf16, {order = #NHWC}>) -> tensor<1x16x2x16xf16, {order = #NHWC}> {
    return %arg0 : tensor<1x16x2x16xf16, {order = #NHWC}>
  }

  // PACK-NOT: @unused_compute
  // PACK: func.func @main
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// 7 static trips with 6 full source tiles. Auto-unrolling uses factor 6 / 3 = 2. Splitting after all 6 full
// tiles would leave the partial tail as a single-trip residual loop, which canonicalization promotes to
// straight-line code, dropping the attributes that submit the main loop's command list. One more main-loop
// iteration is moved into the residual loop instead.
module @StaticSingleTripTail {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x14x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x14x16xf16, {order = #NHWC}>) -> tensor<1x16x14x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c13 = arith.constant 13 : index
    %0 = tensor.empty() : tensor<1x16x14x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c13 step %c2 iter_args(%acc = %0) -> (tensor<1x16x14x16xf16, {order = #NHWC}>) {
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x14x16xf16, {order = #NHWC}> to tensor<1x16x2x16xf16, {order = #NHWC}>
      %3 = tensor.insert_slice %2 into %acc[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x2x16xf16, {order = #NHWC}> into tensor<1x16x14x16xf16, {order = #NHWC}>
      scf.yield %3 : tensor<1x16x14x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x14x16xf16, {order = #NHWC}>
  }

  // AUTO-LABEL: module @StaticSingleTripTail
  // AUTO-DAG: [[C0_TAIL:%.+]] = arith.constant 0 : index
  // AUTO-DAG: [[C2_TAIL:%.+]] = arith.constant 2 : index
  // AUTO-DAG: [[C4_TAIL:%.+]] = arith.constant 4 : index
  // AUTO-DAG: [[C8_TAIL:%.+]] = arith.constant 8 : index
  // AUTO-DAG: [[C13_TAIL:%.+]] = arith.constant 13 : index
  // AUTO: [[MAIN_TAIL:%.+]] = scf.for {{%.+}} = [[C0_TAIL]] to [[C8_TAIL]] step [[C4_TAIL]]
  // AUTO: } {no_await_all = true}
  // AUTO: [[RESIDUAL_TAIL:%.+]] = scf.for {{%.+}} = [[C8_TAIL]] to [[C13_TAIL]] step [[C2_TAIL]] iter_args({{%.+}} = [[MAIN_TAIL]])
  // AUTO: } {no_reset_cmdlist = true}
  // AUTO: return [[RESIDUAL_TAIL]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// 7 static trips without a partial tail. Auto-unrolling uses factor 7 / 3 = 2, so loopUnrollByFactor alone
// would leave a single-trip epilogue. The loop is split explicitly so that the residual loop runs 3 trips.
module @StaticSingleTripEpilogue {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x14x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x16xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x14x16xf16, {order = #NHWC}>) -> tensor<1x16x14x16xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c14 = arith.constant 14 : index
    %0 = tensor.empty() : tensor<1x16x14x16xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c14 step %c2 iter_args(%acc = %0) -> (tensor<1x16x14x16xf16, {order = #NHWC}>) {
      %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x14x16xf16, {order = #NHWC}> to tensor<1x16x2x16xf16, {order = #NHWC}>
      %3 = tensor.insert_slice %2 into %acc[0, 0, %i, 0] [1, 16, 2, 16] [1, 1, 1, 1]
          : tensor<1x16x2x16xf16, {order = #NHWC}> into tensor<1x16x14x16xf16, {order = #NHWC}>
      scf.yield %3 : tensor<1x16x14x16xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x14x16xf16, {order = #NHWC}>
  }

  // AUTO-LABEL: module @StaticSingleTripEpilogue
  // AUTO-DAG: [[C0_EPI:%.+]] = arith.constant 0 : index
  // AUTO-DAG: [[C2_EPI:%.+]] = arith.constant 2 : index
  // AUTO-DAG: [[C4_EPI:%.+]] = arith.constant 4 : index
  // AUTO-DAG: [[C8_EPI:%.+]] = arith.constant 8 : index
  // AUTO-DAG: [[C14_EPI:%.+]] = arith.constant 14 : index
  // AUTO: [[MAIN_EPI:%.+]] = scf.for {{%.+}} = [[C0_EPI]] to [[C8_EPI]] step [[C4_EPI]]
  // AUTO: } {no_await_all = true}
  // AUTO: [[RESIDUAL_EPI:%.+]] = scf.for {{%.+}} = [[C8_EPI]] to [[C14_EPI]] step [[C2_EPI]] iter_args({{%.+}} = [[MAIN_EPI]])
  // AUTO: } {no_reset_cmdlist = true}
  // AUTO: return [[RESIDUAL_EPI]]
}
