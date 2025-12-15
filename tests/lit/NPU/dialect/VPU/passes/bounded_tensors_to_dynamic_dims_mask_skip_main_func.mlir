//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile" --bounded-tensors-to-dynamic-dims-mask %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @EmptyFunction
module @EmptyFunction{
  net.NetworkInfo entryPoint : @EmptyFunction
  inputsInfo : {
    DataInfo "input" : tensor<?x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<?x?x64xf16>
  }

  func.func @EmptyFunction(%arg0: tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
      return %arg0 : tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
  }

  // CHECK: func.func [[EMPTY_FUNC:@.+]]([[_:%.+]]: tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = #CHW}>) -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = #CHW}>
}

// -----

// CHECK-LABEL: @ScheduleEltwiseNHWC
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 90)>
module @ScheduleEltwiseNHWC {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x?x1000xf16>
    DataInfo "input2" : tensor<1x16x?x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x?x1000xf16>
  }

  func.func @main_func0_static(%arg0: tensor<1x16x90x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x90x1000xf16, {order = #NHWC}>
    return %0 : tensor<1x16x90x1000xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
    %c90 = arith.constant 90 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c2 : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %dim_0 step %c90 iter_args(%arg3 = %0) -> (tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim_0]
      %3 = arith.cmpi ne, %2, %c90 : index
      %4 = scf.if %3 -> (index) {
        %6 = arith.subi %c90, %2 : index                                                                                                                                                                                                                                                                                            %7 = arith.cmpi slt, %arg2, %6 : index
        cf.assert %7, "Not enough elements to backtrack in scf.for loop"                                                                                                                                                                                                                                                            %8 = arith.subi %arg2, %6 : index
        scf.yield %8 : index
      } else {
        scf.yield %arg2 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %4, 0] [1, 16, %c90, 1000] [1, 1, 1, 1] : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 1000]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %extracted_slice : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x90x1000xf16, {order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, %4, 0] [1, 16, %c90, 1000] [1, 1, 1, 1] : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 1000]> : tensor<4xsi64>, order = #NHWC}>
      %cast_2 = tensor.cast %extracted_slice_1 : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x90x1000xf16, {order = #NHWC}>
      %5 = func.call @main_func0_static(%cast, %cast_2) : (tensor<1x16x90x1000xf16, {order = #NHWC}>, tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}>
      %cast_3 = tensor.cast %5 : tensor<1x16x90x1000xf16, {order = #NHWC}> to tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 1000]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %cast_3 into %arg3[0, 0, %4, 0] [1, 16, %c90, 1000] [1, 1, 1, 1] : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 1000]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: func.func [[STATIC_FUNC:@.+]]([[_:%.+]]: tensor<1x16x90x1000xf16, {order = #NHWC}>, [[_:%.+]]: tensor<1x16x90x1000xf16, {order = #NHWC}>) -> tensor<1x16x90x1000xf16, {order = #NHWC}>

  // CHECK: func.func [[MAIN:@.+]]([[_:%.+]]: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, [[_:%.+]]: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
}
