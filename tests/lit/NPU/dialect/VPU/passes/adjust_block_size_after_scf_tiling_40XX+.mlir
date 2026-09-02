//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --adjust-block-size-for-scf-tiling  --cse --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 100, d0)>

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %dim step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:    func.func nested @main_func0([[ARG0:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[ELTWISE]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @main([[ARG0:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[C100:%.+]] = arith.constant 100 : index
  // CHECK:      [[C0:%.+]] = arith.constant 0 : index
  // CHECK:      [[C3:%.+]] = arith.constant 3 : index
  // CHECK:      [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH:%.+]] = arith.cmpi sge, [[DIM]], [[C100]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[FOR_RESULT:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[DIM]] step [[C100]] iter_args([[ARG3:%.+]] = [[EMPTY]]) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[OFFSET:%.+]] = affine.min #[[$MAP]]([[ARG2]])[[[DIM]]]
  // CHECK:        [[SLICE0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:        [[CAST0:%.+]] = tensor.cast [[SLICE0]] : tensor<1x16x720x100xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[SLICE1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:        [[CAST1:%.+]] = tensor.cast [[SLICE1]] : tensor<1x16x720x100xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[CALL:%.+]] = func.call @main_func0([[CAST0]], [[CAST1]]) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[CAST2:%.+]] = tensor.cast [[CALL]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED:%.+]] = tensor.insert_slice [[CAST2]] into [[ARG3]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }
  // CHECK:      return [[FOR_RESULT]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 720, 44)>

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (676, d0)>

// CHECK-LABEL: @Add
module @Add {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "Add_3" friendlyName = "output" : tensor<1x16x720x1000xf16>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NCHW} : tensor<1x16x720x1000xf16, {order = #NHWC}> -> tensor<1x16x720x1000xf16>
    return %0 : tensor<1x16x720x1000xf16>
  }
  func.func nested @main_func1(%arg0: tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
    return %0 : tensor<1x16x?x1000xf16, {order = #NHWC}>
  }
  func.func nested @main_func2(%arg0: tensor<1x16x?x1000xf16>, %arg1: tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg1, %0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, tilingStrategy = [1, 1, 4, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x?x1000xf16, {order = #NHWC}>
    return %1 : tensor<1x16x?x1000xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x16x720x1000xf16>, %arg1: tensor<1x16x720x1000xf16>) -> tensor<1x16x720x1000xf16> {
    %c44 = arith.constant 44 : index
    %c720 = arith.constant 720 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c720 step %c44 iter_args(%arg3 = %0) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
      %3 = affine.min #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, 0] [1, 16, %3, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16> to tensor<1x16x?x1000xf16>
      %4 = func.call @main_func1(%extracted_slice) : (tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %arg2, 0] [1, 16, %3, 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16> to tensor<1x16x?x1000xf16>
      %5 = func.call @main_func2(%extracted_slice_0, %4) : (tensor<1x16x?x1000xf16>, tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %5 into %arg3[0, 0, %arg2, 0] [1, 16, %3, 1000] [1, 1, 1, 1] : tensor<1x16x?x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x1000xf16, {order = #NHWC}>
    }
    %2 = call @main_func0(%1) : (tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16>
    return %2 : tensor<1x16x720x1000xf16>
  }

  // CHECK:    func.func nested @main_func0([[ARG0:%.+]]: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16> {
  // CHECK:      [[LAYOUTCAST:%.+]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NCHW} : tensor<1x16x720x1000xf16, {order = #NHWC}> -> tensor<1x16x720x1000xf16>
  // CHECK:      return [[LAYOUTCAST]] : tensor<1x16x720x1000xf16>
  // CHECK:    }

  // CHECK:    func.func nested @main_func1([[ARG0:%.+]]: tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
  // CHECK:      [[LAYOUTCAST:%.+]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:      return [[LAYOUTCAST]] : tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @main_func2([[ARG0:%.+]]: tensor<1x16x?x1000xf16>, [[ARG1:%.+]]: tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
  // CHECK:      [[LAYOUTCAST:%.+]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG1]], [[LAYOUTCAST]]) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:      return [[ELTWISE]] : tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @main([[ARG0:%.+]]: tensor<1x16x720x1000xf16>, [[ARG1:%.+]]: tensor<1x16x720x1000xf16>) -> tensor<1x16x720x1000xf16> {
  // CHECK:      [[C44:%.+]] = arith.constant 44 : index
  // CHECK:      [[C720:%.+]] = arith.constant 720 : index
  // CHECK:      [[C0:%.+]] = arith.constant 0 : index
  // CHECK:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:      [[FOR_RESULT:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C720]] step [[C44]] iter_args([[ARG3:%.+]] = [[EMPTY]]) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
  // CHECK:        [[OFFSET:%.+]] = affine.min #[[$MAP]]([[ARG2]])
  // CHECK:        [[SLICE0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[OFFSET]], 0] [1, 16, 44, 1000] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x1000xf16> to tensor<1x16x44x1000xf16>
  // CHECK:        [[CAST0:%.+]] = tensor.cast [[SLICE0]] : tensor<1x16x44x1000xf16> to tensor<1x16x?x1000xf16>
  // CHECK:        [[CALL1:%.+]] = func.call @main_func1([[CAST0]]) : (tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:        [[SLICE1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, [[OFFSET]], 0] [1, 16, 44, 1000] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x1000xf16> to tensor<1x16x44x1000xf16>
  // CHECK:        [[CAST1:%.+]] = tensor.cast [[SLICE1]] : tensor<1x16x44x1000xf16> to tensor<1x16x?x1000xf16>
  // CHECK:        [[CALL2:%.+]] = func.call @main_func2([[CAST1]], [[CALL1]]) : (tensor<1x16x?x1000xf16>, tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:        [[CAST2:%.+]] = tensor.cast [[CALL2]] : tensor<1x16x?x1000xf16, {order = #NHWC}> to tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:        [[INSERTED:%.+]] = tensor.insert_slice [[CAST2]] into [[ARG3]][0, 0, [[OFFSET]], 0] [1, 16, 44, 1000] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x44x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:      }
  // CHECK:      [[FINAL_CALL:%.+]] = call @main_func0([[FOR_RESULT]]) : (tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16>
  // CHECK:      return [[FINAL_CALL]] : tensor<1x16x720x1000xf16>
  // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map5 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 256, d0)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
//CHECK: #[[$MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @ApplyTilingNCEConvDyn
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn inputsInfo : {
    DataInfo "input" : tensor<1x32x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x64xf16>
  }
  func.func nested @ApplyTilingNCEConvDyn_func0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @ApplyTilingNCEConvDyn(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg1)[%dim]
      %3 = affine.max #map1(%arg1)
      %4 = affine.max #map2(%arg1)
      %5 = affine.min #map3()[%4]
      %6 = affine.max #map4(%2, %3)
      %7 = affine.min #map3()[%6]
      %8 = affine.apply #map5(%2, %5, %7)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, 0] [1, 32, %8, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %extracted_slice : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      %9 = func.call @ApplyTilingNCEConvDyn_func0(%cast) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %9 into %arg2[0, 0, %arg1, 0] [1, 256, %2, 64] [1, 1, 1, 1] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn_func0_dims_H_cases_2([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn_func0_dims_H_cases_1([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn_func0_dims_H_cases_0([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @ApplyTilingNCEConvDyn([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[FALSE:%.+]] = arith.constant false
  // CHECK:      [[C3:%.+]] = arith.constant 3 : index
  // CHECK:      [[C1:%.+]] = arith.constant 1 : index
  // CHECK:      [[C256:%.+]] = arith.constant 256 : index
  // CHECK:      [[C0:%.+]] = arith.constant 0 : index
  // CHECK:      [[C2:%.+]] = arith.constant 2 : index
  // CHECK:      [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH:%.+]] = arith.cmpi sge, [[DIM]], [[C256]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[FOR_RESULT:%.+]] = scf.for [[ARG1:%.+]] = [[C0]] to [[DIM]] step [[C256]] iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[OFFSET:%.+]] = affine.min #[[$MAP]]([[ARG1]])[[[DIM]]]
  // CHECK:        [[SLICE_SIZE:%.+]] = affine.min #[[$MAP1]]([[OFFSET]])[[[DIM]]]
  // CHECK:        [[MAX_POS:%.+]] = affine.max #[[$MAP2]]([[OFFSET]])
  // CHECK:        [[MAX_POS_1:%.+]] = affine.max #[[$MAP3]]([[OFFSET]])
  // CHECK:        [[MIN_SIZE_1:%.+]] = affine.min #[[$MAP4]]()[[[MAX_POS_1]]]
  // CHECK:        [[MAX_SIZE:%.+]] = affine.max #[[$MAP5]]([[SLICE_SIZE]], [[MAX_POS]])
  // CHECK:        [[MIN_SIZE_2:%.+]] = affine.min #[[$MAP4]]()[[[MAX_SIZE]]]
  // CHECK:        [[APPLY_SIZE:%.+]] = affine.apply #[[$MAP6]]([[SLICE_SIZE]], [[MIN_SIZE_1]], [[MIN_SIZE_2]])
  // CHECK:        [[IS_ZERO_POS:%.+]] = arith.cmpi eq, [[MAX_POS]], [[C0]] : index
  // CHECK:        [[CASE_OFFSET:%.+]] = scf.if [[IS_ZERO_POS]] -> (index) {
  // CHECK:          [[IS_EXACT:%.+]] = arith.cmpi eq, [[APPLY_SIZE]], [[DIM]] : index
  // CHECK:          [[CASE:%.+]] = arith.select [[IS_EXACT]], [[C3]], [[C2]] : index
  // CHECK:          scf.yield [[CASE]] : index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS:%.+]] = arith.addi [[MAX_POS]], [[APPLY_SIZE]] : index
  // CHECK:          [[IN_BOUNDS:%.+]] = arith.cmpi slt, [[NEXT_POS]], [[DIM]] : index
  // CHECK:          [[CASE_ALT:%.+]] = arith.select [[IN_BOUNDS]], [[C0]], [[C1]] : index
  // CHECK:          scf.yield [[CASE_ALT]] : index
  // CHECK:        }
  // CHECK:        [[H_SHIFTED:%.+]] = arith.shli [[CASE_OFFSET]], [[C2]] : index
  // CHECK:        [[SWITCH_RESULT:%.+]] = scf.index_switch [[H_SHIFTED]] -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          case 0 {
  // CHECK:            [[SLICE0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[MAX_POS]], 0] [1, 32, 258, 64] [1, 1, 1, 1] :
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST0:%.+]] = tensor.cast [[SLICE0]] : tensor<1x32x258x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL0:%.+]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0([[CAST0]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL0]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 4 {
  // CHECK:            [[SLICE1:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[MAX_POS]], 0] [1, 32, 257, 64] [1, 1, 1, 1] :
  // CHECK:            [[CAST1:%.+]] = tensor.cast [[SLICE1]] : tensor<1x32x257x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL1:%.+]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_1([[CAST1]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL1]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 8 {
  // CHECK:            [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[MAX_POS]], 0] [1, 32, 257, 64] [1, 1, 1, 1] :
  // CHECK:            [[CAST2:%.+]] = tensor.cast [[SLICE2]] : tensor<1x32x257x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_2([[CAST2]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL2]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          default {
  // CHECK:            cf.assert [[FALSE]], "Unsupported case"
  // CHECK:            [[SLICE_DEF:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[MAX_POS]], 0] [1, 32, 258, 64] [1, 1, 1, 1] :
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST_DEF:%.+]] = tensor.cast [[SLICE_DEF]] : tensor<1x32x258x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_DEF:%.+]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0([[CAST_DEF]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_DEF]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:        [[CAST_2:%.+]] = tensor.cast [[SWITCH_RESULT]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED:%.+]] = tensor.insert_slice [[CAST_2]] into [[ARG2:%.+]][0, 0, [[OFF:%.+]], 0] [1, 256, 256, 64] [1, 1, 1, 1] : tensor<1x256x256x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        }
  // CHECK:      return [[FOR_RESULT]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
#map2 = affine_map<(d0) -> (0, d0 - 1)>
#map3 = affine_map<(d0) -> (-d0 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map6 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
#map7 = affine_map<(d0, d1) -> (0, d0 + d1 - 638)>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 256, d0)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (s0 - 160, d0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
// CHECK-DAG: #[[$MAP8:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
// CHECK-DAG: #[[$MAP9:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 638)>

// CHECK-LABEL: @ApplyTilingNCEConvDyn2D
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn2D inputsInfo : {
    DataInfo "input" : tensor<1x32x?x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x?xf16>
  }
  func.func nested @ApplyTilingNCEConvDyn2D_func0(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @ApplyTilingNCEConvDyn2D(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> {
    %c160 = arith.constant 160 : index
    %c256 = arith.constant 256 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = scf.for %arg3 = %c0 to %dim_0 step %c160 iter_args(%arg4 = %arg2) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
        %3 = affine.min #map(%arg1)[%dim]
        %4 = affine.min #map1(%arg3)[%dim_0]
        %5 = affine.max #map2(%arg1)
        %6 = affine.max #map3(%arg1)
        %7 = affine.min #map4()[%6]
        %8 = affine.max #map5(%3, %5)
        %9 = affine.min #map4()[%8]
        %10 = affine.apply #map6(%3, %7, %9)
        %11 = affine.max #map2(%arg3)
        %12 = affine.max #map3(%arg3)
        %13 = affine.min #map4()[%12]
        %14 = affine.max #map7(%4, %11)
        %15 = affine.min #map4()[%14]
        %16 = affine.apply #map6(%4, %13, %15)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %5, %11] [1, 32, %10, %16] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        %17 = func.call @ApplyTilingNCEConvDyn2D_func0(%extracted_slice) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %17 into %arg4[0, 0, %arg1, %arg3] [1, 256, %3, %4] [1, 1, 1, 1] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %2 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @ApplyTilingNCEConvDyn2D([[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK-DAG:      [[FALSE:%.+]] = arith.constant false
  // CHECK-DAG:      [[C1:%.+]] = arith.constant 1 : index
  // CHECK-DAG:      [[C160:%.+]] = arith.constant 160 : index
  // CHECK-DAG:      [[C256:%.+]] = arith.constant 256 : index
  // CHECK-DAG:      [[C3:%.+]] = arith.constant 3 : index
  // CHECK-DAG:      [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG:      [[C2:%.+]] = arith.constant 2 : index
  // CHECK:      [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[DIM_0:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[EMPTY:%.+]] = tensor.empty([[DIM]], [[DIM_0]]) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH_0:%.+]] = arith.cmpi sge, [[DIM_0]], [[C160]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_0]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[HAS_ENOUGH_1:%.+]] = arith.cmpi sge, [[DIM]], [[C256]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_1]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[FOR_RESULT:%.+]] = scf.for [[ARG1:%.+]] = [[C0]] to [[DIM]] step [[C256]] iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[FOR_RESULT_1:%.+]] = scf.for [[ARG3:%.+]] = [[C0]] to [[DIM_0]] step [[C160]] iter_args([[ARG4:%.+]] = [[ARG2]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:          [[H_OFFSET:%.+]] = affine.min #[[$MAP]]([[ARG1]])[[[DIM]]]
  // CHECK:          [[W_OFFSET:%.+]] = affine.min #[[$MAP1]]([[ARG3]])[[[DIM_0]]]
  // CHECK:          [[H_TILE_SIZE:%.+]] = affine.min #[[$MAP2]]([[H_OFFSET]])[[[DIM]]]
  // CHECK:          [[W_TILE_SIZE:%.+]] = affine.min #[[$MAP3]]([[W_OFFSET]])[[[DIM_0]]]
  // CHECK:          [[H_INPUT_START:%.+]] = affine.max #[[$MAP4]]([[H_OFFSET]])
  // CHECK:          [[H_INPUT_TOP_PAD:%.+]] = affine.max #[[$MAP5]]([[H_OFFSET]])
  // CHECK:          [[H_TOP_PAD_SIZE:%.+]] = affine.min #[[$MAP6]]()[[[H_INPUT_TOP_PAD]]]
  // CHECK:          [[H_INPUT_BOTTOM_PAD:%.+]] = affine.max #[[$MAP7]]([[H_TILE_SIZE]], [[H_INPUT_START]])
  // CHECK:          [[H_BOTTOM_PAD_SIZE:%.+]] = affine.min #[[$MAP6]]()[[[H_INPUT_BOTTOM_PAD]]]
  // CHECK:          [[H_INPUT_SIZE:%.+]] = affine.apply #[[$MAP8]]([[H_TILE_SIZE]], [[H_TOP_PAD_SIZE]], [[H_BOTTOM_PAD_SIZE]])
  // CHECK:          [[W_INPUT_START:%.+]] = affine.max #[[$MAP4]]([[W_OFFSET]])
  // CHECK:          [[W_INPUT_LEFT_PAD:%.+]] = affine.max #[[$MAP5]]([[W_OFFSET]])
  // CHECK:          [[W_LEFT_PAD_SIZE:%.+]] = affine.min #[[$MAP6]]()[[[W_INPUT_LEFT_PAD]]]
  // CHECK:          [[W_INPUT_RIGHT_PAD:%.+]] = affine.max #[[$MAP9]]([[W_TILE_SIZE]], [[W_INPUT_START]])
  // CHECK:          [[W_RIGHT_PAD_SIZE:%.+]] = affine.min #[[$MAP6]]()[[[W_INPUT_RIGHT_PAD]]]
  // CHECK:          [[W_INPUT_SIZE:%.+]] = affine.apply #[[$MAP8]]([[W_TILE_SIZE]], [[W_LEFT_PAD_SIZE]], [[W_RIGHT_PAD_SIZE]])
  // CHECK:          [[H_IS_AT_TOP:%.+]] = arith.cmpi eq, [[H_INPUT_START]], [[C0]] : index
  // CHECK:          [[W_IS_AT_LEFT:%.+]] = arith.cmpi eq, [[W_INPUT_START]], [[C0]] : index
  // CHECK:          [[W_CASE_IDX:%.+]] = scf.if [[W_IS_AT_LEFT]] -> (index) {
  // CHECK:            [[W_IS_EXACT_FIT:%.+]] = arith.cmpi eq, [[W_INPUT_SIZE]], [[DIM_0]] : index
  // CHECK:            [[W_CASE:%.+]] = arith.select [[W_IS_EXACT_FIT]], [[C3]], [[C2]] : index
  // CHECK:            scf.yield [[W_CASE]] : index
  // CHECK:          } else {
  // CHECK:            [[W_INPUT_END:%.+]] = arith.addi [[W_INPUT_START]], [[W_INPUT_SIZE]] : index
  // CHECK:            [[W_WITHIN_BOUNDS:%.+]] = arith.cmpi slt, [[W_INPUT_END]], [[DIM_0]] : index
  // CHECK:            [[W_CASE_ALT:%.+]] = arith.select [[W_WITHIN_BOUNDS]], [[C0]], [[C1]] : index
  // CHECK:            scf.yield [[W_CASE_ALT]] : index
  // CHECK:          }
  // CHECK:          [[H_CASE_IDX:%.+]] = scf.if [[H_IS_AT_TOP]] -> (index) {
  // CHECK:            [[H_IS_EXACT_FIT:%.+]] = arith.cmpi eq, [[H_INPUT_SIZE]], [[DIM]] : index
  // CHECK:            [[H_CASE:%.+]] = arith.select [[H_IS_EXACT_FIT]], [[C3]], [[C2]] : index
  // CHECK:            scf.yield [[H_CASE]] : index
  // CHECK:          } else {
  // CHECK:            [[H_INPUT_END:%.+]] = arith.addi [[H_INPUT_START]], [[H_INPUT_SIZE]] : index
  // CHECK:            [[H_WITHIN_BOUNDS:%.+]] = arith.cmpi slt, [[H_INPUT_END]], [[DIM]] : index
  // CHECK:            [[H_CASE_ALT:%.+]] = arith.select [[H_WITHIN_BOUNDS]], [[C0]], [[C1]] : index
  // CHECK:            scf.yield [[H_CASE_ALT]] : index
  // CHECK:          }
  // CHECK:          [[H_SHIFT:%.+]] = arith.shli [[H_CASE_IDX]], [[C2]] : index
  // CHECK:          [[COMBINED_CASE_IDX:%.+]] = arith.ori [[H_SHIFT]], [[W_CASE_IDX]] : index
  // CHECK:          [[SWITCH_RESULT_1:%.+]] = scf.index_switch [[COMBINED_CASE_IDX]] -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          case 0 {
  // CHECK:            [[SLICE_00:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 258, 162] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:            [[CAST_00:%.+]] = tensor.cast [[SLICE_00]] : tensor<1x32x258x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_00:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00([[CAST_00]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_00]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 1 {
  // CHECK:            [[SLICE_01:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 258, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_01:%.+]] = tensor.cast [[SLICE_01]] : tensor<1x32x258x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_01:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01([[CAST_01]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_01]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 2 {
  // CHECK:            [[SLICE_02:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 258, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_02:%.+]] = tensor.cast [[SLICE_02]] : tensor<1x32x258x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_02:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02([[CAST_02]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_02]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 4 {
  // CHECK:            [[SLICE_10:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 257, 162] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:            [[CAST_10:%.+]] = tensor.cast [[SLICE_10]] : tensor<1x32x257x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_10:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10([[CAST_10]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_10]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 5 {
  // CHECK:            [[SLICE_10:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK:            [[CAST_11:%.+]] = tensor.cast [[SLICE_10]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_11:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11([[CAST_11]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_11]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 6 {
  // CHECK:            [[SLICE_10:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK:            [[CAST_12:%.+]] = tensor.cast [[SLICE_10]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_12:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12([[CAST_12]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_12]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 8 {
  // CHECK:            [[SLICE_20:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 257, 162] [1, 1, 1, 1]
  // CHECK:            [[CAST_20:%.+]] = tensor.cast [[SLICE_20]] : tensor<1x32x257x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_20:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20([[CAST_20]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_20]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 9 {
  // CHECK:            [[SLICE_20:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK:            [[CAST_21:%.+]] = tensor.cast [[SLICE_20]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_21:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21([[CAST_21]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_21]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 10 {
  // CHECK:            [[SLICE_22:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[H_INPUT_START]], [[W_INPUT_START]]] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK:            [[CAST_22:%.+]] = tensor.cast [[SLICE_22]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_22:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22([[CAST_22]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_22]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          default {
  // CHECK:            cf.assert [[FALSE]], "Unsupported case"
  // CHECK:          }
  // CHECK:        [[CAST_3:%.+]] = tensor.cast [[SWITCH_RESULT_1]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x160xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED_1:%.+]] = tensor.insert_slice [[CAST_3]] into [[ARG4]][0, 0, [[H_OFFSET]], [[W_OFFSET]]] [1, 256, 256, 160] [1, 1, 1, 1] : tensor<1x256x256x160xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED_1]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }
  // CHECK:      scf.yield [[FOR_RESULT_1]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }
  // CHECK:    return [[FOR_RESULT]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map2 = affine_map<(d0) -> (d0 * 2)>

//CHECK: #[[$MAP_BACKTRACK_OFFSET:.+]] = affine_map<(d0)[s0] -> (s0 - 256, d0)>
//CHECK: #[[$MAP_TILE_SIZE:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP_STRIDE_INPUT_START:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
//CHECK: #[[$MAP_STRIDE_INPUT_SIZE:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: @ConvWithStrides2x2
module {
  net.NetworkInfo entryPoint : @ConvWithStrides2x2 inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "/conv_final/Conv" friendlyName = "EditOpenVinoIRResult_48" tensorNames = ["/conv_final/Conv_output_0"] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func nested @ConvWithStrides2x2_func0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [2, 2]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @ConvWithStrides2x2(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %0 = arith.addi %dim, %c1 : index
    %1 = arith.divsi %0, %c2 : index
    %2 = tensor.empty(%1) : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %3 = arith.addi %dim_0, %c1 : index
    %4 = arith.divsi %3, %c2 : index
    %5 = scf.for %arg1 = %c0 to %4 step %c256 iter_args(%arg2 = %2) -> (tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>) {
      %6 = affine.min #map(%arg1)[%4]
      %7 = affine.max #map1(%arg1)
      %8 = affine.apply #map2(%6)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %7, 0] [1, 32, %8, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
      %9 = func.call @ConvWithStrides2x2_func0(%extracted_slice) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %9 into %arg2[0, 0, %arg1, 0] [1, 256, %6, 32] [1, 1, 1, 1] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %5 : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:    func.func nested @ConvWithStrides2x2_func0_dims_H_cases_2([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [2, 2]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func nested @ConvWithStrides2x2_func0_dims_H_cases_1([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
  // CHECK-SAME: strides = [2, 2]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @ConvWithStrides2x2([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[FALSE:%.+]] = arith.constant false
  // CHECK:      [[C3:%.+]] = arith.constant 3 : index
  // CHECK:      [[C256:%.+]] = arith.constant 256 : index
  // CHECK:      [[C0:%.+]] = arith.constant 0 : index
  // CHECK:      [[C1:%.+]] = arith.constant 1 : index
  // CHECK:      [[C2:%.+]] = arith.constant 2 : index
  // CHECK:      [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[ADDI:%.+]] = arith.addi [[DIM]], [[C1]] : index
  // CHECK:      [[DIVSI:%.+]] = arith.divsi [[ADDI]], [[C2]] : index
  // CHECK:      [[EMPTY:%.+]] = tensor.empty([[DIVSI]]) : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH:%.+]] = arith.cmpi sge, [[DIVSI]], [[C256]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[FOR_RESULT:%.+]] = scf.for [[ARG1:%.+]] = [[C0]] to [[DIVSI]] step [[C256]] iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[OFFSET:%.+]] = affine.min #[[$MAP_BACKTRACK_OFFSET]]([[ARG1]])[[[DIVSI]]]
  // CHECK:        [[TILE_SIZE:%.+]] = affine.min #[[$MAP_TILE_SIZE]]([[OFFSET]])[[[DIVSI]]]
  // CHECK:        [[INPUT_START:%.+]] = affine.max #[[$MAP_STRIDE_INPUT_START]]([[OFFSET]])
  // CHECK:        [[INPUT_SIZE:%.+]] = affine.apply #[[$MAP_STRIDE_INPUT_SIZE]]([[TILE_SIZE]])
  // CHECK:        [[IS_AT_TOP:%.+]] = arith.cmpi eq, [[INPUT_START]], [[C0]] : index
  // CHECK:        [[CASE_OFFSET:%.+]] = scf.if [[IS_AT_TOP]] -> (index) {
  // CHECK:          [[IS_EXACT_FIT:%.+]] = arith.cmpi eq, [[INPUT_SIZE]], [[DIM]] : index
  // CHECK:          [[CASE_IDX:%.+]] = arith.select [[IS_EXACT_FIT]], [[C3]], [[C2]] : index
  // CHECK:          scf.yield [[CASE_IDX]] : index
  // CHECK:        } else {
  // CHECK:          [[INPUT_END:%.+]] = arith.addi [[INPUT_START]], [[INPUT_SIZE]] : index
  // CHECK:          [[WITHIN_BOUNDS:%.+]] = arith.cmpi slt, [[INPUT_END]], [[DIM]] : index
  // CHECK:          [[CASE_IDX_ALT:%.+]] = arith.select [[WITHIN_BOUNDS]], [[C0]], [[C1]] : index
  // CHECK:          scf.yield [[CASE_IDX_ALT]] : index
  // CHECK:        }
  // CHECK:        [[H_SHIFTED:%.+]] = arith.shli [[CASE_OFFSET]], [[C2]] : index
  // CHECK:        [[SWITCH_RESULT:%.+]] = scf.index_switch [[H_SHIFTED]] -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          case 4 {
  // CHECK:            [[SLICE1:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[INPUT_START]], 0] [1, 32, 512, 64] [1, 1, 1, 1]
  // CHECK:            [[CAST1:%.+]] = tensor.cast [[SLICE1]] : tensor<1x32x512x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL1:%.+]] = func.call @ConvWithStrides2x2_func0_dims_H_cases_1([[CAST1]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL1]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 8 {
  // CHECK:            [[SLICE1:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[INPUT_START]], 0] [1, 32, 512, 64] [1, 1, 1, 1]
  // CHECK:            [[CAST2:%.+]] = tensor.cast [[SLICE1]] : tensor<1x32x512x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL2:%.+]] = func.call @ConvWithStrides2x2_func0_dims_H_cases_2([[CAST2]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL2]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          default {
  // CHECK:            cf.assert [[FALSE]], "Unsupported case"
  // CHECK:          }
  // CHECK:        [[CAST_FINAL:%.+]] = tensor.cast [[SWITCH_RESULT]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED:%.+]] = tensor.insert_slice [[CAST_FINAL]] into [[ARG2]][0, 0, [[OFFSET]], 0] [1, 256, 256, 32] [1, 1, 1, 1] : tensor<1x256x256x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }
  // CHECK:      return [[FOR_RESULT]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 48)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 320)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 99)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 640)>
#map4 = affine_map<(d0) -> (0, d0 - 1)>
#map5 = affine_map<(d0) -> (-d0 + 1, 0)>
#map6 = affine_map<()[s0] -> (1, s0)>
#map7 = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
#map8 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @permute_convolution
module {
  net.NetworkInfo entryPoint : @permute_convolution inputsInfo : {
    DataInfo "Parameter_14" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "Convolution_16" friendlyName = "Result_17" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func nested @main_func1(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
    %cst_0 = const.Declare tensor<16x1x1x256xi1> = dense<0.0> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %1 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NCHW}>
    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NCHW}>
  }

  // CHECK-DAG: func.func nested @main_func0([[ARG0:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-DAG: func.func nested @main_func1_dims_HW_cases_{{[0-9]+}}([[ARG0:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 641]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK-COUNT-5: func.func nested @main_func1_dims_HW_cases_{{[0-9]+}}(

  func.func @permute_convolution(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> {
    %c640 = arith.constant 640 : index
    %c99 = arith.constant 99 : index
    %c320 = arith.constant 320 : index
    %c48 = arith.constant 48 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %dim_1 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %0 = tensor.empty(%dim, %dim_1) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim step %c48 iter_args(%arg2 = %0) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
      %4 = scf.for %arg3 = %c0 to %dim_1 step %c320 iter_args(%arg4 = %arg2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
        %5 = affine.min #map(%arg1)[%dim]
        %6 = affine.min #map1(%arg3)[%dim_1]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, %arg3] [1, 16, %5, %6] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NCHW}>
        %7 = func.call @main_func0(%extracted_slice) : (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %7 into %arg4[0, 0, %arg1, %arg3] [1, 16, %5, %6] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %4 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    }
    %dim_2 = tensor.dim %1, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %dim_3 = tensor.dim %1, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = tensor.empty(%dim_2, %dim_3) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %3 = scf.for %arg1 = %c0 to %dim_2 step %c99 iter_args(%arg2 = %2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) {
      %4 = scf.for %arg3 = %c0 to %dim_3 step %c640 iter_args(%arg4 = %arg2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) {
        %5 = affine.min #map2(%arg1)[%dim_2]
        %6 = affine.min #map3(%arg3)[%dim_3]
        %7 = affine.max #map4(%arg1)
        %8 = affine.max #map5(%arg1)
        %9 = affine.min #map6()[%8]
        %10 = affine.max #map7(%5, %7)[%dim_2]
        %11 = affine.min #map6()[%10]
        %12 = affine.apply #map8(%5, %9, %11)
        %13 = affine.max #map4(%arg3)
        %14 = affine.max #map5(%arg3)
        %15 = affine.min #map6()[%14]
        %16 = affine.max #map7(%6, %13)[%dim_3]
        %17 = affine.min #map6()[%16]
        %18 = affine.apply #map8(%6, %15, %17)
        %extracted_slice = tensor.extract_slice %1[0, 0, %7, %13] [1, 16, %12, %18] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
        %19 = func.call @main_func1(%extracted_slice) : (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NCHW}>
        %inserted_slice = tensor.insert_slice %19 into %arg4[0, 0, %arg1, %arg3] [1, 16, %5, %6] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
      }
      scf.yield %4 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    }
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }

  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             func.call @main_func0

  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             scf.index_switch
  // CHECK-COUNT-7:       func.call @main_func1_dims_HW_cases_{{[0-9]+}}({{%.+}}) : ({{.+}})
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map2 = affine_map<(d0) -> (d0 * 2)>

//CHECK: #[[$MAP_BACKTRACK_OFFSET:.+]] = affine_map<(d0)[s0] -> (s0 - 256, d0)>
//CHECK: #[[$MAP_TILE_SIZE:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP_STRIDE_INPUT_START:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
//CHECK: #[[$MAP_STRIDE_INPUT_SIZE:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: @EltwiseWith2Inputs
module {
  net.NetworkInfo entryPoint : @EltwiseWith2Inputs inputsInfo : {
    DataInfo "input0" tensorNames = ["input0"] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    DataInfo "input1" tensorNames = ["input1"] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "/conv_final/Conv" friendlyName = "EditOpenVinoIRResult_48" tensorNames = ["/conv_final/Conv_output_0"] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: func.func nested @EltwiseWith2Inputs_func0_dims_H_cases_2([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>,
  // CHECK-SAME [[ARG1:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) ->
  // CHECK-SAME tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK: func.func nested @EltwiseWith2Inputs_func0_dims_H_cases_1([[ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>,
  // CHECK-SAME [[ARG1:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) ->
  // CHECK-SAME tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>

  func.func nested @EltwiseWith2Inputs_func0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [2, 2]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
    %1 = VPU.NCE.Convolution(%arg1, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [2, 2]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
    return %2 : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @EltwiseWith2Inputs(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %0 = arith.addi %dim, %c1 : index
    %1 = arith.divsi %0, %c2 : index
    %2 = tensor.empty(%1) : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
    %5 = scf.for %arg3 = %c0 to %1 step %c256 iter_args(%arg2 = %2) -> (tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>) {
      %6 = affine.min #map(%arg3)[%1]
      %7 = affine.max #map1(%arg3)
      %8 = affine.apply #map2(%6)
      %slice0 = tensor.extract_slice %arg0[0, 0, %7, 0] [1, 32, %8, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
      %slice1 = tensor.extract_slice %arg1[0, 0, %7, 0] [1, 32, %8, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
      %9 = func.call @EltwiseWith2Inputs_func0(%slice0, %slice1) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %9 into %arg2[0, 0, %arg3, 0] [1, 256, %6, 32] [1, 1, 1, 1] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %5 : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: [[SWITCH_RESULT:%.+]] = scf.index_switch {{%.+}} -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-COUNT-2: func.call @EltwiseWith2Inputs_func0_dims_H_cases_{{[0-9]+}}({{%.+}}, {{%.+}}) :
  // CHECK-SAME:     (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>,
  // CHECK-SAME:     tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) ->
  // CHECK-SAME:     tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 540)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
#map_add = affine_map<(d0) -> (d0 + 10, 10)>
#map_add1 = affine_map<(d0) -> (d0 + 5, 5)>

// CHECK-LABEL: @DynamicDimsAsArgsToFuncOp
module {
  net.NetworkInfo entryPoint : @DynamicDimsAsArgsToFuncOp inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" friendlyName = "output/sink_port_0" tensorNames = ["output"] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
  }
  //CHECK: func.func nested @main_func0({{%.+}}: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}>) ->
  //CHECK-SAME:     tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}> {
  func.func nested @main_func0(%arg0: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[SLICE:%.+]] = tensor.extract_slice {{%.+}}[0, 10, 5, 0] [1, 540, 32, 16] [1, 1, 1, 1]
    %extracted_slice = tensor.extract_slice %arg0[0, %arg1, %arg2, 0] [1, %arg3, %arg4, 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}>
    %5 = VPU.PermuteCast(%extracted_slice) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}>
    return %6 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @DynamicDimsAsArgsToFuncOp(%arg0: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> {
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c540 = arith.constant 540 : index
    %c0 = arith.constant 0 : index
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    %dim_0 = tensor.dim %arg0, %c2 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim step %c540 iter_args(%arg2 = %0) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = scf.for %arg3 = %c0 to %dim_0 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>) {
        %3 = affine.min #map(%arg1)[%dim]
        %4 = affine.min #map1(%arg3)[%dim_0]
        %offset1 = affine.min #map_add(%arg1)
        %offset2 = affine.min #map_add1(%arg3)
        %extracted_slice = tensor.extract_slice %arg0[0, %offset1, %offset2, 0] [1, %3, %4, 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}>
        %5 = func.call @main_func0(%extracted_slice, %offset1, %offset2, %3, %4) : (tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}>, index, index, index, index) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, %arg3] [1, 16, %3, %4] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 99)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 640)>
#map4 = affine_map<(d0) -> (0, d0 - 1)>
#map5 = affine_map<(d0) -> (-d0 + 1, 0)>
#map6 = affine_map<()[s0] -> (1, s0)>
#map7 = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
#map8 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @DynamicDimsAsArgsToFuncOpWithSwitch
module {
  net.NetworkInfo entryPoint : @DynamicDimsAsArgsToFuncOpWithSwitch inputsInfo : {
    DataInfo "Parameter_14" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "Convolution_16" friendlyName = "Result_17" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-COUNT-6: func.func nested @main_func1_dims_HW_cases_{{[0-9]+}}({{%.+}}: tensor<{{.+}}>) -> tensor<{{.+}}>
  func.func nested @main_func1(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
    %1 = tensor.extract_slice %arg0[0, 0, %arg1, %arg2] [1, 16, %arg3, %arg4] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Eltwise(%1, %2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @DynamicDimsAsArgsToFuncOpWithSwitch(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %c640 = arith.constant 640 : index
    %c99 = arith.constant 99 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %dim_2 = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %dim_3 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = tensor.empty(%dim_2, %dim_3) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = scf.for %arg1 = %c0 to %dim_2 step %c99 iter_args(%arg2 = %2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
      %4 = scf.for %arg3 = %c0 to %dim_3 step %c640 iter_args(%arg4 = %arg2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
        %5 = affine.min #map2(%arg1)[%dim_2]
        %6 = affine.min #map3(%arg3)[%dim_3]
        %7 = affine.max #map4(%arg1)
        %8 = affine.max #map5(%arg1)
        %9 = affine.min #map6()[%8]
        %10 = affine.max #map7(%5, %7)[%dim_2]
        %11 = affine.min #map6()[%10]
        %12 = affine.apply #map8(%5, %9, %11)
        %13 = affine.max #map4(%arg3)
        %14 = affine.max #map5(%arg3)
        %15 = affine.min #map6()[%14]
        %16 = affine.max #map7(%6, %13)[%dim_3]
        %17 = affine.min #map6()[%16]
        %18 = affine.apply #map8(%6, %15, %17)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %7, %13] [1, 16, %12, %18] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
        %19 = func.call @main_func1(%extracted_slice, %8, %10, %5, %6) : (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, index, index, index, index) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %19 into %arg4[0, 0, %arg1, %arg3] [1, 16, %5, %6] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %4 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             scf.index_switch
  // CHECK-COUNT-7: func.call @main_func1_dims_HW_cases_{{[0-9]+}}({{%.+}})
}


// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 99)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 640)>
#map4 = affine_map<(d0) -> (0, d0 - 1)>
#map5 = affine_map<(d0) -> (-d0 + 1, 0)>
#map6 = affine_map<()[s0] -> (1, s0)>
#map7 = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
#map8 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @DynamicDimsAsArgsToFuncOpWithSwitchTwoInputs
module {
  net.NetworkInfo entryPoint : @DynamicDimsAsArgsToFuncOpWithSwitchTwoInputs inputsInfo : {
    DataInfo "Parameter_14" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "Convolution_16" friendlyName = "Result_17" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-COUNT-6: func.func nested @main_func1_dims_HW_cases_{{[0-9]+}}({{%.+}}: tensor<{{.+}}>, {{%.+}}: tensor<{{.+}}>) -> tensor<{{.+}}>
  func.func nested @main_func1(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x3x3xf16, {order = #NHWC}>
    %1 = tensor.extract_slice %arg0[0, 0, %arg1, %arg2] [1, 16, %arg3, %arg4] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Eltwise(%1, %2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    %4 = VPU.NCE.Eltwise(%3, %arg5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
    return %4 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @DynamicDimsAsArgsToFuncOpWithSwitchTwoInputs(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %c640 = arith.constant 640 : index
    %c99 = arith.constant 99 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %dim_2 = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %dim_3 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = tensor.empty(%dim_2, %dim_3) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = scf.for %arg1 = %c0 to %dim_2 step %c99 iter_args(%arg2 = %2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
      %4 = scf.for %arg3 = %c0 to %dim_3 step %c640 iter_args(%arg4 = %arg2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
        %5 = affine.min #map2(%arg1)[%dim_2]
        %6 = affine.min #map3(%arg3)[%dim_3]
        %7 = affine.max #map4(%arg1)
        %8 = affine.max #map5(%arg1)
        %9 = affine.min #map6()[%8]
        %10 = affine.max #map7(%5, %7)[%dim_2]
        %11 = affine.min #map6()[%10]
        %12 = affine.apply #map8(%5, %9, %11)
        %13 = affine.max #map4(%arg3)
        %14 = affine.max #map5(%arg3)
        %15 = affine.min #map6()[%14]
        %16 = affine.max #map7(%6, %13)[%dim_3]
        %17 = affine.min #map6()[%16]
        %18 = affine.apply #map8(%6, %15, %17)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %7, %13] [1, 16, %12, %18] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
        %19 = func.call @main_func1(%extracted_slice, %8, %10, %5, %6, %extracted_slice) : (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>, index, index, index, index, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %19 into %arg4[0, 0, %arg1, %arg3] [1, 16, %5, %6] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 99, 640]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %4 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }


  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             scf.index_switch
  // CHECK-COUNT-7: func.call @main_func1_dims_HW_cases_{{[0-9]+}}({{%.+}})
}

// -----
module {
  // CHECK-LABEL: @DynamicDimsAndPermuteCast
  net.NetworkInfo entryPoint : @DynamicDimsAndPermuteCast inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  } outputsInfo : {
    DataInfo "output" friendlyName = "output/sink_port_0" tensorNames = ["output"] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  }
  // CHECK-COUNT-9: func.func nested @main_func0_dims_HW_cases_{{[0-9]+}}({{%.+}}: tensor<{{.+}}>) -> tensor<{{.+}}>
  func.func nested @main_func0(%main: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<0.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<0.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %transpose_to_nchw = VPU.PermuteCast(%main) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 63, 512]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %conv1 = VPU.NCE.Convolution(%transpose_to_nchw, %cst) rawFilterShape [16, 16, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 63, 512]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x16x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 63, 512]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %conv2 = VPU.NCE.Convolution(%conv1, %cst_1) rawFilterShape [16, 16, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 63, 512]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x16x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 63, 512]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %Convert_196_2 = VPU.PermuteCast(%conv2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 63, 512]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    return %Convert_196_2 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  }

  func.func @DynamicDimsAndPermuteCast(%input: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c63 = arith.constant 63 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %transpose_to_nchw = tensor.dim %input, %c1 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %transpose_to_nchw_0 = tensor.dim %input, %c2 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %Convert_196 = tensor.empty(%transpose_to_nchw, %transpose_to_nchw_0) : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %Convert_196_1 = scf.for %Convert_196_2 = %c0 to %transpose_to_nchw step %c63 iter_args(%Convert_196_3 = %Convert_196) -> (tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) {
      %Convert_196_4 = scf.for %Convert_196_5 = %c0 to %transpose_to_nchw_0 step %c512 iter_args(%Convert_196_6 = %Convert_196_3) -> (tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) {
        %Convert_196_7 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 63)>(%Convert_196_2)[%transpose_to_nchw]
        %Convert_196_8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%Convert_196_5)[%transpose_to_nchw_0]
        %conv2 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%Convert_196_2)
        %conv2_9 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%Convert_196_2)
        %conv2_10 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv2_9]
        %conv2_11 = affine.max affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>(%Convert_196_7, %conv2)[%transpose_to_nchw]
        %conv2_12 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv2_11]
        %conv2_13 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%Convert_196_5)
        %conv2_14 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%Convert_196_5)
        %conv2_15 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv2_14]
        %conv2_16 = affine.max affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>(%Convert_196_8, %conv2_13)[%transpose_to_nchw_0]
        %conv2_17 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv2_16]
        %conv1 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%conv2)
        %conv1_18 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%conv2)
        %conv1_19 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv1_18]
        %conv1_20 = affine.max affine_map<(d0, d1, d2, d3)[s0] -> (0, d0 + d1 - d2 - d3 - s0 + 4)>(%conv1, %Convert_196_7, %conv2_10, %conv2_12)[%transpose_to_nchw]
        %conv1_21 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv1_20]
        %conv1_22 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>(%conv1_19, %conv1_21, %Convert_196_7, %conv2_10, %conv2_12)
        %conv1_23 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%conv2_13)
        %conv1_24 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%conv2_13)
        %conv1_25 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv1_24]
        %conv1_26 = affine.max affine_map<(d0, d1, d2, d3)[s0] -> (0, d0 + d1 - d2 - d3 - s0 + 4)>(%conv1_23, %Convert_196_8, %conv2_15, %conv2_17)[%transpose_to_nchw_0]
        %conv1_27 = affine.min affine_map<()[s0] -> (1, s0)>()[%conv1_26]
        %conv1_28 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>(%conv1_25, %conv1_27, %Convert_196_8, %conv2_15, %conv2_17)
        %transpose_to_nchw_29 = tensor.extract_slice %input[0, %conv1, %conv1_23, 0] [1, %conv1_22, %conv1_28, 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> to tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        %transpose_to_nchw_30 = func.call @main_func0(%transpose_to_nchw_29) : (tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        %Convert_196_31 = tensor.insert_slice %transpose_to_nchw_30 into %Convert_196_6[0, %Convert_196_2, %Convert_196_5, 0] [1, %Convert_196_7, %Convert_196_8, 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 63, 512, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> into tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        scf.yield %Convert_196_31 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
      }
      // CHECK:         scf.for
      // CHECK:           scf.for
      // CHECK:             scf.index_switch
      // CHECK-COUNT-10: func.call @main_func0_dims_HW_cases_{{[0-9]+}}({{%.+}})
      scf.yield %Convert_196_4 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    }
    return %Convert_196_1 : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 16]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  }
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map_w_size  = affine_map<(d0)[s0] -> (-d0 + s0, 40)>
#map_dep_idx = affine_map<(d0) -> (d0 floordiv 40)>
#map_inv_idx = affine_map<(d0, d1) -> (-d0 - d1 + 22)>
#map_out_off = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @StaticUniformOutputTilesWithIndexSwitch
module @StaticUniformOutputTilesWithIndexSwitch  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x32x80xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x32x40xf16>
  }

  func.func nested @main_func4(
      %arg0: tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 80]> : tensor<4xsi64>, order = #NHWC}>,
      %dep_idx: index,
      %inv_idx: index
  ) -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 40]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x3xf16, {order = #NHWC}>
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 1, 3] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 2]
    } : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 80]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<16x16x1x3xf16, {order = #NHWC}>
        -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 40]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 40]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x32x80xf16, {order = #NHWC}>) -> tensor<1x16x32x40xf16, {order = #NHWC}> {
    %c0  = arith.constant 0  : index
    %c40 = arith.constant 40 : index
    %c80 = arith.constant 80 : index
    %empty = tensor.empty() : tensor<1x16x32x40xf16, {order = #NHWC}>
    %result = scf.for %iv = %c0 to %c80 step %c40 iter_args(%acc = %empty)
        -> (tensor<1x16x32x40xf16, {order = #NHWC}>) {
      %size = affine.min #map_w_size(%iv)[%c80]
      %slice = tensor.extract_slice %arg0[0, 0, 0, %iv][1, 16, 32, %size][1, 1, 1, 1]
          : tensor<1x16x32x80xf16, {order = #NHWC}>
          to tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 80]> : tensor<4xsi64>, order = #NHWC}>
      %dep_idx = affine.apply #map_dep_idx(%iv)
      %c13 = arith.constant 13 : index
      %c9  = arith.constant 9  : index
      %inv_idx = affine.apply #map_inv_idx(%c13, %c9)
      %call = func.call @main_func4(%slice, %dep_idx, %inv_idx)
          : (tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 80]> : tensor<4xsi64>, order = #NHWC}>,
             index, index)
          -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 40]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %call
          : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 40]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x16x32x20xf16, {order = #NHWC}>
      %out_off = affine.apply #map_out_off(%iv)
      %inserted = tensor.insert_slice %cast into %acc[0, 0, 0, %out_off][1, 16, 32, 20][1, 1, 1, 1]
          : tensor<1x16x32x20xf16, {order = #NHWC}> into tensor<1x16x32x40xf16, {order = #NHWC}>
      scf.yield %inserted : tensor<1x16x32x40xf16, {order = #NHWC}>
    }
    return %result : tensor<1x16x32x40xf16, {order = #NHWC}>
  }

  // CHECK-DAG: func.func nested @main_func4_dims_W_cases_2({{.*}}) -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-DAG: func.func nested @main_func4_dims_W_cases_1({{.*}}) -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: scf.for {{.*}} step %c40 {{.*}} -> (tensor<1x16x32x40xf16, {order = #NHWC}>)
  // CHECK: [[SWITCH:%[^ ]+]] = scf.index_switch {{.*}} -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   case 1 {
  // CHECK:     [[CALL_CASE_1:%[^ ]+]] = func.call @main_func4_dims_W_cases_1({{.*}}, %c0) {{.*}} -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-NEXT:     scf.yield [[CALL_CASE_1]] : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   case 2 {
  // CHECK:     [[CALL_CASE_2:%[^ ]+]] = func.call @main_func4_dims_W_cases_2({{.*}}, %c0) {{.*}} -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-NEXT:     scf.yield [[CALL_CASE_2]] : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   [[CAST_TO_STATIC:%cast[^ ]*]] = tensor.cast [[SWITCH]] : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 20]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x32x20xf16, {order = #NHWC}>
  // CHECK:   [[INSERTED:%inserted[^ ]*]] = tensor.insert_slice [[CAST_TO_STATIC]] into {{.*}} [1, 16, 32, 20] {{.*}} into tensor<1x16x32x40xf16, {order = #NHWC}>
  // CHECK-NEXT:   scf.yield [[INSERTED]] : tensor<1x16x32x40xf16, {order = #NHWC}>
  // CHECK: return {{.*}} : tensor<1x16x32x40xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 100)>

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (900, d0)>

// CHECK-LABEL: @SliceSizeFromLoopIVAndFunctionScale
module @SliceSizeFromLoopIVAndFunctionScale {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x720x1000xf16>
    DataInfo "scale" : tensor<1xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x1000xf16>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    return %arg0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>, %scale: tensor<1xf32>) -> tensor<1x16x720x1000xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1000 = arith.constant 1000 : index
    %0 = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
    %extracted = tensor.extract %scale[%c0] : tensor<1xf32>
    %scale_i64 = arith.fptosi %extracted : f32 to i64
    %bound = arith.index_cast %scale_i64 : i64 to index
    %1 = scf.for %arg2 = %c0 to %c1000 step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
      %sizeIn = affine.min #map(%arg2)[%c1000]
      %sizeOut = affine.min #map1(%arg2)[%bound]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %sizeIn] [1, 1, 1, 1] : tensor<1x16x720x1000xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %sizeOut] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x1000xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x720x1000xf16, {order = #NHWC}>
  }

  // CHECK:        tensor.extract_slice {{.*}} [1, 16, 720, 100] {{.*}}
  // CHECK:        tensor.insert_slice {{.*}} [1, 16, 720, 100] {{.*}}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map  = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
#map5 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-DAG: #[[$MAP_BACKTRACK:.+]] = affine_map<(d0)[s0] -> (s0 - 100, d0)>
// CHECK-DAG: #[[$MAP_TILE:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
// CHECK-DAG: #[[$MAP_HSTART:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK-DAG: #[[$MAP_TOPCLAMP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP_MIN1:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP_BOTOVFL:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
// CHECK-DAG: #[[$MAP_INPSIZE:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @ConvMaxPadding
module @ConvMaxPadding {
  net.NetworkInfo entryPoint : @ConvMaxPadding inputsInfo : {
    DataInfo "input" : tensor<1x16x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x?x64xf16>
  }

  func.func nested @ConvMaxPadding_func0(%arg0: tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {
      resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,
      strides = [1, 1]
    } : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<16x16x3x3xf16, {order = #NHWC}>
     -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @ConvMaxPadding(%arg0: tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %c100 = arith.constant 100 : index
    %c0   = arith.constant 0   : index
    %c2   = arith.constant 2   : index
    %dim  = tensor.dim %arg0, %c2 : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
    %empty = tensor.empty(%dim) : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
    %result = scf.for %arg1 = %c0 to %dim step %c100 iter_args(%arg2 = %empty) -> (tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>) {
      %tile_size = affine.min #map(%arg1)[%dim]
      %h_start   = affine.max #map1(%arg1)
      %top_clamp = affine.max #map2(%arg1)
      %top_min   = affine.min #map3()[%top_clamp]
      %bot_ovfl  = affine.max #map4(%tile_size, %h_start)[%dim]
      %bot_min   = affine.min #map3()[%bot_ovfl]
      %inp_size  = affine.apply #map5(%tile_size, %top_min, %bot_min)
      %extracted = tensor.extract_slice %arg0[0, 0, %h_start, 0] [1, 16, %inp_size, 64] [1, 1, 1, 1]
          : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
      %call = func.call @ConvMaxPadding_func0(%extracted) : (tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
      %inserted = tensor.insert_slice %call into %arg2[0, 0, %arg1, 0] [1, 16, %tile_size, 64] [1, 1, 1, 1]
          : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
          into tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %result : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:      func.func nested @ConvMaxPadding_func0_dims_H_cases_2([[ARG0C2:%.+]]: tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:        [[CST2:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
  // CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[ARG0C2]], [[CST2]]) rawFilterShape [16, 16, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        return [[CONV2]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }

  // CHECK:      func.func nested @ConvMaxPadding_func0_dims_H_cases_1([[ARG0C1:%.+]]: tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:        [[CST1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
  // CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[ARG0C1]], [[CST1]]) rawFilterShape [16, 16, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        return [[CONV1]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }

  // CHECK:      func.func nested @ConvMaxPadding_func0_dims_H_cases_0([[ARG0C0:%.+]]: tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 102, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:        [[CST0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
  // CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG0C0]], [[CST0]]) rawFilterShape [16, 16, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 102, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        return [[CONV0]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }

  // CHECK:      func.func @ConvMaxPadding([[MARG0:%.+]]: tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:        [[FALSE:%.+]] = arith.constant false
  // CHECK:        [[C3:%.+]] = arith.constant 3 : index
  // CHECK:        [[C1:%.+]] = arith.constant 1 : index
  // CHECK:        [[C100:%.+]] = arith.constant 100 : index
  // CHECK:        [[C0:%.+]] = arith.constant 0 : index
  // CHECK:        [[C2:%.+]] = arith.constant 2 : index
  // CHECK:        [[DIM:%.+]] = tensor.dim [[MARG0]], [[C2]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[HAS_ENOUGH:%.+]] = arith.cmpi sge, [[DIM]], [[C100]] : index
  // CHECK:        cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:        [[FOR_RESULT:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[DIM]] step [[C100]] iter_args([[ACC:%.+]] = [[EMPTY]]) -> (tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:          [[BACKTRACK:%.+]] = affine.min #[[$MAP_BACKTRACK]]([[IV]])[[[DIM]]]
  // CHECK:          [[TILE:%.+]] = affine.min #[[$MAP_TILE]]([[BACKTRACK]])[[[DIM]]]
  // CHECK:          [[HSTART:%.+]] = affine.max #[[$MAP_HSTART]]([[BACKTRACK]])
  // CHECK:          [[TOP_CLAMP:%.+]] = affine.max #[[$MAP_TOPCLAMP]]([[BACKTRACK]])
  // CHECK:          [[TOP_MIN:%.+]] = affine.min #[[$MAP_MIN1]]()[[[TOP_CLAMP]]]
  // CHECK:          [[BOT_OVFL:%.+]] = affine.max #[[$MAP_BOTOVFL]]([[TILE]], [[HSTART]])[[[DIM]]]
  // CHECK:          [[BOT_MIN:%.+]] = affine.min #[[$MAP_MIN1]]()[[[BOT_OVFL]]]
  // CHECK:          [[INP_SIZE:%.+]] = affine.apply #[[$MAP_INPSIZE]]([[TILE]], [[TOP_MIN]], [[BOT_MIN]])
  // CHECK:          [[IS_AT_TOP:%.+]] = arith.cmpi eq, [[HSTART]], [[C0]] : index
  // CHECK:          [[CASE_IDX:%.+]] = scf.if [[IS_AT_TOP]] -> (index) {
  // CHECK:            [[IS_EXACT:%.+]] = arith.cmpi eq, [[INP_SIZE]], [[DIM]] : index
  // CHECK:            [[CASE_FIRST:%.+]] = arith.select [[IS_EXACT]], [[C3]], [[C2]] : index
  // CHECK:            scf.yield [[CASE_FIRST]] : index
  // CHECK:          } else {
  // CHECK:            [[NEXT:%.+]] = arith.addi [[HSTART]], [[INP_SIZE]] : index
  // CHECK:            [[IN_BOUNDS:%.+]] = arith.cmpi slt, [[NEXT]], [[DIM]] : index
  // CHECK:            [[CASE_ELSE:%.+]] = arith.select [[IN_BOUNDS]], [[C0]], [[C1]] : index
  // CHECK:            scf.yield [[CASE_ELSE]] : index
  // CHECK:          }
  // CHECK:          [[H_SHIFTED:%.+]] = arith.shli [[CASE_IDX]], [[C2]] : index
  // CHECK:          [[SWITCH:%.+]] = scf.index_switch [[H_SHIFTED]] -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            case 0 {
  // CHECK:              [[SLICE0:%.+]] = tensor.extract_slice [[MARG0]][0, 0, [[HSTART]], 0] [1, 16, 102, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x102x64xf16, {order = #NHWC}>
  // CHECK:              [[CAST0:%.+]] = tensor.cast [[SLICE0]] : tensor<1x16x102x64xf16, {order = #NHWC}> to tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 102, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:              [[CALL0:%.+]] = func.call @ConvMaxPadding_func0_dims_H_cases_0([[CAST0]]) : (tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 102, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:              scf.yield [[CALL0]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            }
  // CHECK:            case 4 {
  // CHECK:              [[SLICE4:%.+]] = tensor.extract_slice [[MARG0]][0, 0, [[HSTART]], 0] [1, 16, 101, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x101x64xf16, {order = #NHWC}>
  // CHECK:              [[CAST4:%.+]] = tensor.cast [[SLICE4]] : tensor<1x16x101x64xf16, {order = #NHWC}> to tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:              [[CALL4:%.+]] = func.call @ConvMaxPadding_func0_dims_H_cases_1([[CAST4]]) : (tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:              scf.yield [[CALL4]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            }
  // CHECK:            case 8 {
  // CHECK:              [[SLICE8:%.+]] = tensor.extract_slice [[MARG0]][0, 0, [[HSTART]], 0] [1, 16, 101, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x101x64xf16, {order = #NHWC}>
  // CHECK:              [[CAST8:%.+]] = tensor.cast [[SLICE8]] : tensor<1x16x101x64xf16, {order = #NHWC}> to tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:              [[CALL8:%.+]] = func.call @ConvMaxPadding_func0_dims_H_cases_2([[CAST8]]) : (tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 101, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:              scf.yield [[CALL8]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            }
  // CHECK:            default {
  // CHECK:              cf.assert [[FALSE]], "Unsupported case"
  // CHECK:            }
  // CHECK:          [[CAST_OUT:%.+]] = tensor.cast [[SWITCH]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x100x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          [[INSERTED:%.+]] = tensor.insert_slice [[CAST_OUT]] into [[ACC]][0, 0, [[BACKTRACK]], 0] [1, 16, 100, 64] [1, 1, 1, 1] : tensor<1x16x100x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          scf.yield [[INSERTED]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        }
  // CHECK:        return [[FOR_RESULT]] : tensor<1x16x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1000, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#mapH = affine_map<(d0)[s0] -> (-d0 + s0, 40)>
#mapW = affine_map<(d0)[s0] -> (-d0 + s0, 64)>

// CHECK-DAG: #[[$MAPH:.+]] = affine_map<(d0)[s0] -> (s0 - 40, d0)>
// CHECK-DAG: #[[$MAPW:.+]] = affine_map<(d0)[s0] -> (s0 - 64, d0)>

// CHECK-LABEL: @InterpolateScalesBoundThroughFuncCall
module @InterpolateScalesBoundThroughFuncCall {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>
    DataInfo "scales" : tensor<2xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func nested @main_func0(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}> {
    %aux = VPU.Empty : tensor<1x1x1x1024xui8>
    %interp = VPU.InterpolateDMA(%arg0, %arg1, %aux) {
      attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
      axes_attr = [2, 3]
    } : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, tensor<2xf32>, tensor<1x1x1x1024xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
    return %interp : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func nested @main_func1(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %expand : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, %scales: tensor<2xf32>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c40 = arith.constant 40 : index
    %c64 = arith.constant 64 : index

    %dimH = tensor.dim %arg0, %c2 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>
    %dimW = tensor.dim %arg0, %c3 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>

    %interp = call @main_func0(%arg0, %scales) : (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>

    %scaleH = tensor.extract %scales[%c0] : tensor<2xf32>
    %scaleH_f64 = arith.extf %scaleH : f32 to f64
    %dimH_i64 = arith.index_cast %dimH : index to i64
    %dimH_f64 = arith.sitofp %dimH_i64 : i64 to f64
    %ubH_f64 = arith.mulf %dimH_f64, %scaleH_f64 : f64
    %ubH_i64 = arith.fptosi %ubH_f64 : f64 to i64
    %ubH = arith.index_cast %ubH_i64 : i64 to index

    %scaleW = tensor.extract %scales[%c1] : tensor<2xf32>
    %scaleW_f64 = arith.extf %scaleW : f32 to f64
    %dimW_i64 = arith.index_cast %dimW : index to i64
    %dimW_f64 = arith.sitofp %dimW_i64 : i64 to f64
    %ubW_f64 = arith.mulf %dimW_f64, %scaleW_f64 : f64
    %ubW_i64 = arith.fptosi %ubW_f64 : f64 to i64
    %ubW = arith.index_cast %ubW_i64 : i64 to index

    %empty = tensor.empty(%ubH, %ubW) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
    %result = scf.for %ivH = %c0 to %ubH step %c40 iter_args(%accH = %empty) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>) {
      %innerRes = scf.for %ivW = %c0 to %ubW step %c64 iter_args(%accW = %accH) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>) {
        %sizeH = affine.min #mapH(%ivH)[%ubH]
        %sizeW = affine.min #mapW(%ivW)[%ubW]
        %slice = tensor.extract_slice %interp[0, 0, %ivH, %ivW] [1, 3, %sizeH, %sizeW] [1, 1, 1, 1] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
        %call = func.call @main_func1(%slice) : (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
        %inserted = tensor.insert_slice %call into %accW[0, 0, %ivH, %ivW] [1, 16, %sizeH, %sizeW] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %innerRes : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %result : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:      func.func nested @main_func0([[F0_ARG0:%.+]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, [[F0_SCALES:%.+]]: tensor<2xf32>) ->
  // CHECK-SAME:      tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:        [[AUX:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>
  // CHECK:        [[DMA:%.+]] = VPU.InterpolateDMA([[F0_ARG0]], [[F0_SCALES]], [[AUX]])
  // CHECK-SAME:      axes_attr = [2, 3]
  // CHECK-SAME:      tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, tensor<2xf32>, tensor<1x1x1x1024xui8>
  // CHECK-SAME:      -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        return [[DMA]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      func.func nested @main_func1([[F1_ARG0:%.+]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>) ->
  // CHECK-SAME:      tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:        [[EXPAND:%.+]] = VPU.Expand([[F1_ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
  // CHECK-SAME:      tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-SAME:      -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        return [[EXPAND]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      func.func @main([[ARG0:%.+]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, [[SCALES:%.+]]: tensor<2xf32>) ->
  // CHECK-SAME:      tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
  // CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
  // CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
  // CHECK-DAG:    [[C40:%.+]] = arith.constant 40 : index
  // CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
  // CHECK:        [[DIMH:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[DIMW:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INTERP:%.+]] = call @main_func0([[ARG0]], [[SCALES]]) :
  // CHECK-SAME:      (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NHWC}>, tensor<2xf32>)
  // CHECK-SAME:      -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[SCALEH:%.+]] = tensor.extract [[SCALES]][[[C0]]] : tensor<2xf32>
  // CHECK:        [[SCALEH_F64:%.+]] = arith.extf [[SCALEH]] : f32 to f64
  // CHECK:        [[DIMH_I64:%.+]] = arith.index_cast [[DIMH]] : index to i64
  // CHECK:        [[DIMH_F64:%.+]] = arith.sitofp [[DIMH_I64]] : i64 to f64
  // CHECK:        [[UBH_F64:%.+]] = arith.mulf [[DIMH_F64]], [[SCALEH_F64]] : f64
  // CHECK:        [[UBH_I64:%.+]] = arith.fptosi [[UBH_F64]] : f64 to i64
  // CHECK:        [[UBH:%.+]] = arith.index_cast [[UBH_I64]] : i64 to index
  // CHECK:        [[SCALEW:%.+]] = tensor.extract [[SCALES]][[[C1]]] : tensor<2xf32>
  // CHECK:        [[SCALEW_F64:%.+]] = arith.extf [[SCALEW]] : f32 to f64
  // CHECK:        [[DIMW_I64:%.+]] = arith.index_cast [[DIMW]] : index to i64
  // CHECK:        [[DIMW_F64:%.+]] = arith.sitofp [[DIMW_I64]] : i64 to f64
  // CHECK:        [[UBW_F64:%.+]] = arith.mulf [[DIMW_F64]], [[SCALEW_F64]] : f64
  // CHECK:        [[UBW_I64:%.+]] = arith.fptosi [[UBW_F64]] : f64 to i64
  // CHECK:        [[UBW:%.+]] = arith.index_cast [[UBW_I64]] : i64 to index
  // CHECK:        [[EMPTY:%.+]] = tensor.empty([[UBH]], [[UBW]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[CMPW:%.+]] = arith.cmpi sge, [[UBW]], [[C64]] : index
  // CHECK:        cf.assert [[CMPW]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:        [[CMPH:%.+]] = arith.cmpi sge, [[UBH]], [[C40]] : index
  // CHECK:        cf.assert [[CMPH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:        [[RES:%.+]] = scf.for [[IVH:%.+]] = [[C0]] to [[UBH]] step [[C40]] iter_args([[ACCH:%.+]] = [[EMPTY]]) ->
  // CHECK-SAME:      (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:          [[INNER:%.+]] = scf.for [[IVW:%.+]] = [[C0]] to [[UBW]] step [[C64]] iter_args([[ACCW:%.+]] = [[ACCH]]) ->
  // CHECK-SAME:        (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:            [[OFFH:%.+]] = affine.min #[[$MAPH]]([[IVH]])[[[UBH]]]
  // CHECK:            [[OFFW:%.+]] = affine.min #[[$MAPW]]([[IVW]])[[[UBW]]]
  // CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INTERP]][0, 0, [[OFFH]], [[OFFW]]] [1, 3, 40, 64] [1, 1, 1, 1] :
  // CHECK-SAME:          tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-SAME:          to tensor<1x3x40x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST_IN:%.+]] = tensor.cast [[SLICE]] : tensor<1x3x40x64xf16, {order = #NHWC}>
  // CHECK-SAME:          to tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL:%.+]] = func.call @main_func1([[CAST_IN]]) :
  // CHECK-SAME:          (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 40, 64]> : tensor<4xsi64>, order = #NHWC}>)
  // CHECK-SAME:          -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CAST_OUT:%.+]] = tensor.cast [[CALL]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-SAME:          to tensor<1x16x40x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[INSERTED:%.+]] = tensor.insert_slice [[CAST_OUT]] into [[ACCW]][0, 0, [[OFFH]], [[OFFW]]] [1, 16, 40, 64] [1, 1, 1, 1] :
  // CHECK-SAME:          tensor<1x16x40x64xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-SAME:          into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[INSERTED]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          scf.yield [[INNER]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        }
  // CHECK:        return [[RES]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 80, 128]> : tensor<4xsi64>, order = #NHWC}>
}
