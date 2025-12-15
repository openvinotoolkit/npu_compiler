//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-block-size-for-scf-tiling  --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 100)>

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }
  func.func private @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
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

  // CHECK:    func.func private @main_func0([[ARG0:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[ELTWISE]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @main([[ARG0:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[C100:%.*]] = arith.constant 100 : index
  // CHECK:      [[C0:%.*]] = arith.constant 0 : index
  // CHECK:      [[C3:%.*]] = arith.constant 3 : index
  // CHECK:      [[DIM:%.*]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[EMPTY:%.*]] = tensor.empty([[DIM]]) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH:%.*]] = arith.cmpi sge, [[DIM]], [[C100]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[REMUI:%.*]] = arith.remui [[DIM]], [[C100]] : index
  // CHECK:      [[FOR_RESULT:%.*]] = scf.for [[ARG2:%.*]] = [[C0]] to [[DIM]] step [[C100]] iter_args([[ARG3:%.*]] = [[EMPTY]]) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[MIN_SIZE:%.*]] = affine.min #[[$MAP]]([[ARG2]])[[[DIM]]]
  // CHECK:        [[IS_FIRST:%.*]] = arith.cmpi eq, [[ARG2]], [[C0]] : index
  // CHECK:        [[OFFSET:%.*]] = scf.if [[IS_FIRST]] -> (index) {
  // CHECK:          scf.yield [[ARG2]] : index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS:%.*]] = arith.addi [[ARG2]], [[MIN_SIZE]] : index
  // CHECK:          [[IN_BOUNDS:%.*]] = arith.cmpi slt, [[NEXT_POS]], [[DIM]] : index
  // CHECK:          [[FINAL_OFFSET:%.*]] = scf.if [[IN_BOUNDS]] -> (index) {
  // CHECK:            scf.yield [[ARG2]] : index
  // CHECK:          } else {
  // CHECK:            [[IS_EXACT:%.*]] = arith.cmpi eq, [[MIN_SIZE]], [[C100]] : index
  // CHECK:            [[BACKTRACK_OFFSET:%.*]] = scf.if [[IS_EXACT]] -> (index) {
  // CHECK:              scf.yield [[ARG2]] : index
  // CHECK:            } else {
  // CHECK:              [[REMAINING:%.*]] = affine.apply #[[$MAP1]]([[ARG2]])[[[REMUI]]]
  // CHECK:              scf.yield [[REMAINING]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[BACKTRACK_OFFSET]] : index
  // CHECK:          }
  // CHECK:          scf.yield [[FINAL_OFFSET]] : index
  // CHECK:        }
  // CHECK:        [[SLICE0:%.*]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:        [[CAST0:%.*]] = tensor.cast [[SLICE0]] : tensor<1x16x720x100xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[SLICE1:%.*]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:        [[CAST1:%.*]] = tensor.cast [[SLICE1]] : tensor<1x16x720x100xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[CALL:%.*]] = func.call @main_func0([[CAST0]], [[CAST1]]) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[CAST2:%.*]] = tensor.cast [[CALL]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED:%.*]] = tensor.insert_slice [[CAST2]] into [[ARG3]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }
  // CHECK:      return [[FOR_RESULT]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 720, 44)>

//CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (-d0 + 720, 44)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0 - 28)>

// CHECK-LABEL: @Add
module @Add {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "Add_3" friendlyName = "output" : tensor<1x16x720x1000xf16>
  }
  func.func private @main_func0(%arg0: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NCHW} : tensor<1x16x720x1000xf16, {order = #NHWC}> -> tensor<1x16x720x1000xf16>
    return %0 : tensor<1x16x720x1000xf16>
  }
  func.func private @main_func1(%arg0: tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
    return %0 : tensor<1x16x?x1000xf16, {order = #NHWC}>
  }
  func.func private @main_func2(%arg0: tensor<1x16x?x1000xf16>, %arg1: tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg1, %0) {is_inplace = true, tilingStrategy = [1, 1, 4, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x?x1000xf16, {order = #NHWC}>
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

  // CHECK:    func.func private @main_func0([[ARG0:%.*]]: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16> {
  // CHECK:      [[LAYOUTCAST:%.*]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NCHW} : tensor<1x16x720x1000xf16, {order = #NHWC}> -> tensor<1x16x720x1000xf16>
  // CHECK:      return [[LAYOUTCAST]] : tensor<1x16x720x1000xf16>
  // CHECK:    }

  // CHECK:    func.func private @main_func1([[ARG0:%.*]]: tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
  // CHECK:      [[LAYOUTCAST:%.*]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:      return [[LAYOUTCAST]] : tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @main_func2([[ARG0:%.*]]: tensor<1x16x?x1000xf16>, [[ARG1:%.*]]: tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}> {
  // CHECK:      [[LAYOUTCAST:%.*]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NHWC} : tensor<1x16x?x1000xf16> -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:      [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[ARG1]], [[LAYOUTCAST]]) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:      return [[ELTWISE]] : tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @main([[ARG0:%.*]]: tensor<1x16x720x1000xf16>, [[ARG1:%.*]]: tensor<1x16x720x1000xf16>) -> tensor<1x16x720x1000xf16> {
  // CHECK:      [[C44:%.*]] = arith.constant 44 : index
  // CHECK:      [[C720:%.*]] = arith.constant 720 : index
  // CHECK:      [[C0:%.*]] = arith.constant 0 : index
  // CHECK:      [[EMPTY:%.*]] = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:      [[FOR_RESULT:%.*]] = scf.for [[ARG2:%.*]] = [[C0]] to [[C720]] step [[C44]] iter_args([[ARG3:%.*]] = [[EMPTY]]) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
  // CHECK:        [[MIN_SIZE:%.*]] = affine.min #[[$MAP]]([[ARG2]])
  // CHECK:        [[IS_FIRST:%.*]] = arith.cmpi eq, [[ARG2]], [[C0]] : index
  // CHECK:        [[OFFSET:%.*]] = scf.if [[IS_FIRST]] -> (index) {
  // CHECK:          scf.yield [[ARG2]] : index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS:%.*]] = arith.addi [[ARG2]], [[MIN_SIZE]] : index
  // CHECK:          [[IN_BOUNDS:%.*]] = arith.cmpi slt, [[NEXT_POS]], [[C720]] : index
  // CHECK:          [[FINAL_OFFSET:%.*]] = scf.if [[IN_BOUNDS]] -> (index) {
  // CHECK:            scf.yield [[ARG2]] : index
  // CHECK:          } else {
  // CHECK:            [[IS_EXACT:%.*]] = arith.cmpi eq, [[MIN_SIZE]], [[C44]] : index
  // CHECK:            [[BACKTRACK_OFFSET:%.*]] = scf.if [[IS_EXACT]] -> (index) {
  // CHECK:              scf.yield [[ARG2]] : index
  // CHECK:            } else {
  // CHECK:              [[REMAINING:%.*]] = affine.apply #[[$MAP1]]([[ARG2]])
  // CHECK:              scf.yield [[REMAINING]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[BACKTRACK_OFFSET]] : index
  // CHECK:          }
  // CHECK:          scf.yield [[FINAL_OFFSET]] : index
  // CHECK:        }
  // CHECK:        [[SLICE0:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[OFFSET]], 0] [1, 16, 44, 1000] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x1000xf16> to tensor<1x16x44x1000xf16>
  // CHECK:        [[CAST0:%.*]] = tensor.cast [[SLICE0]] : tensor<1x16x44x1000xf16> to tensor<1x16x?x1000xf16>
  // CHECK:        [[CALL1:%.*]] = func.call @main_func1([[CAST0]]) : (tensor<1x16x?x1000xf16>) -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:        [[SLICE1:%.*]] = tensor.extract_slice [[ARG1]][0, 0, [[OFFSET]], 0] [1, 16, 44, 1000] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x720x1000xf16> to tensor<1x16x44x1000xf16>
  // CHECK:        [[CAST1:%.*]] = tensor.cast [[SLICE1]] : tensor<1x16x44x1000xf16> to tensor<1x16x?x1000xf16>
  // CHECK:        [[CALL2:%.*]] = func.call @main_func2([[CAST1]], [[CALL1]]) : (tensor<1x16x?x1000xf16>, tensor<1x16x?x1000xf16, {order = #NHWC}>) -> tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:        [[CAST2:%.*]] = tensor.cast [[CALL2]] : tensor<1x16x?x1000xf16, {order = #NHWC}> to tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:        [[INSERTED:%.*]] = tensor.insert_slice [[CAST2]] into [[ARG3]][0, 0, [[OFFSET]], 0] [1, 16, 44, 1000] [1, 1, 1, 1]
  // CHECK-SAME:     tensor<1x16x44x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:      }
  // CHECK:      [[FINAL_CALL:%.*]] = call @main_func0([[FOR_RESULT]]) : (tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16>
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

//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 257)>

// CHECK-LABEL: @ApplyTilingNCEConvDyn
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn inputsInfo : {
    DataInfo "input" : tensor<1x32x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x64xf16>
  }
  func.func private @ApplyTilingNCEConvDyn_func0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @ApplyTilingNCEConvDyn(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim_0 step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg1)[%dim_0]
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

  // CHECK:    func.func private @ApplyTilingNCEConvDyn_func0_dims_H_cases_2([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn_func0_dims_H_cases_1([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn_func0_dims_H_cases_0([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @ApplyTilingNCEConvDyn([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[FALSE:%.*]] = arith.constant false
  // CHECK:      [[C257:%.*]] = arith.constant 257 : index
  // CHECK:      [[C3:%.*]] = arith.constant 3 : index
  // CHECK:      [[C1:%.*]] = arith.constant 1 : index
  // CHECK:      [[C256:%.*]] = arith.constant 256 : index
  // CHECK:      [[C0:%.*]] = arith.constant 0 : index
  // CHECK:      [[C2:%.*]] = arith.constant 2 : index
  // CHECK:      [[DIM:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[EMPTY:%.*]] = tensor.empty([[DIM]]) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH:%.*]] = arith.cmpi sge, [[DIM]], [[C256]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[DIM_0:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[REMUI:%.*]] = arith.remui [[DIM_0]], [[C256]] : index
  // CHECK:      [[DIM_1:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH_INPUT:%.*]] = arith.cmpi sge, [[DIM_1]], [[C257]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_INPUT]], "Not enough elements to backtrack in scf.for loop for Input tensor"
  // CHECK:      [[REMUI_1:%.*]] = arith.remui [[DIM_1]], [[C257]] : index
  // CHECK:      [[FOR_RESULT:%.*]] = scf.for [[ARG1:%.*]] = [[C0]] to [[DIM_0]] step [[C256]] iter_args([[ARG2:%.*]] = [[EMPTY]]) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[MIN_SIZE:%.*]] = affine.min #[[$MAP]]([[ARG1]])[[[DIM_0]]]
  // CHECK:        [[MAX_POS:%.*]] = affine.max #[[$MAP1]]([[ARG1]])
  // CHECK:        [[MAX_H:%.*]] = affine.max #[[$MAP2]]([[ARG1]])
  // CHECK:        [[MIN_H:%.*]] = affine.min #[[$MAP3]]()[[[MAX_H]]]
  // CHECK:        [[MAX_SIZE:%.*]] = affine.max #[[$MAP4]]([[MIN_SIZE]], [[MAX_POS]])
  // CHECK:        [[MIN_SIZE_2:%.*]] = affine.min #[[$MAP3]]()[[[MAX_SIZE]]]
  // CHECK:        [[APPLY_SIZE:%.*]] = affine.apply #[[$MAP5]]([[MIN_SIZE]], [[MIN_H]], [[MIN_SIZE_2]])
  // CHECK:        [[IS_FIRST:%.*]] = arith.cmpi eq, [[ARG1]], [[C0]] : index
  // CHECK:        [[OFFSET:%.*]] = scf.if [[IS_FIRST]] -> (index) {
  // CHECK:          scf.yield [[ARG1]] : index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS:%.*]] = arith.addi [[ARG1]], [[MIN_SIZE]] : index
  // CHECK:          [[IN_BOUNDS:%.*]] = arith.cmpi slt, [[NEXT_POS]], [[DIM_0]] : index
  // CHECK:          [[FINAL_OFFSET:%.*]] = scf.if [[IN_BOUNDS]] -> (index) {
  // CHECK:            scf.yield [[ARG1]] : index
  // CHECK:          } else {
  // CHECK:            [[IS_EXACT:%.*]] = arith.cmpi eq, [[MIN_SIZE]], [[C256]] : index
  // CHECK:            [[BACKTRACK_OFFSET:%.*]] = scf.if [[IS_EXACT]] -> (index) {
  // CHECK:              scf.yield [[ARG1]] : index
  // CHECK:            } else {
  // CHECK:              [[REMAINING:%.*]] = affine.apply #[[$MAP6]]([[ARG1]])[[[REMUI]]]
  // CHECK:              scf.yield [[REMAINING]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[BACKTRACK_OFFSET]] : index
  // CHECK:          }
  // CHECK:          scf.yield [[FINAL_OFFSET]] : index
  // CHECK:        }
  // CHECK:        [[IS_ZERO_POS:%.*]] = arith.cmpi eq, [[MAX_POS]], [[C0]] : index
  // CHECK:        [[CASE_OFFSET:%.*]]:2 = scf.if [[IS_ZERO_POS]] -> (index, index) {
  // CHECK:          [[IS_EXACT_CASE:%.*]] = arith.cmpi eq, [[APPLY_SIZE]], [[DIM_1]] : index
  // CHECK:          [[CASE:%.*]] = arith.select [[IS_EXACT_CASE]], [[C3]], [[C2]] : index
  // CHECK:          scf.yield [[CASE]], [[MAX_POS]] : index, index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS_CASE:%.*]] = arith.addi [[MAX_POS]], [[APPLY_SIZE]] : index
  // CHECK:          [[IN_BOUNDS_CASE:%.*]] = arith.cmpi slt, [[NEXT_POS_CASE]], [[DIM_1]] : index
  // CHECK:          [[CASE_ALT:%.*]] = arith.select [[IN_BOUNDS_CASE]], [[C0]], [[C1]] : index
  // CHECK:          [[OFFSET_CASE:%.*]] = scf.if [[IN_BOUNDS_CASE]] -> (index) {
  // CHECK:            scf.yield [[MAX_POS]] : index
  // CHECK:          } else {
  // CHECK:            [[IS_EXACT_ALT:%.*]] = arith.cmpi eq, [[APPLY_SIZE]], [[C257]] : index
  // CHECK:            [[BACKTRACK_OFFSET_ALT:%.*]] = scf.if [[IS_EXACT_ALT]] -> (index) {
  // CHECK:              scf.yield [[MAX_POS]] : index
  // CHECK:            } else {
  // CHECK:              [[REMAINING_ALT:%.*]] = affine.apply #[[$MAP7]]([[MAX_POS]])[[[REMUI_1]]]
  // CHECK:              scf.yield [[REMAINING_ALT]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[BACKTRACK_OFFSET_ALT]] : index
  // CHECK:          }
  // CHECK:          scf.yield [[CASE_ALT]], [[OFFSET_CASE]] : index, index
  // CHECK:        }
  // CHECK:        [[SWITCH_RESULT:%.*]] = scf.index_switch [[CASE_OFFSET:%.*]]#0 -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          case 0 {
  // CHECK:            [[SLICE0:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 258, 64] [1, 1, 1, 1] :
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST0:%.*]] = tensor.cast [[SLICE0]] : tensor<1x32x258x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL0:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0([[CAST0]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL0]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 1 {
  // CHECK:            [[SLICE1:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 257, 64] [1, 1, 1, 1] :
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST1:%.*]] = tensor.cast [[SLICE1]] : tensor<1x32x257x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL1:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_1([[CAST1]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL1]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 2 {
  // CHECK:            [[SLICE2:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 257, 64] [1, 1, 1, 1] :
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST2:%.*]] = tensor.cast [[SLICE2]] : tensor<1x32x257x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL2:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_2([[CAST2]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL2]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          default {
  // CHECK:            cf.assert [[FALSE]], "Unsupported case"
  // CHECK:            [[SLICE_DEF:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 258, 64] [1, 1, 1, 1] :
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST_DEF:%.*]] = tensor.cast [[SLICE_DEF]] : tensor<1x32x258x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_DEF:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0([[CAST_DEF]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_DEF]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:        [[CAST_2:%.*]] = tensor.cast [[SWITCH_RESULT]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED:%.*]] = tensor.insert_slice [[CAST_2]] into [[ARG2:%.*]][0, 0, [[OFF:%.*]], 0] [1, 256, 256, 64] [1, 1, 1, 1] : tensor<1x256x256x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
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

//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP4:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 638)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 160)>
//CHECK: #[[$MAP9:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
//CHECK: #[[$MAP10:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 161)>
//CHECK: #[[$MAP11:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 257)>

// CHECK-LABEL: @ApplyTilingNCEConvDyn2D
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn2D inputsInfo : {
    DataInfo "input" : tensor<1x32x?x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x?xf16>
  }
  func.func private @ApplyTilingNCEConvDyn2D_func0(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
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

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
  // CHECK-SAME:   tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @ApplyTilingNCEConvDyn2D([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[FALSE:%.*]] = arith.constant false
  // CHECK:      [[C161:%.*]] = arith.constant 161 : index
  // CHECK:      [[C257:%.*]] = arith.constant 257 : index
  // CHECK:      [[C1:%.*]] = arith.constant 1 : index
  // CHECK:      [[C160:%.*]] = arith.constant 160 : index
  // CHECK:      [[C256:%.*]] = arith.constant 256 : index
  // CHECK:      [[C3:%.*]] = arith.constant 3 : index
  // CHECK:      [[C0:%.*]] = arith.constant 0 : index
  // CHECK:      [[C2:%.*]] = arith.constant 2 : index
  // CHECK:      [[DIM:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[DIM_0:%.*]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[EMPTY:%.*]] = tensor.empty([[DIM]], [[DIM_0]]) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH_0:%.*]] = arith.cmpi sge, [[DIM_0]], [[C160]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_0]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[HAS_ENOUGH_1:%.*]] = arith.cmpi sge, [[DIM]], [[C256]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_1]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[REMUI:%.*]] = arith.remui [[DIM]], [[C256]] : index
  // CHECK:      [[REMUI_1:%.*]] = arith.remui [[DIM_0]], [[C160]] : index
  // CHECK:      [[DIM_3:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH_INPUT:%.*]] = arith.cmpi sge, [[DIM_3]], [[C257]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_INPUT]], "Not enough elements to backtrack in scf.for loop for Input tensor"
  // CHECK:      [[REMUI_2:%.*]] = arith.remui [[DIM_3]], [[C257]] : index
  // CHECK:      [[DIM_4:%.*]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH_INPUT_1:%.*]] = arith.cmpi sge, [[DIM_4]], [[C161]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_INPUT_1]], "Not enough elements to backtrack in scf.for loop for Input tensor"
  // CHECK:      [[REMUI_3:%.*]] = arith.remui [[DIM_4]], [[C161]] : index
  // CHECK:      [[FOR_RESULT:%.*]] = scf.for [[ARG1:%.*]] = [[C0]] to [[DIM]] step [[C256]] iter_args([[ARG2:%.*]] = [[EMPTY]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[FOR_RESULT_1:%.*]] = scf.for [[ARG3:%.*]] = [[C0]] to [[DIM_0]] step [[C160]] iter_args([[ARG4:%.*]] = [[ARG2]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:          [[MIN_SIZE:%.*]] = affine.min #[[$MAP]]([[ARG1]])[[[DIM]]]
  // CHECK:          [[MIN_SIZE_1:%.*]] = affine.min #[[$MAP1]]([[ARG3]])[[[DIM_0]]]
  // CHECK:          [[MAX_POS:%.*]] = affine.max #[[$MAP2]]([[ARG1]])
  // CHECK:          [[MAX_POS_1:%.*]] = affine.max #[[$MAP3]]([[ARG1]])
  // CHECK:          [[MIN_SIZE_2:%.*]] = affine.min #[[$MAP4]]()[[[MAX_POS_1]]]
  // CHECK:          [[MAX_SIZE:%.*]] = affine.max #[[$MAP5]]([[MIN_SIZE]], [[MAX_POS]])
  // CHECK:          [[MIN_SIZE_3:%.*]] = affine.min #[[$MAP4]]()[[[MAX_SIZE]]]
  // CHECK:          [[APPLY_SIZE:%.*]] = affine.apply #[[$MAP6]]([[MIN_SIZE]], [[MIN_SIZE_2]], [[MIN_SIZE_3]])
  // CHECK:          [[MAX_POS_2:%.*]] = affine.max #[[$MAP2]]([[ARG3]])
  // CHECK:          [[MAX_POS_3:%.*]] = affine.max #[[$MAP3]]([[ARG3]])
  // CHECK:          [[MIN_SIZE_4:%.*]] = affine.min #[[$MAP4]]()[[[MAX_POS_3]]]
  // CHECK:          [[MAX_SIZE_1:%.*]] = affine.max #[[$MAP7]]([[MIN_SIZE_1]], [[MAX_POS_2]])
  // CHECK:          [[MIN_SIZE_5:%.*]] = affine.min #[[$MAP4]]()[[[MAX_SIZE_1]]]
  // CHECK:          [[APPLY_SIZE_1:%.*]] = affine.apply #[[$MAP6]]([[MIN_SIZE_1]], [[MIN_SIZE_4]], [[MIN_SIZE_5]])
  // CHECK:          [[IS_FIRST:%.*]] = arith.cmpi eq, [[ARG1]], [[C0]] : index
  // CHECK:          [[IS_FIRST_1:%.*]] = arith.cmpi eq, [[ARG3]], [[C0]] : index
  // CHECK:          [[OFFSET:%.*]] = scf.if [[IS_FIRST_1]] -> (index) {
  // CHECK:            scf.yield [[ARG3]] : index
  // CHECK:          } else {
  // CHECK:            [[NEXT_POS:%.*]] = arith.addi [[ARG3]], [[MIN_SIZE_1]] : index
  // CHECK:            [[IN_BOUNDS:%.*]] = arith.cmpi slt, [[NEXT_POS]], [[DIM_0]] : index
  // CHECK:            [[FINAL_OFFSET:%.*]] = scf.if [[IN_BOUNDS]] -> (index) {
  // CHECK:              scf.yield [[ARG3]] : index
  // CHECK:            } else {
  // CHECK:              [[IS_EXACT:%.*]] = arith.cmpi eq, [[MIN_SIZE_1]], [[C160]] : index
  // CHECK:              [[BACKTRACK_OFFSET:%.*]] = scf.if [[IS_EXACT]] -> (index) {
  // CHECK:                scf.yield [[ARG3]] : index
  // CHECK:              } else {
  // CHECK:                [[REMAINING:%.*]] = affine.apply #[[$MAP8]]([[ARG3]])[[[REMUI_1]]]
  // CHECK:                scf.yield [[REMAINING]] : index
  // CHECK:              }
  // CHECK:              scf.yield [[BACKTRACK_OFFSET]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[FINAL_OFFSET]] : index
  // CHECK:          }
  // CHECK:          [[OFFSET_1:%.*]] = scf.if [[IS_FIRST]] -> (index) {
  // CHECK:            scf.yield [[ARG1]] : index
  // CHECK:          } else {
  // CHECK:            [[NEXT_POS_1:%.*]] = arith.addi [[ARG1]], [[MIN_SIZE]] : index
  // CHECK:            [[IN_BOUNDS_1:%.*]] = arith.cmpi slt, [[NEXT_POS_1]], [[DIM]] : index
  // CHECK:            [[FINAL_OFFSET_1:%.*]] = scf.if [[IN_BOUNDS_1]] -> (index) {
  // CHECK:              scf.yield [[ARG1]] : index
  // CHECK:            } else {
  // CHECK:              [[IS_EXACT_1:%.*]] = arith.cmpi eq, [[MIN_SIZE]], [[C256]] : index
  // CHECK:              [[BACKTRACK_OFFSET_1:%.*]] = scf.if [[IS_EXACT_1]] -> (index) {
  // CHECK:                scf.yield [[ARG1]] : index
  // CHECK:              } else {
  // CHECK:                [[REMAINING_1:%.*]] = affine.apply #[[$MAP9]]([[ARG1]])[[[REMUI]]]
  // CHECK:                scf.yield [[REMAINING_1]] : index
  // CHECK:              }
  // CHECK:              scf.yield [[BACKTRACK_OFFSET_1]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[FINAL_OFFSET_1]] : index
  // CHECK:          }
  // CHECK:          [[IS_ZERO_POS:%.*]] = arith.cmpi eq, [[MAX_POS]], [[C0]] : index
  // CHECK:          [[IS_ZERO_POS_1:%.*]] = arith.cmpi eq, [[MAX_POS_2]], [[C0]] : index
  // CHECK:          [[CASE_OFFSET:%.*]]:2 = scf.if [[IS_ZERO_POS_1]] -> (index, index) {
  // CHECK:            [[IS_EXACT_2:%.*]] = arith.cmpi eq, [[APPLY_SIZE_1]], [[DIM_4]] : index
  // CHECK:            [[CASE:%.*]] = arith.select [[IS_EXACT_2]], [[C3]], [[C2]] : index
  // CHECK:            scf.yield [[CASE]], [[MAX_POS_2]] : index, index
  // CHECK:          } else {
  // CHECK:            [[NEXT_POS_2:%.*]] = arith.addi [[MAX_POS_2]], [[APPLY_SIZE_1]] : index
  // CHECK:            [[IN_BOUNDS_2:%.*]] = arith.cmpi slt, [[NEXT_POS_2]], [[DIM_4]] : index
  // CHECK:            [[CASE_ALT:%.*]] = arith.select [[IN_BOUNDS_2]], [[C0]], [[C1]] : index
  // CHECK:            [[OFFSET_CASE:%.*]] = scf.if [[IN_BOUNDS_2]] -> (index) {
  // CHECK:              scf.yield [[MAX_POS_2]] : index
  // CHECK:            } else {
  // CHECK:              [[IS_EXACT_ALT:%.*]] = arith.cmpi eq, [[APPLY_SIZE_1]], [[C161]] : index
  // CHECK:              [[BACKTRACK_OFFSET_ALT:%.*]] = scf.if [[IS_EXACT_ALT]] -> (index) {
  // CHECK:                scf.yield [[MAX_POS_2]] : index
  // CHECK:              } else {
  // CHECK:                [[REMAINING_ALT:%.*]] = affine.apply #[[$MAP10]]([[MAX_POS_2]])[[[REMUI_3]]]
  // CHECK:                scf.yield [[REMAINING_ALT]] : index
  // CHECK:              }
  // CHECK:              scf.yield [[BACKTRACK_OFFSET_ALT]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[CASE_ALT]], [[OFFSET_CASE]] : index, index
  // CHECK:          }
  // CHECK:          [[CASE_OFFSET_1:%.*]]:2 = scf.if [[IS_ZERO_POS]] -> (index, index) {
  // CHECK:            [[IS_EXACT_3:%.*]] = arith.cmpi eq, [[APPLY_SIZE]], [[DIM_3]] : index
  // CHECK:            [[CASE_1:%.*]] = arith.select [[IS_EXACT_3]], [[C3]], [[C2]] : index
  // CHECK:            scf.yield [[CASE_1]], [[MAX_POS]] : index, index
  // CHECK:          } else {
  // CHECK:            [[NEXT_POS_3:%.*]] = arith.addi [[MAX_POS]], [[APPLY_SIZE]] : index
  // CHECK:            [[IN_BOUNDS_3:%.*]] = arith.cmpi slt, [[NEXT_POS_3]], [[DIM_3]] : index
  // CHECK:            [[CASE_ALT_1:%.*]] = arith.select [[IN_BOUNDS_3]], [[C0]], [[C1]] : index
  // CHECK:            [[OFFSET_CASE_1:%.*]] = scf.if [[IN_BOUNDS_3]] -> (index) {
  // CHECK:              scf.yield [[MAX_POS]] : index
  // CHECK:            } else {
  // CHECK:              [[IS_EXACT_ALT_1:%.*]] = arith.cmpi eq, [[APPLY_SIZE]], [[C257]] : index
  // CHECK:              [[BACKTRACK_OFFSET_ALT_1:%.*]] = scf.if [[IS_EXACT_ALT_1]] -> (index) {
  // CHECK:                scf.yield [[MAX_POS]] : index
  // CHECK:              } else {
  // CHECK:                [[REMAINING_ALT_1:%.*]] = affine.apply #[[$MAP11]]([[MAX_POS]])[[[REMUI_2]]]
  // CHECK:                scf.yield [[REMAINING_ALT_1]] : index
  // CHECK:              }
  // CHECK:              scf.yield [[BACKTRACK_OFFSET_ALT_1]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[CASE_ALT_1]], [[OFFSET_CASE_1]] : index, index
  // CHECK:          }
  // CHECK:          [[SHLI:%.*]] = arith.shli [[CASE_OFFSET_1]]#0, [[C2]] : index
  // CHECK:          [[ORI:%.*]] = arith.ori [[SHLI]], [[CASE_OFFSET]]#0 : index
  // CHECK:          [[SWITCH_RESULT_1:%.*]] = scf.index_switch [[ORI]] -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          case 0 {
  // CHECK:            [[SLICE_00:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 258, 162] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:            [[CAST_00:%.*]] = tensor.cast [[SLICE_00]] : tensor<1x32x258x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_00:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00([[CAST_00]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_00]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 1 {
  // CHECK:            [[SLICE_01:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 258, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_01:%.*]] = tensor.cast [[SLICE_01]] : tensor<1x32x258x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_01:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01([[CAST_01]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_01]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 2 {
  // CHECK:            [[SLICE_02:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 258, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_02:%.*]] = tensor.cast [[SLICE_02]] : tensor<1x32x258x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_02:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02([[CAST_02]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_02]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 4 {
  // CHECK:            [[SLICE_10:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 257, 162] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:            [[CAST_10:%.*]] = tensor.cast [[SLICE_10]] : tensor<1x32x257x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_10:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10([[CAST_10]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_10]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 5 {
  // CHECK:            [[SLICE_11:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_11:%.*]] = tensor.cast [[SLICE_11]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_11:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11([[CAST_11]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_11]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 6 {
  // CHECK:            [[SLICE_12:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_12:%.*]] = tensor.cast [[SLICE_12]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_12:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12([[CAST_12]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_12]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 8 {
  // CHECK:            [[SLICE_20:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 257, 162] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:            [[CAST_20:%.*]] = tensor.cast [[SLICE_20]] : tensor<1x32x257x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_20:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20([[CAST_20]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_20]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 9 {
  // CHECK:            [[SLICE_21:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_21:%.*]] = tensor.cast [[SLICE_21]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_21:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21([[CAST_21]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_21]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 10 {
  // CHECK:            [[SLICE_22:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 257, 161] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:            [[CAST_22:%.*]] = tensor.cast [[SLICE_22]] : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_22:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22([[CAST_22]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_22]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          default {
  // CHECK:            cf.assert [[FALSE]], "Unsupported case"
  // CHECK:            [[SLICE_DEF:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET_1]]#1, [[CASE_OFFSET]]#1] [1, 32, 258, 162] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:            [[CAST_DEF:%.*]] = tensor.cast [[SLICE_DEF]] : tensor<1x32x258x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_DEF:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00([[CAST_DEF]]) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_DEF]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:        [[CAST_3:%.*]] = tensor.cast [[SWITCH_RESULT_1]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x160xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED_1:%.*]] = tensor.insert_slice [[CAST_3]] into [[ARG4]][0, 0, [[OFFSET_1]], [[OFFSET]]] [1, 256, 256, 160] [1, 1, 1, 1] : tensor<1x256x256x160xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
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

//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (d0 * 2)>
//CHECK: #[[$MAP3:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 512)>

// CHECK-LABEL: @ConvWithStrides2x2
module {
  net.NetworkInfo entryPoint : @ConvWithStrides2x2 inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "/conv_final/Conv" friendlyName = "EditOpenVinoIRResult_48" tensorNames = ["/conv_final/Conv_output_0"] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func private @ConvWithStrides2x2_func0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [2, 2]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
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

  // CHECK:    func.func private @ConvWithStrides2x2_func0_dims_H_cases_2([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [2, 2]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func private @ConvWithStrides2x2_func0_dims_H_cases_1([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:      [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [2, 2]}
  // CHECK-SAME:   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @ConvWithStrides2x2([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[FALSE:%.*]] = arith.constant false
  // CHECK:      [[C512:%.*]] = arith.constant 512 : index
  // CHECK:      [[C3:%.*]] = arith.constant 3 : index
  // CHECK:      [[C256:%.*]] = arith.constant 256 : index
  // CHECK:      [[C0:%.*]] = arith.constant 0 : index
  // CHECK:      [[C1:%.*]] = arith.constant 1 : index
  // CHECK:      [[C2:%.*]] = arith.constant 2 : index
  // CHECK:      [[DIM:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[ADDI:%.*]] = arith.addi [[DIM]], [[C1]] : index
  // CHECK:      [[DIVSI:%.*]] = arith.divsi [[ADDI]], [[C2]] : index
  // CHECK:      [[EMPTY:%.*]] = tensor.empty([[DIVSI]]) : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH:%.*]] = arith.cmpi sge, [[DIVSI]], [[C256]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH]], "Not enough elements to backtrack in scf.for loop for Output tensor"
  // CHECK:      [[DIM_0:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[ADDI_1:%.*]] = arith.addi [[DIM_0]], [[C1]] : index
  // CHECK:      [[DIVSI_1:%.*]] = arith.divsi [[ADDI_1]], [[C2]] : index
  // CHECK:      [[REMUI:%.*]] = arith.remui [[DIVSI_1]], [[C256]] : index
  // CHECK:      [[DIM_1:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[HAS_ENOUGH_INPUT:%.*]] = arith.cmpi sge, [[DIM_1]], [[C512]] : index
  // CHECK:      cf.assert [[HAS_ENOUGH_INPUT]], "Not enough elements to backtrack in scf.for loop for Input tensor"
  // CHECK:      [[REMUI_1:%.*]] = arith.remui [[DIM_1]], [[C512]] : index
  // CHECK:      [[FOR_RESULT:%.*]] = scf.for [[ARG1:%.*]] = [[C0]] to [[DIVSI_1]] step [[C256]] iter_args([[ARG2:%.*]] = [[EMPTY]]) -> (tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:        [[MIN_SIZE:%.*]] = affine.min #[[$MAP]]([[ARG1]])[[[DIVSI_1]]]
  // CHECK:        [[MAX_POS:%.*]] = affine.max #[[$MAP1]]([[ARG1]])
  // CHECK:        [[APPLY_SIZE:%.*]] = affine.apply #[[$MAP2]]([[MIN_SIZE]])
  // CHECK:        [[IS_FIRST:%.*]] = arith.cmpi eq, [[ARG1]], [[C0]] : index
  // CHECK:        [[OFFSET:%.*]] = scf.if [[IS_FIRST]] -> (index) {
  // CHECK:          scf.yield [[ARG1]] : index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS:%.*]] = arith.addi [[ARG1]], [[MIN_SIZE]] : index
  // CHECK:          [[IN_BOUNDS:%.*]] = arith.cmpi slt, [[NEXT_POS]], [[DIVSI_1]] : index
  // CHECK:          [[FINAL_OFFSET:%.*]] = scf.if [[IN_BOUNDS]] -> (index) {
  // CHECK:            scf.yield [[ARG1]] : index
  // CHECK:          } else {
  // CHECK:            [[IS_EXACT:%.*]] = arith.cmpi eq, [[MIN_SIZE]], [[C256]] : index
  // CHECK:            [[BACKTRACK_OFFSET:%.*]] = scf.if [[IS_EXACT]] -> (index) {
  // CHECK:              scf.yield [[ARG1]] : index
  // CHECK:            } else {
  // CHECK:              [[REMAINING:%.*]] = affine.apply #[[$MAP3]]([[ARG1]])[[[REMUI]]]
  // CHECK:              scf.yield [[REMAINING]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[BACKTRACK_OFFSET]] : index
  // CHECK:          }
  // CHECK:          scf.yield [[FINAL_OFFSET]] : index
  // CHECK:        }
  // CHECK:        [[IS_ZERO_POS:%.*]] = arith.cmpi eq, [[MAX_POS]], [[C0]] : index
  // CHECK:        [[CASE_OFFSET:%.*]]:2 = scf.if [[IS_ZERO_POS]] -> (index, index) {
  // CHECK:          [[IS_EXACT_CASE:%.*]] = arith.cmpi eq, [[APPLY_SIZE]], [[DIM_1]] : index
  // CHECK:          [[CASE:%.*]] = arith.select [[IS_EXACT_CASE]], [[C3]], [[C2]] : index
  // CHECK:          scf.yield [[CASE]], [[MAX_POS]] : index, index
  // CHECK:        } else {
  // CHECK:          [[NEXT_POS_CASE:%.*]] = arith.addi [[MAX_POS]], [[APPLY_SIZE]] : index
  // CHECK:          [[IN_BOUNDS_CASE:%.*]] = arith.cmpi slt, [[NEXT_POS_CASE]], [[DIM_1]] : index
  // CHECK:          [[CASE_ALT:%.*]] = arith.select [[IN_BOUNDS_CASE]], [[C0]], [[C1]] : index
  // CHECK:          [[OFFSET_CASE:%.*]] = scf.if [[IN_BOUNDS_CASE]] -> (index) {
  // CHECK:            scf.yield [[MAX_POS]] : index
  // CHECK:          } else {
  // CHECK:            [[IS_EXACT_ALT:%.*]] = arith.cmpi eq, [[APPLY_SIZE]], [[C512]] : index
  // CHECK:            [[BACKTRACK_OFFSET_ALT:%.*]] = scf.if [[IS_EXACT_ALT]] -> (index) {
  // CHECK:              scf.yield [[MAX_POS]] : index
  // CHECK:            } else {
  // CHECK:              [[REMAINING_ALT:%.*]] = affine.apply #[[$MAP4]]([[MAX_POS]])[[[REMUI_1]]]
  // CHECK:              scf.yield [[REMAINING_ALT]] : index
  // CHECK:            }
  // CHECK:            scf.yield [[BACKTRACK_OFFSET_ALT]] : index
  // CHECK:          }
  // CHECK:          scf.yield [[CASE_ALT]], [[OFFSET_CASE]] : index, index
  // CHECK:        }
  // CHECK:        [[SWITCH_RESULT:%.*]] = scf.index_switch [[CASE_OFFSET]]#0 -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          case 1 {
  // CHECK:            [[SLICE1:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 512, 64] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x512x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST1:%.*]] = tensor.cast [[SLICE1]] : tensor<1x32x512x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL1:%.*]] = func.call @ConvWithStrides2x2_func0_dims_H_cases_1([[CAST1]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL1]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          case 2 {
  // CHECK:            [[SLICE2:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 512, 64] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x512x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST2:%.*]] = tensor.cast [[SLICE2]] : tensor<1x32x512x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL2:%.*]] = func.call @ConvWithStrides2x2_func0_dims_H_cases_2([[CAST2]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL2]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:          default {
  // CHECK:            cf.assert [[FALSE]], "Unsupported case"
  // CHECK:            [[SLICE_DEF:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[CASE_OFFSET]]#1, 0] [1, 32, 512, 64] [1, 1, 1, 1]
  // CHECK-SAME:       tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x512x64xf16, {order = #NHWC}>
  // CHECK:            [[CAST_DEF:%.*]] = tensor.cast [[SLICE_DEF]] : tensor<1x32x512x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            [[CALL_DEF:%.*]] = func.call @ConvWithStrides2x2_func0_dims_H_cases_1([[CAST_DEF]]) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:            scf.yield [[CALL_DEF]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:          }
  // CHECK:        [[CAST_FINAL:%.*]] = tensor.cast [[SWITCH_RESULT]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 32]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        [[INSERTED:%.*]] = tensor.insert_slice [[CAST_FINAL]] into [[ARG2]][0, 0, [[OFFSET]], 0] [1, 256, 256, 32] [1, 1, 1, 1] : tensor<1x256x256x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:        scf.yield [[INSERTED]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      }
  // CHECK:      return [[FOR_RESULT]] : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 32]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:    }
}
