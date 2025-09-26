//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-dynamic-to-static-kernels  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
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
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %dim_0 step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim_0]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:  func.func @main_func0_static([[ARGS0:%.*]]: tensor<1x16x720x100xf16, {order = #NHWC}>, [[ARGS1:%.*]]: tensor<1x16x720x100xf16, {order = #NHWC}>) -> tensor<1x16x720x100xf16, {order = #NHWC}> {
  // CHECK:   [[ADDRESULT:%.*]] = VPU.NCE.Eltwise([[ARGS0]], [[ARGS1]])
  // CHECK:   return [[ADDRESULT]] : tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:  }
  // CHECK:  func.func @main([[ARG0:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {

  // CHECK:  [[MIN:%.*]] = affine.min #map(%arg2)[%dim_0]
  // CHECK:  [[COND:%.*]] = arith.cmpi ne, [[MIN]], %c100 : index
  // CHECK:  [[INDEX:%.*]] = scf.if [[COND]] -> (index) {
  // CHECK:  [[ELEMENTS:%.*]] = arith.subi %c100, [[MIN]] : index
  // CHECK:  [[CHECK:%.*]] = arith.cmpi sgt, %arg2, [[ELEMENTS]] : index
  // CHECK:  cf.assert [[CHECK]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:  [[FINAL_INDEX:%.*]] = arith.subi %arg2, [[ELEMENTS]] : index
  // CHECK:  scf.yield [[FINAL_INDEX]] : index
  // CHECK:  } else {
  // CHECK:  scf.yield %arg2 : index
  // CHECK:  }
  // CHECK:  [[SLICE0:%.*]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[INDEX]]] [1, 16, 720, %c100] [1, 1, 1, 1] :
  // CHECK:  [[IN0:%.*]] = tensor.cast [[SLICE0]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : {{.*}}}> to tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:  [[SLICE1:%.*]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[INDEX]]] [1, 16, 720, %c100] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : {{.*}}}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : {{.*}}}>
  // CHECK:  [[IN1:%.*]] = tensor.cast [[SLICE1]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : {{.*}}}> to tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:  [[CALL_OUTPUT:%.*]]  = func.call @main_func0_static([[IN0]], [[IN1]]) : (tensor<1x16x720x100xf16, {order = #NHWC}>, tensor<1x16x720x100xf16, {order = #NHWC}>) -> tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:  [[OUT:%.*]] = tensor.cast [[CALL_OUTPUT]] : tensor<1x16x720x100xf16, {order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : {{.*}}}>
  // CHECK:  [[INSERT_SLICE:%.*]]  = tensor.insert_slice [[OUT]] into %arg3[0, 0, 0, [[INDEX]]] [1, 16, 720, %c100] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : {{.*}}}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : {{.*}}}>
  // CHECK:  scf.yield [[INSERT_SLICE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @CopyInputOutput
module @CopyInputOutput {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x720x1000x16xf16>
    DataInfo "input2" : tensor<1x720x1000x16xf16>                                                                                                                                                                                                                                                                                                                          } outputsInfo : {
    DataInfo "output" : tensor<1x720x1000x16xf16>
  }
  func.func private @main_func0(%arg0: memref<1x90x1000x16xf16>, %arg1: memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x90x1000x16xf16>) outputs(%arg1 : memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
    return %0 : memref<1x90x1000x16xf16>
  }
  func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
    %c90 = arith.constant 90 : index
    %c720 = arith.constant 720 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x720x1000x16xf16>
    %0 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %alloc) -> (memref<1x720x1000x16xf16>) {
      %subview = memref.subview %arg0[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %1 = builtin.unrealized_conversion_cast %subview : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %2 = builtin.unrealized_conversion_cast %subview_0 : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %3 = func.call @main_func0(%1, %2) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
      %subview_1 = memref.subview %arg3[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      memref.copy %3, %subview_1 : memref<1x90x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      scf.yield %arg3 : memref<1x720x1000x16xf16>
    }
    return %0 : memref<1x720x1000x16xf16>
  }

  // CHECK-NOT: func.func @main_func0_static
  // CHECK-NOT: scf.if
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 720, 44)>
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

  // CHECK:  func.func private @main_func0([[ARG0:%.*]]: tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16> {
  // CHECK:       [[RESULT:%.*]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NCHW} : tensor<1x16x720x1000xf16, {order = #NHWC}> -> tensor<1x16x720x1000xf16>
  // CHECK:       return [[RESULT]] : tensor<1x16x720x1000xf16>
  // CHECK:  }

  // CHECK:  func.func @main_func1_static([[ARG0:%.*]]: tensor<1x16x44x1000xf16>) -> tensor<1x16x44x1000xf16, {order = #NHWC}> {
  // CHECK:       [[RESULT:%.*]] = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x44x1000xf16> -> tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:       return [[RESULT]] : tensor<1x16x44x1000xf16, {order = #NHWC}>

  // CHECK:  func.func @main_func2_static([[ARG0:%.*]]: tensor<1x16x44x1000xf16>, [[ARG1:%.*]]: tensor<1x16x44x1000xf16, {order = #NHWC}>) -> tensor<1x16x44x1000xf16, {order = #NHWC}> {
  // CHECK:       [[CAST:%.*]] = VPU.LayoutCast([[ARG0]]) {dst_order = #NHWC} : tensor<1x16x44x1000xf16> -> tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:       [[ADD_RESULT:%.*]] = VPU.NCE.Eltwise([[ARG1]], [[CAST]]) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:       return [[ADD_RESULT]] : tensor<1x16x44x1000xf16, {order = #NHWC}>

  // CHECK:  func.func @main([[ARG0:%.*]]: tensor<1x16x720x1000xf16>, [[ARG1:%.*]]: tensor<1x16x720x1000xf16>) -> tensor<1x16x720x1000xf16> {
  // CHECK:       [[C44:%.*]] = arith.constant 44 : index
  // CHECK:       [[C720:%.*]] = arith.constant 720 : index
  // CHECK:       [[C0:%.*]] = arith.constant 0 : index
  // CHECK:       [[EMPTY_BUFFER:%.*]] = tensor.empty() : tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_BUFFER:%.*]] = scf.for [[LOOP_ITER:%.*]] = [[C0]] to [[C720]] step [[C44]] iter_args([[ARG3:%.*]] = [[EMPTY_BUFFER]]) -> (tensor<1x16x720x1000xf16, {order = #NHWC}>) {
  // CHECK:           [[ORIG_OFFSET:%.*]] = affine.min #map([[LOOP_ITER]])
  // CHECK:           [[STEP_CHECK:%.*]] = arith.cmpi ne, [[ORIG_OFFSET]], [[C44]] : index
  // CHECK:           [[ADJ_OFFSET:%.*]] = scf.if [[STEP_CHECK]] -> (index) {
  // CHECK:               [[CUR_SIZE:%.*]] = arith.subi [[C44]], [[ORIG_OFFSET]] : index
  // CHECK:               [[BACKTRACK_CHECK:%.*]] = arith.cmpi sgt, [[LOOP_ITER]], [[CUR_SIZE]] : index
  // CHECK:               cf.assert [[BACKTRACK_CHECK]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:               [[OFFSET:%.*]] = arith.subi [[LOOP_ITER]], [[CUR_SIZE]] : index
  // CHECK:                   scf.yield [[OFFSET]] : index
  // CHECK:               } else {
  // CHECK:                   scf.yield [[LOOP_ITER]] : index
  // CHECK:               }
  // CHECK:           [[INPUT_SLICE:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[ADJ_OFFSET]], 0] [1, 16, [[C44]], 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16> to tensor<1x16x?x1000xf16>
  // CHECK:           [[STATIC_INPUT_SLICE:%.*]] = tensor.cast [[INPUT_SLICE]] : tensor<1x16x?x1000xf16> to tensor<1x16x44x1000xf16>
  // CHECK:           [[TMP_OUTPUT:%.*]] = func.call @main_func1_static([[STATIC_INPUT_SLICE]]) : (tensor<1x16x44x1000xf16>) -> tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:           [[DYN_OUTPUT_SLICE:%.*]] = tensor.cast [[TMP_OUTPUT]] : tensor<1x16x44x1000xf16, {order = #NHWC}> to tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:           [[INPUT1_SLICE:%.*]] = tensor.extract_slice [[ARG1]][0, 0, [[ADJ_OFFSET]], 0] [1, 16, [[C44]], 1000] [1, 1, 1, 1] : tensor<1x16x720x1000xf16> to tensor<1x16x?x1000xf16>
  // CHECK:           [[STATIC_INPUT1_SLICE:%.*]] = tensor.cast [[INPUT1_SLICE]] : tensor<1x16x?x1000xf16> to tensor<1x16x44x1000xf16>
  // CHECK:           [[TMP_OUTPUT1:%.*]] = func.call @main_func2_static([[STATIC_INPUT1_SLICE]], [[TMP_OUTPUT]]) : (tensor<1x16x44x1000xf16>, tensor<1x16x44x1000xf16, {order = #NHWC}>) -> tensor<1x16x44x1000xf16, {order = #NHWC}>
  // CHECK:           [[DYN_OUTPUT1_SLICE:%.*]] = tensor.cast [[TMP_OUTPUT1]] : tensor<1x16x44x1000xf16, {order = #NHWC}> to tensor<1x16x?x1000xf16, {order = #NHWC}>
  // CHECK:           [[OUTPUT:%.*]] = tensor.insert_slice [[DYN_OUTPUT1_SLICE]] into [[ARG3]][0, 0, [[ADJ_OFFSET]], 0] [1, 16, [[C44]], 1000] [1, 1, 1, 1] : tensor<1x16x?x1000xf16, {order = #NHWC}> into tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:           scf.yield [[OUTPUT]] : tensor<1x16x720x1000xf16, {order = #NHWC}>
  // CHECK:       }
  // CHECK:       [[RESULT:%.*]] = call @main_func0([[OUTPUT_BUFFER]]) : (tensor<1x16x720x1000xf16, {order = #NHWC}>) -> tensor<1x16x720x1000xf16>
  // CHECK:       return [[RESULT]] : tensor<1x16x720x1000xf16>
  // CHECK: }
}
