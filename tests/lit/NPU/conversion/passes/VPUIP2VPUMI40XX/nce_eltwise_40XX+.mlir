//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @mainModule {

  net.NetworkInfo entryPoint : @singleEltwise inputsInfo : {
    DataInfo "input_0" : tensor<1x32x56x56xui8>
    DataInfo "input_1" : tensor<1x32x56x56xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x56x56xui8>
  }

func.func @singleEltwise(%arg0: memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg2: memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
  %0 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %5 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %6 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  VPURT.Task updates(%5 : !VPURT.Barrier) {
    %7 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%5 : !VPURT.Barrier) {
    %7 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  }
  VPURT.Task waits(%6 : !VPURT.Barrier) {
    %7 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
  }
  VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
    %7 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{eltwise_type = #VPU.eltwise_type<ADD>, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weights(%1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_input(%3 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_output(%4 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [55, 55, 31], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
  }
  return %arg2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
}
}

// CHECK: func.func @singleEltwise
// CHECK: VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <1, -1> -> !VPURegMapped.Index<0:0:1>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.DPUInvariant
// CHECK-SAME: eltwise_type = #VPU.eltwise_type<ADD>
// CHECK-SAME: nce_task_type = #VPUIP.nce_task_type<ELTWISE>
// CHECK: VPUMI40XX.DPUVariant

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
net.NetworkInfo entryPoint : @singleDPU inputsInfo : {
  DataInfo "dummy_input" : tensor<1x50x1x1xf16>
} outputsInfo : {
  DataInfo "dummy_output" : tensor<1x50x1x1xf16>
}
func.func @singleDPU() {
    %256 = VPURT.DeclareBuffer <CMX_NN> [0] <206592> -> memref<1x64x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[INPUT:%.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: -> [[INPUT_TYPE:.+]]
    %252 = VPURT.DeclareBuffer <CMX_NN> [0] <206592> -> memref<1x64x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[WEIGHTS:%.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: -> [[WEIGHTS_TYPE:.+]]
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <173824> -> memref<1x64x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT:%.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE:.+]]

    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 1
    // CHECK-SAME: producer_count = 0
    // CHECK-SAME: <0, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 0
    // CHECK-SAME: producer_count = 1
    // CHECK-SAME: <1, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:1>

    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %364 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_permute_quantize, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%256 : memref<1x64x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%252 : memref<1x64x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%256 : memref<1x64x3x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%37 : memref<1x64x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]>) outputs(%37 : memref<1x64x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]>) -> memref<1x64x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 2, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 1.000000e+00 : f64>}
      }
    }
    // CHECK-NOT: VPURT.Task
    // CHECK-NOT: VPUIP.NCEClusterTask
    // CHECK-NOT: DPUTask
    // CHECK: [[INVARIANT:%.+]] = VPUMI40XX.DPUInvariant
    // CHECK-SAME: clean_after = 0
    // CHECK-SAME: is_permute_quantize
    // CHECK-SAME: mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>
    // CHECK-SAME: nce_task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME: start_after = 0
    // CHECK-SAME: input([[INPUT]] : [[INPUT_TYPE]])
    // CHECK-SAME: weights([[WEIGHTS]] : [[WEIGHTS_TYPE]])
    // CHECK-SAME: outputs([[OUTPUT]] : [[OUTPUT_TYPE]])
    // CHECK-SAME: waits([[BAR0]] : !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: updates([[BAR1]] : !VPURegMapped.Index<0:0:1>)
    // CHECK-SAME: -> <0:0:0>

    // CHECK: VPUMI40XX.PPETask
    // CHECK-SAME: mode = <ADD>
    // CHECK-SAME: clamp_low = -2147483648
    // CHECK-SAME: clamp_high = 2147483647
    // CHECK-SAME: lrelu_mult = 1
    // CHECK-SAME: lrelu_shift = 0
    // CHECK-SAME: quant_scale = [5.000000e-01]
    // CHECK-SAME: fp_prelu_alpha = 1.000000e+00

    // CHECK: VPUMI40XX.DPUVariant
    // CHECK-SAME: calls([[INVARIANT]] : <0:0:0>)
    // CHECK-SAME: weights([[WEIGHTS]] : [[WEIGHTS_TYPE]])
    // CHECK-SAME: end = [15, 2, 63]
    // CHECK-SAME: inEnd = [15, 2, 63]
    // CHECK-SAME: inStart = [0, 0, 0]
    // CHECK-SAME: mpe_mode = #VPU.mpe_mode<CUBOID_16x16>
    // CHECK-SAME: nce_task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME: pad = #VPU.Padding<
    // CHECK-SAME:   left = 0
    // CHECK-SAME:   right = 0
    // CHECK-SAME:   top = 0
    // CHECK-SAME:   bottom = 0
    // CHECK-SAME: >
    // CHECK-SAME: -> <0:0:0>
    return
}
