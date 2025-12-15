//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode="allow-custom-values=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.032067115634095436:128>

// CHECK-LABEL: @TestConvolution
module @TestConvolution {
    config.Resources 3 of @NCE at 1.700000e+03 MHz

    // CHECK-DAG:  {{  }}config.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:  {{  }}config.Resources {activity_factor = {{[0-9]+.[0-9]+}} : f64} 3 of @NCE at 1.700000e+03 MHz
    // CHECK-DAG:  {{    }}config.ExecutorResource 1 of @DPU
    // CHECK-DAG:  {{    }}config.ExecutorResource 2 of @SHAVE_ACT

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x28x28xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x28x28xf16>
    }

    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
        config.Option @config.ReduceSupported : true
    }

    // CHECK: func.func @main
    func.func @main(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x16x28x28xf16> {

    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-2.05229545> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.0362618> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-2.05229545> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.0362618> : tensor<1x1x1x1xf32>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x28x28xf16>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x28x28xf16>
    %2 = IE.ReduceMean(%1) { axes_value = [1], keep_dims } :
        tensor<1x64x28x28xf16>
        -> tensor<1x1x28x28xf16>
    %3 = IE.Swish(%2) : tensor<1x1x28x28xf16> -> tensor<1x1x28x28xf16>

    %5 = IE.FakeQuantize(%3, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x28x28xf16>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x28x28xf16>
    %6 = IE.Expand(%5) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x28x28xf16> -> tensor<1x16x28x28xf16>
    return %6 : tensor<1x16x28x28xf16>

    // Quantization ELTWISE

    // CHECK:       VPURT.Task waits([[barrier_1:%.+]] : !VPURT.Barrier) updates([[barrier_2:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[input_0:%.+]] : memref<1x64x10x28xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[output_0:%.+]] : memref<1x64x10x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x10x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPURT.Task waits([[barrier_1:%.+]] : !VPURT.Barrier) updates([[barrier_2:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:              input([[input_1:%.+]] : memref<1x64x9x28xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:         outputs([[output_1:%.+]] : memref<1x64x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 1]>

    // CHECK:       VPURT.Task waits([[barrier_1:%.+]] : !VPURT.Barrier) updates([[barrier_2:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[input_2:%.+]] : memref<1x64x9x28xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:          outputs([[output_2:%.+]] : memref<1x64x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 2]>) -> memref<1x64x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 2]>


    // Dequantization ELTWISE

    // CHECK:       VPURT.Task waits([[barrier_2:%.+]] : !VPURT.Barrier) updates([[barrier_3:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[input_3:%.+]] : memref<1x64x10x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[output_3:%.+]] : memref<1x64x10x28xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x10x28xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPURT.Task waits([[barrier_2:%.+]] : !VPURT.Barrier) updates([[barrier_3:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:              input([[input_4:%.+]] : memref<1x64x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:         outputs([[output_4:%.+]] : memref<1x64x9x28xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x9x28xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK:       VPURT.Task waits([[barrier_2:%.+]] : !VPURT.Barrier) updates([[barrier_3:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[input_5:%.+]] : memref<1x64x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:          outputs([[output_5:%.+]] : memref<1x64x9x28xf16, #NHWC, [@CMX_NN, 2]>) -> memref<1x64x9x28xf16, #NHWC, [@CMX_NN, 2]>


    // ReduceMean Op

    // CHECK:       VPURT.Task waits([[barrier_3:%.+]] : !VPURT.Barrier) updates([[barrier_4:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<REDUCEMEAN>}
    // CHECK-SAME:          -> memref<1x16x10x28xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPURT.Task waits([[barrier_3:%.+]] : !VPURT.Barrier) updates([[barrier_4:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<REDUCEMEAN>}
    // CHECK-SAME:          -> memref<1x16x9x28xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK:       VPURT.Task waits([[barrier_3:%.+]] : !VPURT.Barrier) updates([[barrier_4:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<REDUCEMEAN>}
    // CHECK-SAME:          -> memref<1x16x9x28xf16, #NHWC, [@CMX_NN, 2]>


    // Quantization ELTWISE

    // CHECK:       VPURT.Task waits([[barrier_5:%.+]] : !VPURT.Barrier) updates([[barrier_6:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:       input([[input_9:%.+]] : memref<1x16x10x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       outputs([[output_9:%.+]] : memref<1x16x10x28xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x10x28xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPURT.Task waits([[barrier_5:%.+]] : !VPURT.Barrier) updates([[barrier_6:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[input_10:%.+]] : memref<1x16x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:      outputs([[input_10:%.+]] : memref<1x16x9x28xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x9x28xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK:       VPURT.Task waits([[barrier_5:%.+]] : !VPURT.Barrier) updates([[barrier_6:%.+]] : !VPURT.Barrier)
    // CHECK:       VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[input_11:%.+]] : memref<1x16x9x28x!qElemType{{[0-9]*}}, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:      outputs([[output_11:%.+]] : memref<1x16x9x28xf16, #NHWC, [@CMX_NN, 2]>) -> memref<1x16x9x28xf16, #NHWC, [@CMX_NN, 2]>

  }
}

// -----

// CHECK-LABEL: @BatchedGroupConvWithBroadcast
module @BatchedGroupConvWithBroadcast {
  // CHECK-DAG:  {{  }}config.ExecutorResource 2 of @DMA_NN
  // CHECK-DAG:  {{  }}config.Resources {activity_factor = {{[0-9]+.[0-9]+}} : f64} 3 of @NCE at 2.100000e+03 MHz
  // CHECK-DAG:  {{    }}config.ExecutorResource 1 of @DPU
  // CHECK-DAG:  {{    }}config.ExecutorResource 2 of @SHAVE_ACT

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x1x2x2xf16>
    DataInfo "Parameter_2" : tensor<1x1x3x3xf16>
  } outputsInfo : {
    DataInfo "GroupConvolution_10" : tensor<4x1x2x2xf16>
  }

  // CHECK:       @main(
  // CHECK-SAME:      [[ARG0:%.+]]: memref<4x1x2x2xf16, @DDR>,
  // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1x3x3xf16, @DDR>,
  // CHECK-SAME:      -> memref<4x1x2x2xf16, @DDR>

  func.func @main(%arg0: tensor<4x1x2x2xf16>, %arg1: tensor<1x1x3x3xf16>) -> tensor<4x1x2x2xf16> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
         } : tensor<4x1x2x2xf16>, tensor<1x1x3x3xf16> -> tensor<4x1x2x2xf16>
    return %0 : tensor<4x1x2x2xf16>

    // CHECK: VPURT.Task waits([[BAR_0:%.+]] : !VPURT.Barrier) updates([[BAR_1:%.+]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK: VPUIP.NCEClusterTask {is_superdense,
    // CHECK:   kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK:   kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK:   input([[INPUT_0:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 0]>) weights([[WEIGHTS_0:%.+]] : memref<4x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   weight_table([[WT_0:%.+]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK:   parent_input([[INPUT_0:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   parent_output([[OUT_0:%.+]] : memref<1x4x2x2xf16, [@CMX_NN, 0]>) outputs([[OUT_0:%.+]] : memref<1x4x2x2xf16, [@CMX_NN, 0]>)
    // CHECK:   -> memref<1x4x2x2xf16, [@CMX_NN, 0]> variants : {
    // CHECK:     DPUTask {cluster_id = 0 : i64
    // CHECK:   PPETask {ppe = #VPU.PPEFp<mode = <NOOP>

    // CHECK: VPURT.Task waits([[BAR_0:%.+]] : !VPURT.Barrier) updates([[BAR_1:%.+]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK: VPUIP.NCEClusterTask {
    // CHECK:   kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK:   kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK:   input([[INPUT_1:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 1]>) weights([[WEIGHTS_1:%.+]] : memref<4x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK:   weight_table([[WT_1:%.+]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    // CHECK:   parent_input([[INPUT_1:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK:   parent_output([[OUT_1:%.+]] : memref<1x4x2x2xf16, [@CMX_NN, 1]>) outputs([[OUT_1:%.+]] : memref<1x4x2x2xf16, [@CMX_NN, 1]>)
    // CHECK:   -> memref<1x4x2x2xf16, [@CMX_NN, 1]> variants : {
    // CHECK:     DPUTask {cluster_id = 1 : i64
    // CHECK:   PPETask {ppe = #VPU.PPEFp<mode = <NOOP>

    // CHECK: VPURT.Task waits([[BAR_0:%.+]] : !VPURT.Barrier) updates([[BAR_1:%.+]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK: VPUIP.NCEClusterTask {
    // CHECK:   kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK:   kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK:   input([[INPUT_2:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 2]>) weights([[WEIGHTS_2:%.+]] : memref<4x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK:   weight_table([[WT_2:%.+]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    // CHECK:   parent_input([[INPUT_2:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK:   parent_output([[OUT_2:%.+]] : memref<1x4x2x2xf16, [@CMX_NN, 2]>) outputs([[OUT_2:%.+]] : memref<1x4x2x2xf16, [@CMX_NN, 2]>)
    // CHECK:   -> memref<1x4x2x2xf16, [@CMX_NN, 2]> variants : {
    // CHECK:     DPUTask {cluster_id = 2 : i64
    // CHECK:   PPETask {ppe = #VPU.PPEFp<mode = <NOOP>
  }
}
