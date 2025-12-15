//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --reduce-exceeding-active-count-barriers="num-barriers=16 max-variant-count=128 share-wait-and-update-barriers=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.006069572766621908:128>
module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.MetadataMaxInvariantCount : 32
  }

// CHECK-LABEL: @NotCallLegalizeScheduleForNonWlmWhenWlmFlagIsTrue
func.func @NotCallLegalizeScheduleForNonWlmWhenWlmFlagIsTrue(%arg0: memref<1x32x8x32x8x180xf32, @DDR>) -> memref<1x32x8x32x8x180xf32, @DDR> attributes {inliner_dispatch = #VPUIP.VPUIPInlinerDispatch} {
  %dummy_barrier_0 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_2 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_3 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_4 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_5 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_6 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_7 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_8 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_9 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_10 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %dummy_barrier_11 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier

  %barrier_0 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_2 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_3 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_4 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_5 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_6 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %barrier_7 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier

  %dma_out = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  %dma_in = VPURT.DeclareBuffer <NetworkInput> [2] <0> -> memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>

  %conv_in = VPURT.DeclareBuffer <CMX_NN> [0] <1161216> -> memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>
  %conv_weights = VPURT.DeclareBuffer <CMX_NN> [0] <800768> -> memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
  %weights_table = VPURT.DeclareBuffer <CMX_NN> [0] <1383424> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  %conv_out = VPURT.DeclareBuffer <CMX_NN> [0] <90112> -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>

  VPURT.Task updates(%barrier_0 : !VPURT.Barrier)  {
    %dma = VPUIP.NNDMA {port = 0 : i64} inputs(%dma_in : memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>) outputs(%dma_out : memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier_0 : !VPURT.Barrier) updates(%barrier_3 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_3 : !VPURT.Barrier) updates(%barrier_4, %barrier_1, %barrier_2, %barrier_5 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)  {
    %dma = VPUIP.NNDMA {port = 0 : i64} inputs(%dma_in : memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>) outputs(%dma_out : memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier_3 : !VPURT.Barrier) updates(%barrier_4, %barrier_1, %barrier_2, %barrier_5 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)  {
    %dma = VPUIP.NNDMA {port = 0 : i64} inputs(%dma_in : memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>) outputs(%dma_out : memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_1, %barrier_2 : !VPURT.Barrier, !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_4 : !VPURT.Barrier) updates(%barrier_6 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_6 : !VPURT.Barrier) updates(%barrier_2, %barrier_5 : !VPURT.Barrier, !VPURT.Barrier)  {
    %dma = VPUIP.NNDMA {port = 0 : i64} inputs(%dma_in : memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>) outputs(%dma_out : memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier_6 : !VPURT.Barrier) updates(%barrier_2, %barrier_5 : !VPURT.Barrier, !VPURT.Barrier)  {
    %dma = VPUIP.NNDMA {port = 0 : i64} inputs(%dma_in : memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>) outputs(%dma_out : memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
     %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  VPURT.Task waits(%barrier_2 : !VPURT.Barrier) updates(%barrier_5 : !VPURT.Barrier)  {
    %conv = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%conv_weights : memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%conv_in : memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%conv_out : memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
  }

  return %arg0 : memref<1x32x8x32x8x180xf32, @DDR>

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  // CHECK: [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier

  // CHECK: [[DMA_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1024x64x4xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK: [[DMA_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [2] <0> -> memref<1x1024x64x4xf16, {order = #NHWC, strides = [11796480, 1, 184320, 1024]}, @DDR>
  // CHECK: [[CONV_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1161216> -> memref<1x32x16x4x!qElemType, #NHWC, [@CMX_NN, 0]>
  // CHECK: [[CONV_WEIGHTS:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <800768> -> memref<64x32x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
  // CHECK: [[CONV_WEIGHTS_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1383424> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  // CHECK: [[CONV_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <90112> -> memref<1x64x16x4xf16, #NHWC, [@CMX_NN, 0]>

  // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NNDMA
  // CHECK: }

  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK: }

  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NNDMA
  // CHECK: }

  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NNDMA
  // CHECK: }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  /// Fixed incorrect barrier assignment. A task may have more than one wait barrier when wlmFlag=true.
  // CHECK-NOT: VPURT.Task waits([[BAR2]], [[BAR4]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NNDMA
  // CHECK: }

  // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NNDMA
  // CHECK: }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)  {
  // CHECK:     VPUIP.NCEClusterTask
  // CHECK  }

  // CHECK: return
}
}