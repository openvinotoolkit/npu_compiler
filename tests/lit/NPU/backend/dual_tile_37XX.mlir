//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s
// REQUIRES: arch-NPU37XX

//
// This file generates a blob that runs convolutions on two tiles, used to
// demonstrate that the runtime can handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qtype = !quant.uniform<u8:f32, 1.000000e+00>

module @dual_tile attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xui8>
    } outputsInfo : {
      DataInfo "output_0" : tensor<2x16x16x16xf16>
    }

  config.MemoryResource 31457280 bytes of @DDR {config.bandwidth = 8, config.derateFactor = 6.000000e-01}
  config.ExecutorResource 2 of @DMA_NN
  config.Resources 1 of @NCE {
      config.MemoryResource 2097152 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
      config.ExecutorResource 1 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
  }

  func.func @main(
        %input_arg: memref<1x16x16x16x!qtype, #NHWC, @DDR>,
        %output_arg: memref<2x16x16x16xf16, #NHWC, @DDR>
      ) -> memref<2x16x16x16xf16, #NHWC, @DDR> {
    %weights_constant = const.Declare memref<16x1x1x16x!qtype, #NHWC, @DDR> =
      dense<1> : tensor<16x1x1x16xui8>, [#const.CastElemType<!qtype>, #const.Reorder<#NHWC>]
    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <12544>
      -> memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <12544>
      -> memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 1]>

    %input_0 = VPURT.DeclareBuffer <CMX_NN> [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>
    %output_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output_ddr_0 = VPURT.DeclareBuffer <DDR> <0>
      -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %parent_input_0 = VPURT.DeclareBuffer <CMX_NN> [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>
    %parent_output_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>

    %input_1 = VPURT.DeclareBuffer <CMX_NN> [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>
    %output_1 = VPURT.DeclareBuffer <CMX_NN> [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %output_ddr_1 = VPURT.DeclareBuffer <DDR> <8192>
      -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %parent_input_1 = VPURT.DeclareBuffer <CMX_NN> [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>
    %parent_output_1 = VPURT.DeclareBuffer <CMX_NN> [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

    %output_ddr = VPURT.DeclareBuffer <DDR> <0>
      -> memref<2x16x16x16xf16, #NHWC, @DDR>

    %weight_table_constant = const.Declare memref<16x1x1x4xsi32, #NHWC, @DDR> =
      dense<[[[[12544, 16777215, 1073761792, 0]]], [[[12560, 16777215, 1073761792, 0]]], [[[12576, 16777215, 1073761792, 0]]], [[[12592, 16777215, 1073761792, 0]]], [[[12608, 16777215, 1073761792, 0]]], [[[12624, 16777215, 1073761792, 0]]], [[[12640, 16777215, 1073761792, 0]]], [[[12656, 16777215, 1073761792, 0]]], [[[12672, 16777215, 1073761792, 0]]], [[[12688, 16777215, 1073761792, 0]]], [[[12704, 16777215, 1073761792, 0]]], [[[12720, 16777215, 1073761792, 0]]], [[[12736, 16777215, 1073761792, 0]]], [[[12752, 16777215, 1073761792, 0]]], [[[12768, 16777215, 1073761792, 0]]], [[[12784, 16777215, 1073761792, 0]]]]> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %weight_table0 = VPURT.DeclareBuffer <CMX_NN> [0] <12288>
      -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %weight_table1 = VPURT.DeclareBuffer <CMX_NN> [1] <12288>
      -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

    %inputs_ready = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %conv_complete = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %output_ready = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, @DDR>)
        outputs(%input_0 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, @DDR>)
        outputs(%input_1 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weights_constant : memref<16x1x1x16x!qtype, #NHWC, @DDR>)
        outputs(%weights0 : memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        -> memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weights_constant : memref<16x1x1x16x!qtype, #NHWC, @DDR>)
        outputs(%weights1 : memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        -> memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weight_table_constant : memref<16x1x1x4xsi32, #NHWC, @DDR>)
        outputs(%weight_table0 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
        -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weight_table_constant : memref<16x1x1x4xsi32, #NHWC, @DDR>)
        outputs(%weight_table1 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>)
        -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task waits(%inputs_ready : !VPURT.Barrier) updates(%conv_complete : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 8],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%input_0 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        weights(%weights0 : memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weight_table0 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
        parent_input(%parent_input_0 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        parent_output(%parent_output_0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%output_0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
        variants : {
          DPUTask {
            outEnd = [15, 15, 15],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            outStart = [0, 0, 0]
          }
        }
        PPE : {
        }
    }

    VPURT.Task waits(%inputs_ready : !VPURT.Barrier) updates(%conv_complete : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 8],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%input_1 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        weights(%weights1 : memref<16x1x1x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        weight_table(%weight_table1 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>)
        parent_input(%parent_input_1 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        parent_output(%parent_output_1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>)
        outputs(%output_1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>)
        -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
        variants : {
          DPUTask {
            outEnd = [15, 15, 15],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            outStart = [0, 0, 0]
          }
        }
        PPE : {
        }
    }

    VPURT.Task waits(%conv_complete : !VPURT.Barrier) updates(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%output_ddr_0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
        -> memref<1x16x16x16xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%conv_complete : !VPURT.Barrier) updates(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>)
        outputs(%output_ddr_1 : memref<1x16x16x16xf16, #NHWC, @DDR>)
        -> memref<1x16x16x16xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_ddr : memref<2x16x16x16xf16, #NHWC, @DDR>)
        outputs(%output_arg : memref<2x16x16x16xf16, #NHWC, @DDR>)
        -> memref<2x16x16x16xf16, #NHWC, @DDR>
    }

    return %output_arg : memref<2x16x16x16xf16, #NHWC, @DDR>
  }
}
