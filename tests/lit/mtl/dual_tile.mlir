// RUN: vpux-opt %s

//
// This file generates a blob that runs convolutions on two tiles, used to
// demonstrate that the runtime can handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//
// To generate a blob, use:
//
//    vpux-translate --export-VPUIP < dual_tile.mlir > dual_tile.blob
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qtype = type !quant.uniform<u8:f32, 1.000000e+00>

module @dual_tile attributes {VPU.arch = "MTL", VPU.compilationMode = "DefaultHW"} {
  IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xui8, {order = #NHWC}>
    } outputsInfo : {
      DataInfo "output_0" : tensor<2x16x16x16xf16, {order = #NHWC}>
    }

  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
  IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

  IE.RunTimeResources
    executors :  {
      ExecutorResource 1 of "DMA_UPA"
      ExecutorResource 1 of "SHAVE_NN"
      ExecutorResource 1 of "NCE"  {
        ExecutorResource 1 of "DPU"
      }
      ExecutorResource 2 of "DMA_NN"
    }

  func @main(
        %input_arg: memref<1x16x16x16x!qtype, #NHWC, @DDR>,
        %output_arg: memref<2x16x16x16xf16, #NHWC, @DDR>
      ) -> memref<2x16x16x16xf16, #NHWC, @DDR> {
    %weights_constant = const.Declare memref<16x1x1x16x!qtype, #NHWC, @DDR> =
      #const.Content<dense<1> : tensor<16x1x1x16xui8>, [#const.QuantCast<!qtype>, #const.Reorder<#NHWC>]>
    %weights = VPURT.DeclareBuffer "CMX_NN" [0] <12544>
      -> memref<16x1x1x16x!qtype, #NHWC, @CMX_NN>

    %input_0 = VPURT.DeclareBuffer "CMX_NN" [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>
    %output_0 = VPURT.DeclareBuffer "CMX_NN" [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    %output_ddr_0 = VPURT.DeclareBuffer "DDR" <0>
      -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %parent_input_0 = VPURT.DeclareBuffer "CMX_NN" [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>
    %parent_output_0 = VPURT.DeclareBuffer "CMX_NN" [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    %input_1 = VPURT.DeclareBuffer "CMX_NN" [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>
    %output_1 = VPURT.DeclareBuffer "CMX_NN" [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    %output_ddr_1 = VPURT.DeclareBuffer "DDR" <8192>
      -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %parent_input_1 = VPURT.DeclareBuffer "CMX_NN" [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>
    %parent_output_1 = VPURT.DeclareBuffer "CMX_NN" [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    %output_ddr = VPURT.DeclareBuffer "DDR" <0>
      -> memref<2x16x16x16xf16, #NHWC, @DDR>

    %weight_table_constant = const.Declare memref<16x1x1x4xsi32, #NHWC, @DDR> =
      #const.Content<dense<[[[[12544, 16777215, 1073761792, 0]]], [[[12560, 16777215, 1073761792, 0]]], [[[12576, 16777215, 1073761792, 0]]], [[[12592, 16777215, 1073761792, 0]]], [[[12608, 16777215, 1073761792, 0]]], [[[12624, 16777215, 1073761792, 0]]], [[[12640, 16777215, 1073761792, 0]]], [[[12656, 16777215, 1073761792, 0]]], [[[12672, 16777215, 1073761792, 0]]], [[[12688, 16777215, 1073761792, 0]]], [[[12704, 16777215, 1073761792, 0]]], [[[12720, 16777215, 1073761792, 0]]], [[[12736, 16777215, 1073761792, 0]]], [[[12752, 16777215, 1073761792, 0]]], [[[12768, 16777215, 1073761792, 0]]], [[[12784, 16777215, 1073761792, 0]]]]> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <12288>
      -> memref<16x1x1x4xsi32, #NHWC, @CMX_NN>

    %inputs_ready = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %conv_complete = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %output_ready = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, @DDR>)
        outputs(%input_0 : memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>)
        -> memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>
    }
    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, @DDR>)
        outputs(%input_1 : memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>)
        -> memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weights_constant : memref<16x1x1x16x!qtype, #NHWC, @DDR>)
        outputs(%weights : memref<16x1x1x16x!qtype, #NHWC, @CMX_NN>)
        -> memref<16x1x1x16x!qtype, #NHWC, @CMX_NN>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weight_table_constant : memref<16x1x1x4xsi32, #NHWC, @DDR>)
        outputs(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        -> memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
    }

    VPURT.Task waits(%inputs_ready : !VPURT.Barrier) updates(%conv_complete : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          kernel_size = [1, 8],
          kernel_strides = [1, 1],
          task_type = "CONV"
        }
        input(%input_0 : memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x1x1x16x!qtype, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%parent_input_0 : memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>)
        parent_output(%parent_output_0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
        outputs(%output_0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
        -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
        variants : {
          DPUTask {
            end = [15, 15, 15],
            mpe_mode = "CUBOID_16x16",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            start = [0, 0, 0]
          }
        }
        PPE : {
        }
    }

    VPURT.Task waits(%inputs_ready : !VPURT.Barrier) updates(%conv_complete : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          kernel_size = [1, 8],
          kernel_strides = [1, 1],
          task_type = "CONV"
        }
        input(%input_1 : memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x1x1x16x!qtype, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%parent_input_1 : memref<1x16x16x16x!qtype, #NHWC, @CMX_NN>)
        parent_output(%parent_output_1 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
        outputs(%output_1 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
        -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
        variants : {
          DPUTask {
            end = [15, 15, 15],
            mpe_mode = "CUBOID_16x16",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            start = [0, 0, 0]
          }
        }
        PPE : {
        }
    }

    VPURT.Task waits(%conv_complete : !VPURT.Barrier) updates(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
        outputs(%output_ddr_0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
        -> memref<1x16x16x16xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%conv_complete : !VPURT.Barrier) updates(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_1 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
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
