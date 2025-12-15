//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --compute-nce-input-workloads %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input1_CMX = tensor<1x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Input2_CMX = tensor<1x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x3x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @AddInputWorkloadsOC
module @AddInputWorkloadsOC  {

  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input1" : tensor<1x16x4x4xf16>
    DataInfo "input2" : tensor<1x16x4x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x3x4x4xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input1_CMX, %arg1: !Input2_CMX) -> !Output_CMX {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        input_padding = [0, 13, 0, 0],
        output_padding = [0, 0, 0, 0],
        ppe = #VPU.PPEInt<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 4, 4] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<MATRIX>
    }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.Eltwise

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 4, 4]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 3, 4, 4]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <MATRIX>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x16x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x3x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<3x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @DWConvInputWorkloadsAutopaddingODU
module @DWConvInputWorkloadsAutopaddingODU  {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : false
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x60x60xf16>
    DataInfo "weights" : tensor<3x16x1x1xf16>
    DataInfo "weightsTable" : tensor<16x1x1x4xsi32>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x3x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX) -> !Output_CMX {
    %0 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
            input_padding = [0, 13, 0, 0],
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [3, 1, 1, 1],
            strides = [1, 1]
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 30, 60]  #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
            VPU.DPU.Workload outOffsets [0, 0, 30, 0] outSizes [1, 3, 30, 60] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.DepthConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 30, 60]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 3, 30, 60]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 30, 0]
  // CHECK-SAME:          inSizes [1, 16, 30, 60]
  // CHECK-SAME:          outOffsets [0, 0, 30, 0]
  // CHECK-SAME:          outSizes [1, 3, 30, 60]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x16x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x3x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @MaxPoolInputWorkloadsAutopaddingODU
module @MaxPoolInputWorkloadsAutopaddingODU  {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : false
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x60x60xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x3x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX) -> !Output_CMX {
    %0 = VPU.NCE.MaxPool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 30, 60]  #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
            VPU.DPU.Workload outOffsets [0, 0, 30, 0] outSizes [1, 3, 30, 60] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.MaxPool

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 30, 60]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 3, 30, 60]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 30, 0]
  // CHECK-SAME:          inSizes [1, 16, 30, 60]
  // CHECK-SAME:          outOffsets [0, 0, 30, 0]
  // CHECK-SAME:          outSizes [1, 3, 30, 60]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x16x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x3x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @AveragePoolInputWorkloadsAutopaddingODU
module @AveragePoolInputWorkloadsAutopaddingODU  {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : false
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x60x60xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x3x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX) -> !Output_CMX {
    %0 = VPU.NCE.AveragePool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 30, 60]  #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
            VPU.DPU.Workload outOffsets [0, 0, 30, 0] outSizes [1, 3, 30, 60] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.AveragePool

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 30, 60]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 3, 30, 60]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 30, 0]
  // CHECK-SAME:          inSizes [1, 16, 30, 60]
  // CHECK-SAME:          outOffsets [0, 0, 30, 0]
  // CHECK-SAME:          outSizes [1, 3, 30, 60]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input0_CMX = !VPU.DistributedTensor<1x4096x1x1xf16, #NHWC, @CMX_NN, {
  mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 1376, 1, 1], [1, 1360, 1, 1], [1, 1360, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 1376, 0, 0], [0, 2736, 0, 0]],
  memory_shapes = [[1, 1376, 1, 1], [1, 1360, 1, 1], [1, 1360, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 1376, 0, 0], [0, 2736, 0, 0]]}>
!Input1_CMX = !VPU.DistributedTensor<1x4096x1x1xf16, #NHWC, @CMX_NN, {
  mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 1376, 1, 1], [1, 1360, 1, 1], [1, 1360, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 1376, 0, 0], [0, 2736, 0, 0]],
  memory_shapes = [[1, 1376, 1, 1], [1, 1360, 1, 1], [1, 1360, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 1376, 0, 0], [0, 2736, 0, 0]]}>
!Output_CMX = !VPU.DistributedTensor<1x4096x1x1xf16, #NHWC, @CMX_NN, {
  mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 1376, 1, 1], [1, 1360, 1, 1], [1, 1360, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 1376, 0, 0], [0, 2736, 0, 0]],
  memory_shapes = [[1, 1376, 1, 1], [1, 1360, 1, 1], [1, 1360, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 1376, 0, 0], [0, 2736, 0, 0]]}>

// CHECK-LABEL: @EltwiseInputWorkloads
module @EltwiseInputWorkloads  {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : false
  }

  func.func @main(%arg0: !Input0_CMX, %arg1: !Input1_CMX) -> !Output_CMX {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>
      } -> !Output_CMX {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1376, 1, 1] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_8x16> attributes {cluster_id = 0 : i64}
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1360, 1, 1] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_8x16> attributes {cluster_id = 1 : i64}
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1360, 1, 1] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_8x16> attributes {cluster_id = 2 : i64}
    }
    return %0 : !Output_CMX
  }
  //CHECK:            VPU.NCE.Eltwise

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 1376, 1, 1]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 1376, 1, 1]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_8x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 1360, 1, 1]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 1360, 1, 1]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_8x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 1360, 1, 1]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 1360, 1, 1]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_8x16>
}
