//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --compute-nce-input-workloads %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x48x31x31xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvInputWorkloadsHeight
module @ConvInputWorkloadsHeight  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x31x31xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX) -> !Output_CMX {
    %0 =  VPU.NCE.Convolution(%arg0, %arg1) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [48, 16, 3, 3],
            strides = [2, 2]
        } : !Input_CMX, !Weights_CMX -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 16, 31] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            VPU.DPU.Workload outOffsets [0, 0, 16, 0] outSizes [1, 48, 15, 31] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.Convolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 32, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 48, 16, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_4x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 31, 0]
  // CHECK-SAME:          inSizes [1, 16, 31, 62]
  // CHECK-SAME:          outOffsets [0, 0, 16, 0]
  // CHECK-SAME:          outSizes [1, 48, 15, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK-SAME:          <CUBOID_4x16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x48x31x31xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvInputWorkloadsOC
module @ConvInputWorkloadsOC  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x31x31xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX) -> !Output_CMX {
    %0 = VPU.NCE.Convolution(%arg0, %arg1) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [48, 16, 3, 3],
            strides = [2, 2]
        } : !Input_CMX, !Weights_CMX -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 24, 31, 31] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16>
            VPU.DPU.Workload outOffsets [0, 24, 0, 0] outSizes [1, 24, 31, 31] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16>
    }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.Convolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 24, 31, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          <CUBOID_4x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 24, 0, 0]
  // CHECK-SAME:          outSizes [1, 24, 31, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          <CUBOID_4x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x32x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x32x31x31xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @DWConvInputWorkloadsHeight
module @DWConvInputWorkloadsHeight  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x32x62x62xf16>
    DataInfo "weights" : tensor<32x16x1x1xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x32x31x31xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX) -> !Output_CMX {
    %0 = VPU.NCE.DepthConvolution(%arg0, %arg1) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [32, 1, 3, 3],
            strides = [2, 2]
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 31]  #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
            VPU.DPU.Workload outOffsets [0, 0, 16, 0] outSizes [1, 32, 15, 31] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.DepthConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 32, 32, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 32, 16, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 31, 0]
  // CHECK-SAME:          inSizes [1, 32, 31, 62]
  // CHECK-SAME:          outOffsets [0, 0, 16, 0]
  // CHECK-SAME:          outSizes [1, 32, 15, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x48x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x48x31x31xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @DWConvInputWorkloadsOC
module @DWConvInputWorkloadsOC  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x48x62x62xf16>
    DataInfo "weights" : tensor<48x16x1x1xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x31x31xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX) -> !Output_CMX {
    %0 = VPU.NCE.DepthConvolution(%arg0, %arg1) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [48, 1, 3, 3],
            strides = [2, 2]
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 31, 31] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16>
            VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 31, 31] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.DepthConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 32, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 32, 31, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 32, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 32, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 31, 31]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPU.DistributedTensor<
    1x16x62x62xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [5, 5],
    pads = #VPU.Padding<left = 2 , right = 2, top = 2, bottom = 2>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x48x62x62xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [5, 5],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!Output_NCE = tensor<1x48x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

!Weights_CMX = tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!Input_DDR = tensor<1x16x62x62xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvInputWorkloadsSOHExtraLines
module @ConvInputWorkloadsSOHExtraLines  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x62x62xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_DDR, %arg1: !Weights_CMX) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output_cmx = VPU.NCE.Convolution(%input_cmx, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [48, 16, 3, 3],
              strides = [1, 1]
          } : !Input_CMX, !Weights_CMX -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 11,  62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 11, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 21, 0] outSizes [1, 48, 11, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 32, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 42, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 2 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 52, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 2 : i64}
        }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Convolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 48, 11,  62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 10, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 11, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 1, 0]
  // CHECK-SAME:          inSizes [1, 16, 13, 62]
  // CHECK-SAME:          outOffsets [0, 0, 21, 0]
  // CHECK-SAME:          outSizes [1, 48, 11,  62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 12, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 32, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 1, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 42, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 2

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 11, 0]
  // CHECK-SAME:          inSizes [1, 16, 11, 62]
  // CHECK-SAME:          outOffsets [0, 0, 52, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 2

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputData_CMX = !VPU.DistributedTensor<
    1x16x62x62xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [5, 5],
    pads = #VPU.Padding<left = 2 , right = 2, top = 2, bottom = 2>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!InputSM_CMX = !VPU.DistributedTensor<
    1x16x62x62xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [5, 5],
    pads = #VPU.Padding<left = 2 , right = 2, top = 2, bottom = 2>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!Input_CMX = !VPU.SparseTensor<data=!InputData_CMX,
                               sparsity_map=!InputSM_CMX>

!OutputData_CMX = !VPU.DistributedTensor<
    1x48x62x62xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!OutputSM_CMX = !VPU.DistributedTensor<
    1x48x62x62xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.SparseTensor<data=!OutputData_CMX,
                                sparsity_map=!OutputSM_CMX>

!OutputStub_CMX = !VPU.SparseTensor<data=tensor<1x48x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                    sparsity_map=tensor<1x48x62x62xi1, {mem_space = @CMX_NN, order = #NHWC}>>
!Weights_CMX = tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!InputDataStub_CMX = tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSMStub_CMX = tensor<1x16x62x62xi1, {mem_space = @CMX_NN, order = #NHWC}>
!InputStub_CMX = !VPU.SparseTensor<data=!InputDataStub_CMX,
                                   sparsity_map=!InputSMStub_CMX>

// CHECK-LABEL: @SparseConvInputWorkloadsSOHExtraLines
module @SparseConvInputWorkloadsSOHExtraLines  {
  func.func @main(%arg0: !InputData_CMX, %arg1: !Weights_CMX, %arg3: !InputSM_CMX) -> !Output_CMX {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg3) -> !Input_CMX

    %output_cmx = VPU.NCE.Convolution(%input_sparse, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [48, 16, 3, 3],
              strides = [1, 1]
          } : !Input_CMX, !Weights_CMX -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 11,  62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 11, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 21, 0] outSizes [1, 48, 11, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 32, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 42, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 2 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 52, 0] outSizes [1, 48, 10, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 2 : i64}
        }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Convolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 48, 11,  62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 10, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 11, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 1, 0]
  // CHECK-SAME:          inSizes [1, 16, 13, 62]
  // CHECK-SAME:          outOffsets [0, 0, 21, 0]
  // CHECK-SAME:          outSizes [1, 48, 11,  62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 12, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 32, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 1, 0]
  // CHECK-SAME:          inSizes [1, 16, 12, 62]
  // CHECK-SAME:          outOffsets [0, 0, 42, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 2

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 11, 0]
  // CHECK-SAME:          inSizes [1, 16, 11, 62]
  // CHECK-SAME:          outOffsets [0, 0, 52, 0]
  // CHECK-SAME:          outSizes [1, 48, 10, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 2

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPU.DistributedTensor<
    1x16x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x64x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Output_NCE = tensor<1x64x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<64x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!Input_DDR = tensor<1x16x62x62xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvInputWorkloadsSOK
module @ConvInputWorkloadsSOK  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<64x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x64x62x62xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_DDR, %arg1: !Weights_CMX) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output_cmx =  VPU.NCE.Convolution(%input_cmx, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [64, 16, 3, 3],
              strides = [1, 1]
          } : !Input_CMX, !Weights_CMX -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
          }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Convolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 16, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 32, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 48, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPU.DistributedTensor<
    1x16x62x62xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x48x62x62xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Output_NCE = tensor<1x48x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!Input_DDR = tensor<1x16x62x62xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvInputWorkloadsSOHNoExtraLines
module @ConvInputWorkloadsSOHNoExtraLines  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x62x62xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_DDR, %arg1: !Weights_CMX) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output_cmx =  VPU.NCE.Convolution(%input_cmx, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [48, 16, 3, 3],
              strides = [1, 1]
          } : !Input_CMX, !Weights_CMX -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 16, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 16, 0] outSizes [1, 48, 15, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 31, 0] outSizes [1, 48, 16, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 47, 0] outSizes [1, 48, 15, 62] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
          }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Convolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 17, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 48, 16, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 15, 0]
  // CHECK-SAME:          inSizes [1, 16, 17, 62]
  // CHECK-SAME:          outOffsets [0, 0, 16, 0]
  // CHECK-SAME:          outSizes [1, 48, 15, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 18, 62]
  // CHECK-SAME:          outOffsets [0, 0, 31, 0]
  // CHECK-SAME:          outSizes [1, 48, 16, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 16, 0]
  // CHECK-SAME:          inSizes [1, 16, 16, 62]
  // CHECK-SAME:          outOffsets [0, 0, 47, 0]
  // CHECK-SAME:          outSizes [1, 48, 15, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPU.DistributedTensor<
    1x80x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x80x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = tensor<80x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

!Input_DDR = tensor<1x80x62x62xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x80x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = tensor<1x80x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @DWInputWorkloadsSOKSEGSEG
module @DWInputWorkloadsSOKSEGSEG  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x80x62x62xf16>
    DataInfo "weights" : tensor<80x16x1x1xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x80x62x62xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_DDR, %arg1: !Weights_CMX) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output_cmx =  VPU.NCE.DepthConvolution(%input_cmx, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [80, 1, 3, 3],
              strides = [1, 1]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
          }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.DepthConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 32, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 32, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 32, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 32, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 32, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 32, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPU.DistributedTensor<
    1x80x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x80x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = tensor<80x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

!Input_DDR = tensor<1x80x62x62xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x80x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = tensor<1x80x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @DWInputWorkloadsSOKSEGDUP
module @DWInputWorkloadsSOKSEGDUP  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x80x62x62xf16>
    DataInfo "weights" : tensor<80x16x1x1xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x80x62x62xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_DDR, %arg1: !Weights_CMX) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output_cmx =  VPU.NCE.DepthConvolution(%input_cmx, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [80, 1, 3, 3],
              strides = [1, 1]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
          }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.DepthConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 32, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 32, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 32, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 32, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 48, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 16, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 64, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPU.DistributedTensor<
    1x80x62x62xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x80x62x62xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = tensor<80x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

!Input_DDR = tensor<1x80x62x62xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x80x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = tensor<1x80x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @DWInputWorkloadsSOKDUPSEG
module @DWInputWorkloadsSOKDUPSEG  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x80x62x62xf16>
    DataInfo "weights" : tensor<80x16x1x1xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x80x62x62xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_DDR, %arg1: !Weights_CMX) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output_cmx =  VPU.NCE.DepthConvolution(%input_cmx, %arg1) {
              ppe = #VPU.PPEStub<>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              rawFilterShape = [80, 1, 3, 3],
              strides = [1, 1]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
              VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 62, 62] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_4x16> attributes {cluster_id = 1 : i64}
          }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.DepthConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 32, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 32, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 32, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 32, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 48, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 64, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 62, 62]
  // CHECK-SAME:          outOffsets [0, 16, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 62, 62]
  // CHECK-SAME:          <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = tensor<1x4x209x416xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Output_CMX = tensor<1x32x104x208xf16, {mem_space = @CMX_NN, order = #NHWC}>
!Weights_CMX = tensor<32x1x1x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvInputWorkloadsHeight
module @ConvInputWorkloadsHeight  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x4x209x416xf16>
    DataInfo "weights" : tensor<32x1x1x32xf16>
    DataInfo "weightsTable" : tensor<32x1x1x4xsi32>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x32x104x208xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX) -> !Output_CMX {
    %0 =  VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
            cm_sp_pattern = 15 : i64, minimumHardwareExecutionCost = 4294967398 : i64,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 4, 3, 3], strides = [2, 2]
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 104, 208] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16>
        }
    return %0 : !Output_CMX
  }

  //CHECK:            VPU.NCE.CompressConvolution

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 208, 416]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 32, 104, 208]
  // CHECK-SAME:          <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:          <CUBOID_16x16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteInputWorkloads
func.func @NCEPermuteInputWorkloads(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
            dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
            minimumHardwareExecutionCost = 4294967300 : i64, ppe = #VPU.PPEStub<>
        } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
        }
    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    //CHECK:            VPU.NCE.Permute

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 0, 0, 0]
    // CHECK-SAME:          inSizes [1, 3, 224, 224]
    // CHECK-SAME:          outOffsets [0, 0, 0, 0]
    // CHECK-SAME:          outSizes [1, 4, 224, 224]
    // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2
}>

!Input_DDR = tensor<1x3x224x224xf16>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>
!OutputStub_CMX = tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteNoExtraLinesAtInput
func.func @NCEPermuteNoExtraLinesAtInput(%arg0: !Input_DDR) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%input_cmx) {
                dstElemType = !quant.uniform<u8:f16, 1.000000e+00>,
                dstOrder = #NHWC,
                expandedChannels = 4 : i64, minimumHardwareExecutionCost = 4294967300 : i64,
                ppe = #VPU.PPEStub<>
        } -> !Output_CMX {
          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
          VPU.DPU.Workload outOffsets [0, 0, 112, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output : !Output_CMX

    //CHECK:            VPU.NCE.Permute

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 0, 0, 0]
    // CHECK-SAME:          inSizes [1, 3, 112, 224]
    // CHECK-SAME:          outOffsets [0, 0, 0, 0]
    // CHECK-SAME:          outSizes [1, 4, 112, 224]
    // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>
    // CHECK-SAME:          cluster_id = 0 : i64

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 0, 0, 0]
    // CHECK-SAME:          inSizes [1, 3, 112, 224]
    // CHECK-SAME:          outOffsets [0, 0, 112, 0]
    // CHECK-SAME:          outSizes [1, 4, 112, 224]
    // CHECK-SAME:          left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>
    // CHECK-SAME:          cluster_id = 1 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2
}>

!Input_DDR = tensor<1x3x224x224xf16>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>
!OutputStub_CMX = tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteWithAdjustedInputWorkloadForExtraLines
func.func @NCEPermuteWithAdjustedInputWorkloadForExtraLines(%arg0: !Input_DDR) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%input_cmx) {
                dstElemType = !quant.uniform<u8:f16, 1.000000e+00>,
                dstOrder = #NHWC,
                expandedChannels = 4 : i64, minimumHardwareExecutionCost = 4294967300 : i64,
                ppe = #VPU.PPEStub<>
        } -> !Output_CMX {
          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
          VPU.DPU.Workload outOffsets [0, 0, 112, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output : !Output_CMX

    //CHECK:            VPU.NCE.Permute

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 0, 0, 0]
    // CHECK-SAME:          inSizes [1, 3, 112, 224]
    // CHECK-SAME:          outOffsets [0, 0, 0, 0]
    // CHECK-SAME:          outSizes [1, 4, 112, 224]
    // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>
    // CHECK-SAME:          cluster_id = 0 : i64

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 0, 1, 0]
    // CHECK-SAME:          inSizes [1, 3, 112, 224]
    // CHECK-SAME:          outOffsets [0, 0, 112, 0]
    // CHECK-SAME:          outSizes [1, 4, 112, 224]
    // CHECK-SAME:          left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>
    // CHECK-SAME:          cluster_id = 1 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x128x32x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
}>

!Output_CMX = !VPU.DistributedTensor<
    1x128x32x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
}>

!Input_DDR = tensor<1x128x32x64xf16>
!InputStub_CMX = tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteInputWorkloadsSOC
func.func @NCEPermuteInputWorkloadsSOC(%arg0: !Input_DDR) -> !Output_CMX {
    %input_cmx = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%input_cmx) {
                dstElemType = f16,
                dstOrder = #NHWC,
                expandedChannels = 128 : i64, minimumHardwareExecutionCost = 5442 : i64,
                ppe = #VPU.PPEStub<>
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 3 : i64}
        }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = f16, dstOrder = #NHWC, expandedChannels = 128 : i64
    // CHECK-SAME:      -> !VPU.DistributedTensor<
    // CHECK-SAME:          1x128x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 4, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          alignment = [1, 16, 1, 1],
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
    // CHECK-SAME:          }> {

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      inOffsets [0, 0, 0, 0] inSizes [1, 32, 32, 64]
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      inOffsets [0, 0, 0, 0] inSizes [1, 32, 32, 64]
    // CHECK-SAME:      outOffsets [0, 32, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      inOffsets [0, 0, 0, 0] inSizes [1, 32, 32, 64]
    // CHECK-SAME:      outOffsets [0, 64, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 2

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      inOffsets [0, 0, 0, 0] inSizes [1, 32, 32, 64]
    // CHECK-SAME:      outOffsets [0, 96, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 3
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteInputWorkloadsChannels
func.func @NCEPermuteInputWorkloadsChannels(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
            dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
            minimumHardwareExecutionCost = 4294967300 : i64, ppe = #VPU.PPEStub<>
        } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 2, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 2, 0, 0] outSizes [1, 2, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }
    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    //CHECK:            VPU.NCE.Permute

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 0, 0, 0]
    // CHECK-SAME:          inSizes [1, 2, 224, 224]
    // CHECK-SAME:          outOffsets [0, 0, 0, 0]
    // CHECK-SAME:          outSizes [1, 2, 224, 224]
    // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>

    // CHECK:           VPU.DPU.Workload
    // CHECK-SAME:          inOffsets [0, 2, 0, 0]
    // CHECK-SAME:          inSizes [1, 1, 224, 224]
    // CHECK-SAME:          outOffsets [0, 2, 0, 0]
    // CHECK-SAME:          outSizes [1, 2, 224, 224]
    // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputData_CMX = !VPU.DistributedTensor<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputSM_CMX = !VPU.DistributedTensor<
    1x16x64x64xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputSET_CMX = !VPU.DistributedTensor<
    1x1x64x64xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_CMX = !VPU.SparseTensor<data=!InputData_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSET_CMX,
                  #VPU.SEInterpolate<
                      mode = <NEAREST>,
                      coordinate_transformation_mode = <ASYMMETRIC>,
                      scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                      nearest_mode = <FLOOR>,
                      initial_input_shape = [1, 16, 32, 32],
                      initial_output_shape = [1, 16, 64, 64]
              >>

!Output_CMX = !VPU.DistributedTensor<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Weights_CMX = tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!InputDataStub_CMX = tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSMStub_CMX = tensor<1x16x64x64xi1, {mem_space = @CMX_NN, order = #NHWC}>
!InputSETStub_CMX = tensor<1x1x64x64xi32, {mem_space = @CMX_NN, order = #NHWC}>
!InputStub_CMX = !VPU.SparseTensor<data=!InputDataStub_CMX, sparsity_map=!InputSMStub_CMX, storage_element_table=!InputSETStub_CMX,
                      #VPU.SEInterpolate<
                          mode = <NEAREST>,
                          coordinate_transformation_mode = <ASYMMETRIC>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]
                  >>

// CHECK-LABEL: @SparseNearestNCEInterpolateInputWorkloadsSOHExtraLines
module @SparseNearestNCEInterpolateInputWorkloadsSOHExtraLines  {
  func.func @main(%arg0: !InputData_CMX, %arg1: !InputSM_CMX, %arg2: !InputSET_CMX, %arg3: !Weights_CMX, %arg4: !WeightsTable_CMX) -> !Output_CMX {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1, %arg2) {
                        seAttr = #VPU.SEInterpolate<
                          mode = <NEAREST>,
                          coordinate_transformation_mode = <ASYMMETRIC>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]>}
                    -> !Input_CMX

    %output_cmx = VPU.NCE.Interpolate(%input_sparse, %arg3, %arg4) {
              minimumHardwareExecutionCost = 2886 : i64,
              mode = #VPU.nce_interpolate_mode<NEAREST>,
              ppe = #VPU.PPEStub<>,
              rawFilterShape = [16, 16, 1, 1],
              strides = [1, 1]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 32, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Interpolate

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 32, 64]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 32, 64]
  // CHECK-SAME:          outOffsets [0, 0, 32, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputData_CMX = !VPU.DistributedTensor<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]
}>

!InputSM_CMX = !VPU.DistributedTensor<
    1x16x64x64xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 32, 64], [1, 16, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 16, 32, 64], [1, 16, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
}>

!InputSET_CMX = !VPU.DistributedTensor<
    1x1x64x64xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 32, 64], [1, 1, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 1, 32, 64], [1, 1, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
}>

!Input_CMX = !VPU.SparseTensor<data=!InputData_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSET_CMX,
                  #VPU.SEInterpolate<
                      mode = <NEAREST>,
                      coordinate_transformation_mode = <ASYMMETRIC>,
                      scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                      nearest_mode = <FLOOR>,
                      initial_input_shape = [1, 16, 32, 32],
                      initial_output_shape = [1, 16, 64, 64]
              >>

!Output_CMX = !VPU.DistributedTensor<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 32, 64], [1, 16, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 16, 32, 64], [1, 16, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
}>

!Weights_CMX = tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!InputDataStub_CMX = tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSMStub_CMX = tensor<1x16x64x64xi1, {mem_space = @CMX_NN, order = #NHWC}>
!InputSETStub_CMX = tensor<1x1x64x64xi32, {mem_space = @CMX_NN, order = #NHWC}>
!InputStub_CMX = !VPU.SparseTensor<data=!InputDataStub_CMX, sparsity_map=!InputSMStub_CMX, storage_element_table=!InputSETStub_CMX,
                      #VPU.SEInterpolate<
                          mode = <NEAREST>,
                          coordinate_transformation_mode = <ASYMMETRIC>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]
                  >>

// CHECK-LABEL: @SparseNearestNCEInterpolateInputWorkloadsSOHExtraLinesWithExplicitOffset
module @SparseNearestNCEInterpolateInputWorkloadsSOHExtraLinesWithExplicitOffset  {
  func.func @main(%arg0: !InputData_CMX, %arg1: !InputSM_CMX, %arg2: !InputSET_CMX, %arg3: !Weights_CMX, %arg4: !WeightsTable_CMX) -> !Output_CMX {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1, %arg2) {
                        seAttr = #VPU.SEInterpolate<
                          mode = <NEAREST>,
                          coordinate_transformation_mode = <ASYMMETRIC>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]>}
                    -> !Input_CMX

    %output_cmx = VPU.NCE.Interpolate(%input_sparse, %arg3, %arg4) {
              minimumHardwareExecutionCost = 2886 : i64,
              mode = #VPU.nce_interpolate_mode<NEAREST>,
              ppe = #VPU.PPEStub<>,
              rawFilterShape = [16, 16, 1, 1],
              strides = [1, 1]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 32, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Interpolate

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 32, 64]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 32, 64]
  // CHECK-SAME:          outOffsets [0, 0, 32, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputData_CMX = !VPU.DistributedTensor<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [4, 4],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [2, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments
}>

!InputSM_CMX = !VPU.DistributedTensor<
    1x16x130x130xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [4, 4],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [2, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments
}>

!InputSET_CMX = !VPU.DistributedTensor<
    1x1x130x130xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [4, 4],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [2, 2],
    num_clusters = 2 : i64,
    uniform_distributed_segments
}>

!Input_CMX = !VPU.SparseTensor<data=!InputData_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSET_CMX,
                  #VPU.SEInterpolate<
                      mode = <BILINEAR>,
                      coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                      scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                      nearest_mode = <FLOOR>,
                      initial_input_shape = [1, 16, 32, 32],
                      initial_output_shape = [1, 16, 64, 64]
              >>

!Output_CMX = !VPU.DistributedTensor<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Weights_CMX = tensor<16x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!InputDataStub_CMX = tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSMStub_CMX = tensor<1x16x130x130xi1, {mem_space = @CMX_NN, order = #NHWC}>
!InputSETStub_CMX = tensor<1x1x130x130xi32, {mem_space = @CMX_NN, order = #NHWC}>
!InputStub_CMX = !VPU.SparseTensor<data=!InputDataStub_CMX, sparsity_map=!InputSMStub_CMX, storage_element_table=!InputSETStub_CMX,
                      #VPU.SEInterpolate<
                          mode = <BILINEAR>,
                          coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]
                  >>

// CHECK-LABEL: @SparseBilinearNCEInterpolateInputWorkloadsSOHExtraLines
module @SparseBilinearNCEInterpolateInputWorkloadsSOHExtraLines  {
  func.func @main(%arg0: !InputData_CMX, %arg1: !InputSM_CMX, %arg2: !InputSET_CMX, %arg3: !Weights_CMX, %arg4: !WeightsTable_CMX) -> !Output_CMX {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1, %arg2) {
                        seAttr = #VPU.SEInterpolate<
                          mode = <BILINEAR>,
                          coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]>}
                    -> !Input_CMX

    %output_cmx = VPU.NCE.Interpolate(%input_sparse, %arg3, %arg4) {
              minimumHardwareExecutionCost = 14721 : i64,
              mode = #VPU.nce_interpolate_mode<BILINEAR>,
              ppe = #VPU.PPEStub<>,
              rawFilterShape = [16, 16, 4, 4],
              strides = [2, 2]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 32, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Interpolate

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 66, 130]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 66, 130]
  // CHECK-SAME:          outOffsets [0, 0, 32, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputData_CMX = !VPU.DistributedTensor<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 17, 32], [1, 16, 17, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    uniform_distributed_segments
}>

!InputSM_CMX = !VPU.DistributedTensor<
    1x16x130x130xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 65, 130], [1, 16, 65, 130]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 65, 0]],
    memory_shapes = [[1, 16, 66, 130], [1, 16, 66, 130]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    uniform_distributed_segments
}>

!InputSET_CMX = !VPU.DistributedTensor<
    1x1x130x130xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 65, 130], [1, 1, 65, 130]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 65, 0]],
    memory_shapes = [[1, 1, 66, 130], [1, 1, 66, 130]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    uniform_distributed_segments
}>

!Input_CMX = !VPU.SparseTensor<data=!InputData_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSET_CMX,
                  #VPU.SEInterpolate<
                      mode = <BILINEAR>,
                      coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                      scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                      nearest_mode = <FLOOR>,
                      initial_input_shape = [1, 16, 32, 32],
                      initial_output_shape = [1, 16, 64, 64]
              >>

!Output_CMX = !VPU.DistributedTensor<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 32, 64], [1, 16, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 16, 32, 64], [1, 16, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    uniform_distributed_segments
}>

!Weights_CMX = tensor<16x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTable_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

!InputDataStub_CMX = tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSMStub_CMX = tensor<1x16x130x130xi1, {mem_space = @CMX_NN, order = #NHWC}>
!InputSETStub_CMX = tensor<1x1x130x130xi32, {mem_space = @CMX_NN, order = #NHWC}>
!InputStub_CMX = !VPU.SparseTensor<data=!InputDataStub_CMX, sparsity_map=!InputSMStub_CMX, storage_element_table=!InputSETStub_CMX,
                      #VPU.SEInterpolate<
                          mode = <BILINEAR>,
                          coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]
                  >>

// CHECK-LABEL: @SparseBilinearNCEInterpolateInputWorkloadsSOHExtraLines
module @SparseBilinearNCEInterpolateInputWorkloadsSOHExtraLines  {
  func.func @main(%arg0: !InputData_CMX, %arg1: !InputSM_CMX, %arg2: !InputSET_CMX, %arg3: !Weights_CMX, %arg4: !WeightsTable_CMX) -> !Output_CMX {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1, %arg2) {
                        seAttr = #VPU.SEInterpolate<
                          mode = <BILINEAR>,
                          coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                          nearest_mode = <FLOOR>,
                          initial_input_shape = [1, 16, 32, 32],
                          initial_output_shape = [1, 16, 64, 64]>}
                    -> !Input_CMX

    %output_cmx = VPU.NCE.Interpolate(%input_sparse, %arg3, %arg4) {
              minimumHardwareExecutionCost = 14721 : i64,
              mode = #VPU.nce_interpolate_mode<BILINEAR>,
              ppe = #VPU.PPEStub<>,
              rawFilterShape = [16, 16, 4, 4],
              strides = [2, 2]
          } -> !Output_CMX {
              VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
              VPU.DPU.Workload outOffsets [0, 0, 32, 0] outSizes [1, 16, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output_cmx : !Output_CMX
  }

  //CHECK:            VPU.NCE.Interpolate

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 66, 130]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 0

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 66, 130]
  // CHECK-SAME:          outOffsets [0, 0, 32, 0]
  // CHECK-SAME:          outSizes [1, 16, 32, 64]
  // CHECK-SAME:          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:          cluster_id = 1

}
