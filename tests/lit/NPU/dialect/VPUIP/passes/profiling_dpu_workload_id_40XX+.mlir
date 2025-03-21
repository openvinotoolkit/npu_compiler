//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --dpu-profiling %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfiling
module @DpuProfiling  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = memref.alloc() : !Output_CMX
    %1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }  input(%arg0 : !Input_CMX)
            weights(%arg1 : !Weights_CMX)
            weight_table(%arg2 : !WeightsTable_CMX)
            parent_input(%arg0 : !Input_CMX)
            parent_output(%0 : !Output_CMX)
            outputs(%0 : !Output_CMX)
            -> !Output_CMX variants :  {
            DPUTask {
                outEnd = [59, 59, 47],
                mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                outStart = [0, 0, 0]
            }
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.Copy inputs(%1 : !Output_CMX) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        VPUIP.NCEClusterTask
    //CHECK:        DPUTask
    //CHECK-SAME:   workload_id = 0 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x48x60x60xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingWithMulticlustering
module @DpuProfilingWithMulticlustering  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.AllocDistributed -> !OutputDistributed
    %1 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [3, 3],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
      } input(%arg0 : !Input_CMX)
        weights(%arg1 : !Weights_CMX)
        weight_table(%arg2 : !WeightsTable_CMX)
        parent_input(%arg0 : !Input_CMX)
        parent_output(%0 : !OutputDistributed)
        outputs(%0 : !OutputDistributed)
            -> !OutputDistributed variants :  {
      DPUTask {cluster_id = 0 : i64, outEnd = [59, 14, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      DPUTask {cluster_id = 1 : i64, outEnd = [59, 29, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 15, 0]}
      DPUTask {cluster_id = 2 : i64, outEnd = [59, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 30, 0]}
      DPUTask {cluster_id = 3 : i64, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 45, 0]}
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.Copy inputs(%1 : !OutputDistributed) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.Copy inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        VPUIP.NCEClusterTask
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 0
    //CHECK-SAME:   workload_id = 0 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 1
    //CHECK-SAME:   workload_id = 0 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 2
    //CHECK-SAME:   workload_id = 0 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 3
    //CHECK-SAME:   workload_id = 0 : i64
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0_CMX = memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>
!Output0_CMX = memref<1x48x222x222xf16, #NHWC, [@CMX_NN, 0]>
!Weights0_CMX = memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTable0_CMX = memref<48x1x1x4xsi32, [@CMX_NN, 0]>

!Weights1_CMX = memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTable1_CMX = memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!Output1_CMX = memref<1x32x55x55xf16, #NHWC, [@CMX_NN, 0]>
!Output2_CMX = memref<1x32x55x55xf16, #NHWC, @CMX_NN>

!OutputDistributed = !VPUIP.DistributedBuffer<1x32x55x55xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64
}>

!Output_DDR = memref<1x32x55x55xf16, #NHWC>

// CHECK-LABEL: @DpuProfilingMultipleOps
module @DpuProfilingMultipleOps  {

  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x3x224x224xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x32x55x55xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: memref<1x3x224x224xf16, #NHWC>, %arg1: !Output_DDR) -> !Output_DDR {
    //CHECK:        func.func @main

    %0 = memref.alloc() : !Input0_CMX
    %1 = memref.alloc() : !Output0_CMX
    %2 = memref.alloc() : !Weights0_CMX
    %3 = memref.alloc() : !WeightsTable0_CMX
    %4 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [3, 3],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
        } input(%0 : !Input0_CMX)
          weights(%2 : !Weights0_CMX)
          weight_table(%3 : !WeightsTable0_CMX)
          parent_input(%0 : !Input0_CMX)
          parent_output(%1 : !Output0_CMX)
          outputs(%1 : !Output0_CMX)
          -> !Output0_CMX variants :
        {
          DPUTask {outEnd = [111, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
          DPUTask {outEnd = [221, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [112, 0, 0]}
          DPUTask {outEnd = [111, 89, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 45, 0]}
          DPUTask {outEnd = [221, 89, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [112, 45, 0]}
        } PPE :  {
    }
    //CHECK:        VPUIP.NCEClusterTask
    //CHECK:        DPUTask
    //CHECK-NOT:    cluster_id
    //CHECK-SAME:   workload_id = 0 : i64
    //CHECK:        DPUTask
    //CHECK-NOT:    cluster_id
    //CHECK-SAME:   workload_id = 1 : i64
    //CHECK:        DPUTask
    //CHECK-NOT:    cluster_id
    //CHECK-SAME:   workload_id = 2 : i64
    //CHECK:        DPUTask
    //CHECK-NOT:    cluster_id
    //CHECK-SAME:   workload_id = 3 : i64

    %15 = VPURT.AllocDistributed -> !OutputDistributed
    %16 = VPUIP.NCEClusterTask {
        task_type = #VPUIP.nce_task_type<ELTWISE>
        }
        input(%4 : !Output0_CMX)
        weights(%4 : !Output0_CMX)
        parent_input(%4 : !Output0_CMX)
        parent_output(%15 : !OutputDistributed)
        outputs(%15 : !OutputDistributed)
        -> !OutputDistributed variants :  {
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 10, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 21, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 11, 0]}
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 32, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 22, 0]}
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 43, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 33, 0]}
      DPUTask {cluster_id = 1 : i64, outEnd = [54, 43, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 33, 0]}
      DPUTask {cluster_id = 2 : i64, outEnd = [54, 54, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 44, 0]}
    } PPE :  {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    //CHECK:        VPUIP.NCEClusterTask
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 0
    //CHECK-SAME:   workload_id = 0 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 0
    //CHECK-SAME:   workload_id = 1 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 0
    //CHECK-SAME:   workload_id = 2 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 0
    //CHECK-SAME:   workload_id = 3 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 1
    //CHECK-SAME:   workload_id = 0 : i64
    //CHECK:        DPUTask
    //CHECK-SAME:   cluster_id = 2
    //CHECK-SAME:   workload_id = 0 : i64

    %17 = memref.alloc() : !Output_DDR
    %18 = VPUIP.Copy inputs(%16 : !OutputDistributed) outputs(%17 : !Output_DDR) -> !Output_DDR
    %19 = VPUIP.Copy inputs(%17 : !Output_DDR) outputs(%arg1 : !Output_DDR) -> !Output_DDR

    return %19: !Output_DDR

  }
}
