//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --dpu-profiling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfiling
module @DpuProfiling  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
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
    %3 = VPUIP.NNDMA inputs(%1 : !Output_CMX) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.NNDMA inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.*]]x[[PROFDATA_INFO_TENSOR_TYPE:.*]]>
    //CHECK:        func.func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

    //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
    //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc()
    //CHECK-SAME:   {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.*]]} : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
    //CHECK:        [[PROF_VIEW:%.+]] = VPUIP.SubView [[PROF_BUF_CMX]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]> to memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>

    //CHECK:        [[NCE_RES:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>
    //CHECK-SAME:   profiling_data([[PROF_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>)

    //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]> to memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
    //CHECK:        [[PROF_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#1 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) -> memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
    //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.NNDMA {profiling_buffer_mgmt} inputs([[PROF_CONCAT]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

    //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>
    //CHECK:        [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NNDMA inputs([[NCE_RES]]#0 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.NNDMA inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)

    //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) outputs(%arg4 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

    //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
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

  net.NetworkInfo entryPoint : @main inputsInfo :  {
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
    } PPE : {}
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.NNDMA inputs(%1 : !OutputDistributed) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.NNDMA inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    return %4 : !Output_DDR
  }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.*]]x[[PROFDATA_INFO_TENSOR_TYPE:.*]]>
    //CHECK:        func.func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

    //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
    //CHECK:        [[PROF_BUF_CMX:%.+]] =   VPURT.AllocDistributed {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.*]]} -> !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64, uniform_distributed_segments}>
    //CHECK:        [[PROF_BUF_VIEW_CMX:%.+]] =   VPUIP.SubView [[PROF_BUF_CMX]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]]

    //CHECK:        [[NCE_RES:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 1 : i64>
    //CHECK-SAME:   input(%arg0 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   weights(%arg1 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:   weight_table(%arg2 : memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
    //CHECK-SAME:   outputs([[OUTPUT_BUF_CMX]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    //CHECK-SAME:   profiling_data([[PROF_BUF_VIEW_CMX]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN

    //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
    //CHECK:        [[PROF_VIEW_CMX_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#1
    //CHECK-SAME:       outputs([[PROF_BUF_CMX]]

    //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.NNDMA {profiling_buffer_mgmt}
    //CHECK-SAME:       inputs([[PROF_VIEW_CMX_CONCAT]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN
    //CHECK-SAME:       outputs([[PROF_OUTPUT_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

    //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>

    //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NNDMA
    //CHECK-SAME:       inputs([[NCE_RES]]#0 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
    //CHECK-SAME:       outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)

    //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.NNDMA inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)
    //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) outputs(%arg4 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

    //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
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
  config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource  26214400 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x3x224x224xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x32x55x55xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: memref<1x3x224x224xf16, #NHWC>, %arg1: !Output_DDR) -> !Output_DDR {
    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.*]]x[[PROFDATA_INFO_TENSOR_TYPE:.*]]>
    //CHECK:        func.func @main(%arg0: memref<1x3x224x224xf16, #NHWC>, %arg1: memref<1x32x55x55xf16, #NHWC>, %arg2: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x32x55x55xf16, #NHWC>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)
    //CHECK:        [[BUFFER_D:%.*]] = VPURT.AllocDistributed {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.*]]} -> !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SPLIT_0:.*]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK:        [[BUFFER_0:%.+]] = memref.alloc() {alignment = [[PROF_BUF_CMX_ALIGN_ATTR]]} : memref<[[PROFDATA_INFO_TENSOR_SPLIT_1:.*]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>

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
          DPUTask {outEnd = [111, 134, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 90, 0]}
          DPUTask {outEnd = [221, 134, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [112, 90, 0]}
          DPUTask {outEnd = [111, 179, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 135, 0]}
          DPUTask {outEnd = [221, 179, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [112, 135, 0]}
          DPUTask {outEnd = [111, 221, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 180, 0]}
          DPUTask {outEnd = [221, 221, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [112, 180, 0]}
        } PPE :  {
    }
    //CHECK:        [[PROF_VIEW_OP_0:%.+]] = VPUIP.SubView [[BUFFER_0:%.+]] [[[VIEW_OFFSET_0:.*]]] [[[VIEW_SIZE_0:.*]]] : memref<[[BUFF_SPLIT:.*]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]> to memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_0:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 10 : i64, numVariants = 10 : i64, clusterId = 0 : i64>
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_0]] : memref<[[VIEW_SIZE_0:.*]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>)

    %5 = memref.alloc() : !Output0_CMX
    %6 = VPUIP.NCEClusterTask {
          task_type = #VPUIP.nce_task_type<ELTWISE>
          } input(%4 : !Output0_CMX)
            weights(%4 : !Output0_CMX)
            parent_input(%4 : !Output0_CMX)
            parent_output(%5 : !Output0_CMX)
            outputs(%5 : !Output0_CMX)
            -> !Output0_CMX variants :
          {
            DPUTask {outEnd = [31, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
            DPUTask {outEnd = [63, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [32, 0, 0]}
            DPUTask {outEnd = [95, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [64, 0, 0]}
            DPUTask {outEnd = [127, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [96, 0, 0]}
            DPUTask {outEnd = [159, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [128, 0, 0]}
            DPUTask {outEnd = [191, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [160, 0, 0]}
            DPUTask {outEnd = [221, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [192, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
    }

    //CHECK:        [[PROF_VIEW_OP_1:%.+]] = VPUIP.SubView [[BUFFER_0:%.+]] [[[VIEW_OFFSET_0:.*]]] [[[VIEW_SIZE_0:.*]]] : memref<[[BUFF_SPLIT:.*]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]> to memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_1:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 2 : i64, maxVariants = 7 : i64, numVariants = 7 : i64, clusterId = 0 : i64>
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_1]] : memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>)

    %7 = memref.alloc() : !Weights1_CMX
    %8 = memref.alloc() : !WeightsTable1_CMX
    %9 = memref.alloc() : !Output1_CMX
    %10 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [4, 4],
        task_type = #VPUIP.nce_task_type<CONV>
        } input(%6 : !Output0_CMX)
          weights(%7 : !Weights1_CMX)
          weight_table(%8 : !WeightsTable1_CMX)
          parent_input(%6 : !Output0_CMX)
          parent_output(%9 : !Output1_CMX)
          outputs(%9 : !Output1_CMX)
          -> !Output1_CMX variants :
        {
          DPUTask {outEnd = [11, 18, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
          DPUTask {outEnd = [23, 18, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [12, 0, 0]}
          DPUTask {outEnd = [35, 18, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [24, 0, 0]}
    } PPE :  {
    }

    //CHECK:        [[PROF_VIEW_OP_2:%.+]] = VPUIP.SubView [[BUFFER_0:%.+]] [[[VIEW_OFFSET_0:.*]]] [[[VIEW_SIZE_0:.*]]] : memref<[[BUFF_SPLIT:.*]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]> to memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
    //CHECK:        [[OP_RESULT_2:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 3 : i64, maxVariants = 3 : i64, numVariants = 3 : i64, clusterId = 0 : i64>
    //CHECK-SAME:   profiling_data([[PROF_VIEW_OP_2]] : memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>)

    //CHECK:        [[DDR_VIEW_0:%.*]] = VPUIP.SubView %arg2 [[[VIEW_OFFSET_0:.*]]] [[[VIEW_SIZE_0:.*]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]> to memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
    //CHECK:        [[PROF_CONCAT_0:%.*]] = VPUIP.ConcatView inputs([[PROF_CONCAT_OPS:.*]] : memref[[PROF_CONCAT_TYPES:.*]]) outputs([[BUFFER_0]] : memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) -> memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
    //CHECK:        [[COPY_PROF_TO_DDR_0:%.*]] = VPUIP.NNDMA {profiling_buffer_mgmt} inputs([[PROF_CONCAT_0]] : memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[DDR_VIEW_0]] : memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> memref<[[VIEW_SIZE_0]]x[[PROFDATA_INFO_TENSOR_TYPE]]>

    %11 = memref.alloc() : !Output1_CMX

    %12 = VPUIP.NCEClusterTask {
        task_type = #VPUIP.nce_task_type<ELTWISE>
        }
        input(%10 : !Output1_CMX)
        weights(%10 : !Output1_CMX)
        parent_input(%10 : !Output1_CMX)
        parent_output(%11 : !Output1_CMX)
        outputs(%11 : !Output1_CMX)
        -> !Output1_CMX variants :  {
      DPUTask {outEnd = [54, 10, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      DPUTask {outEnd = [54, 21, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 11, 0]}
      DPUTask {outEnd = [54, 32, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 22, 0]}
      DPUTask {outEnd = [54, 43, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 33, 0]}
      DPUTask {outEnd = [54, 54, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 44, 0]}
    } PPE :  {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %13 = VPURT.AllocDistributed -> !OutputDistributed
    %14 =  VPUIP.NCEClusterTask {
        task_type = #VPUIP.nce_task_type<ELTWISE>
        }
        input(%12 : !Output1_CMX)
        weights(%12 : !Output1_CMX)
        parent_input(%12 : !Output1_CMX)
        parent_output(%13 : !OutputDistributed)
        outputs(%13 : !OutputDistributed)
        -> !OutputDistributed variants :  {
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 10, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 21, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 11, 0]}
      DPUTask {cluster_id = 0 : i64, outEnd = [54, 32, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 22, 0]}
      DPUTask {cluster_id = 1 : i64, outEnd = [54, 43, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 33, 0]}
      DPUTask {cluster_id = 2 : i64, outEnd = [54, 54, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 44, 0]}
    } PPE :  {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    //CHECK:        [[PROF_VIEW_OP_4:%.+]] = VPUIP.SubView [[BUFFER_D]] [0] [[[PROF_VIEW_OP_4_SIZE:.*]]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SPLIT_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK-SAME:     to !VPUIP.DistributedBuffer<[[PROF_VIEW_OP_4_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK:        [[OP_RESULT_4:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:     profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 1 : i64, maxVariants = 3 : i64>
    //CHECK-SAME:     profiling_data([[PROF_VIEW_OP_4]] : !VPUIP.DistributedBuffer<[[PROF_VIEW_OP_4_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN

    %15 = VPURT.AllocDistributed -> !OutputDistributed
    %16 = VPUIP.NCEClusterTask {
        task_type = #VPUIP.nce_task_type<ELTWISE>
        }
        input(%14 : !OutputDistributed)
        weights(%14 : !OutputDistributed)
        parent_input(%14 : !OutputDistributed)
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

    //CHECK:        [[PROF_VIEW_OP_5:%.+]] = VPUIP.SubView [[BUFFER_D]] [[[PROF_VIEW_OFFSET:.*]]] [[[PROF_VIEW_OP_5_SIZE:.*]]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SPLIT_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK-SAME:     to !VPUIP.DistributedBuffer<[[PROF_VIEW_OP_5_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK:        [[OP_RESULT_5:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:     profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 2 : i64, maxVariants = 4 : i64>
    //CHECK-SAME:     profiling_data([[PROF_VIEW_OP_5]] : !VPUIP.DistributedBuffer<[[PROF_VIEW_OP_5_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN


    %17 = memref.alloc() : !Output_DDR
    %18 = VPUIP.NNDMA inputs(%16 : !OutputDistributed) outputs(%17 : !Output_DDR) -> !Output_DDR
    %19 = VPUIP.NNDMA inputs(%17 : !Output_DDR) outputs(%arg1 : !Output_DDR) -> !Output_DDR


    //CHECK:        [[DDR_VIEW_2:%.*]] = VPUIP.SubView %arg2 [[[DDR_VIEW_2_OFFSET:.*]]] [[[PROFDATA_INFO_TENSOR_SPLIT_0]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]> to memref<[[PROFDATA_INFO_TENSOR_SPLIT_0]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
    //CHECK:        [[PROF_CONCAT_3:%.*]] = VPUIP.ConcatView inputs([[OP_RESULT_4]]#1, [[OP_RESULT_5]]#1
    //CHECK-SAME:     !VPUIP.DistributedBuffer<[[PROF_VIEW_OP_4_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK-SAME:     !VPUIP.DistributedBuffer<[[PROF_VIEW_OP_5_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3], num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:     inputs([[PROF_CONCAT_3]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SPLIT_0]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN
    //CHECK-SAME:     outputs([[DDR_VIEW_2]] : memref<[[PROFDATA_INFO_TENSOR_SPLIT_0]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> memref<[[PROFDATA_INFO_TENSOR_SPLIT_0]]x[[PROFDATA_INFO_TENSOR_TYPE]]>

    //CHECK:        [[PROF_RESULT:%.*]] = VPUIP.ConcatView inputs([[CONCAT_VIEW_INPUTS:.*]] : [[CONCAT_VIEW_TYPES:.*]]) outputs(%arg2 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>

    return %19: !Output_DDR

    //CHECK:        return
    //CHECK-SAME:   [[PROF_RESULT]]
    //CHECK-SAME:   memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
  }
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>
!Output_DDR_SM = memref<1x48x60x60xi1, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Output_CMX_SM = memref<1x48x60x60xi1, #NHWC, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingSparse
module @DpuProfilingSparse  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = memref.alloc() : !Output_CMX
    %sm = memref.alloc() : !Output_CMX_SM

    %1:2 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }  input(%arg0 : !Input_CMX)
            weights(%arg1 : !Weights_CMX)
            weight_table(%arg2 : !WeightsTable_CMX)
            parent_input(%arg0 : !Input_CMX)
            parent_output(%0 : !Output_CMX)
            parent_output_sparsity_map(%sm : !Output_CMX_SM)
            outputs(%0 : !Output_CMX)
            output_sparsity_map(%sm : !Output_CMX_SM)
            -> !Output_CMX, !Output_CMX_SM variants :  {
            DPUTask {
                outEnd = [59, 59, 47],
                mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                outStart = [0, 0, 0]
            }
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %osm = memref.alloc() : !Output_DDR_SM
    %3 = VPUIP.NNDMA inputs(%1#0 : !Output_CMX) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.NNDMA inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    %5 = VPUIP.NNDMA inputs(%1#1 : !Output_CMX_SM) outputs(%osm : !Output_DDR_SM) -> !Output_DDR_SM
    return %4 : !Output_DDR
  }

  //CHECK:        profilingOutputsInfo
  //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.*]]x[[PROFDATA_INFO_TENSOR_TYPE:.*]]>
  //CHECK:        func.func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
  //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc() {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.*]]} : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
  //CHECK:        [[SPARSITY_MAP_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xi1, #NHWC, @CMX_NN>
  //CHECK:        [[PROF_VIEW:%.+]] = VPUIP.SubView [[PROF_BUF_CMX]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]> to memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>

  //CHECK:        [[NCE_RES:%[0-9]+]]:3 = VPUIP.NCEClusterTask
  //CHECK-SAME:   #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>
  //CHECK-SAME:   output_sparsity_map([[SPARSITY_MAP_BUF_CMX]] : memref<1x48x60x60xi1, #NHWC, @CMX_NN>)
  //CHECK-SAME:   profiling_data([[PROF_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>)

  //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]> to memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
  //CHECK:        [[PROF_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#2 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) -> memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
  //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.NNDMA {profiling_buffer_mgmt} inputs([[PROF_CONCAT]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>
  //CHECK:        [[OUTPUT_SPARSITY_MAP_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xi1, #NHWC, @DDR>
  //CHECK:        [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NNDMA inputs([[NCE_RES]]#0 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
  //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.NNDMA inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)
  //CHECK:        [[COPY_SPARSITY_MAP_TO_DDR:%.*]] = VPUIP.NNDMA inputs([[NCE_RES]]#1 : memref<1x48x60x60xi1, #NHWC, @CMX_NN>) outputs([[OUTPUT_SPARSITY_MAP_DDR]] : memref<1x48x60x60xi1, #NHWC, @DDR>)

  //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) outputs(%arg4 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x48x60x60xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!OutputDistributed_SM = !VPUIP.DistributedBuffer<
    1x48x60x60xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>
!Output_DDR_SM = memref<1x48x60x60xi1, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, @CMX_NN>
!Output_CMX_SM = memref<1x48x60x60xi1, #NHWC, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTable_CMX = memref<48x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @DpuProfilingSparseWithMulticlustering
module @DpuProfilingSparseWithMulticlustering  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
    DataInfo "weightsTable" : tensor<48x1x1x4xsi32>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg2: !WeightsTable_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.AllocDistributed -> !OutputDistributed
    %sm = VPURT.AllocDistributed -> !OutputDistributed_SM
    %1:2 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [3, 3],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
      } input(%arg0 : !Input_CMX)
        weights(%arg1 : !Weights_CMX)
        weight_table(%arg2 : !WeightsTable_CMX)
        parent_input(%arg0 : !Input_CMX)
        parent_output(%0 : !OutputDistributed)
        parent_output_sparsity_map(%sm : !OutputDistributed_SM)
        outputs(%0 : !OutputDistributed)
        output_sparsity_map(%sm : !OutputDistributed_SM)
            -> !OutputDistributed, !OutputDistributed_SM variants :  {
      DPUTask {cluster_id = 0 : i64, outEnd = [59, 14, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      DPUTask {cluster_id = 1 : i64, outEnd = [59, 29, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 15, 0]}
      DPUTask {cluster_id = 2 : i64, outEnd = [59, 44, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 30, 0]}
      DPUTask {cluster_id = 3 : i64, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 45, 0]}
    } PPE :  {
    }
    %2 = memref.alloc() : !Output_DDR
    %3 = VPUIP.NNDMA inputs(%1#0 : !OutputDistributed) outputs(%2 : !Output_DDR) -> !Output_DDR
    %4 = VPUIP.NNDMA inputs(%3 : !Output_DDR) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    %osm = memref.alloc() : !Output_DDR_SM
    %5 = VPUIP.NNDMA inputs(%1#1 : !OutputDistributed_SM) outputs(%osm : !Output_DDR_SM) -> !Output_DDR_SM
    return %4 : !Output_DDR
  }

  //CHECK:        profilingOutputsInfo
  //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.*]]x[[PROFDATA_INFO_TENSOR_TYPE:.*]]>
  //CHECK:        func.func @main(%arg0: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %arg1: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %arg2: memref<48x1x1x4xsi32, #NHWC, @CMX_NN>, %arg3: memref<1x48x60x60xf16, #NHWC, @DDR>, %arg4: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_CMX:%.+]]   = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
  //CHECK:        [[PROF_BUF_CMX:%.+]]     = VPURT.AllocDistributed {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.*]]} -> !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64, uniform_distributed_segments}>
  //CHECK:        [[SPARSITY_MAP_BUF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
  //CHECK:        [[PROF_BUF_VIEW_CMX:%.+]] =   VPUIP.SubView [[PROF_BUF_CMX]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]]

  //CHECK:        [[NCE_RES:%[0-9]+]]:3 = VPUIP.NCEClusterTask
  //CHECK-SAME:   #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 1 : i64>
  //CHECK-SAME:   input(%arg0 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:   weights(%arg1 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
  //CHECK-SAME:   weight_table(%arg2 : memref<48x1x1x4xsi32, #NHWC, @CMX_NN>)
  //CHECK-SAME:   outputs([[OUTPUT_BUF_CMX]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
  //CHECK-SAME:   output_sparsity_map([[SPARSITY_MAP_BUF_CMX]] : !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN
  //CHECK-SAME:   profiling_data([[PROF_BUF_VIEW_CMX]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN

  //CHECK:        [[PROF_OUTPUT_VIEW:%.*]] = VPUIP.SubView %arg4 [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
  //CHECK:        [[PROF_VIEW_CMX_CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_RES]]#2
  //CHECK-SAME:       outputs([[PROF_BUF_CMX]]

  //CHECK:        [[COPY_PROF_TO_DDR:%.*]] = VPUIP.NNDMA {profiling_buffer_mgmt}
  //CHECK-SAME:       inputs([[PROF_VIEW_CMX_CONCAT]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN
  //CHECK-SAME:       outputs([[PROF_OUTPUT_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, #NHWC, @DDR>

  //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NNDMA
  //CHECK-SAME:       inputs([[NCE_RES]]#0 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
  //CHECK-SAME:       outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)

  //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.*]] = VPUIP.NNDMA inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC, @DDR>)

  //CHECK:        [[SPARSITY_MAP_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xi1, #NHWC, @DDR>
  //CHECK:        [[COPY_SPARSITY_MAP_TO_DDR:%.+]] = VPUIP.NNDMA
  //CHECK-SAME:       inputs([[NCE_RES]]#1 : !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
  //CHECK-SAME:       outputs([[SPARSITY_MAP_BUF_DDR]] : memref<1x48x60x60xi1, #NHWC, @DDR>)

  //CHECK:        [[PROF_RES:%.*]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) outputs(%arg4 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, #NHWC, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>

}
