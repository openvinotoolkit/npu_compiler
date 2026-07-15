//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --dpu-profiling %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, {order = #NHWC}, @DDR>
!Output_DDR_SM = memref<1x48x60x60xi1, {order = #NHWC}, @DDR>

!Input_CMX = memref<1x16x62x62xf16, {order = #NHWC}, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, {order = #NHWC}, @CMX_NN>
!Output_CMX_SM = memref<1x48x60x60xi1, {order = #NHWC}, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @DpuProfilingSparse
module @DpuProfilingSparse  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = memref.alloc() : !Output_CMX
    %sm = memref.alloc() : !Output_CMX_SM

    %1:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>  input(%arg0 : !Input_CMX)
            weights(%arg1 : !Weights_CMX)
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
  //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.+]]x[[PROFDATA_INFO_TENSOR_TYPE:.+]]>
  //CHECK:        func.func @main([[ARG0:%.+]]: memref<1x16x62x62xf16, {order = #NHWC}, @CMX_NN>, [[ARG1:%.+]]: memref<48x16x3x3xf16, {order = #NHWC}, @CMX_NN>, [[ARG3:%.+]]: memref<1x48x60x60xf16, {order = #NHWC}, @DDR>, [[ARG4:%.+]]: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x48x60x60xf16, {order = #NHWC}, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xf16, {order = #NHWC}, @CMX_NN>
  //CHECK:        [[PROF_BUF_CMX:%.+]] = memref.alloc() {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.+]]} : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
  //CHECK:        [[SPARSITY_MAP_BUF_CMX:%.+]] = memref.alloc() : memref<1x48x60x60xi1, {order = #NHWC}, @CMX_NN>
  //CHECK:        [[PROF_VIEW:%.+]] = VPUIP.SubView [[PROF_BUF_CMX]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]> to memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>

  //CHECK:        [[NCE_RES:%[0-9]+]]:3 = VPUIP.NCEClusterTask
  //CHECK-SAME:   #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>
  //CHECK-SAME:   output_sparsity_map([[SPARSITY_MAP_BUF_CMX]] : memref<1x48x60x60xi1, {order = #NHWC}, @CMX_NN>)
  //CHECK-SAME:   profiling_data([[PROF_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>)

  //CHECK:        [[PROF_OUTPUT_VIEW:%.+]] = VPUIP.SubView [[ARG4]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]> to memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
  //CHECK:        [[PROF_CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCE_RES]]#2 : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[PROF_BUF_CMX]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) -> memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>
  //CHECK:        [[COPY_PROF_TO_DDR:%.+]] = VPUIP.NNDMA <{profiling_buffer_mgmt}> inputs([[PROF_CONCAT]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], [@CMX_NN, 0]>) outputs([[PROF_OUTPUT_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>
  //CHECK:        [[OUTPUT_SPARSITY_MAP_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xi1, {order = #NHWC}, @DDR>
  //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NNDMA inputs([[NCE_RES]]#0 : memref<1x48x60x60xf16, {order = #NHWC}, @CMX_NN>) outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>)
  //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.+]] = VPUIP.NNDMA inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>) outputs([[ARG3]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>)
  //CHECK:        [[COPY_SPARSITY_MAP_TO_DDR:%.+]] = VPUIP.NNDMA inputs([[NCE_RES]]#1 : memref<1x48x60x60xi1, {order = #NHWC}, @CMX_NN>) outputs([[OUTPUT_SPARSITY_MAP_DDR]] : memref<1x48x60x60xi1, {order = #NHWC}, @DDR>)

  //CHECK:        [[PROF_RES:%.+]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) outputs([[ARG4]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
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

!Output_DDR = memref<1x48x60x60xf16, {order = #NHWC}, @DDR>
!Output_DDR_SM = memref<1x48x60x60xi1, {order = #NHWC}, @DDR>

!Input_CMX = memref<1x16x62x62xf16, {order = #NHWC}, @CMX_NN>
!Output_CMX = memref<1x48x60x60xf16, {order = #NHWC}, @CMX_NN>
!Output_CMX_SM = memref<1x48x60x60xi1, {order = #NHWC}, @CMX_NN>
!Weights_CMX = memref<48x16x3x3xf16, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @DpuProfilingSparseWithMulticlustering
module @DpuProfilingSparseWithMulticlustering  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.AllocDistributed -> !OutputDistributed
    %sm = VPURT.AllocDistributed -> !OutputDistributed_SM
    %1:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [3, 3],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
      }> input(%arg0 : !Input_CMX)
        weights(%arg1 : !Weights_CMX)
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
  //CHECK-NEXT:   DataInfo "dpu" : tensor<[[PROFDATA_INFO_TENSOR_SIZE:.+]]x[[PROFDATA_INFO_TENSOR_TYPE:.+]]>
  //CHECK:        func.func @main([[ARG0:%.+]]: memref<1x16x62x62xf16, {order = #NHWC}, @CMX_NN>, [[ARG1:%.+]]: memref<48x16x3x3xf16, {order = #NHWC}, @CMX_NN>, [[ARG3:%.+]]: memref<1x48x60x60xf16, {order = #NHWC}, @DDR>, [[ARG4:%.+]]: memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) -> (memref<1x48x60x60xf16, {order = #NHWC}, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_CMX:%.+]]   = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
  //CHECK:        [[PROF_BUF_CMX:%.+]]     = VPURT.AllocDistributed {alignment = [[PROF_BUF_CMX_ALIGN_ATTR:.+]]} -> !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4], num_clusters = 4 : i64, uniform_distributed_segments}>
  //CHECK:        [[SPARSITY_MAP_BUF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
  //CHECK:        [[PROF_BUF_VIEW_CMX:%.+]] =   VPUIP.SubView [[PROF_BUF_CMX]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]]

  //CHECK:        [[NCE_RES:%[0-9]+]]:3 = VPUIP.NCEClusterTask
  //CHECK-SAME:   #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 1 : i64, maxVariants = 1 : i64>
  //CHECK-SAME:   input([[ARG0]] : memref<1x16x62x62xf16, {order = #NHWC}, @CMX_NN>)
  //CHECK-SAME:   weights([[ARG1]] : memref<48x16x3x3xf16, {order = #NHWC}, @CMX_NN>)
  //CHECK-SAME:   outputs([[OUTPUT_BUF_CMX]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN
  //CHECK-SAME:   output_sparsity_map([[SPARSITY_MAP_BUF_CMX]] : !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN
  //CHECK-SAME:   profiling_data([[PROF_BUF_VIEW_CMX]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN

  //CHECK:        [[PROF_OUTPUT_VIEW:%.+]] = VPUIP.SubView [[ARG4]] [0] [[[PROFDATA_INFO_TENSOR_SIZE]]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>
  //CHECK:        [[PROF_VIEW_CMX_CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCE_RES]]#2
  //CHECK-SAME:       outputs([[PROF_BUF_CMX]]

  //CHECK:        [[COPY_PROF_TO_DDR:%.+]] = VPUIP.NNDMA <{profiling_buffer_mgmt}>
  //CHECK-SAME:       inputs([[PROF_VIEW_CMX_CONCAT]] : !VPUIP.DistributedBuffer<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]], #C, @CMX_NN
  //CHECK-SAME:       outputs([[PROF_OUTPUT_VIEW]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        [[OUTPUT_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>

  //CHECK:        [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.NNDMA
  //CHECK-SAME:       inputs([[NCE_RES]]#0 : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
  //CHECK-SAME:       outputs([[OUTPUT_BUF_DDR]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>)

  //CHECK:        [[COPY_OUTPUT_TO_RESULT:%.+]] = VPUIP.NNDMA inputs([[COPY_OUTPUT_TO_DDR]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>) outputs([[ARG3]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>)

  //CHECK:        [[SPARSITY_MAP_BUF_DDR:%.+]] = memref.alloc() : memref<1x48x60x60xi1, {order = #NHWC}, @DDR>
  //CHECK:        [[COPY_SPARSITY_MAP_TO_DDR:%.+]] = VPUIP.NNDMA
  //CHECK-SAME:       inputs([[NCE_RES]]#1 : !VPUIP.DistributedBuffer<1x48x60x60xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
  //CHECK-SAME:       outputs([[SPARSITY_MAP_BUF_DDR]] : memref<1x48x60x60xi1, {order = #NHWC}, @DDR>)

  //CHECK:        [[PROF_RES:%.+]] = VPUIP.ConcatView inputs([[COPY_PROF_TO_DDR]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>) outputs([[ARG4]] : memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>)

  //CHECK:        return [[COPY_OUTPUT_TO_RESULT]], [[PROF_RES]] : memref<1x48x60x60xf16, {order = #NHWC}, @DDR>, memref<[[PROFDATA_INFO_TENSOR_SIZE]]x[[PROFDATA_INFO_TENSOR_TYPE]]>

}
