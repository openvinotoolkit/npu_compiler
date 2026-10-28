//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --dpu-profiling %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, [@CMX_NN, 0]>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, [@CMX_NN, 0]>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

// Single NCE task with 3 DPU variants fits entirely within the pipelining subbuffer
// (1024 bytes reserved), so no CMX->DDR flush is inserted mid-task.
// The pass allocates a profiling buffer in CMX and a pipelined DMA to drain it to DDR.

// CHECK-LABEL: module @DpuProfilingSimple
module @DpuProfilingSimple  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  // CHECK:       profilingOutputsInfo : {
  // CHECK-NEXT:    DataInfo "dpu" : tensor<6xui64>
  }
  config.Resources 4 of @NCE at 1.850000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DpuProfilingReservedMemory {
        config.MemoryResource 2048 bytes of @CMX_NN offset 1472000
      }
    }
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> !Output_CMX
    %1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %11 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>,
            profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 0 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>
        }> input(%arg0 : !Input_CMX)
              weights(%arg1 : !Weights_CMX)
              parent_input(%arg0 : !Input_CMX)
              parent_output(%0 : !Output_CMX)
              outputs(%0 : !Output_CMX)
              -> !Output_CMX variants :  {
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
      } PPE :  {
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA inputs(%0 : !Output_CMX) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    }
    return %arg3 : !Output_DDR
  }
}

// CHECK:       [[BARRIER_NCE:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:       [[PROF_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<6xui64, [@CMX_NN, 0]>
// CHECK:       [[BARRIER_FLUSH:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH]] : !VPURT.Barrier, !VPURT.Barrier)
// CHECK-NEXT:    profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 0 : i64, maxVariants = 3 : i64, numVariants = 3 : i64, clusterId = 0 : i64>{{.*}}profiling_data([[PROF_CMX]] : memref<6xui64, [@CMX_NN, 0]>)
// CHECK:       DPUTask {{.*}}workload_id = 0 : i64{{.*}}
// CHECK:       DPUTask {{.*}}workload_id = 1 : i64{{.*}}
// CHECK:       DPUTask {{.*}}workload_id = 2 : i64{{.*}}
// CHECK:       [[PROF_CMX_READ:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<6xui64, [@CMX_NN, 0]>
// CHECK:       [[PROF_DDR:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<6xui64, @DDR>
// CHECK:       VPURT.Task waits([[BARRIER_FLUSH]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}> inputs([[PROF_CMX_READ]] : memref<6xui64, [@CMX_NN, 0]>) outputs([[PROF_DDR]] : memref<6xui64, @DDR>)

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x48x60x60xf16, #NHWC, @DDR>

!Input_CMX = memref<1x16x62x62xf16, #NHWC, [@CMX_NN, 0]>
!Output_CMX = memref<1x48x60x60xf16, #NHWC, [@CMX_NN, 0]>
!Weights_CMX = memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

// Two NCE tasks each with 2 DPU variants, but only 64 bytes (4x ui64) are reserved as the
// pipelining subbuffer. Each task needs 4 ui64 (2 variants x 2 ui64 per variant), exactly
// filling the subbuffer. The second task cannot reuse the subbuffer until the first task's
// profiling data has been flushed to DDR via a CMX->DDR DMA. The pass inserts an
// intermediate barrier and flush DMA between the two NCE tasks.

// CHECK-LABEL: module @DpuProfilingWithCMX2DDRFlush
module @DpuProfilingWithCMX2DDRFlush  {

  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x16x62x62xf16>
    DataInfo "weights" : tensor<48x16x3x3xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16>
  } profilingOutputsInfo :  {
  // CHECK:       profilingOutputsInfo : {
  // CHECK-NEXT:    DataInfo "dpu" : tensor<8xui64>
  }
  config.Resources 4 of @NCE at 1.850000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DpuProfilingReservedMemory {
        config.MemoryResource 64 bytes of @CMX_NN offset 1472000
      }
    }
  }

  func.func @main(%arg0: !Input_CMX, %arg1: !Weights_CMX, %arg3: !Output_DDR) -> !Output_DDR {

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> !Output_CMX
    %1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %11 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>,
            profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 0 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>
        }> input(%arg0 : !Input_CMX)
              weights(%arg1 : !Weights_CMX)
              parent_input(%arg0 : !Input_CMX)
              parent_output(%0 : !Output_CMX)
              outputs(%0 : !Output_CMX)
              -> !Output_CMX variants :  {
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
      } PPE :  {
      }
    }
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %11 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>,
            profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 0 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>
        }> input(%arg0 : !Input_CMX)
              weights(%arg1 : !Weights_CMX)
              parent_input(%arg0 : !Input_CMX)
              parent_output(%0 : !Output_CMX)
              outputs(%0 : !Output_CMX)
              -> !Output_CMX variants :  {
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
              DPUTask { cluster_id = 0, outEnd = [59, 59, 47], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
      } PPE :  {
      }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA inputs(%0 : !Output_CMX) outputs(%arg3 : !Output_DDR) -> !Output_DDR
    }
    return %arg3 : !Output_DDR
  }
}

// CHECK:       [[BARRIER_NCE:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:       [[PROF0_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<4xui64, [@CMX_NN, 0]>
// CHECK:       [[BARRIER_FLUSH0:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH0]] : !VPURT.Barrier, !VPURT.Barrier)
// CHECK-NEXT:    profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 0 : i64, taskId = 0 : i64, maxVariants = 2 : i64, numVariants = 2 : i64, clusterId = 0 : i64>{{.*}}profiling_data([[PROF0_CMX]] : memref<4xui64, [@CMX_NN, 0]>)
// CHECK:       DPUTask {{.*}}workload_id = 0 : i64{{.*}}
// CHECK:       DPUTask {{.*}}workload_id = 1 : i64{{.*}}
// CHECK:       [[PROF0_CMX_READ:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472000> -> memref<4xui64, [@CMX_NN, 0]>
// CHECK:       [[PROF0_DDR:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui64, @DDR>
// CHECK:       VPURT.Task waits([[BARRIER_FLUSH0]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}> inputs([[PROF0_CMX_READ]] : memref<4xui64, [@CMX_NN, 0]>) outputs([[PROF0_DDR]] : memref<4xui64, @DDR>)
// CHECK:       [[PROF1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472032> -> memref<4xui64, [@CMX_NN, 0]>
// CHECK:       [[BARRIER_FLUSH1:%.+]] = VPURT.DeclareVirtualBarrier
// CHECK:       VPURT.Task updates([[BARRIER_NCE]], [[BARRIER_FLUSH1]] : !VPURT.Barrier, !VPURT.Barrier)
// CHECK-NEXT:    profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 1 : i64, maxVariants = 2 : i64, numVariants = 2 : i64, clusterId = 0 : i64>{{.*}}profiling_data([[PROF1_CMX]] : memref<4xui64, [@CMX_NN, 0]>)
// CHECK:       DPUTask {{.*}}workload_id = 0 : i64{{.*}}
// CHECK:       DPUTask {{.*}}workload_id = 1 : i64{{.*}}
// CHECK:       [[PROF1_CMX_READ:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1472032> -> memref<4xui64, [@CMX_NN, 0]>
// CHECK:       [[PROF1_DDR:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <32> -> memref<4xui64, @DDR>
// CHECK:       VPURT.Task waits([[BARRIER_FLUSH1]] : !VPURT.Barrier)
// CHECK-NEXT:    VPUIP.NNDMA <{port = 0 : i64, profiling_buffer_mgmt}> inputs([[PROF1_CMX_READ]] : memref<4xui64, [@CMX_NN, 0]>) outputs([[PROF1_DDR]] : memref<4xui64, @DDR>)
