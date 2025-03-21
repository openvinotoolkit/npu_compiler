//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --constant-dpu-prof-hwp-base %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputType = memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>
!WeightType = memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputType = memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]>
!ProfType = memref<4xui64, [@CMX_NN, 0]>
!ProfOutputType = memref<4xui64>


func.func @main(%input: !InputType, %weights: !WeightType, %weightsTable: !WeightTableType) -> (!OutputType, !ProfOutputType) {

  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %profBuf = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> !ProfType
  %output = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> !OutputType
  %profOutput = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> !ProfOutputType


  VPURT.Task updates(%bar0 : !VPURT.Barrier) {
    %0:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], minimumHardwareExecutionCost = 1 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    input(%input : !InputType)
    weights(%weights : !WeightType)
    weight_table(%weightsTable : !WeightTableType )
    parent_input(%input : !InputType)
    parent_output(%output : !OutputType)
    outputs(%output: !OutputType)
    profiling_data(%profBuf : !ProfType)
    -> !OutputType, !ProfType variants : {
      DPUTask {inEnd = [63, 63, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, workload_id = 0 : i64}
    } PPE : {
    }
  }

  VPURT.Task waits(%bar0: !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%profBuf : !ProfType) outputs(%profOutput : !ProfOutputType) -> !ProfOutputType
  }

  return %output, %profOutput : !OutputType, !ProfOutputType

  //CHECK:    [[PROF_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<4xui64, [@CMX_NN, 0]>
  //CHECK:    [[NEW_PROF_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4xui64, [@CMX_NN, 0]>
  //CHECK:    [[OUT_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]>
  //CHECK:        VPUIP.NCEClusterTask
  //CHECK-SAME:   profiling_data([[NEW_PROF_BUF]] : memref<4xui64, [@CMX_NN, 0]>)
  //CHECK:        DPUTask
  //CHECK-SAME:   workload_id = 32 : i64

  //CHECK:        VPUIP.NNDMA
  //CHECK-SAME:   inputs([[PROF_BUF]] : memref<4xui64, [@CMX_NN, 0]>)
}
