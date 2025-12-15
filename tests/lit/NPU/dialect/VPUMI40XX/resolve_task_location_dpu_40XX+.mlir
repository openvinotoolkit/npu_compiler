//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-task-location %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @multiple_clusters_dpu_soh_f16_f16_f16() {
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>

  %8 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>

  %11 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>

  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>

  %14 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

  %31 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
  }
  %32 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} input(%13 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) weights(%7 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%15 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) outputs(%10 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) -> <1:0:0> PPE : {
  }
  %33 = VPUMI40XX.DPUVariant calls(%31 : <0:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:0>
  %34 = VPUMI40XX.DPUVariant calls(%32 : <1:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<1:0:0>

  %mi = VPUMI40XX.MappedInference invariants(%31, %32 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) variants(%33, %34 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 1, 0, 0, 0, 0]) variantCount([1, 1, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: func.func @multiple_clusters_dpu_soh_f16_f16_f16
//CHECK-DAG: [[TB0:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB1:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<1:0:0>
//CHECK-DAG: [[TB2:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB3:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<1:0:0>

//CHECK: [[IVAR0:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TB2]] : !VPURegMapped.Index<0:0:0>)
//CHECK: [[IVAR1:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TB3]] : !VPURegMapped.Index<1:0:0>)

//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TB0]]
    //CHECK-SAME: calls([[IVAR0]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TB1]]
    //CHECK-SAME: calls([[IVAR1]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @manyDPUTasks() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

  %i0 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i1 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:1> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i2 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:2> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i3 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:3> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i4 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:4> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i5 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:5> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i6 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:6> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i7 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:7> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i8 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:8> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i9 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:9> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i10 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:10> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i11 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:11> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i12 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:12> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i13 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:13> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i14 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:14> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i15 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:15> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i16 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:16> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i17 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:17> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i18 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:18> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i19 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:19> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i20 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:20> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i21 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:21> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i22 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:22> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i23 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:23> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i24 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:24> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i25 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:25> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i26 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:26> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i27 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:27> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i28 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:28> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i29 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:29> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i30 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:30> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i31 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:31> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i32 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:32> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i33 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:33> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i34 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:34> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i35 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:35> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i36 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:36> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i37 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:37> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i38 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:38> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i39 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:39> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i40 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:40> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i41 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:41> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i42 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:42> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i43 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:43> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i44 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:44> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i45 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:45> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i46 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:46> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i47 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:47> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i48 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:48> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i49 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:49> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i50 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:50> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i51 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:51> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i52 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:52> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i53 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:53> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i54 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:54> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i55 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:55> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i56 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:56> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i57 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:57> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i58 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:58> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i59 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:59> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i60 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:60> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i61 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:61> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i62 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:62> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i63 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:63> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i64 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:64> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }
  %i65 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:65> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }

  %v0 = VPUMI40XX.DPUVariant calls(%i0 : <0:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:0>
  %v1 = VPUMI40XX.DPUVariant calls(%i1 : <0:0:1>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:1>
  %v2 = VPUMI40XX.DPUVariant calls(%i2 : <0:0:2>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:2>
  %v3 = VPUMI40XX.DPUVariant calls(%i3 : <0:0:3>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:3>
  %v4 = VPUMI40XX.DPUVariant calls(%i4 : <0:0:4>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:4>
  %v5 = VPUMI40XX.DPUVariant calls(%i5 : <0:0:5>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:5>
  %v6 = VPUMI40XX.DPUVariant calls(%i6 : <0:0:6>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:6>
  %v7 = VPUMI40XX.DPUVariant calls(%i7 : <0:0:7>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:7>
  %v8 = VPUMI40XX.DPUVariant calls(%i8 : <0:0:8>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:8>
  %v9 = VPUMI40XX.DPUVariant calls(%i9 : <0:0:9>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:9>
  %v10 = VPUMI40XX.DPUVariant calls(%i10 : <0:0:10>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:10>
  %v11 = VPUMI40XX.DPUVariant calls(%i11 : <0:0:11>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:11>
  %v12 = VPUMI40XX.DPUVariant calls(%i12 : <0:0:12>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:12>
  %v13 = VPUMI40XX.DPUVariant calls(%i13 : <0:0:13>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:13>
  %v14 = VPUMI40XX.DPUVariant calls(%i14 : <0:0:14>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:14>
  %v15 = VPUMI40XX.DPUVariant calls(%i15 : <0:0:15>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:15>
  %v16 = VPUMI40XX.DPUVariant calls(%i16 : <0:0:16>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:16>
  %v17 = VPUMI40XX.DPUVariant calls(%i17 : <0:0:17>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:17>
  %v18 = VPUMI40XX.DPUVariant calls(%i18 : <0:0:18>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:18>
  %v19 = VPUMI40XX.DPUVariant calls(%i19 : <0:0:19>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:19>
  %v20 = VPUMI40XX.DPUVariant calls(%i20 : <0:0:20>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:20>
  %v21 = VPUMI40XX.DPUVariant calls(%i21 : <0:0:21>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:21>
  %v22 = VPUMI40XX.DPUVariant calls(%i22 : <0:0:22>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:22>
  %v23 = VPUMI40XX.DPUVariant calls(%i23 : <0:0:23>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:23>
  %v24 = VPUMI40XX.DPUVariant calls(%i24 : <0:0:24>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:24>
  %v25 = VPUMI40XX.DPUVariant calls(%i25 : <0:0:25>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:25>
  %v26 = VPUMI40XX.DPUVariant calls(%i26 : <0:0:26>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:26>
  %v27 = VPUMI40XX.DPUVariant calls(%i27 : <0:0:27>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:27>
  %v28 = VPUMI40XX.DPUVariant calls(%i28 : <0:0:28>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:28>
  %v29 = VPUMI40XX.DPUVariant calls(%i29 : <0:0:29>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:29>
  %v30 = VPUMI40XX.DPUVariant calls(%i30 : <0:0:30>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:30>
  %v31 = VPUMI40XX.DPUVariant calls(%i31 : <0:0:31>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:31>
  %v32 = VPUMI40XX.DPUVariant calls(%i32 : <0:0:32>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:32>
  %v33 = VPUMI40XX.DPUVariant calls(%i33 : <0:0:33>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:33>
  %v34 = VPUMI40XX.DPUVariant calls(%i34 : <0:0:34>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:34>
  %v35 = VPUMI40XX.DPUVariant calls(%i35 : <0:0:35>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:35>
  %v36 = VPUMI40XX.DPUVariant calls(%i36 : <0:0:36>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:36>
  %v37 = VPUMI40XX.DPUVariant calls(%i37 : <0:0:37>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:37>
  %v38 = VPUMI40XX.DPUVariant calls(%i38 : <0:0:38>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:38>
  %v39 = VPUMI40XX.DPUVariant calls(%i39 : <0:0:39>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:39>
  %v40 = VPUMI40XX.DPUVariant calls(%i40 : <0:0:40>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:40>
  %v41 = VPUMI40XX.DPUVariant calls(%i41 : <0:0:41>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:41>
  %v42 = VPUMI40XX.DPUVariant calls(%i42 : <0:0:42>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:42>
  %v43 = VPUMI40XX.DPUVariant calls(%i43 : <0:0:43>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:43>
  %v44 = VPUMI40XX.DPUVariant calls(%i44 : <0:0:44>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:44>
  %v45 = VPUMI40XX.DPUVariant calls(%i45 : <0:0:45>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:45>
  %v46 = VPUMI40XX.DPUVariant calls(%i46 : <0:0:46>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:46>
  %v47 = VPUMI40XX.DPUVariant calls(%i47 : <0:0:47>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:47>
  %v48 = VPUMI40XX.DPUVariant calls(%i48 : <0:0:48>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:48>
  %v49 = VPUMI40XX.DPUVariant calls(%i49 : <0:0:49>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:49>
  %v50 = VPUMI40XX.DPUVariant calls(%i50 : <0:0:50>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:50>
  %v51 = VPUMI40XX.DPUVariant calls(%i51 : <0:0:51>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:51>
  %v52 = VPUMI40XX.DPUVariant calls(%i52 : <0:0:52>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:52>
  %v53 = VPUMI40XX.DPUVariant calls(%i53 : <0:0:53>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:53>
  %v54 = VPUMI40XX.DPUVariant calls(%i54 : <0:0:54>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:54>
  %v55 = VPUMI40XX.DPUVariant calls(%i55 : <0:0:55>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:55>
  %v56 = VPUMI40XX.DPUVariant calls(%i56 : <0:0:56>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:56>
  %v57 = VPUMI40XX.DPUVariant calls(%i57 : <0:0:57>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:57>
  %v58 = VPUMI40XX.DPUVariant calls(%i58 : <0:0:58>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:58>
  %v59 = VPUMI40XX.DPUVariant calls(%i59 : <0:0:59>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:59>
  %v60 = VPUMI40XX.DPUVariant calls(%i60 : <0:0:60>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:60>
  %v61 = VPUMI40XX.DPUVariant calls(%i61 : <0:0:61>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:61>
  %v62 = VPUMI40XX.DPUVariant calls(%i62 : <0:0:62>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:62>
  %v63 = VPUMI40XX.DPUVariant calls(%i63 : <0:0:63>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:63>
  %v64 = VPUMI40XX.DPUVariant calls(%i64 : <0:0:64>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:64>
  %v65 = VPUMI40XX.DPUVariant calls(%i65 : <0:0:65>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:65>

  %mi = VPUMI40XX.MappedInference invariants(%i0 : !VPURegMapped.Index<0:0:0>) variants(%v0 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([66, 0, 0, 0, 0, 0]) variantCount([66, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: func.func @manyDPUTasks

//CHECK-DAG: [[TBI0:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TBI1:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TBI2:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
//CHECK-DAG: [[TBI3:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:3>
//CHECK-DAG: [[TBI4:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:4>
//CHECK-DAG: [[TBI5:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:5>
//CHECK-DAG: [[TBI6:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:6>
//CHECK-DAG: [[TBI7:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:7>
//CHECK-DAG: [[TBI8:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:8>
//CHECK-DAG: [[TBI9:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:9>
//CHECK-DAG: [[TBI10:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:10>
//CHECK-DAG: [[TBI11:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:11>
//CHECK-DAG: [[TBI12:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:12>
//CHECK-DAG: [[TBI13:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:13>
//CHECK-DAG: [[TBI14:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:14>
//CHECK-DAG: [[TBI15:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:15>
//CHECK-DAG: [[TBI16:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:16>
//CHECK-DAG: [[TBI17:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:17>
//CHECK-DAG: [[TBI18:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:18>
//CHECK-DAG: [[TBI19:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:19>
//CHECK-DAG: [[TBI20:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:20>
//CHECK-DAG: [[TBI21:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:21>
//CHECK-DAG: [[TBI22:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:22>
//CHECK-DAG: [[TBI23:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:23>
//CHECK-DAG: [[TBI24:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:24>
//CHECK-DAG: [[TBI25:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:25>
//CHECK-DAG: [[TBI26:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:26>
//CHECK-DAG: [[TBI27:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:27>
//CHECK-DAG: [[TBI28:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:28>
//CHECK-DAG: [[TBI29:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:29>
//CHECK-DAG: [[TBI30:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:30>
//CHECK-DAG: [[TBI31:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:31>
//CHECK-DAG: [[TBI32:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:32>
//CHECK-DAG: [[TBI33:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:33>
//CHECK-DAG: [[TBI34:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:34>
//CHECK-DAG: [[TBI35:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:35>
//CHECK-DAG: [[TBI36:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:36>
//CHECK-DAG: [[TBI37:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:37>
//CHECK-DAG: [[TBI38:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:38>
//CHECK-DAG: [[TBI39:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:39>
//CHECK-DAG: [[TBI40:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:40>
//CHECK-DAG: [[TBI41:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:41>
//CHECK-DAG: [[TBI42:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:42>
//CHECK-DAG: [[TBI43:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:43>
//CHECK-DAG: [[TBI44:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:44>
//CHECK-DAG: [[TBI45:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:45>
//CHECK-DAG: [[TBI46:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:46>
//CHECK-DAG: [[TBI47:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:47>
//CHECK-DAG: [[TBI48:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:48>
//CHECK-DAG: [[TBI49:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:49>
//CHECK-DAG: [[TBI50:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:50>
//CHECK-DAG: [[TBI51:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:51>
//CHECK-DAG: [[TBI52:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:52>
//CHECK-DAG: [[TBI53:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:53>
//CHECK-DAG: [[TBI54:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:54>
//CHECK-DAG: [[TBI55:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:55>
//CHECK-DAG: [[TBI56:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:56>
//CHECK-DAG: [[TBI57:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:57>
//CHECK-DAG: [[TBI58:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:58>
//CHECK-DAG: [[TBI59:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:59>
//CHECK-DAG: [[TBI60:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:60>
//CHECK-DAG: [[TBI61:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:61>
//CHECK-DAG: [[TBI62:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:62>
//CHECK-DAG: [[TBI63:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUInvariant> -> !VPURegMapped.Index<0:0:63>

//CHECK-DAG: [[TBV0:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TBV1:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TBV2:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:2>
//CHECK-DAG: [[TBV3:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:3>
//CHECK-DAG: [[TBV4:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:4>
//CHECK-DAG: [[TBV5:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:5>
//CHECK-DAG: [[TBV6:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:6>
//CHECK-DAG: [[TBV7:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:7>
//CHECK-DAG: [[TBV8:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:8>
//CHECK-DAG: [[TBV9:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:9>
//CHECK-DAG: [[TBV10:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:10>
//CHECK-DAG: [[TBV11:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:11>
//CHECK-DAG: [[TBV12:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:12>
//CHECK-DAG: [[TBV13:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:13>
//CHECK-DAG: [[TBV14:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:14>
//CHECK-DAG: [[TBV15:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:15>
//CHECK-DAG: [[TBV16:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:16>
//CHECK-DAG: [[TBV17:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:17>
//CHECK-DAG: [[TBV18:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:18>
//CHECK-DAG: [[TBV19:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:19>
//CHECK-DAG: [[TBV20:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:20>
//CHECK-DAG: [[TBV21:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:21>
//CHECK-DAG: [[TBV22:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:22>
//CHECK-DAG: [[TBV23:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:23>
//CHECK-DAG: [[TBV24:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:24>
//CHECK-DAG: [[TBV25:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:25>
//CHECK-DAG: [[TBV26:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:26>
//CHECK-DAG: [[TBV27:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:27>
//CHECK-DAG: [[TBV28:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:28>
//CHECK-DAG: [[TBV29:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:29>
//CHECK-DAG: [[TBV30:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:30>
//CHECK-DAG: [[TBV31:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:31>
//CHECK-DAG: [[TBV32:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:32>
//CHECK-DAG: [[TBV33:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:33>
//CHECK-DAG: [[TBV34:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:34>
//CHECK-DAG: [[TBV35:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:35>
//CHECK-DAG: [[TBV36:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:36>
//CHECK-DAG: [[TBV37:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:37>
//CHECK-DAG: [[TBV38:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:38>
//CHECK-DAG: [[TBV39:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:39>
//CHECK-DAG: [[TBV40:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:40>
//CHECK-DAG: [[TBV41:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:41>
//CHECK-DAG: [[TBV42:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:42>
//CHECK-DAG: [[TBV43:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:43>
//CHECK-DAG: [[TBV44:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:44>
//CHECK-DAG: [[TBV45:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:45>
//CHECK-DAG: [[TBV46:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:46>
//CHECK-DAG: [[TBV47:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:47>
//CHECK-DAG: [[TBV48:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:48>
//CHECK-DAG: [[TBV49:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:49>
//CHECK-DAG: [[TBV50:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:50>
//CHECK-DAG: [[TBV51:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:51>
//CHECK-DAG: [[TBV52:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:52>
//CHECK-DAG: [[TBV53:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:53>
//CHECK-DAG: [[TBV54:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:54>
//CHECK-DAG: [[TBV55:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:55>
//CHECK-DAG: [[TBV56:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:56>
//CHECK-DAG: [[TBV57:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:57>
//CHECK-DAG: [[TBV58:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:58>
//CHECK-DAG: [[TBV59:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:59>
//CHECK-DAG: [[TBV60:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:60>
//CHECK-DAG: [[TBV61:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:61>
//CHECK-DAG: [[TBV62:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:62>
//CHECK-DAG: [[TBV63:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:63>
//CHECK-DAG: [[TBV64:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:64>
//CHECK-DAG: [[TBV65:%.*]] = VPUMI40XX.DeclareTaskBuffer {offset = {{.*}}}  <DPUVariant> -> !VPURegMapped.Index<0:0:65>

//CHECK-DAG: VPURegMapped.TaskBufferLayout

//CHECK: [[IVAR0:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI0]]
//CHECK: [[IVAR1:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI1]]
//CHECK: [[IVAR2:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI2]]
//CHECK: [[IVAR3:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI3]]
//CHECK: [[IVAR4:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI4]]
//CHECK: [[IVAR5:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI5]]
//CHECK: [[IVAR6:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI6]]
//CHECK: [[IVAR7:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI7]]
//CHECK: [[IVAR8:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI8]]
//CHECK: [[IVAR9:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI9]]
//CHECK: [[IVAR10:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI10]]
//CHECK: [[IVAR11:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI11]]
//CHECK: [[IVAR12:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI12]]
//CHECK: [[IVAR13:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI13]]
//CHECK: [[IVAR14:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI14]]
//CHECK: [[IVAR15:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI15]]
//CHECK: [[IVAR16:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI16]]
//CHECK: [[IVAR17:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI17]]
//CHECK: [[IVAR18:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI18]]
//CHECK: [[IVAR19:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI19]]
//CHECK: [[IVAR20:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI20]]
//CHECK: [[IVAR21:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI21]]
//CHECK: [[IVAR22:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI22]]
//CHECK: [[IVAR23:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI23]]
//CHECK: [[IVAR24:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI24]]
//CHECK: [[IVAR25:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI25]]
//CHECK: [[IVAR26:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI26]]
//CHECK: [[IVAR27:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI27]]
//CHECK: [[IVAR28:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI28]]
//CHECK: [[IVAR29:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI29]]
//CHECK: [[IVAR30:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI30]]
//CHECK: [[IVAR31:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI31]]
//CHECK: [[IVAR32:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI32]]
//CHECK: [[IVAR33:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI33]]
//CHECK: [[IVAR34:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI34]]
//CHECK: [[IVAR35:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI35]]
//CHECK: [[IVAR36:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI36]]
//CHECK: [[IVAR37:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI37]]
//CHECK: [[IVAR38:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI38]]
//CHECK: [[IVAR39:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI39]]
//CHECK: [[IVAR40:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI40]]
//CHECK: [[IVAR41:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI41]]
//CHECK: [[IVAR42:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI42]]
//CHECK: [[IVAR43:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI43]]
//CHECK: [[IVAR44:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI44]]
//CHECK: [[IVAR45:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI45]]
//CHECK: [[IVAR46:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI46]]
//CHECK: [[IVAR47:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI47]]
//CHECK: [[IVAR48:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI48]]
//CHECK: [[IVAR49:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI49]]
//CHECK: [[IVAR50:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI50]]
//CHECK: [[IVAR51:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI51]]
//CHECK: [[IVAR52:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI52]]
//CHECK: [[IVAR53:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI53]]
//CHECK: [[IVAR54:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI54]]
//CHECK: [[IVAR55:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI55]]
//CHECK: [[IVAR56:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI56]]
//CHECK: [[IVAR57:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI57]]
//CHECK: [[IVAR58:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI58]]
//CHECK: [[IVAR59:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI59]]
//CHECK: [[IVAR60:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI60]]
//CHECK: [[IVAR61:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI61]]
//CHECK: [[IVAR62:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI62]]
//CHECK: [[IVAR63:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI63]]
//CHECK: [[IVAR64:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI0]]
//CHECK: [[IVAR65:%.*]] = VPUMI40XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TBI1]]

//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV0]]
    //CHECK-SAME: calls([[IVAR0]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV1]]
    //CHECK-SAME: calls([[IVAR1]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV2]]
    //CHECK-SAME: calls([[IVAR2]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV3]]
    //CHECK-SAME: calls([[IVAR3]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV4]]
    //CHECK-SAME: calls([[IVAR4]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV5]]
    //CHECK-SAME: calls([[IVAR5]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV6]]
    //CHECK-SAME: calls([[IVAR6]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV7]]
    //CHECK-SAME: calls([[IVAR7]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV8]]
    //CHECK-SAME: calls([[IVAR8]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV9]]
    //CHECK-SAME: calls([[IVAR9]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV10]]
    //CHECK-SAME: calls([[IVAR10]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV11]]
    //CHECK-SAME: calls([[IVAR11]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV12]]
    //CHECK-SAME: calls([[IVAR12]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV13]]
    //CHECK-SAME: calls([[IVAR13]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV14]]
    //CHECK-SAME: calls([[IVAR14]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV15]]
    //CHECK-SAME: calls([[IVAR15]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV16]]
    //CHECK-SAME: calls([[IVAR16]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV17]]
    //CHECK-SAME: calls([[IVAR17]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV18]]
    //CHECK-SAME: calls([[IVAR18]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV19]]
    //CHECK-SAME: calls([[IVAR19]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV20]]
    //CHECK-SAME: calls([[IVAR20]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV21]]
    //CHECK-SAME: calls([[IVAR21]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV22]]
    //CHECK-SAME: calls([[IVAR22]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV23]]
    //CHECK-SAME: calls([[IVAR23]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV24]]
    //CHECK-SAME: calls([[IVAR24]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV25]]
    //CHECK-SAME: calls([[IVAR25]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV26]]
    //CHECK-SAME: calls([[IVAR26]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV27]]
    //CHECK-SAME: calls([[IVAR27]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV28]]
    //CHECK-SAME: calls([[IVAR28]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV29]]
    //CHECK-SAME: calls([[IVAR29]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV30]]
    //CHECK-SAME: calls([[IVAR30]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV31]]
    //CHECK-SAME: calls([[IVAR31]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV32]]
    //CHECK-SAME: calls([[IVAR32]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV33]]
    //CHECK-SAME: calls([[IVAR33]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV34]]
    //CHECK-SAME: calls([[IVAR34]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV35]]
    //CHECK-SAME: calls([[IVAR35]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV36]]
    //CHECK-SAME: calls([[IVAR36]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV37]]
    //CHECK-SAME: calls([[IVAR37]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV38]]
    //CHECK-SAME: calls([[IVAR38]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV39]]
    //CHECK-SAME: calls([[IVAR39]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV40]]
    //CHECK-SAME: calls([[IVAR40]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV41]]
    //CHECK-SAME: calls([[IVAR41]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV42]]
    //CHECK-SAME: calls([[IVAR42]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV43]]
    //CHECK-SAME: calls([[IVAR43]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV44]]
    //CHECK-SAME: calls([[IVAR44]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV45]]
    //CHECK-SAME: calls([[IVAR45]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV46]]
    //CHECK-SAME: calls([[IVAR46]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV47]]
    //CHECK-SAME: calls([[IVAR47]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV48]]
    //CHECK-SAME: calls([[IVAR48]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV49]]
    //CHECK-SAME: calls([[IVAR49]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV50]]
    //CHECK-SAME: calls([[IVAR50]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV51]]
    //CHECK-SAME: calls([[IVAR51]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV52]]
    //CHECK-SAME: calls([[IVAR52]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV53]]
    //CHECK-SAME: calls([[IVAR53]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV54]]
    //CHECK-SAME: calls([[IVAR54]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV55]]
    //CHECK-SAME: calls([[IVAR55]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV56]]
    //CHECK-SAME: calls([[IVAR56]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV57]]
    //CHECK-SAME: calls([[IVAR57]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV58]]
    //CHECK-SAME: calls([[IVAR58]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV59]]
    //CHECK-SAME: calls([[IVAR59]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV60]]
    //CHECK-SAME: calls([[IVAR60]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV61]]
    //CHECK-SAME: calls([[IVAR61]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV62]]
    //CHECK-SAME: calls([[IVAR62]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV63]]
    //CHECK-SAME: calls([[IVAR63]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV64]]
    //CHECK-SAME: calls([[IVAR64]]
//CHECK: VPUMI40XX.DPUVariant
    //CHECK-SAME: taskLocation([[TBV65]]
    //CHECK-SAME: calls([[IVAR65]]
