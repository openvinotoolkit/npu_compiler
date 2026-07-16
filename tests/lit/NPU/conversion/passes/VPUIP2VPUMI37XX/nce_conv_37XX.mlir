//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x32xf16>
    DataInfo "output_1" : tensor<1x64x16x32xf16>
  }

func.func nested @multiple_clusters_dpu_soh_f16_f16_f16(%arg0: memref<1x32x32x32xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x16x32xf16, {order = #NHWC}, @DDR>, %arg2: memref<1x64x16x32xf16, {order = #NHWC}, @DDR>) -> (memref<1x64x16x32xf16, {order = #NHWC}, @DDR>, memref<1x64x16x32xf16, {order = #NHWC}, @DDR>) {
  %2 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x16x32xf16, {order = #NHWC}, [@DDR, 0]>
  %3 = VPURT.DeclareBuffer <NetworkInput> [0] <32768> -> memref<1x32x16x32xf16, {order = #NHWC}, [@DDR, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <102400> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, {order = #NHWC, strides = [4, 1, 4, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%11 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%8 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }
  }
  VPURT.Task {
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]> variants : {
      DPUTask {outEnd = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 16, 0]}
    } PPE : {
    }
  }
  return %arg1, %arg2 : memref<1x64x16x32xf16, {order = #NHWC}, @DDR>, memref<1x64x16x32xf16, {order = #NHWC}, @DDR>
}
}

//CHECK-LABEL: @multiple_clusters_dpu_soh_f16_f16_f16
//CHECK: [[VAL2:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> [[TYPE2:.+]]
//CHECK-NEXT: [[VAL3:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <32768> -> [[TYPE3:.+]]
//CHECK-NEXT: [[VAL4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
//CHECK-NEXT: [[VAL5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[TYPE5:.+]]
//CHECK-NEXT: [[VAL6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[TYPE6:.+]]
//CHECK-NEXT: [[VAL7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> [[TYPE7:.+]]
//CHECK-NEXT: [[VAL8:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[TYPE8:.+]]
//CHECK-NEXT: [[VAL9:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> [[TYPE9:.+]]
//CHECK-NEXT: [[VAL10:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> [[TYPE10:.+]]
//CHECK-NEXT: [[VAL11:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> [[TYPE11:.+]]
//CHECK-NEXT: [[VAL12:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> [[TYPE12:.+]]
//CHECK-NEXT: [[VAL13:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> [[TYPE13:.+]]
//CHECK-NEXT: [[VAL14:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> [[TYPE14:.+]]
//CHECK-NEXT: [[VAL15:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <102400> -> [[TYPE15:.+]]
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: VPUMI37XX.DPUInvariant {{.+}} is_segmented
                //CHECK-DAG: input([[VAL11]] : [[TYPE11]])
                //CHECK-DAG: weights([[VAL5]] : [[TYPE5]])
                //CHECK-DAG: weight_table([[VAL13]] : [[TYPE13]])
                //CHECK-DAG: parent_input([[VAL10]] : [[TYPE10]])
                //CHECK-DAG: parent_output([[VAL7]] : [[TYPE7]])
                //CHECK-DAG: outputs([[VAL8]] : [[TYPE8]])
//CHECK-NOT: DPUTask
//CHECK: VPUMI37XX.DPUVariant
//CHECK-NEXT: VPUMI37XX.DPUInvariant {{.+}} is_segmented
                //CHECK-DAG: input([[VAL12]] : [[TYPE12]])
                //CHECK-DAG: weights([[VAL6]] : [[TYPE6]])
                //CHECK-DAG: weight_table([[VAL14]] : [[TYPE14]])
                //CHECK-DAG: parent_input([[VAL10]] : [[TYPE10]])
                //CHECK-DAG: parent_output([[VAL7]] : [[TYPE7]])
                //CHECK-DAG: outputs([[VAL9]] : [[TYPE9]])
//CHECK-NOT: DPUTask
//CHECK: VPUMI37XX.DPUVariant

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  net.NetworkInfo entryPoint : @multiple_clusters_dpu_sok_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x16xf16>
    DataInfo "output_1" : tensor<1x64x16x16xf16>
  }

  func.func nested @multiple_clusters_dpu_sok_f16_f16_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, %arg2: memref<1x64x16x16xf16, {order = #NHWC}, @DDR>) -> (memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, memref<1x64x16x16xf16, {order = #NHWC}, @DDR>) {
    %1 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> !VPUIP.DistributedBuffer<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <51200> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [1] <51200> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>

    VPURT.Task {
      %15 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%8 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%2 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%10 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%7 : !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) parent_output(%4 : !VPUIP.DistributedBuffer<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%5 : !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants : {
        DPUTask {outEnd = [15, 15, 31], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      } PPE : {
      }
    }
    VPURT.Task {
      %15 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%9 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) weights(%3 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) weight_table(%11 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>) parent_input(%7 : !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) parent_output(%4 : !VPUIP.DistributedBuffer<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%6 : !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants : {
        DPUTask {outEnd = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 32]}
      } PPE : {
      }
    }
    return %arg1, %arg2 : memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, memref<1x64x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multiple_clusters_dpu_sok_f16_f16_f16

//CHECK: [[VAL1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> [[TYPE1:.+]]
//CHECK-NEXT: [[VAL2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[TYPE2:.+]]
//CHECK-NEXT: [[VAL3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[TYPE3:.+]]
//CHECK-NEXT: [[VAL4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> [[TYPE4:.+]]
//CHECK-NEXT: [[VAL5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> [[TYPE5:.+]]
//CHECK-NEXT: [[VAL6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> [[TYPE6:.+]]
//CHECK-NEXT: [[VAL7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> [[TYPE7:.+]]
//CHECK-NEXT: [[VAL8:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> [[TYPE8:.+]]
//CHECK-NEXT: [[VAL9:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> [[TYPE9:.+]]
//CHECK-NEXT: [[VAL10:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <51200> -> [[TYPE10:.+]]
//CHECK-NEXT: [[VAL11:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <51200> -> [[TYPE11:.+]]
//CHECK-NEXT: [[VAL12:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> [[TYPE12:.+]]
//CHECK-NEXT: [[VAL13:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> [[TYPE13:.+]]
//CHECK-NEXT: [[VAL14:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> [[TYPE14:.+]]
//CHECK-NEXT: [[VAL15:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> [[TYPE15:.+]]

//CHECK-NOT: VPURT.Task
//CHECK: VPUMI37XX.DPUInvariant
                //CHECK-DAG: input([[VAL8]] : [[TYPE8]])
                //CHECK-DAG: weights([[VAL2]] : [[TYPE2]])
                //CHECK-DAG: weight_table([[VAL10]] : [[TYPE10]])
                //CHECK-DAG: parent_input([[VAL7]] : [[TYPE7]])
                //CHECK-DAG: parent_output([[VAL4]] : [[TYPE4]])
                //CHECK-DAG: outputs([[VAL14]], [[VAL15]] : [[TYPE14]], [[TYPE15]])
//CHECK-NOT: DPUTask
//CHECK: VPUMI37XX.DPUVariant

//CHECK-NEXT: [[VAL26:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> [[TYPE26:.+]]
//CHECK-NEXT: [[VAL27:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> [[TYPE27:.+]]
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: VPUMI37XX.DPUInvariant
                //CHECK-DAG: input([[VAL9]] : [[TYPE9]])
                //CHECK-DAG: weights([[VAL3]] : [[TYPE3]])
                //CHECK-DAG: weight_table([[VAL11]] : [[TYPE11]])
                //CHECK-DAG: parent_input([[VAL7]] : [[TYPE7]])
                //CHECK-DAG: parent_output([[VAL4]] : [[TYPE4]])
                //CHECK-DAG: outputs([[VAL26]], [[VAL27]] : [[TYPE26]], [[TYPE27]])
//CHECK-NOT: DPUTask
//CHECK: VPUMI37XX.DPUVariant

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  func.func nested @sparse_conv(%arg0: memref<1x128x32x32xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x32x32x32xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<512x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <73728> -> memref<1x128x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <335872> -> memref<32x1x1x128xi1, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <336384> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %8 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%2 : memref<1x128x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%1 : memref<32x1x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights_sparsity_map(%3 : memref<32x1x1x128xi1, [@CMX_NN, 0]>) weight_table(%5 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%2 : memref<1x128x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%4 : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%4 : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [31, 31, 31], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>
  }
  net.NetworkInfo entryPoint : @sparse_conv inputsInfo : {
    DataInfo "input_0" : tensor<1x128x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x32x32xf16>
  }
}

//CHECK-LABEL: @sparse_conv
//CHECK: [[VALSPS:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <335872> -> memref<32x1x1x128xi1, [@CMX_NN, 0]>
//CHECK: VPUMI37XX.DPUInvariant
//CHECK-SAME: weights_sparsity_map([[VALSPS]] : memref<32x1x1x128xi1, [@CMX_NN, 0]>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @se_table_dpu_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func nested @se_table_dpu_f16_f16_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x32x16x16xi1, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <35328> -> memref<1x1x16x16xi32, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) input_sparsity_map(%2 : memref<1x32x16x16xi1, [@CMX_NN, 0]>) input_storage_element_table(%3 : memref<1x1x16x16xi32, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%4 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input_sparsity_map(%2 : memref<1x32x16x16xi1, [@CMX_NN, 0]>) parent_input_storage_element_table(%3 : memref<1x1x16x16xi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%5 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @se_table_dpu_f16_f16_f16
//CHECK: [[VALSMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<1x32x16x16xi1
//CHECK: [[VALSET:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<1x1x16x16xi32
//CHECK: VPUMI37XX.DPUInvariant
//CHECK-SAME: input_sparsity_map([[VALSMP]] : memref<1x32x16x16xi1
//CHECK-SAME: input_storage_element_table([[VALSET]] : memref<1x1x16x16xi32

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  func.func @ComputNCESeSizes(%arg0: memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x16x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>,
                        %arg2: memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>)
          -> (memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>) {
      %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
      %1 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
      VPURT.Task {
          %2:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>, is_superdense, input_se_size = 16 : i64, output_se_size = 32 : i64}>
          input(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          input_sparsity_map(%arg1 : memref<1x16x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>)
          weights(%0 : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          weight_table(%1 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
          parent_input(%arg0 : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          parent_input_sparsity_map(%arg1 : memref<1x16x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>)
          parent_output(%arg2 : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          parent_output_sparsity_map(%arg3 : memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>)
          outputs(%arg2 : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          output_sparsity_map(%arg3 : memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>)
          -> memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]> variants : {
              DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
          }
      }
      return %arg2, %arg3 : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>
  }
}

// CHECK-LABEL: @ComputNCESeSizes
// CHECK-SAME: [[ARG_3:%[^:]+]]: memref<1x32x32x32xi1, {order = #NHWC}, [@CMX_NN, 0]>

//CHECK: [[VALSMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<32x16x1x1xf16
//CHECK: [[VALSET:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<32x1x1x4xsi32
//CHECK: VPUMI37XX.DPUInvariant
//CHECK-SAME: input_se_size = 16 : i64
//CHECK-SAME: is_superdense
//CHECK-SAME: output_se_size = 32 : i64
//CHECK-SAME: parent_output_sparsity_map([[ARG_3]] : memref<1x32x32x32xi1
//CHECK-SAME: output_sparsity_map_buff([[ARG_3]] : memref<1x32x32x32xi1
