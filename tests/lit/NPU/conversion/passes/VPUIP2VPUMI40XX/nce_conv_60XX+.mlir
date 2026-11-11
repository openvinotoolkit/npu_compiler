//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_16_16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_16_16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 16], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_16_16
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: weight_table_offset
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: weight_table_offset

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_16_16_dataptr inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_16_16_dataptr(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table_data_ptr(%2 : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 16], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 256 : i64}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_16_16_dataptr
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 256 : i64

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_64_64_32_16_dataptr inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_64_64_32_16_dataptr(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<176x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<176x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table_data_ptr(%2 : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 256 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 159], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 512 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 175], outStart = [0, 0, 160], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 768 : i64}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_64_64_32_16_dataptr
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 256 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 512 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 768 : i64

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_16_16_wzp8 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_16_16_wzp8(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <37376> -> memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<32x32x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_zero_points(%3 : memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 16], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 32 : i64}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_16_16_wzp8
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 32 : i64

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f32, 1.600000e+01>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_64_64_32_16_wzp8 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_64_64_32_16_wzp8(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<176x32x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <37376> -> memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<176x32x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_zero_points(%3 : memref<32x1x1x1xi8, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 64 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 159], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 128 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 175], outStart = [0, 0, 160], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 160 : i64}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_64_64_32_16_wzp8
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 64 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 128 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 160 : i64

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f32, 1.600000e+01>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_64_64_32_16_wzp4 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_64_64_32_16_wzp4(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<176x32x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <37376> -> memref<32x1x1x1xi4, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<176x32x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_zero_points(%3 : memref<32x1x1x1xi4, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 32 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 159], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 64 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 175], outStart = [0, 0, 160], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 96 : i64}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_64_64_32_16_wzp4
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 32 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 64 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 96 : i64

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @multivar_z_16_16_dataptr_splith inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @multivar_z_16_16_dataptr_splith(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table_data_ptr(%2 : memref<32x1x1x1xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 7, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 15], outStart = [0, 8, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 0 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 7, 31], outStart = [0, 0, 16], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 256 : i64}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 8, 16], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, weight_table_offset = 256 : i64}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @multivar_z_16_16_dataptr_splith
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 0 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 256 : i64
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: weight_table_offset = 256 : i64

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @dynShapeZMConv_f16_f16_force_inv_read inputsInfo : {
    DataInfo "input_0" : tensor<1x16x256x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x256x16xf16>
  }

  func.func @dynShapeZMConv_f16_f16_force_inv_read(%arg0: memref<1x16x256x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x16x256x16xf16, {order = #NHWC}, @DDR>) -> memref<1x16x256x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <131616> -> memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <131584> -> memref<1x1x1x1xi32, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %4 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, dynamic_scale_config = #VPUIP.DynamicScaleConfigAttr<dynamicSkipMode = <SPATIAL_ONLY>, dynamicAdjustMode = <DISABLED>>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%0 : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%2 : memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) dynamic_sequence_length(%3 : memref<1x1x1x1xi32, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x256x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 63, 15], is_dummy, inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {inEnd = [15, 63, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, variant_primitive_id = [0]}
        DPUTask {inEnd = [15, 127, 15], inStart = [0, 64, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 127, 15], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, variant_primitive_id = [1]}
        DPUTask {inEnd = [15, 191, 15], inStart = [0, 128, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 191, 15], outStart = [0, 128, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, variant_primitive_id = [2]}
        DPUTask {inEnd = [15, 255, 15], inStart = [0, 192, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 255, 15], outStart = [0, 192, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, variant_primitive_id = [3]}
      } PPE : {
      }
    }
    return %arg1 : memref<1x16x256x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @dynShapeZMConv_f16_f16_force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: force_inv_read
