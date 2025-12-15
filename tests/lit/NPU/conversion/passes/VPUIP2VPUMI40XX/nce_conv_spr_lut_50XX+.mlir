//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @non_spr_lut_dpu_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func private @non_spr_lut_dpu_f16_f16_f16(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x32x16x16xf16, #NHWC, @DDR>) -> memref<1x32x16x16xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task {
      %4 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%1 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)   weights(%0 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%2 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)   parent_output(%3 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, #NHWC, @DDR>
  }
}

//CHECK-LABEL: @non_spr_lut_dpu_f16_f16_f16
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: spr_lut_read

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @spr_lut_dpu_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func private @spr_lut_dpu_f16_f16_f16(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x32x16x16xf16, #NHWC, @DDR>) -> memref<1x32x16x16xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <19456> -> memref<1x1x10x16xui16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <20480> -> memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%1 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)  weights(%0 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%3 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) spr_lookup_table(%2 : memref<1x1x10x16xui16, #NHWC, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)  parent_output(%4 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [0, 0, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, #NHWC, @DDR>
  }
}

//CHECK-LABEL: @spr_lut_dpu_f16_f16_f16
//CHECK: %[[VALSMP:.*]] = VPURT.DeclareBuffer <CMX_NN> {{.*}} -> memref<1x1x10x16xui16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: spr_lookup_table(%[[VALSMP]] : memref<1x1x10x16xui16
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: spr_lut_read
//CHECK-NOT: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-DAG: spr_lut_read
//CHECK-DAG: force_inv_read

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @spr_lut_dpu_f16_f16_f16_double_var inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func private @spr_lut_dpu_f16_f16_f16_double_var(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x32x16x16xf16, #NHWC, @DDR>) -> memref<1x32x16x16xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <19456> -> memref<1x1x10x16xui16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <20480> -> memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%1 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)  weights(%0 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%3 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) spr_lookup_table(%2 : memref<1x1x10x16xui16, #NHWC, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)  parent_output(%4 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [1, 1, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, #NHWC, @DDR>
  }
}

//CHECK-LABEL: @spr_lut_dpu_f16_f16_f16_double_var
//CHECK: %[[VALSMP:.*]] = VPURT.DeclareBuffer <CMX_NN> {{.*}} -> memref<1x1x10x16xui16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: spr_lookup_table(%[[VALSMP]] : memref<1x1x10x16xui16
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: spr_lut_read
//CHECK-NOT: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-DAG: spr_lut_read
//CHECK-DAG: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: spr_lut_read
//CHECK-NOT: force_inv_read
