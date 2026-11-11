//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @non_pallet_lut_dpu_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @non_pallet_lut_dpu_f16_f16_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %4 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @non_pallet_lut_dpu_f16_f16_f16
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: pallet_lut_read


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @pallet_lut_dpu_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @pallet_lut_dpu_f16_f16_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <19456> -> memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <20480> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %6 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)  weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%3 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) pallet_lookup_table(%2 : memref<1x1x1x16xui16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)  parent_output(%4 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%4 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], is_dummy, inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @pallet_lut_dpu_f16_f16_f16
//CHECK: [[VALSMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<1x1x1x16xui16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: pallet_lookup_table([[VALSMP]] : memref<1x1x1x16xui16
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: pallet_lut_read
//CHECK-NOT: force_inv_read
//CHECK: VPUMI40XX.DPUVariant
//CHECK-DAG: pallet_lut_read
//CHECK-DAG: force_inv_read


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @out_reduce_max_per_xy_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @out_reduce_max_per_xy_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x1x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task {
      %5:2 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 1, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) max_per_xy_buff(%4 : memref<1x1x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @out_reduce_max_per_xy_f16
//CHECK: [[VAL4:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<1x1x16x16xf16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: max_per_xy([[VAL4]] :  memref<1x1x16x16xf16


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @out_reduce_min_per_xy_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @out_reduce_min_per_xy_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x1x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task {
      %5:2 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 1, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) min_per_xy_buff(%4 : memref<1x1x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @out_reduce_min_per_xy_f16
//CHECK: [[VAL4:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<1x1x16x16xf16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: min_per_xy([[VAL4]] :  memref<1x1x16x16xf16


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @out_reduce_min_max_per_tensor_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @out_reduce_min_max_per_tensor_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task {
      %5:2 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 1>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) min_max_per_tensor_buff(%4 : memref<1x1x2x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x2x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @out_reduce_min_max_per_tensor_f16
//CHECK: [[VAL4:%.+]] = VPURT.DeclareBuffer <CMX_NN> {{.+}} -> memref<1x1x2x1xf16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: min_max_per_tensor([[VAL4]] :  memref<1x1x2x1xf16


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @out_reduce_min_max_per_tensor_2_tiles_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @out_reduce_min_max_per_tensor_2_tiles_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <36864> -> memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [1] <36868> -> memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 1]>

    VPURT.Task {
      %6:3 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 2>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)   parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) min_max_per_tensor_buff(%4, %5 : memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 1]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @out_reduce_min_max_per_tensor_2_tiles_f16
//CHECK: [[VAL4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] {{.+}} -> memref<1x1x2x1xf16
//CHECK: [[VAL5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] {{.+}} -> memref<1x1x2x1xf16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK-SAME: min_max_per_tensor([[VAL4]], [[VAL5]] :  memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x1x2x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @local_region_no_suppress_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @local_region_no_suppress_f16_f16_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %4 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>,
        local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 8 : i64, yStart = 0 : i64, yEnd = 8 : i64>
    }>
      input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @local_region_no_suppress_f16_f16_f16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK:        local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 8 : i64, yStart = 0 : i64, yEnd = 8 : i64>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @local_region_suppress_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x16x16xf16>
  }

  func.func @local_region_suppress_f16_f16_f16(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %4 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>,
        local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 8 : i64, yStart = 0 : i64, yEnd = 8 : i64, suppressOutput = unit>
    }>
      input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      weights(%0 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      weight_table(%2 : memref<32x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_output(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      outputs(%3 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    return %arg1 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @local_region_suppress_f16_f16_f16
//CHECK: VPUMI40XX.DPUInvariant
//CHECK:        local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 8 : i64, yStart = 0 : i64, yEnd = 8 : i64, suppressOutput = unit>
