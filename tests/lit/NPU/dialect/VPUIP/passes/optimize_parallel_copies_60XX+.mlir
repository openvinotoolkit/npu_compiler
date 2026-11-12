//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: func.func @OptimizeScaleBiasCopyWith4BitWeights
func.func @OptimizeScaleBiasCopyWith4BitWeights(
        %input0: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
        %input1: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        -> (memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>) {
    %weights = const.Declare memref<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<64x32x1x1xi4>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %scale_cst = const.Declare memref<64x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
    %bias_cst = const.Declare memref<64x1x1x1xf32> = dense<0.000000e+00> : tensor<64x1x1x1xf32>

    %scale_buf0 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %scale_cmx0 = VPUIP.Copy
            inputs(%scale_cst : memref<64x1x1x1xf32>)
            outputs(%scale_buf0 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %bias_buf0 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %bias_cmx0 = VPUIP.Copy
            inputs(%bias_cst : memref<64x1x1x1xf32>)
            outputs(%bias_buf0 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %weights_buf0 = memref.alloc() : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx0 = VPUIP.Copy
            inputs(%weights : memref<64x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf0 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf0 = memref.alloc() : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx0 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx0 : memref<64x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx0 : memref<64x1x1x1xf32, @CMX_NN>)
        parent_input(%input0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf0 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf0 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    // %scale_cmx1 and %bias_cmx1 are same consts as %scale_cmx0 and %bias_cmx0, will be removed.
    %scale_buf1 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %scale_cmx1 = VPUIP.Copy
            inputs(%scale_cst : memref<64x1x1x1xf32>)
            outputs(%scale_buf1 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %bias_buf1 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %bias_cmx1 = VPUIP.Copy
            inputs(%bias_cst : memref<64x1x1x1xf32>)
            outputs(%bias_buf1 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %weights_buf1 = memref.alloc() : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx1 = VPUIP.Copy
            inputs(%weights : memref<64x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf1 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf1 = memref.alloc() : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx1 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx1 : memref<64x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx1 : memref<64x1x1x1xf32, @CMX_NN>)
        parent_input(%input1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf1 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf1 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    return %conv0, %conv1 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[SCALE_CST:%.+]] = const.Declare memref<64x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<64x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[SCALE_BUF0:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SCALE_CST]] : memref<64x1x1x1xf32>)
    // CHECK-SAME:      outputs([[SCALE_BUF0]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK:       [[BIAS_BUF0:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[BIAS_CST]] : memref<64x1x1x1xf32>)
    // CHECK-SAME:      outputs([[BIAS_BUF0]] : memref<64x1x1x1xf32, @CMX_NN>)

    // CHECK:       [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX0]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX0]] : memref<64x1x1x1xf32, @CMX_NN>)

    // Second scale/bias copies are eliminated; conv1 reuses scale_cmx0 / bias_cmx0

    // CHECK:       [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX0]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX0]] : memref<64x1x1x1xf32, @CMX_NN>)
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NotOptimizeScaleBiasCopyWith8BitWeights
func.func @NotOptimizeScaleBiasCopyWith8BitWeights(
        %input0: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
        %input1: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        -> (memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>) {
    %weights = const.Declare memref<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<64x32x1x1xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %scale_cst = const.Declare memref<64x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
    %bias_cst = const.Declare memref<64x1x1x1xf32> = dense<0.000000e+00> : tensor<64x1x1x1xf32>

    %scale_buf0 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %scale_cmx0 = VPUIP.Copy
            inputs(%scale_cst : memref<64x1x1x1xf32>)
            outputs(%scale_buf0 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %bias_buf0 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %bias_cmx0 = VPUIP.Copy
            inputs(%bias_cst : memref<64x1x1x1xf32>)
            outputs(%bias_buf0 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %weights_buf0 = memref.alloc() : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx0 = VPUIP.Copy
            inputs(%weights : memref<64x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf0 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf0 = memref.alloc() : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx0 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx0 : memref<64x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx0 : memref<64x1x1x1xf32, @CMX_NN>)
        parent_input(%input0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf0 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf0 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    // %scale_cmx1 and %bias_cmx1 are same consts, but 8-bit weights: should NOT be removed.
    %scale_buf1 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %scale_cmx1 = VPUIP.Copy
            inputs(%scale_cst : memref<64x1x1x1xf32>)
            outputs(%scale_buf1 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %bias_buf1 = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %bias_cmx1 = VPUIP.Copy
            inputs(%bias_cst : memref<64x1x1x1xf32>)
            outputs(%bias_buf1 : memref<64x1x1x1xf32, @CMX_NN>)
            -> memref<64x1x1x1xf32, @CMX_NN>
    %weights_buf1 = memref.alloc() : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx1 = VPUIP.Copy
            inputs(%weights : memref<64x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf1 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf1 = memref.alloc() : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx1 : memref<64x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx1 : memref<64x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx1 : memref<64x1x1x1xf32, @CMX_NN>)
        parent_input(%input1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf1 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf1 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    return %conv0, %conv1 : memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[SCALE_CST:%.+]] = const.Declare memref<64x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<64x1x1x1xf32> = dense<0.000000e+00>

    // Both scale and bias copies are preserved for 8-bit weights
    // CHECK:       [[SCALE_BUF0:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SCALE_CST]] : memref<64x1x1x1xf32>)
    // CHECK-SAME:      outputs([[SCALE_BUF0]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK:       [[BIAS_BUF0:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[BIAS_CST]] : memref<64x1x1x1xf32>)
    // CHECK-SAME:      outputs([[BIAS_BUF0]] : memref<64x1x1x1xf32, @CMX_NN>)

    // CHECK:       [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX0]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX0]] : memref<64x1x1x1xf32, @CMX_NN>)

    // CHECK:       [[SCALE_BUF1:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SCALE_CST]] : memref<64x1x1x1xf32>)
    // CHECK-SAME:      outputs([[SCALE_BUF1]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK:       [[BIAS_BUF1:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_CMX1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[BIAS_CST]] : memref<64x1x1x1xf32>)
    // CHECK-SAME:      outputs([[BIAS_BUF1]] : memref<64x1x1x1xf32, @CMX_NN>)

    // CHECK:       [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX1]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX1]] : memref<64x1x1x1xf32, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @OptimizeWeightCopyWithConcatAttrWeights
func.func @OptimizeWeightCopyWithConcatAttrWeights(
        %input0: memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>,
        %input1: memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        -> (memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>) {
    %weights = const.Declare memref<32x32x1x1x!qElemType, {order = #NHWC}> = dense<> : tensor<0xf80>,
        [#const.Concat<axis = <0 : i64>, offsets = <[[0, 0, 0, 0], [16, 0, 0, 0]]>,
            constants = <[
                dense<1> : tensor<16x16x1x1xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>],
                dense<2> : tensor<16x16x1x1xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
            ]>>,
         #const.PadWithZero<[0, 0, 0, 0], [0, 16, 0, 0]>]

    %weights_buf0 = memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx0 = VPUIP.Copy
            inputs(%weights : memref<32x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf0 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf0 = memref.alloc() : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input0 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx0 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_input(%input0 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 31], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    %weights_buf1 = memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx1 = VPUIP.Copy
            inputs(%weights : memref<32x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf1 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf1 = memref.alloc() : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input1 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx1 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_input(%input1 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 31], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    return %conv0, %conv1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1x!qElemType, {order = #NHWC}> = dense<> : tensor<0xf80>, [#const.Concat<
    // CHECK:       [[WEIGHTS_BUF0:%.+]] = memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[WEIGHTS_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[WEIGHTS]] : memref<32x32x1x1x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:      outputs([[WEIGHTS_BUF0]] : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK:       [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weights([[WEIGHTS_CMX0]] : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-NOT:   memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weights([[WEIGHTS_CMX0]] : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NoCrashConcatPadWeightsCanWeightsBeCompressed
func.func @NoCrashConcatPadWeightsCanWeightsBeCompressed(
        %input0: memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>,
        %input1: memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        -> (memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>) {
    %weights = const.Declare memref<32x32x1x1x!qElemType, {order = #NHWC}> = dense<> : tensor<0xf80>,
        [#const.Concat<axis = <0 : i64>, offsets = <[[0, 0, 0, 0], [16, 0, 0, 0]]>,
            constants = <[
                dense<1> : tensor<16x16x1x1xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>],
                dense<2> : tensor<16x16x1x1xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
            ]>>,
         #const.PadWithZero<[0, 0, 0, 0], [0, 16, 0, 0]>]

    %weights_buf0 = memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx0 = VPUIP.Copy
            inputs(%weights : memref<32x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf0 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf0 = memref.alloc() : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input0 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx0 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_input(%input0 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf0 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 31], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    %weights_buf1 = memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %weights_cmx1 = VPUIP.Copy
            inputs(%weights : memref<32x32x1x1x!qElemType, {order = #NHWC}>)
            outputs(%weights_buf1 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
            -> memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %out_buf1 = memref.alloc() : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
    %conv1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input1 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        weights(%weights_cmx1 : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_input(%input1 : memref<1x16x8x8xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [7, 7, 31], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        }
        PPE : {
        }

    return %conv0, %conv1 : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>, memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1x!qElemType, {order = #NHWC}> = dense<> : tensor<0xf80>, [#const.Concat<
    // CHECK:       [[WEIGHTS_BUF0:%.+]] = memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[WEIGHTS_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[WEIGHTS]] : memref<32x32x1x1x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:      outputs([[WEIGHTS_BUF0]] : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK:       [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weights([[WEIGHTS_CMX0]] : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-NOT:   memref.alloc() : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weights([[WEIGHTS_CMX0]] : memref<32x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
}

// -----

// Positive: two NCE tasks with weights from SubViews of the same DDR buffer.
// Bias consts are identical splat values — tile-1 bias alloc+copy is eliminated.
// Scale consts are also identical but are NOT merged by SplatConstRewriter.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseSplatBiasForTiledWeights
func.func @FuseSplatBiasForTiledWeights(
        %input0: memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>,
        %input1: memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>,
        %weight_ddr: memref<16x32x1x1x!qElemType, {order = #NHWC}>)
        -> (memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>,
            memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
{
    // Constants first — each tile has a separate const.Declare with a different
    // SubView offset, preventing CSE while keeping the same splat value.
    %scale_cst0 = const.Declare memref<8x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 1]>]
    %bias_cst0   = const.Declare memref<8x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 1]>]
    %scale_cst1 = const.Declare memref<8x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 1]>]
    %bias_cst1   = const.Declare memref<8x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 1]>]

    // --- Tile 0 ---
    %wt_sv0 = VPUIP.SubView %weight_ddr [0, 0, 0, 0] [8, 32, 1, 1]
        : memref<16x32x1x1x!qElemType, {order = #NHWC}>
          to memref<8x32x1x1x!qElemType, {order = #NHWC}>
    %wt_buf0 = memref.alloc() : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %wt_cmx0 = VPUIP.Copy
        inputs(%wt_sv0 : memref<8x32x1x1x!qElemType, {order = #NHWC}>)
        outputs(%wt_buf0 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        -> memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %scale_buf0 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %scale_cmx0 = VPUIP.Copy
        inputs(%scale_cst0 : memref<8x1x1x1xf32>)
        outputs(%scale_buf0 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %bias_buf0 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %bias_cmx0 = VPUIP.Copy
        inputs(%bias_cst0 : memref<8x1x1x1xf32>)
        outputs(%bias_buf0 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %out_buf0 = memref.alloc() : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
    %nce0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>}>
        input(%input0 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        weights(%wt_cmx0 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx0 : memref<8x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx0 : memref<8x1x1x1xf32, @CMX_NN>)
        parent_input(%input0 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf0 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf0 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [3, 3, 7],
                      mpe_mode = #VPU.mpe_mode<CUBOID_4x16>,
                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        } PPE : {}

    // --- Tile 1: different SubView of same weight buffer; same splat scale/bias ---
    %wt_sv1 = VPUIP.SubView %weight_ddr [8, 0, 0, 0] [8, 32, 1, 1]
        : memref<16x32x1x1x!qElemType, {order = #NHWC}>
          to memref<8x32x1x1x!qElemType, {order = #NHWC}>
    %wt_buf1 = memref.alloc() : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %wt_cmx1 = VPUIP.Copy
        inputs(%wt_sv1 : memref<8x32x1x1x!qElemType, {order = #NHWC}>)
        outputs(%wt_buf1 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        -> memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %scale_buf1 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %scale_cmx1 = VPUIP.Copy
        inputs(%scale_cst1 : memref<8x1x1x1xf32>)
        outputs(%scale_buf1 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %bias_buf1 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %bias_cmx1 = VPUIP.Copy
        inputs(%bias_cst1 : memref<8x1x1x1xf32>)
        outputs(%bias_buf1 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %out_buf1 = memref.alloc() : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
    %nce1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>}>
        input(%input1 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        weights(%wt_cmx1 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx1 : memref<8x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx1 : memref<8x1x1x1xf32, @CMX_NN>)
        parent_input(%input1 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf1 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf1 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [3, 3, 7],
                      mpe_mode = #VPU.mpe_mode<CUBOID_4x16>,
                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        } PPE : {}

    return %nce0, %nce1
        : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>, memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[SCALE_CST0:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:   [[SCALE_CST1:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<0.000000e+00>
    // Scale is NOT merged: tile-0 gets its own alloc+copy.
    // CHECK:       [[SCALE_BUF0:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[SCALE_BUF0]] :
    // Bias is merged: one alloc+copy shared by both tiles.
    // CHECK:       [[BIAS_BUF:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[BIAS_CST]] :
    // CHECK-SAME:      outputs([[BIAS_BUF]] :
    // CHECK:       [[NCE0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX0]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX]] : memref<8x1x1x1xf32, @CMX_NN>)
    // Tile-1: separate scale alloc+copy; bias copy eliminated (reuses [[BIAS_CMX]]).
    // CHECK:       [[SCALE_BUF1:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[SCALE_BUF1]] :
    // CHECK-NOT:   memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[NCE1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX1]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK:       return [[NCE0]], [[NCE1]]
}

// -----

// Negative: bias consts with different splat values are NOT fused.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: func.func @DoNotFuseDifferentSplatBiasForTiledWeights
func.func @DoNotFuseDifferentSplatBiasForTiledWeights(
        %input0: memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>,
        %input1: memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        -> (memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>,
            memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
{
    %weight_ddr = const.Declare memref<16x32x1x1x!qElemType, {order = #NHWC}> = dense<1>
        : tensor<16x32x1x1xi4>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %scale_cst0 = const.Declare memref<8x1x1x1xf32> = dense<5.000000e-01> : tensor<8x1x1x1xf32>
    %bias_cst0   = const.Declare memref<8x1x1x1xf32> = dense<0.000000e+00> : tensor<8x1x1x1xf32>
    %scale_cst1 = const.Declare memref<8x1x1x1xf32> = dense<2.000000e+00> : tensor<8x1x1x1xf32>
    %bias_cst1   = const.Declare memref<8x1x1x1xf32> = dense<1.000000e+00> : tensor<8x1x1x1xf32>

    // --- Tile 0: scale=0.5, bias=0.0 ---
    %wt_sv0 = VPUIP.SubView %weight_ddr [0, 0, 0, 0] [8, 32, 1, 1]
        : memref<16x32x1x1x!qElemType, {order = #NHWC}>
          to memref<8x32x1x1x!qElemType, {order = #NHWC}>
    %wt_buf0 = memref.alloc() : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %wt_cmx0 = VPUIP.Copy
        inputs(%wt_sv0 : memref<8x32x1x1x!qElemType, {order = #NHWC}>)
        outputs(%wt_buf0 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        -> memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %scale_buf0 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %scale_cmx0 = VPUIP.Copy
        inputs(%scale_cst0 : memref<8x1x1x1xf32>)
        outputs(%scale_buf0 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %bias_buf0 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %bias_cmx0 = VPUIP.Copy
        inputs(%bias_cst0 : memref<8x1x1x1xf32>)
        outputs(%bias_buf0 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %out_buf0 = memref.alloc() : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
    %nce0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>}>
        input(%input0 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        weights(%wt_cmx0 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx0 : memref<8x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx0 : memref<8x1x1x1xf32, @CMX_NN>)
        parent_input(%input0 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf0 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf0 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [3, 3, 7],
                      mpe_mode = #VPU.mpe_mode<CUBOID_4x16>,
                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        } PPE : {}

    // --- Tile 1: scale=2.0 (differs), bias=1.0 (differs) ---
    %wt_sv1 = VPUIP.SubView %weight_ddr [8, 0, 0, 0] [8, 32, 1, 1]
        : memref<16x32x1x1x!qElemType, {order = #NHWC}>
          to memref<8x32x1x1x!qElemType, {order = #NHWC}>
    %wt_buf1 = memref.alloc() : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %wt_cmx1 = VPUIP.Copy
        inputs(%wt_sv1 : memref<8x32x1x1x!qElemType, {order = #NHWC}>)
        outputs(%wt_buf1 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        -> memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
    %scale_buf1 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %scale_cmx1 = VPUIP.Copy
        inputs(%scale_cst1 : memref<8x1x1x1xf32>)
        outputs(%scale_buf1 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %bias_buf1 = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    %bias_cmx1 = VPUIP.Copy
        inputs(%bias_cst1 : memref<8x1x1x1xf32>)
        outputs(%bias_buf1 : memref<8x1x1x1xf32, @CMX_NN>)
        -> memref<8x1x1x1xf32, @CMX_NN>
    %out_buf1 = memref.alloc() : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
    %nce1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>}>
        input(%input1 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        weights(%wt_cmx1 : memref<8x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
        weight_table_scale(%scale_cmx1 : memref<8x1x1x1xf32, @CMX_NN>)
        weight_table_bias(%bias_cmx1 : memref<8x1x1x1xf32, @CMX_NN>)
        parent_input(%input1 : memref<1x32x4x4xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%out_buf1 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%out_buf1 : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outStart = [0, 0, 0], outEnd = [3, 3, 7],
                      mpe_mode = #VPU.mpe_mode<CUBOID_4x16>,
                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
        } PPE : {}

    return %nce0, %nce1
        : memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>, memref<1x8x4x4xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[SCALE_CST0:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<5.000000e-01>
    // CHECK-DAG:   [[BIAS_CST0:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<0.000000e+00>
    // CHECK-DAG:   [[SCALE_CST1:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<2.000000e+00>
    // CHECK-DAG:   [[BIAS_CST1:%.+]] = const.Declare memref<8x1x1x1xf32> = dense<1.000000e+00>
    // CHECK:       [[SCALE_BUF0:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SCALE_CST0]] :
    // CHECK-SAME:      outputs([[SCALE_BUF0]] :
    // CHECK:       [[BIAS_BUF0:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_CMX0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[BIAS_CST0]] :
    // CHECK-SAME:      outputs([[BIAS_BUF0]] :
    // CHECK:       [[NCE0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX0]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX0]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK:       [[SCALE_BUF1:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[SCALE_CMX1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SCALE_CST1]] :
    // CHECK-SAME:      outputs([[SCALE_BUF1]] :
    // CHECK:       [[BIAS_BUF1:%.+]] = memref.alloc() : memref<8x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_CMX1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[BIAS_CST1]] :
    // CHECK-SAME:      outputs([[BIAS_BUF1]] :
    // CHECK:       [[NCE1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table_scale([[SCALE_CMX1]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_CMX1]] : memref<8x1x1x1xf32, @CMX_NN>)
    // CHECK:       return [[NCE0]], [[NCE1]]
}
