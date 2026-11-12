//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.platform = #config.platform<NPU6010>} {

func.func @ParsePrintClusterTaskWtSplit1(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    %wt_data_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_sp_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_scale = const.Declare memref<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %wt_bias = const.Declare memref<64x1x1x1xf32> = dense<0.0> : tensor<64x1x1x1xf32>
    %wt_data_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_sp_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_data_ptr_cmx = VPUIP.Copy inputs(%wt_data_ptr : memref<64x1x1x1xsi32>) outputs(%wt_data_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_sp_ptr_cmx = VPUIP.Copy inputs(%wt_sp_ptr : memref<64x1x1x1xsi32>) outputs(%wt_sp_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_cmx = VPUIP.Copy inputs(%wt_scale : memref<64x1x1x1xf32>) outputs(%wt_scale_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_cmx = VPUIP.Copy inputs(%wt_bias : memref<64x1x1x1xf32>) outputs(%wt_bias_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>  input(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
                weight_table_data_ptr(%wt_data_ptr_cmx: memref<64x1x1x1xsi32, @CMX_NN>)
                weight_table_sp_ptr(%wt_sp_ptr_cmx: memref<64x1x1x1xsi32, @CMX_NN>)
                weight_table_scale(%wt_scale_cmx: memref<64x1x1x1xf32, @CMX_NN>)
                weight_table_bias(%wt_bias_cmx: memref<64x1x1x1xf32, @CMX_NN>)
                parent_input(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                parent_output(%out_buff_cmx : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
                outputs(%out_buff_cmx : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
                    -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {
                    outStart = [0, 0, 0], outEnd = [31, 15, 15],
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
                } PPE :  {
                }
        async.yield %0 : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    }

    %0 = async.await %r1 : !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
    return %0 : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:        [[OUT_BUFF_CMX:%.+]] = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:        [[WT_DATA_PTR:%.+]] = const.Declare memref<64x1x1x1xsi32>
    // CHECK:        [[WT_SP_PTR:%.+]] = const.Declare memref<64x1x1x1xsi32>
    // CHECK:        [[WT_SCALE:%.+]] = const.Declare memref<64x1x1x1xf32>
    // CHECK:        [[WT_BIAS:%.+]] = const.Declare memref<64x1x1x1xf32>

    // CHECK:        [[WT_DATA_PTR_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    // CHECK:        [[WT_SP_PTR_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    // CHECK:        [[WT_SCALE_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:        [[WT_BIAS_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:        [[WT_DATA_PTR_CMX:%.+]] = VPUIP.Copy
    // CHECK:        [[WT_SP_PTR_CMX:%.+]] = VPUIP.Copy
    // CHECK:        [[WT_SCALE_CMX:%.+]] = VPUIP.Copy
    // CHECK:        [[WT_BIAS_CMX:%.+]] = VPUIP.Copy

    // CHECK:        {{[^:]+}}, [[RESULTS:%.+]] = async.execute -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:        [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:           input({{[^:]+}} : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           weights([[WEIGHTS]] : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           weight_table_data_ptr([[WT_DATA_PTR_CMX]] : memref<64x1x1x1xsi32, @CMX_NN>)
    // CHECK-SAME:           weight_table_sp_ptr([[WT_SP_PTR_CMX]] : memref<64x1x1x1xsi32, @CMX_NN>)
    // CHECK-SAME:           weight_table_scale([[WT_SCALE_CMX]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:           weight_table_bias([[WT_BIAS_CMX]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:           parent_input({{[^:]+}} : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           parent_output([[OUT_BUFF_CMX]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           outputs([[OUT_BUFF_CMX]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:            async.yield [[NCE_TASK]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:        }
    // CHECK:        [[WAIT_RESULTS:%.+]] = async.await [[RESULTS]] : !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
    // CHECK:        return [[WAIT_RESULTS]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.platform = #config.platform<NPU6010>} {

func.func @ParsePrintClusterTaskWtSplit2(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    %wt_sp_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_scale = const.Declare memref<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %wt_bias = const.Declare memref<64x1x1x1xf32> = dense<0.0> : tensor<64x1x1x1xf32>
    %wt_zp = const.Declare memref<64x1x1x1xi8> = dense<0> : tensor<64x1x1x1xi8>
    %wt_sp_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_buff_cmx = memref.alloc() : memref<64x1x1x1xi8, @CMX_NN>
    %wt_sp_ptr_cmx = VPUIP.Copy inputs(%wt_sp_ptr : memref<64x1x1x1xsi32>) outputs(%wt_sp_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_cmx = VPUIP.Copy inputs(%wt_scale : memref<64x1x1x1xf32>) outputs(%wt_scale_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_cmx = VPUIP.Copy inputs(%wt_bias : memref<64x1x1x1xf32>) outputs(%wt_bias_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_cmx = VPUIP.Copy inputs(%wt_zp : memref<64x1x1x1xi8>) outputs(%wt_zp_buff_cmx : memref<64x1x1x1xi8, @CMX_NN>) -> memref<64x1x1x1xi8, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>  input(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
                weight_table_sp_ptr(%wt_sp_ptr_cmx: memref<64x1x1x1xsi32, @CMX_NN>)
                weight_table_scale(%wt_scale_cmx: memref<64x1x1x1xf32, @CMX_NN>)
                weight_table_bias(%wt_bias_cmx: memref<64x1x1x1xf32, @CMX_NN>)
                weight_zero_points(%wt_zp_cmx: memref<64x1x1x1xi8, @CMX_NN>)
                parent_input(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                parent_output(%out_buff_cmx : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
                outputs(%out_buff_cmx : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
                    -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {
                    outStart = [0, 0, 0], outEnd = [31, 15, 15],
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
                } PPE :  {
                }
        async.yield %0 : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    }

    %0 = async.await %r1 : !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
    return %0 : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:        [[OUT_BUFF_CMX:%.+]] = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:        [[WT_SP_PTR:%.+]] = const.Declare memref<64x1x1x1xsi32>
    // CHECK:        [[WT_SCALE:%.+]] = const.Declare memref<64x1x1x1xf32>
    // CHECK:        [[WT_BIAS:%.+]] = const.Declare memref<64x1x1x1xf32>
    // CHECK:        [[WT_ZP:%.+]] = const.Declare memref<64x1x1x1xi8>

    // CHECK:        [[WT_SP_PTR_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    // CHECK:        [[WT_SCALE_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:        [[WT_BIAS_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    // CHECK:        [[WT_ZP_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xi8, @CMX_NN>
    // CHECK:        [[WT_SP_PTR_CMX:%.+]] = VPUIP.Copy
    // CHECK:        [[WT_SCALE_CMX:%.+]] = VPUIP.Copy
    // CHECK:        [[WT_BIAS_CMX:%.+]] = VPUIP.Copy
    // CHECK:        [[WT_ZP_CMX:%.+]] = VPUIP.Copy

    // CHECK:        {{[^:]+}}, [[RESULTS:%.+]] = async.execute -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:        [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:           input({{[^:]+}} : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           weights([[WEIGHTS]] : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           weight_table_sp_ptr([[WT_SP_PTR_CMX]] : memref<64x1x1x1xsi32, @CMX_NN>)
    // CHECK-SAME:           weight_table_scale([[WT_SCALE_CMX]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:           weight_table_bias([[WT_BIAS_CMX]] : memref<64x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:           weight_zero_points([[WT_ZP_CMX]] : memref<64x1x1x1xi8, @CMX_NN>)
    // CHECK-SAME:           parent_input({{[^:]+}} : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           parent_output([[OUT_BUFF_CMX]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:           outputs([[OUT_BUFF_CMX]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:            async.yield [[NCE_TASK]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:        }
    // CHECK:        [[WAIT_RESULTS:%.+]] = async.await [[RESULTS]] : !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
    // CHECK:        return [[WAIT_RESULTS]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
}
}
