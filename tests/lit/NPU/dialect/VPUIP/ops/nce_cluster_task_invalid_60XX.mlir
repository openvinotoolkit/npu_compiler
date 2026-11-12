//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.platform = #config.platform<NPU6010>} {

func.func @ParsePrintClusterTaskWtSplitInvalid1(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    %wt_sp_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_scale = const.Declare memref<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %wt_zp = const.Declare memref<64x1x1x1xi8> = dense<0> : tensor<64x1x1x1xi8>
    %wt_sp_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_buff_cmx = memref.alloc() : memref<64x1x1x1xi8, @CMX_NN>
    %wt_sp_ptr_cmx = VPUIP.Copy inputs(%wt_sp_ptr : memref<64x1x1x1xsi32>) outputs(%wt_sp_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_cmx = VPUIP.Copy inputs(%wt_scale : memref<64x1x1x1xf32>) outputs(%wt_scale_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_cmx = VPUIP.Copy inputs(%wt_zp : memref<64x1x1x1xi8>) outputs(%wt_zp_buff_cmx : memref<64x1x1x1xi8, @CMX_NN>) -> memref<64x1x1x1xi8, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        // expected-error@+1 {{weight_table scale and bias are required together for NCETaskType : 'CONV'}}
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>  input(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
                weight_table_sp_ptr(%wt_sp_ptr_cmx: memref<64x1x1x1xsi32, @CMX_NN>)
                weight_table_scale(%wt_scale_cmx: memref<64x1x1x1xf32, @CMX_NN>)
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
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.platform = #config.platform<NPU6010>} {

func.func @ParsePrintClusterTaskWtSplitInvalid2(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    %weight_table = const.Declare memref<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %wt_sp_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_scale = const.Declare memref<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %wt_bias = const.Declare memref<64x1x1x1xf32> = dense<0.0> : tensor<64x1x1x1xf32>
    %wt_zp = const.Declare memref<64x1x1x1xi8> = dense<0> : tensor<64x1x1x1xi8>
    %weight_table_buff_cmx = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %wt_sp_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_buff_cmx = memref.alloc() : memref<64x1x1x1xi8, @CMX_NN>
    %weight_table_cmx = VPUIP.Copy inputs(%weight_table : memref<64x1x1x4xsi32>) outputs(%weight_table_buff_cmx : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    %wt_sp_ptr_cmx = VPUIP.Copy inputs(%wt_sp_ptr : memref<64x1x1x1xsi32>) outputs(%wt_sp_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_cmx = VPUIP.Copy inputs(%wt_scale : memref<64x1x1x1xf32>) outputs(%wt_scale_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_cmx = VPUIP.Copy inputs(%wt_bias : memref<64x1x1x1xf32>) outputs(%wt_bias_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_cmx = VPUIP.Copy inputs(%wt_zp : memref<64x1x1x1xi8>) outputs(%wt_zp_buff_cmx : memref<64x1x1x1xi8, @CMX_NN>) -> memref<64x1x1x1xi8, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        // expected-error@+1 {{weight_table legacy and split format are mutually exclusive for NCETaskType : 'CONV'}}
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>  input(%arg0 : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%weight_table_cmx : memref<64x1x1x4xsi32, @CMX_NN>)
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
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.platform = #config.platform<NPU6010>} {

func.func @ParsePrintClusterTaskWtSplitInvalid3(%arg0: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    %wt_data_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_sp_ptr = const.Declare memref<64x1x1x1xsi32> = dense<10> : tensor<64x1x1x1xsi32>
    %wt_scale = const.Declare memref<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %wt_bias = const.Declare memref<64x1x1x1xf32> = dense<0.0> : tensor<64x1x1x1xf32>
    %wt_zp = const.Declare memref<64x1x1x1xi8> = dense<0> : tensor<64x1x1x1xi8>
    %wt_data_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_sp_ptr_buff_cmx = memref.alloc() : memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_buff_cmx = memref.alloc() : memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_buff_cmx = memref.alloc() : memref<64x1x1x1xi8, @CMX_NN>
    %wt_data_ptr_cmx = VPUIP.Copy inputs(%wt_data_ptr : memref<64x1x1x1xsi32>) outputs(%wt_data_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_sp_ptr_cmx = VPUIP.Copy inputs(%wt_sp_ptr : memref<64x1x1x1xsi32>) outputs(%wt_sp_ptr_buff_cmx : memref<64x1x1x1xsi32, @CMX_NN>) -> memref<64x1x1x1xsi32, @CMX_NN>
    %wt_scale_cmx = VPUIP.Copy inputs(%wt_scale : memref<64x1x1x1xf32>) outputs(%wt_scale_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_bias_cmx = VPUIP.Copy inputs(%wt_bias : memref<64x1x1x1xf32>) outputs(%wt_bias_buff_cmx : memref<64x1x1x1xf32, @CMX_NN>) -> memref<64x1x1x1xf32, @CMX_NN>
    %wt_zp_cmx = VPUIP.Copy inputs(%wt_zp : memref<64x1x1x1xi8>) outputs(%wt_zp_buff_cmx : memref<64x1x1x1xi8, @CMX_NN>) -> memref<64x1x1x1xi8, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        // expected-error@+1 {{weight_table data pointers and zero points only are mutually exclusive for NCETaskType : 'CONV'}}
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
}
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input= memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!InputSM = memref<1x32x6x6xi1, @CMX_NN>
!InputSE = memref<1x1x6x6xi32, {order = #NHWC}, @CMX_NN>
!Weights = memref<32x32x1x1xf16, {order = #NHWC}, @CMX_NN>

!Output = memref<1x32x6x6xf16, {order = #NHWC}, @CMX_NN>


module attributes {config.platform = #config.platform<NPU6010>} {

func.func @checkConvDCIMLimitation(%input_data: !Input, %input_sm: !InputSM, %input_se: !InputSE,
                                     %weights_data: !Weights, %out_cmx: !Output)
                                      -> !Output {

    %t1, %r1 = async.execute
                -> !async.value<!Output>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // expected-error@+1 {{DCIM is not supported with Sparsity or SEP for op: 'CONV'}}
    %conv_out = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
        task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input_data : !Input)
        input_sparsity_map(%input_sm : !InputSM)
        input_storage_element_table(%input_se : !InputSE)
        weights(%weights_data : !Weights)
        parent_input(%input_data : !Input)
        parent_input_sparsity_map(%input_sm : !InputSM)
        parent_input_storage_element_table(%input_se : !InputSE)
        parent_output(%out_cmx : !Output)
        outputs(%out_cmx : !Output)
    -> !Output variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [16, 6, 6],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
        async.yield %conv_out : !Output

        }
    %conv_out = async.await %r1 : !async.value<!Output>
    return %conv_out : !Output

}
}
