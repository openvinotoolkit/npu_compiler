//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @checkDimensionLimit(%arg0: memref<1x32x16384x16xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x16384xf16, #NHWC, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %out_buff_cmx = memref.alloc() : memref<1x64x14x16384xf16, #NHWC, @CMX_NN>
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
                -> !async.value<memref<1x64x14x16384xf16, #NHWC, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        // expected-error@+1 {{Op dimensions exceed VPU_DIMENSION_LIMIT: [1, 32, 16384, 16]}}
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }  input(%arg0 : memref<1x32x16384x16xf16, #NHWC, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
                weight_table_sp_ptr(%wt_sp_ptr_cmx: memref<64x1x1x1xsi32, @CMX_NN>)
                weight_table_scale(%wt_scale_cmx: memref<64x1x1x1xf32, @CMX_NN>)
                weight_zero_points(%wt_zp_cmx: memref<64x1x1x1xi8, @CMX_NN>)
                parent_input(%arg0 : memref<1x32x16384x16xf16, #NHWC, @CMX_NN>)
                parent_output(%out_buff_cmx : memref<1x64x14x16384xf16, #NHWC, @CMX_NN>)
                outputs(%out_buff_cmx : memref<1x64x14x16384xf16, #NHWC, @CMX_NN>)
                    -> memref<1x64x14x16384xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {
                    outStart = [0, 0, 0], outEnd = [31, 15, 16383],
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
                } PPE :  {
                }
        async.yield %0 : memref<1x64x14x16384xf16, #NHWC, @CMX_NN>
    }

    %0 = async.await %r1 : !async.value<memref<1x64x14x16384xf16, #NHWC, @CMX_NN>>
    return %0 : memref<1x64x14x16384xf16, #NHWC, @CMX_NN>
  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @checkOutputDimensionLimit(%arg0: memref<1x32x8190x16xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x8194xf16, #NHWC, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %out_buff_cmx = memref.alloc() : memref<1x64x14x8194xf16, #NHWC, @CMX_NN>
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
                -> !async.value<memref<1x64x14x8194xf16, #NHWC, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        // expected-error@+1 {{Op dimensions exceed VPU_DIMENSION_LIMIT: [1, 64, 14, 8194]}}
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }  input(%arg0 : memref<1x32x8190x16xf16, #NHWC, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
                weight_table_sp_ptr(%wt_sp_ptr_cmx: memref<64x1x1x1xsi32, @CMX_NN>)
                weight_table_scale(%wt_scale_cmx: memref<64x1x1x1xf32, @CMX_NN>)
                weight_zero_points(%wt_zp_cmx: memref<64x1x1x1xi8, @CMX_NN>)
                parent_input(%arg0 : memref<1x32x8190x16xf16, #NHWC, @CMX_NN>)
                parent_output(%out_buff_cmx : memref<1x64x14x8194xf16, #NHWC, @CMX_NN>)
                outputs(%out_buff_cmx : memref<1x64x14x8194xf16, #NHWC, @CMX_NN>)
                    -> memref<1x64x14x8194xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {
                    outStart = [0, 0, 0], outEnd = [31, 15, 8193],
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
                }
                } PPE :  {
                }
        async.yield %0 : memref<1x64x14x8194xf16, #NHWC, @CMX_NN>
    }

    %0 = async.await %r1 : !async.value<memref<1x64x14x8194xf16, #NHWC, @CMX_NN>>
    return %0 : memref<1x64x14x8194xf16, #NHWC, @CMX_NN>
}
