//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --swizzling="enable-activation-swizzling=true enable-weights-swizzling=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Weights_DDR = memref<16x16x1x1xf16, {order = #NHWC}, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, {order = #NHWC}, @DDR>
!WeightsStub_CMX = memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, {order = #NHWC}, @CMX_NN>

func.func @SetSwizzlingForDpuToDpuBufferOnly(%in : memref<1x16x56x56xf16, {order = #NHWC}, @DDR>) -> memref<1x16x56x56xf16, {order = #NHWC}, @DDR> {
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %buf0 = memref.alloc() : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x56x56xf16, {order = #NHWC}, @DDR>
    %buf5 = memref.alloc() : !WeightsTableStub_CMX
    %buf6 = memref.alloc() : !WeightsStub_CMX

    %5 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf5 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %6 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf6 : !WeightsStub_CMX) -> !WeightsStub_CMX

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x56x56xf16, {order = #NHWC}, @DDR>)
            outputs(%buf0 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>

    %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%0 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        weights(%6 : memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%5 : memref<16x1x1x4xsi32, {order = #NHWC}, @CMX_NN>)
        parent_input(%0 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%1 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        weights(%6 : memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%5 : memref<16x1x1x4xsi32, {order = #NHWC}, @CMX_NN>)
        parent_input(%1 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%buf2 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%buf3 : memref<1x16x56x56xf16, {order = #NHWC}, @DDR>)
             -> memref<1x16x56x56xf16, {order = #NHWC}, @DDR>

    return %3 : memref<1x16x56x56xf16, {order = #NHWC}, @DDR>

    // Verify that swizzling for DPU to DPU buffer only

    // CHECK:   VPUIP.NCEClusterTask <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK-SAME:                   task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK:   input({{[^:]+}} : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK:   weights({{[^:]+}} : memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK:   weight_table({{[^:]+}} : memref<16x1x1x4xsi32, {order = #NHWC}, @CMX_NN>)
    // CHECK:   parent_input({{[^:]+}} : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK:   parent_output({{[^:]+}} : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:   outputs({{[^:]+}} : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
}
