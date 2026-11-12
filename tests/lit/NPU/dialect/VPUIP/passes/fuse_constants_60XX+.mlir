//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --fuse-constants %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input_DDR = memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x1x1x4xf16, {order = #NHWC}, @DDR>
!ScaleTable_DDR = memref<16x1x1x1xf32, @DDR>
!BiasTable_DDR = memref<16x1x1x1xf32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x1x1x4xf16, {order = #NHWC}, @CMX_NN>
!ScaleTableStub_CMX = memref<16x1x1x1xf32, @CMX_NN>
!BiasTableStub_CMX = memref<16x1x1x1xf32, @CMX_NN>

// CHECK-LABEL: @FuseConstantsConv
module @FuseConstantsConv {

// CHECK-LABEL: @main
// CHECK-SAME: ([[IN:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>)
func.func @main(%in : !Input_DDR) -> !OutputStub_CMX {

    %input_CMX = memref.alloc() : !InputStub_CMX
    %output_CMX = memref.alloc() : !OutputStub_CMX
    %scale_CMX = memref.alloc() : !ScaleTableStub_CMX
    %bias_CMX = memref.alloc() : !BiasTableStub_CMX
    %weights_CMX = memref.alloc() : !WeightsStub_CMX

    %scale_table_const = const.Declare !ScaleTable_DDR = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_const = const.Declare !BiasTable_DDR = dense<2.0> : tensor<16x1x1x1xf32>
    %weights_const = const.Declare !Weights_DDR = dense<3.0> : tensor<16x1x1x4xf16>, [#const.Reorder<#NHWC>]

    %scale_table = VPUIP.Copy inputs(%scale_table_const : !ScaleTable_DDR) outputs(%scale_CMX : !ScaleTableStub_CMX) -> !ScaleTableStub_CMX
    %bias_table = VPUIP.Copy inputs(%bias_table_const : !BiasTable_DDR) outputs(%bias_CMX : !BiasTableStub_CMX) -> !BiasTableStub_CMX
    %input = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%input_CMX : !InputStub_CMX) -> !InputStub_CMX
    %weights = VPUIP.Copy inputs(%weights_const : !Weights_DDR) outputs(%weights_CMX : !WeightsStub_CMX) -> !WeightsStub_CMX
    %NCEOp = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%input : !InputStub_CMX)
        weights(%weights : !WeightsStub_CMX)
        weight_table_scale(%scale_table : !ScaleTableStub_CMX)
        weight_table_bias(%bias_table : !BiasTableStub_CMX)
        parent_input(%input : !InputStub_CMX)
        parent_output(%output_CMX : !OutputStub_CMX)
        outputs(%output_CMX : !OutputStub_CMX) -> !OutputStub_CMX
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 10, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }
    return %NCEOp : !OutputStub_CMX

    // CHECK:       [[INPUT_CMX:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[OUTPUT_CMX:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[SCALE_TABLE_CMX:%.+]] = memref.alloc() : memref<16x1x1x1xf32, @CMX_NN>
    // CHECK:       [[BIAS_TABLE_CMX:%.+]] = memref.alloc() : memref<16x1x1x1xf32, @CMX_NN>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = memref.alloc() : memref<16x1x1x4xf16, {order = #NHWC}, @CMX_NN>

    // CHECK-DAG:   [[SCALE_TABLE_CONST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:   [[BIAS_TABLE_CONST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:   [[WEIGHTS_CONST:%.+]] = const.Declare memref<16x1x1x4xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[SCALE_TABLE:%.+]] = VPUIP.Copy inputs([[SCALE_TABLE_CONST]] : memref<16x1x1x1xf32, @DDR>)
    // CHECK-SAME:      outputs([[SCALE_TABLE_CMX]]  : memref<16x1x1x1xf32, @CMX_NN>) -> memref<16x1x1x1xf32, @CMX_NN>

    // CHECK:       [[BIAS_TABLE:%.+]] = VPUIP.Copy inputs([[BIAS_TABLE_CONST]] : memref<16x1x1x1xf32, @DDR>)
    // CHECK-SAME:      outputs([[BIAS_TABLE_CMX]]  : memref<16x1x1x1xf32, @CMX_NN>) -> memref<16x1x1x1xf32, @CMX_NN>

    // CHECK:       [[INPUT:%.+]] = VPUIP.Copy inputs([[IN]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[INPUT_CMX]]  : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[WEIGHTS:%.+]] = VPUIP.Copy inputs([[WEIGHTS_CONST]] : memref<16x1x1x4xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[WEIGHTS_CMX]] : memref<16x1x1x4xf16, {order = #NHWC}, @CMX_NN>) -> memref<16x1x1x4xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[NCE_OP:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      weights([[WEIGHTS]] : memref<16x1x1x4xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      weight_table_scale([[SCALE_TABLE]] : memref<16x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      weight_table_bias([[BIAS_TABLE]] : memref<16x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUTPUT_CMX]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUTPUT_CMX]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:       return [[NCE_OP]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
}

}
