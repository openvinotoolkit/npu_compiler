//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-constants %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x1x1x4xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x1x1x4xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @FuseConstantsConv
func.func @FuseConstantsConv(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x1x1x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
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
    return %3 : !OutputStub_CMX

	// CHECK-DAG:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x384xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[OUT_BUF1]]  : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x384xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x384xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x384xui8, [@CMX_NN, 0]>) -> memref<1x1x1x384xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] :
    // CHECK-SAME:		memref<1x1x1x384xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 128] :
    // CHECK-SAME:		memref<1x1x1x384xui8, [@CMX_NN, 0]> to memref<1x1x1x128xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] : memref<1x1x1x128xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<16x1x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:       return [[VAR6]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x1x1x4x!qElemType, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x1x1x4x!qElemType, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @FuseConstantsConv
func.func @FuseConstantsConv(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x1x1x4xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
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
    return %3 : !OutputStub_CMX

    // CHECK-DAG:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x288xui8> = dense<1> : tensor<16x1x1x4xsi32>, [#const.FuseWeights<tensor<1x1x1x288xui8>, weightsTable = <dense<1> : tensor<16x1x1x4xsi32>>, weights = <dense<1.000000e+00> : tensor<16x1x1x4xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]>>]
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[OUT_BUF1]]  : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x288xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x288xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x288xui8, [@CMX_NN, 0]>) -> memref<1x1x1x288xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] :
    // CHECK-SAME:		memref<1x1x1x288xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [288, 288, 288, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [288, 288, 288, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 32] :
    // CHECK-SAME:		memref<1x1x1x288xui8, [@CMX_NN, 0]> to memref<1x1x1x32xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [288, 288, 288, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] : memref<1x1x1x32xui8,
    // CHECK-SAME:		{order = #NCHW, strides = [288, 288, 288, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4x!qElemType, #NHWC, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<16x1x1x4x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:       return [[VAR6]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x1x1x4xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x1x1x4xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @FuseConstantsMaxPool
func.func @FuseConstantsMaxPool(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %2 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %3 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%2 : !InputStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%2 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
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
    return %3 : !OutputStub_CMX

    // CHECK-DAG:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x256xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[OUT_BUF1]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x256xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x256xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x256xui8, [@CMX_NN, 0]>) -> memref<1x1x1x256xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR1]] : memref<1x1x1x256xui8,
    // CHECK-SAME:		[@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:       return [[VAR6]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>, order = #NHWC}, @DDR>
!WeightsSM_DDR = memref<16x1x1x128xi1, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>
!WeightsSMStub_CMX = memref<16x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @FuseConstantsConvSparseWeights
func.func @FuseConstantsConvSparseWeights(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX
    %buf4 = memref.alloc() : !WeightsSMStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.Copy inputs(%weights_sm : !WeightsSM_DDR) outputs(%buf4 : !WeightsSMStub_CMX) -> !WeightsSMStub_CMX
    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weights_sparsity_map(%3 : !WeightsSMStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
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
    return %4 : !OutputStub_CMX

    // CHECK-DAG:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x1024xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR
    // CHECK-NOT:   [[WEIGHTS_SM:%.+]] = const.Declare !WeightsSM_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[OUT_BUF1]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1024xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x1024xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1024xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1024xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] :
    // CHECK-SAME:      memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 512] :
    // CHECK-SAME:      memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x512xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] :
    // CHECK-SAME:      memref<1x1x1x512xui8, {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to
    // CHECK-SAME:      memref<16x16x1x1xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>}, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 768] [1, 1, 1, 256] :
    // CHECK-SAME:      memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR7:%.*]] = VPUIP.ViewOp [[VAR6]] : memref<1x1x1x256xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to memref<16x1x1x128xi1, @CMX_NN>

    // CHECK:       [[VAR8:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<16x16x1x1xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>}, @CMX_NN>)
    // CHECK-SAME:      weights_sparsity_map([[VAR7]] : memref<16x1x1x128xi1, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:       return [[VAR8]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @FuseDuplicatedConstants
func.func @FuseDuplicatedConstants(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weight_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed


    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
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

    %5 = VPUIP.Copy { out_mem_space = @DDR } inputs(%4: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %5 : !Output_DDR

    // Verify that constants are fused by checking no other cst is present other than fused constant

    // CHECK-DAG:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x1280xf16>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:   [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX]]

    // CHECK:   [[FUSED_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1280xf16, {order = #NCHW, strides = [1280, 1280, 1280, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[COPY_FUSED:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[FUSED_CONSTANT]] : memref<1x1x1x1280xf16>)
    // CHECK-SAME:      outputs([[FUSED_CMX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x16x112x28xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x112x28xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @FuseDuplicatedConstantsWithShapeCast
func.func @FuseDuplicatedConstantsWithShapeCast(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed0
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %11 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed0) -> !InputDistributed0

    %1 = VPUIP.ShapeCast {shape = [1, 16, 56, 56]} inputs(%11 : !InputDistributed0) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weight_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed


    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
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

    %5 = VPUIP.Copy { out_mem_space = @DDR } inputs(%4: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %5 : !Output_DDR

    // Verify that constants are fused by checking no other cst is present other than fused constant

    // CHECK-DAG:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x1280xf16>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:   [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x112x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x16x112x28xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX]]

    // CHECK:   [[FUSED_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1280xf16, {order = #NCHW, strides = [1280, 1280, 1280, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[COPY_FUSED:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[FUSED_CONSTANT]] : memref<1x1x1x1280xf16>)
    // CHECK-SAME:      outputs([[FUSED_CMX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    memory_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[32, 32, 1, 1], [32, 32, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 32, 1, 1], [32, 32, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    memory_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]]
}>

!Input_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @FuseDuplicatedConstantsWithExplicitDistributedAttr
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x56x56xf16, #NHWC, @DDR>)
func.func @FuseDuplicatedConstantsWithExplicitDistributedAttr(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed)
        -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weight_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed)
        -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed)
            -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
            {
                cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                outEnd = [55, 27, 15], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
            }
            DPUTask
            {
                cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                outEnd = [55, 55, 15], outStart = [0, 28, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
            }
        }
        PPE :
        {
        }

    %5 = VPUIP.Copy { out_mem_space = @DDR } inputs(%4: !OutputDistributed) outputs(%output: !Output_DDR)
        -> !Output_DDR

    return %5 : !Output_DDR

    // Verify that constants are fused by checking no other cst is present other than fused constant

    // CHECK-DAG:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x1280xf16>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:   [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]]}>

    // CHECK:   [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 28, 56], [1, 16, 28, 56]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]]}>

    // CHECK:   [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX]]

    // CHECK:       [[FUSED_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1280xf16,
    // CHECK-SAME:          {order = #NCHW, strides = [1280, 1280, 1280, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1280], [1, 1, 1, 1280]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1280], [1, 1, 1, 1280]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_FUSED:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[FUSED_CONSTANT]] : memref<1x1x1x1280xf16>)
    // CHECK-SAME:      outputs([[FUSED_CMX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input_0_DDR = memref<76x64x1x1xf16, #NHWC, @DDR>
!Input_1_DDR = memref<1x64x1x4xf16, #NHWC, @DDR>
!Output_DDR = memref<1x80x1x4xf16, #NHWC, @DDR>

// CHECK-LABEL: @DoNotFuseWeightsFromNonConstant
// CHECK-SAME: ([[INPUT:%.+]]: memref<76x64x1x1xf16, #NHWC, @DDR>, [[INPUT_0:%.+]]: memref<1x64x1x4xf16, #NHWC, @DDR>)
func.func @DoNotFuseWeightsFromNonConstant(%input : !Input_0_DDR, %input1 : !Input_1_DDR) -> !Output_DDR {
    %cst = const.Declare memref<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>
    %alloc = memref.alloc() : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>
    %0 = VPUIP.Copy
        inputs(%input1 : !Input_1_DDR)
        outputs(%alloc : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<80x64x1x1xf16, #NHWC, @DDR>
    %1 = VPUIP.ExpandDMA {
            pads_begin = [0, 0, 0, 0],
            pads_end = [4, 0, 0, 0]
        }
        inputs(%input : !Input_0_DDR)
        outputs(%alloc_0 : memref<80x64x1x1xf16, #NHWC, @DDR>)
        -> memref<80x64x1x1xf16, #NHWC, @DDR>
    %alloc_1 = memref.alloc() : memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPUIP.Copy
        inputs(%1 : memref<80x64x1x1xf16, #NHWC, @DDR>)
        outputs(%alloc_1 : memref<80x64x1x1xf16, affine_map<(d0, d1, d2, d3)
        -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<80x1x1x4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy
        inputs(%cst : memref<80x1x1x4xsi32>)
        outputs(%alloc_2 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
        -> memref<80x1x1x4xsi32, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            minimumHardwareExecutionCost = 4294967195 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%2 : memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%3 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%0 : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%alloc_3 : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%alloc_3 : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]> variants :
        {
            DPUTask {
                inEnd = [3, 0, 63],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                outEnd = [3, 0, 79],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }

    %alloc_5 = memref.alloc() : !Output_DDR
    %5 = VPUIP.Copy
        inputs(%4 : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%alloc_5 : !Output_DDR)
        -> !Output_DDR

    return %5 : !Output_DDR

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<80x1x1x4xsi32>

    // CHECK:       [[BUF_OUT_0_CMX:%.+]]  = memref.alloc() : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : memref<1x64x1x4xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUF_OUT_0_CMX]] : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:     -> memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[BUF_OUT_1_CMX:%.+]] = memref.alloc() : memref<80x64x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[EXPAND_INPUT:%.+]] = VPUIP.ExpandDMA
    // CHECK-SAME{LITERAL}:     pads_begin = [0, 0, 0, 0],
    // CHECK-SAME{LITERAL}:     pads_end = [4, 0, 0, 0]
    // CHECK-SAME:      inputs([[INPUT]] : memref<76x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX]] : memref<80x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      -> memref<80x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[BUF_OUT_2_CMX:%.+]] = memref.alloc() : memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[COPY_INPUT_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[EXPAND_INPUT]] : memref<80x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF_OUT_2_CMX]] : memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      -> memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[BUF_OUT_3_CMX:%.+]] = memref.alloc() : memref<80x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[COPY_INPUT_2:%.+]]  = VPUIP.Copy
    // CHECK-SAME:      inputs([[CST]] : memref<80x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[BUF_OUT_3_CMX]] : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      -> memref<80x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:   [[BUF_OUT_4_CMX:%.+]] = memref.alloc() : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[NCE_TASK:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          minimumHardwareExecutionCost = 4294967195 : i64,
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:   input([[COPY_INPUT]] : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:   weights([[COPY_INPUT_1]]  : memref<80x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:   weight_table([[COPY_INPUT_2]] : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:   parent_input([[COPY_INPUT]] : memref<1x64x1x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:   parent_output([[BUF_OUT_4_CMX]] : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:   outputs([[BUF_OUT_4_CMX]] : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:   -> memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]> variants :
    // CHECK:        DPUTask
    // CHECK-SAME:          inEnd = [3, 0, 63],
    // CHECK-SAME:          inStart = [0, 0, 0],
    // CHECK-SAME:          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
    // CHECK-SAME:          outEnd = [3, 0, 79],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:   [[BUF_OUT_5_CMX:%.+]] = memref.alloc() : memref<1x80x1x4xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_INPUT_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCE_TASK]] : memref<1x80x1x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[BUF_OUT_5_CMX]] : memref<1x80x1x4xf16, #NHWC, @DDR>)
    // CHECK-SAME:      -> memref<1x80x1x4xf16, #NHWC, @DDR>

    // CHECK:       return [[COPY_INPUT_3]] : memref<1x80x1x4xf16, #NHWC, @DDR>
}


// -----

!qElemType = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseSignedQuantizedWeightsMixedPrecision
func.func @FuseSignedQuantizedWeightsMixedPrecision(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>) -> memref<1x16x16x16xf16, #NHWC, @DDR> {
    %weights = const.Declare memref<16x16x1x1x!qElemType, #NHWC> = dense<"0x00C9A3BEF9486844534800C970C7E8C3FE4811C747C514C823C046C67CC02DC2CA4221C05B48363AED4677BE78C36943DB37E9C586BC904769C5BAC88239D24224489EBED9BAB23C8BBD31C764480AC6914504465244AF480E4677C312C8B143CF3587474D47E447954626C898469CC88AC49AC668C89047D24204C8723F50BEDE4294480FBF4E390DC6AD4335C461C328C77643AB45B146FFBED1C8A9C80145F640E3486D42F74408C464C44FBCC9458FC5EFC744C82BBCECB816480AC821C409483FC49CC766C7F037CEC82AC827432B48C4C51B48B0C405C465B1C03E77C8463DEE3D8F4011C79148253FC8C4FE4361C5F4C75A39E0BE8048C74371B0DEBE7F3A80C84F45BE398CC88D4233C7C434D9457248B4C8ED3EAA470948873A40C729BC37C7D8472646E6C018C0263AB1C6184246488DC117C2AE3D04458341854479C7AB479C43F240E9410545D8C10BC244459AC4BDC1EB470E45C1BDA047A648E2C88A42D7A8DE4043C8B5C7BD457F485CC802BCCAC1453E6B4859BFCFC042C5424509486F45E53DF2C3F9C87445B040F1C6EFC24A3E5438E9C8B8472E44B6C1B3B816484B45EF4038B9D7C89FC44B48A246A3431B3CF9484DC88EC667B842C7DB44534829C6DF43B6B957C865C51C4547311445D4C53B4882C83B44093684C78EC6CDC826C0BDC8DAC7B8C8473213C5F5C733473AC4373A5DC53A3CAD480049"> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weight_table = const.Declare memref<16x1x1x4xsi32> = dense<[[[[0, 0, 1006699012, 0]]], [[[16, 0, 1006699012, 0]]], [[[32, 0, 1006699012, 0]]], [[[48, 0, 1006699012, 0]]], [[[64, 0, 1006699012, 0]]], [[[80, 0, 1006699012, 0]]], [[[96, 0, 1006699012, 0]]], [[[112, 0, 1006699012, 0]]], [[[128, 0, 1006699012, 0]]], [[[144, 0, 1006699012, 0]]], [[[160, 0, 1006699012, 0]]], [[[176, 0, 1006699012, 0]]], [[[192, 0, 1006699012, 0]]], [[[208, 0, 1006699012, 0]]], [[[224, 0, 1006699012, 0]]], [[[240, 0, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>

    %alloc_input = memref.alloc() : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %input_cmx = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%alloc_input : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_wt = memref.alloc() : memref<16x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights_cmx = VPUIP.Copy inputs(%weights : memref<16x16x1x1x!qElemType, #NHWC>) outputs(%alloc_wt : memref<16x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %alloc_wt_table = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table_cmx = VPUIP.Copy inputs(%weight_table : memref<16x1x1x4xsi32>) outputs(%alloc_wt_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %out_alloc = memref.alloc() : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %out_cmx = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 4294967195 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%input_cmx : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights_cmx : memref<16x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weights_table_cmx : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%input_cmx : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out_alloc : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%out_alloc : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %out_ddr = VPUIP.Copy inputs(%out_cmx : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) -> memref<1x16x16x16xf16, #NHWC, @DDR>
    return %out_ddr : memref<1x16x16x16xf16, #NHWC, @DDR>
	// CHECK-DAG:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x512xui8>
    // CHECK-SAME:  [#const.FuseWeights<tensor<1x1x1x512xui8>, weightsTable =
    // CHECK-SAME   weights =
    // CHECK-SAME   [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]>>]
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare memref<16x16x1x1x!qElemType
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32>

    // CHECK: [[FUSED_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x512xui8, [@CMX_NN, 0]>
    // CHECK: [[FUSED_CMX:%.+]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x512xui8>) outputs([[FUSED_ALLOC]] : memref<1x1x1x512xui8, [@CMX_NN, 0]>) -> memref<1x1x1x512xui8, [@CMX_NN, 0]>
    // CHECK: [[WT_TABLE_SUBVIEW:%.+]] = VPUIP.SubView [[FUSED_CMX]] [0, 0, 0, 0] [1, 1, 1, 256] : memref<1x1x1x512xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8, {order = #NCHW, strides = [512, 512, 512, 1]}, [@CMX_NN, 0]>
    // CHECK: [[WT_TABLE_CMX:%.+]] = VPUIP.ViewOp [[WT_TABLE_SUBVIEW]] : memref<1x1x1x256xui8, {order = #NCHW, strides = [512, 512, 512, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK: [[WT_SUBVIEW:%.+]] = VPUIP.SubView [[FUSED_CMX]] [0, 0, 0, 256] [1, 1, 1, 256] : memref<1x1x1x512xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8, {order = #NCHW, strides = [512, 512, 512, 1]}, [@CMX_NN, 0]>
    // CHECK: [[WT_CMX:%.+]] = VPUIP.ViewOp [[WT_SUBVIEW]] : memref<1x1x1x256xui8, {order = #NCHW, strides = [512, 512, 512, 1]}, [@CMX_NN, 0]> to memref<16x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:    weights([[WT_CMX]] : memref<16x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:    weight_table([[WT_TABLE_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseWeightsWithMajorityOfF16Type
func.func @FuseWeightsWithMajorityOfF16Type(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>) -> memref<1x16x16x16xf16, #NHWC, @DDR> {
    %weights = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weight_table = const.Declare memref<16x1x1x4xsi32> = dense<[[[[0, 0, 1006699012, 0]]], [[[16, 0, 1006699012, 0]]], [[[32, 0, 1006699012, 0]]], [[[48, 0, 1006699012, 0]]], [[[64, 0, 1006699012, 0]]], [[[80, 0, 1006699012, 0]]], [[[96, 0, 1006699012, 0]]], [[[112, 0, 1006699012, 0]]], [[[128, 0, 1006699012, 0]]], [[[144, 0, 1006699012, 0]]], [[[160, 0, 1006699012, 0]]], [[[176, 0, 1006699012, 0]]], [[[192, 0, 1006699012, 0]]], [[[208, 0, 1006699012, 0]]], [[[224, 0, 1006699012, 0]]], [[[240, 0, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>

    %alloc_input = memref.alloc() : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %input_cmx = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%alloc_input : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_wt = memref.alloc() : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights_cmx = VPUIP.Copy inputs(%weights : memref<16x16x1x1xf16, #NHWC>) outputs(%alloc_wt : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_wt_table = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table_cmx = VPUIP.Copy inputs(%weight_table : memref<16x1x1x4xsi32>) outputs(%alloc_wt_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %out_alloc = memref.alloc() : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %out_cmx = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 4294967195 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%input_cmx : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights_cmx : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weights_table_cmx : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%input_cmx : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out_alloc : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%out_alloc : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %out_ddr = VPUIP.Copy inputs(%out_cmx : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) -> memref<1x16x16x16xf16, #NHWC, @DDR>
    return %out_ddr : memref<1x16x16x16xf16, #NHWC, @DDR>
	// CHECK-DAG:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x384xf16>
    // CHECK-SAME:  [#const.FuseWeights<tensor<1x1x1x384xf16>, weightsTable =
    // CHECK-SAME:  weights = <dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>>]
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare memref<16x16x1x1xf16
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32>

    // CHECK: [[FUSED_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x384xf16, [@CMX_NN, 0]>
    // CHECK: [[FUSED_CMX:%.+]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x384xf16>) outputs([[FUSED_ALLOC]] : memref<1x1x1x384xf16, [@CMX_NN, 0]>) -> memref<1x1x1x384xf16, [@CMX_NN, 0]>
    // CHECK: [[WT_TABLE_SUBVIEW:%.+]] = VPUIP.SubView [[FUSED_CMX]] [0, 0, 0, 0] [1, 1, 1, 128] : memref<1x1x1x384xf16, [@CMX_NN, 0]> to memref<1x1x1x128xf16, {order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]>
    // CHECK: [[WT_TABLE_CMX:%.+]] = VPUIP.ViewOp [[WT_TABLE_SUBVIEW]] : memref<1x1x1x128xf16, {order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK: [[WT_SUBVIEW:%.+]] = VPUIP.SubView [[FUSED_CMX]] [0, 0, 0, 128] [1, 1, 1, 256] : memref<1x1x1x384xf16, [@CMX_NN, 0]> to memref<1x1x1x256xf16, {order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]>
    // CHECK: [[WT_CMX:%.+]] = VPUIP.ViewOp [[WT_SUBVIEW]] : memref<1x1x1x256xf16, {order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]> to memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:    weights([[WT_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:    weight_table([[WT_TABLE_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
}
