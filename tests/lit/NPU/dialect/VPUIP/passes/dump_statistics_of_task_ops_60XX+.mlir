//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env IE_NPU_LOG_FILTER="dump-statistics-of-task-ops" vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --dump-statistics-of-task-ops -o /dev/null %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU6010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!Weights = memref<32x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!Output = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>

func.func @testSCLConvNumOfOps(%input: !Input, %weights: !Weights, %out: !Output) -> !Output {
    %dcimConv = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }>
    input(%input : !Input)
    weights(%weights : !Weights)
    parent_input(%input : !Input)
    parent_output(%out : !Output)
    outputs(%out : !Output)
    -> !Output variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 31],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}
    return %dcimConv : !Output
}

// CHECK:   Input size -
// CHECK:   DDR heap size -
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 1 ops
// CHECK:     NCETask Operations - 1 ops
// CHECK:       Dense - 1 ops
// CHECK:     SCL Task Op - 1 ops
// CHECK:       opCount - 82944 ops


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!Weights = memref<32x32x1x1xf16, {order = #NHWC}, @CMX_NN>
!Output = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>

func.func @testDCIMConvNumOfOps(%input: !Input, %weights: !Weights, %out: !Output) -> !Output {
    %dcimConv = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }>
    input(%input : !Input)
    weights(%weights : !Weights)
    parent_input(%input : !Input)
    parent_output(%out : !Output)
    outputs(%out : !Output)
    -> !Output variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 31],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}
    return %dcimConv : !Output
}

// CHECK:   Input size -
// CHECK:   DDR heap size -
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 1 ops
// CHECK:     NCETask Operations - 1 ops
// CHECK:       Dense - 1 ops
// CHECK:     DCIM Task Op - 1 ops
// CHECK:       opCount - 9216 ops

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!Weights = memref<32x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!Output = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>

!Input2 = memref<1x64x3x3xf16, {order = #NHWC}, @CMX_NN>
!Weights2 = memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>
!Output2 = memref<1x64x3x3xf16, {order = #NHWC}, @CMX_NN>


func.func @testSCLAndDCIMConvNumOfOps(%input: !Input, %weights: !Weights, %out: !Output, %input2: !Input2, %weights2: !Weights2, %out2: !Output2) -> !Output2 {
    %dcimConv = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }>
    input(%input : !Input)
    weights(%weights : !Weights)
    parent_input(%input : !Input)
    parent_output(%out : !Output)
    outputs(%out : !Output)
    -> !Output variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 31],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}
    %sclConv = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }>
    input(%input2 : !Input2)
    weights(%weights2 : !Weights2)
    parent_input(%input2 : !Input2)
    parent_output(%out2 : !Output2)
    outputs(%out2 : !Output2)
    -> !Output2 variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 63],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}
    return %sclConv : !Output2
}

// CHECK:   Input size -
// CHECK:   DDR heap size -
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 2 ops
// CHECK:     NCETask Operations - 2 ops
// CHECK:       Dense - 2 ops
// CHECK:     DCIM Task Op - 1 ops
// CHECK:       opCount - 82944 ops
// CHECK:     SCL Task Op - 1 ops
// CHECK:       opCount - 36864 ops



// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvInput = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!ConvWeights = memref<32x32x3x3xf16, {order = #NHWC}, @CMX_NN>
!ConvOutput = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>

!DwInput = memref<1x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!DwWeights = memref<16x32x1x1xf16, {order = #NHWC}, @CMX_NN>
!DwOutput = memref<1x32x3x3xf16, {order = #NHWC}, @CMX_NN>

!OtherInput = memref<1x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!OtherWeights = memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>
!OtherOutput = memref<1x16x3x3xf16, {order = #NHWC}, @CMX_NN>

func.func @testSCLAndDCIMAllOps(%convInput: !ConvInput, %convWeights : !ConvWeights, %convOutput : !ConvOutput, %dwInput : !DwInput, %dwWeights : !DwWeights, %dwOutput : !DwOutput, %otherInput : !OtherInput, %otherWeights : !OtherWeights,%otherOutput : !OtherOutput) -> !OtherOutput {

    %dcimConv = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }>
    input(%convInput : !ConvInput)
    weights(%convWeights : !ConvWeights)
    parent_input(%convInput : !ConvInput)
    parent_output(%convOutput : !ConvOutput)
    outputs(%convOutput : !ConvOutput)
    -> !ConvOutput variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 31],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}


    %dcimConv2 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }>
    input(%convInput : !ConvInput)
    weights(%convWeights : !ConvWeights)
    parent_input(%convInput : !ConvInput)
    parent_output(%convOutput : !ConvOutput)
    outputs(%convOutput : !ConvOutput)
    -> !ConvOutput variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 31],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}


    %dwConv = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<DWCONV>
    }>
    input(%dwInput : !DwInput)
    weights(%dwWeights : !DwWeights)
    parent_input(%dwInput : !DwInput)
    parent_output(%dwOutput : !DwOutput)
    outputs(%dwOutput : !DwOutput)
    -> !DwOutput variants : {
        DPUTask {cluster_id = 0 : i64,
         outStart = [0, 0, 0], outEnd = [2, 2, 31],
         mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}


    %eltwiseOp = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<ELTWISE>
    }>
    input(%otherInput : !OtherInput)
    weights(%otherInput : !OtherInput)
    parent_input(%otherInput : !OtherInput)
    parent_output(%otherOutput : !OtherOutput)
    outputs(%otherOutput : !OtherOutput)
    -> !OtherOutput variants : {
        DPUTask { cluster_id = 0 : i64,
        outStart = [0, 0, 0], outEnd = [2, 2, 15],
        mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    }
    PPE:{}


    %poolingOp = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<MAXPOOL>
    }>
    input(%otherInput : !OtherInput)
    parent_input(%otherInput : !OtherInput)
    parent_output(%otherOutput : !OtherOutput)
    outputs(%otherOutput : !OtherOutput)
    -> !OtherOutput variants : {
        DPUTask {cluster_id = 0 : i64,
        mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outStart = [0, 0, 0] ,outEnd = [2, 2, 15],
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    }
    PPE:{}

    return %otherOutput : !OtherOutput
}


// CHECK:   Input size -
// CHECK:   DDR heap size -
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 5 ops
// CHECK:     NCETask Operations - 5 ops
// CHECK:       Dense - 5 ops
// CHECK:     DCIM Task Op - 2 ops
// CHECK:       opCount - 165888 ops
// CHECK:     SCL Task Op - 3 ops
// CHECK:       opCount - 576 ops
