//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelConstCopies(
        %output1: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output2: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
         -> (memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = const.Declare memref<1x16x112x112xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<1x16x112x112xf16>, [#const.Reorder<#NHWC>]
    %1 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            outputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %6 = VPUIP.Copy
            inputs(%5 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %8 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %10 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%8 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%8 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%9 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%9 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %11 = VPUIP.Copy
            inputs(%10 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

}

// CHECK-LABEL: func.func @OptimizeParallelConstCopies

// CHECK:       [[VAR0:%.+]] =  const.Declare memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
// CHECK:       [[VAR1:%.+]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
// CHECK:       [[VAR2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR3:%.+]] =  VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// CHECK:       [[VAR4:%.+]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
// CHECK:       [[VAR5:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR4]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR6:%.+]] =  VPUIP.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelActCopies(
        %input: memref<1x32x1x12544xf16, {order = #NHWC}, @DDR>,
        %output1: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output2: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
        -> (memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>){
    %weights_table = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>

    %reshape = VPUIP.GenericReshape inputs(%input: memref<1x32x1x12544xf16, {order = #NHWC}, @DDR>) -> memref<1x32x112x112xf16, {order = #NHWC}, @DDR>
    %subview1 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc_in1 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %copy_in1 = VPUIP.Copy
            inputs(%subview1 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
            outputs(%alloc_in1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %alloc_out1 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %nce1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%copy_in1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%copy_in1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%alloc_out1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%alloc_out1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %copy_out1 = VPUIP.Copy
            inputs(%nce1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %subview2 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc_in2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %copy_in2 = VPUIP.Copy
            inputs(%subview2 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
            outputs(%alloc_in2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %alloc_out2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %nce2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%copy_in2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%copy_in2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%alloc_out2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%alloc_out2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %copy_out2 = VPUIP.Copy
            inputs(%nce2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output2 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    return %copy_out1, %copy_out1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

}

// CHECK-LABEL: func.func @OptimizeParallelActCopies

// CHECK:       [[RESHAPE:%.+]] =  VPUIP.GenericReshape

// CHECK:       [[SUBVIEW:%.+]] =  VPUIP.SubView [[RESHAPE]] [0, 0, 0, 0] [1, 16, 112, 112]
// CHECK:       [[ALLOC:%.+]] = memref.alloc()
// CHECK:       [[COPY:%.+]] =  VPUIP.Copy inputs([[SUBVIEW]]
// CHECK-SAME:                             outputs([[ALLOC]]

// CHECK:       [[NCE1:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[COPY]]
// CHECK:       [[COPY_OUT1:%.+]] =  VPUIP.Copy inputs([[NCE1]]

// CHECK:       [[NCE2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[COPY]]
// CHECK:       [[COPY_OUT1:%.+]] =  VPUIP.Copy inputs([[NCE2]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeParallelSubViewInputCopies
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x16x112x113xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
func.func @OptimizeParallelSubViewInputCopies(
        %input: memref<1x16x112x113xf16, {order = #NHWC}, @DDR>,
        %output1: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output2: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
         -> (memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>

    %2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %3 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %5 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%5 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%5 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.Copy
            inputs(%6 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %8 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %9 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %10 = VPUIP.Copy
            inputs(%9 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
            outputs(%8 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %11 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %12 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%10 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%10 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%11 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%11 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %13 = VPUIP.Copy
            inputs(%12 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    return %7, %13 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
}

// CHECK:       [[ALLOC_CMX:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

// CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 16, 112, 112] :
// CHECK-SAME:        memref<1x16x112x113xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>

// CHECK:       [[DDR2CMX:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] :
// CHECK-SAME:        memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK-SAME:        outputs([[ALLOC_CMX]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

// CHECK:       [[ALLOC_NCE_CMX_1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[NCE_1:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input([[DDR2CMX]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:        outputs([[ALLOC_NCE_CMX_1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

// CHECK:       [[CMX2DDR_1:%.+]] = VPUIP.Copy
// CHECK-SAME:        inputs([[NCE_1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:        outputs([[ARG_1]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

// CHECK:       [[ALLOC_NCE_CMX_2:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[NCE_2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input([[DDR2CMX]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:        outputs([[ALLOC_NCE_CMX_2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

// CHECK:       [[CMX2DDR_2:%.+]] = VPUIP.Copy
// CHECK-SAME:        inputs([[NCE_2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:        outputs([[ARG_1]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

// CHECK:       return [[CMX2DDR_1]], [[CMX2DDR_2]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x32x28x28xf16, {order = #NHWC}, @DDR>
!Output_CMX = memref<1x32x28x28xf16, {order = #NHWC}, @CMX_NN>
!Output = memref<1x32x28x28xf16, {order = #NHWC}>
!Weights_CMX = memref<128x32x1x1xf16, {order = #NHWC}, @CMX_NN>
!Output_CONV = memref<1x128x28x28xf16, {order = #NHWC}, @CMX_NN>

!CopyOutput_Distributed = !VPUIP.DistributedBuffer<
  1x32x28x28xf16, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!ConvOutput_Distributed = !VPUIP.DistributedBuffer<
  1x128x28x28xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

func.func @OptimizeParallelMulticlusterCopies() -> (!ConvOutput_Distributed, !ConvOutput_Distributed) {
    %0 = memref.alloc() : !Output_DDR
    %1 = VPURT.AllocDistributed -> !CopyOutput_Distributed
    %2 = VPURT.AllocDistributed -> !CopyOutput_Distributed
    %3 = memref.alloc() : !Output_CMX
    %4 = VPUIP.Copy
        inputs(%3 : !Output_CMX)
        outputs(%0 : !Output_DDR) -> !Output_DDR
    %5 = VPUIP.Copy
        inputs(%4 : !Output_DDR)
        outputs(%1 : !CopyOutput_Distributed) -> !CopyOutput_Distributed
    %6 = VPURT.AllocDistributed -> !ConvOutput_Distributed
    %7 = memref.alloc() : !Weights_CMX
    %9 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
           }>
        input(%5 : !CopyOutput_Distributed)
        weights(%7 : !Weights_CMX)
        parent_input(%5 : !CopyOutput_Distributed)
        parent_output(%6 : !ConvOutput_Distributed)
        outputs(%6 : !ConvOutput_Distributed)
            -> !ConvOutput_Distributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }
    %10 = VPUIP.Copy
        inputs(%4 : !Output_DDR)
        outputs(%2 : !CopyOutput_Distributed) -> !CopyOutput_Distributed
    %12 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
           }>
        input(%10 : !CopyOutput_Distributed)
        weights(%7 : !Weights_CMX)
        parent_input(%10 : !CopyOutput_Distributed)
        parent_output(%6 : !ConvOutput_Distributed)
        outputs(%6 : !ConvOutput_Distributed)
            -> !ConvOutput_Distributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }
    return %9, %12: !ConvOutput_Distributed, !ConvOutput_Distributed
}

// CHECK-LABEL: @OptimizeParallelMulticlusterCopies

//CHECK: [[DISTR_BUFFER0:%.+]] = VPURT.AllocDistributed
//CHECK: [[COMMON_ROOT:%.+]] = VPUIP.Copy

//CHECK: [[BRANCH1_COPY:%.+]] = VPUIP.Copy
//CHECK-SAME:  inputs([[COMMON_ROOT]] : memref<1x32x28x28xf16, {order = #NHWC}, @DDR>)
//CHECK-SAME:  outputs([[DISTR_BUFFER0]] : !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

//CHECK: [[BRANCH1_CONSUMER:%.+]] = VPUIP.NCEClusterTask <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1]
//CHECK-SAME: input([[BRANCH1_COPY]] : !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

//CHECK-NOT: VPUIP.Copy

//CHECK: [[BRANCH2_CONSUMER:%.+]] = VPUIP.NCEClusterTask
//CHECK-SAME: input([[BRANCH1_COPY]]

//CHECK:  return [[BRANCH1_CONSUMER]], [[BRANCH2_CONSUMER]]

// -----

// CHECK-LABEL: @OptimizeParallelMultiShaveCopies

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
    func.func nested @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OptimizeParallelMultiShaveCopies(%arg0: memref<387072x3xf16, @DDR>,
                                            %arg1: memref<1x96768xsi32, @DDR>,
                                            %arg2: memref<1x96768xsi32, @DDR>,
                                            %arg3: memref<1x96768x1xf16, @DDR>,
                                            %arg4: memref<1x96768x1xf16, @DDR>)
                                            -> (memref<1x96768x1xf16, @DDR>, memref<1x96768x1xf16, @DDR>) {
    %0 = VPUIP.SubView %arg0 [0, 0] [387072, 1] : memref<387072x3xf16, @DDR> to memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>
    %alloc = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%0 : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>) outputs(%alloc : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x96768xsi32, @DDR>) outputs(%alloc_0 : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SubView %2 [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %4 = VPUIP.SubView %alloc_1 [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    %5 = VPUIP.SubView %2 [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %6 = VPUIP.SubView %alloc_1 [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    inputs( %1 as %arg5: memref<387072x1xf16, [@CMX_NN, 0]>,
            %3 as %arg6: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
            %1 as %arg7: memref<387072x1xf16, [@CMX_NN, 0]>,
            %5 as %arg8: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    outputs(%4 as %arg9: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
            %6 as %arg10: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg5, %arg6, %arg9) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg7, %arg8, %arg10) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    }

    %7 = VPUIP.ConcatView inputs(%results#0, %results#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)
        outputs(%alloc_1 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%7 : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs(%arg3 : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>

    %alloc_2 = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    %10 = VPUIP.Copy inputs(%0 : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>) outputs(%alloc_2 : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    %11 = VPUIP.Copy inputs(%arg2 : memref<1x96768xsi32, @DDR>) outputs(%alloc_3 : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    %12 = VPUIP.SubView %11 [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %13 = VPUIP.SubView %alloc_4 [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    %14 = VPUIP.SubView %11 [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %15 = VPUIP.SubView %alloc_4 [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    %results_5:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    inputs( %10 as %arg5: memref<387072x1xf16, [@CMX_NN, 0]>,
            %12 as %arg6: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
            %10 as %arg7: memref<387072x1xf16, [@CMX_NN, 0]>,
            %14 as %arg8: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    outputs(%13 as %arg9: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    %15 as %arg10: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg5, %arg6, %arg9) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg7, %arg8, %arg10) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    }

    %16 = VPUIP.ConcatView inputs(%results_5#0, %results_5#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)
        outputs(%alloc_4 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    %17 = VPUIP.Copy inputs(%16 : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs(%arg4 : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>
    return %8, %17 : memref<1x96768x1xf16, @DDR>, memref<1x96768x1xf16, @DDR>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0] [387072, 1] : memref<387072x3xf16, @DDR> to memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>
    // CHECK:       [[ALLOC_CMX0:%.+]] = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW0]] : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>) outputs([[ALLOC_CMX0]] : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_CMX1:%.+]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32, @DDR>) outputs([[ALLOC_CMX1]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC_CMX2:%.+]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY1]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC_CMX2]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[COPY1]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW4:%.+]] = VPUIP.SubView [[ALLOC_CMX2]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[GATHER0:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:      inputs([[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW1]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW3]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[SUBVIEW2]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW4]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    // CHECK-SAME:      (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[GATHER0]]#0, [[GATHER0]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC_CMX2]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[CONCAT0]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs({{[^:]+}} : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>

    // CHECK:       [[ALLOC_CMX3:%.+]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32, @DDR>) outputs([[ALLOC_CMX3]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC_CMX4:%.+]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW5:%.+]] = VPUIP.SubView [[COPY3]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC_CMX4]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW7:%.+]] = VPUIP.SubView [[COPY3]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW8:%.+]] = VPUIP.SubView [[ALLOC_CMX4]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[GATHER1:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:      inputs([[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW5]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW7]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[SUBVIEW6]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW8]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    // CHECK-SAME:      (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:        }

    // CHECK:       [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[GATHER1]]#0, [[GATHER1]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC_CMX4]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY4:%.+]] = VPUIP.Copy inputs([[CONCAT1]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs({{[^:]+}} : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>

    // CHECK:       return [[COPY2]], [[COPY4]] : memref<1x96768x1xf16, @DDR>, memref<1x96768x1xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

module @VPU.SW {
  func.func nested @builtin_convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
}

// CHECK-LABEL: @OptimizeParallelSubViewWithDistributedCopies
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
func.func @OptimizeParallelSubViewWithDistributedCopies(
        %input: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>,
        %weights: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_convert
        inputs(%input as %input_0: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>)
        outputs(%0 as %output_0: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>) on tile 0 -> memref<1x144x128x128xf16, {order = #NHWC}, @DDR> {
        VPUIP.SW.Kernel.run (%input_0, %output_0) : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>, memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    }

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%3 : !OutputDistributed) -> !OutputDistributed
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
      <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV> }>
        input(%4 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%4 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.Copy
        inputs(%7 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%8 : !OutputDistributed) -> !OutputDistributed
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV> }>
        input(%9 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%9 : !OutputDistributed)
        parent_output(%10 : !OutputDistributed)
        outputs(%10 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

    return %6, %11 : !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.+]] = memref.alloc() : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[CONVERT:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_convert
    // CHECK-SAME:          inputs([[ARG_0]]
    // CHECK-SAME:          outputs([[IN_BUFFER]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, {order = #NHWC}, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]]
    // CHECK-SAME:      weights([[ARG_1]]
    // CHECK-SAME:      outputs([[BUFFER_2]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUFFER_3:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]]
    // CHECK-SAME:      weights([[ARG_1]]
    // CHECK-SAME:      outputs([[BUFFER_3]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>


    // CHECK:       return [[NCE_1]], [[NCE_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:                                     !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x2x512x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!SubViewDistributed = !VPUIP.DistributedBuffer<
    1x1x512x1xf16, {
    order = #NHWC,
    strides = [1024, 1, 2, 2]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @NotOptimizeParallelDistributedCopiesWithSubviewHasDiffOffset
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x1x512x1xf16, @DDR>)
func.func @NotOptimizeParallelDistributedCopiesWithSubviewHasDiffOffset(
        %arg0: memref<1x1x512x1xf16, @DDR>) -> !OutputDistributed {
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x1x512x1xf16, @DDR>) -> memref<1x1x512x1xf16, {order = #NHWC}, @DDR>
    %2 = VPURT.AllocDistributed -> !OutputDistributed

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 1, 512, 1] : !OutputDistributed to !SubViewDistributed
    %4 = VPUIP.Copy
        inputs(%1 : memref<1x1x512x1xf16, {order = #NHWC}, @DDR>)
        outputs(%3 : !SubViewDistributed) -> !SubViewDistributed

    %5 = VPUIP.SubView %2 [0, 1, 0, 0] [1, 1, 512, 1] : !OutputDistributed to !SubViewDistributed
    %6 = VPUIP.Copy
        inputs(%1 : memref<1x1x512x1xf16, {order = #NHWC}, @DDR>)
        outputs(%5 : !SubViewDistributed) -> !SubViewDistributed

    %7 = VPUIP.ConcatView
            inputs(%4, %6 : !SubViewDistributed, !SubViewDistributed)
            outputs(%2 : !OutputDistributed) -> !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[ARG_0]] : memref<1x1x512x1xf16, @DDR>) -> memref<1x1x512x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[OUT_BUFFER:%.+]] = VPURT.AllocDistributed

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 1, 512, 1]
    // CHECK:       [[CLUSTER_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[PERMUTECAST]]
    // CHECK-SAME:      outputs([[SUBVIEW_0]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 1, 0, 0] [1, 1, 512, 1]
    // CHECK:       [[CLUSTER_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[PERMUTECAST]]
    // CHECK-SAME:      outputs([[SUBVIEW_1]]

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CLUSTER_0]], [[CLUSTER_1]]
    // CHECK-SAME:      outputs([[OUT_BUFFER]]

    // CHECK:       return [[CONCATVIEW]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

module @VPU.SW {
  func.func nested @builtin_convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
}

// CHECK-LABEL: @OptimizeParallelSubViewWithParallelDistributedCopies
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
func.func @OptimizeParallelSubViewWithParallelDistributedCopies(
        %input: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>,
        %weights: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_convert
        inputs(%input as %input_0: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>)
        outputs(%0 as %output_0: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>) on tile 0 -> memref<1x144x128x128xf16, {order = #NHWC}, @DDR> {
        VPUIP.SW.Kernel.run (%input_0, %output_0) : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>, memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    }

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%3 : !OutputDistributed) -> !OutputDistributed

    %100 = VPURT.AllocDistributed -> !OutputDistributed
    %101 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%100 : !OutputDistributed) -> !OutputDistributed


    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV> }>
        input(%4 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%4 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

	    %102 = VPURT.AllocDistributed -> !OutputDistributed
    %103 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV> }>
        input(%101 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%101 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }


    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.Copy
        inputs(%7 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%8 : !OutputDistributed) -> !OutputDistributed
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV> }>
        input(%9 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%9 : !OutputDistributed)
        parent_output(%10 : !OutputDistributed)
        outputs(%10 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

    return %6, %11, %103 : !OutputDistributed, !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.+]] = memref.alloc() : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[CONVERT:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_convert
    // CHECK-SAME:          inputs([[ARG_0]]
    // CHECK-SAME:          outputs([[IN_BUFFER]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, {order = #NHWC}, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights([[ARG_1]] : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_4]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights([[ARG_1]] : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_4]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_6:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights([[ARG_1]] : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_6]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>


    // CHECK:       return [[NCE_1]], [[NCE_3]], [[NCE_2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CopyOutputDistributed = !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

module @VPU.SW {
  func.func nested @builtin_convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
}

// CHECK-LABEL: @OptimizeParallelDistributedCopiesWithInPlaceNCEEltwise
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x128x104x104xf32, {order = #NHWC}, @DDR>)
func.func @OptimizeParallelDistributedCopiesWithInPlaceNCEEltwise(%arg0:
memref<1x128x104x104xf32, {order = #NHWC}, @DDR>) -> (!CopyOutputDistributed, !OutputDistributed) {
    %0 = memref.alloc() : memref<1x128x104x104xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_convert
        inputs(%arg0 as %input_0: memref<1x128x104x104xf32, {order = #NHWC}, @DDR>)
        outputs(%0 as %output_0: memref<1x128x104x104xf16, {order = #NHWC}, @DDR>) on tile 0 -> memref<1x128x104x104xf16, {order = #NHWC}, @DDR> {
        VPUIP.SW.Kernel.run (%input_0, %output_0) : memref<1x128x104x104xf32, {order = #NHWC}, @DDR>, memref<1x128x104x104xf16, {order = #NHWC}, @DDR>
    }
    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 128, 52, 104] :
                memref<1x128x104x104xf16, {order = #NHWC}, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
        outputs(%3 : !OutputDistributed) -> !OutputDistributed
    %5 = VPURT.AllocDistributed -> !CopyOutputDistributed
    %6 = VPUIP.Copy
        inputs(%4 : !OutputDistributed)
        outputs(%5 : !CopyOutputDistributed) -> !CopyOutputDistributed
    %7 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 128, 52, 104] :
        memref<1x128x104x104xf16, {order = #NHWC}, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %8 = VPUIP.SubView %1 [0, 0, 52, 0] [1, 128, 52, 104] :
        memref<1x128x104x104xf16, {order = #NHWC}, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %9 = VPURT.AllocDistributed -> !OutputDistributed
    %10 = VPUIP.Copy
        inputs(%7 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
        outputs(%9 : !OutputDistributed) -> !OutputDistributed
    %11 = VPURT.AllocDistributed -> !OutputDistributed
    %12 = VPUIP.Copy
        inputs(%8 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
        outputs(%11 : !OutputDistributed) -> !OutputDistributed
    %13 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4294967400 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                is_inplace = true,
                task_type = #VPUIP.nce_task_type<ELTWISE>
             }>
        input(%10 : !OutputDistributed)
        weights(%12 : !OutputDistributed)
        parent_input(%10 : !OutputDistributed)
        parent_output(%9 : !OutputDistributed)
        outputs(%9 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 25, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> }
    } PPE : {
    }
    return %6, %13: !CopyOutputDistributed, !OutputDistributed
    // CHECK:       [[MEM_REF:%.+]] = memref.alloc() : memref<1x128x104x104xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[CONVERT:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_convert
    // CHECK-SAME:          inputs([[ARG_0]]
    // CHECK-SAME:          outputs([[MEM_REF]]
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 0, 0] [1, 128, 52, 104] :
    // CHECK-SAME:        memref<1x128x104x104xf16, {order = #NHWC}, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    // CHECK:       [[DISTBUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE:%.+]] = VPUIP.Copy
    // CHECK-SAME:        inputs([[SUBVIEW]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
    // CHECK-SAME:        outputs([[DISTBUFF]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[DISTBUFF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:        inputs([[NCE]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        outputs([[DISTBUFF_1]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 52, 0] [1, 128, 52, 104] :
    // CHECK-SAME:        memref<1x128x104x104xf16, {order = #NHWC}, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    // CHECK:       [[DISTBUFF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:        inputs([[SUBVIEW_0]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
    // CHECK-SAME:        outputs([[DISTBUFF_2]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_3:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4294967400 : i64} <{is_inplace = true
    // CHECK-SAME:        input([[NCE]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        weights([[NCE_2]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        outputs([[DISTBUFF]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       return [[NCE_1]], [[NCE_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!bufDdr = memref<1x88x52x52xf16, {order = #NCHW, strides = [475904, 2704, 52, 1]}, @DDR>
!typeCst = memref<1x88x52x12xf16>
!bufDistributed = !VPUIP.DistributedBuffer<
  1x88x52x64xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64
}>

!bufDistributed0 = !VPUIP.DistributedBuffer<
  1x88x52x52xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64
}>
!bufCompact0 = memref<1x88x52x52xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN>

!bufDistributed1 = !VPUIP.DistributedBuffer<
  1x88x52x12xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64
}>
!bufCompact1 = memref<1x88x52x12xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN>

func.func @DoNotOptimizeCopiesIfUserIsConcat(%arg0: !bufDdr, %arg1: !bufDdr) -> (!bufDistributed, !bufDistributed) {

  %cst = const.Declare !typeCst = dense<0.000000e+00> : tensor<54912xf16>, [#const.Reshape<[1, 88, 52, 12]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>]

  %buf0 = VPURT.AllocDistributed -> !bufDistributed
  %subview0_0 = VPUIP.SubView %buf0 [0, 0, 0, 0] [1, 88, 52, 52] : !bufDistributed to !bufDistributed0
  %copy0_0 = VPUIP.Copy
      inputs(%arg0 : !bufDdr)
      outputs(%subview0_0 : !bufDistributed0) -> !bufDistributed0
  %subview0_1 = VPUIP.SubView %buf0 [0, 0, 0, 52] [1, 88, 52, 12] : !bufDistributed to !bufDistributed1
  %copy0_1 = VPUIP.Copy inputs(%cst : !typeCst) outputs(%subview0_1 : !bufDistributed1) -> !bufDistributed1
  %concat0 = VPUIP.ConcatView inputs(%copy0_0, %copy0_1 : !bufDistributed0, !bufDistributed1) outputs(%buf0 : !bufDistributed) -> !bufDistributed

  %buf1 = VPURT.AllocDistributed -> !bufDistributed
  %subview1_0 = VPUIP.SubView %buf1 [0, 0, 0, 0] [1, 88, 52, 52] : !bufDistributed to !bufDistributed0
  %copy1_0 = VPUIP.Copy
      inputs(%arg1 : !bufDdr)
      outputs(%subview1_0 : !bufDistributed0) -> !bufDistributed0
  %subview1_1 = VPUIP.SubView %buf1 [0, 0, 0, 52] [1, 88, 52, 12] : !bufDistributed to !bufDistributed1
  %copy1_1 = VPUIP.Copy inputs(%cst : !typeCst) outputs(%subview1_1 : !bufDistributed1) -> !bufDistributed1
  %concat1 = VPUIP.ConcatView inputs(%copy1_0, %copy1_1 : !bufDistributed0, !bufDistributed1) outputs(%buf1 : !bufDistributed) -> !bufDistributed

  return %concat0, %concat1: !bufDistributed, !bufDistributed

  // CHECK-LABEL: @DoNotOptimizeCopiesIfUserIsConcat

  // CHECK: [[CST:%.+]] = const.Declare
  // CHECK: [[BUF0:%.+]] = VPURT.AllocDistributed
  // CHECK: [[SUBVIEW0_0:%.+]] = VPUIP.SubView [[BUF0]]
  // CHECK: [[COPY0_0:%.+]] = VPUIP.Copy
  // CHECK-SAME: outputs([[SUBVIEW0_0]]

  // CHECK: [[SUBVIEW0_1:%.+]] = VPUIP.SubView [[BUF0]]
  // CHECK: [[COPY0_1:%.+]] = VPUIP.Copy inputs([[CST]]
  // CHECK-SAME: outputs([[SUBVIEW0_1]]

  // CHECK: [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[COPY0_0]], [[COPY0_1]]

  // CHECK: [[BUF1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[SUBVIEW1_0:%.+]] = VPUIP.SubView [[BUF1]]
  // CHECK: [[COPY1_0:%.+]] = VPUIP.Copy
  // CHECK-SAME: outputs([[SUBVIEW1_0]]

  // CHECK: [[SUBVIEW1_1:%.+]] = VPUIP.SubView [[BUF1]]
  // CHECK: [[COPY1_1:%.+]] = VPUIP.Copy inputs([[CST]]
  // CHECK-SAME: outputs([[SUBVIEW1_1]]

  // CHECK: [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[COPY1_0]], [[COPY1_1]]

  // CHECK:  return [[CONCAT0]], [[CONCAT1]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistrType = !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
!ConstDistrType = !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

module @VPU.SW {
  func.func nested @builtin_PRelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "prelu_fp16.cpp", VPU.kernel_entry = "prelu_fp16"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OptimizeConstCopyToSoftKernel() -> (!DistrType, !DistrType)  {
    %cst = const.Declare memref<1x18x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<18xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 18, 1, 1]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %0 = VPURT.AllocDistributed -> !DistrType
    %1 = VPURT.AllocDistributed -> !ConstDistrType
    %2 = VPURT.AllocDistributed -> !DistrType

    %3 = VPUIP.Copy
        inputs(%cst : memref<1x18x1x1xf16, {order = #NHWC}>)
        outputs(%1 : !ConstDistrType) -> !ConstDistrType

    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
                inputs(%0 as %arg3: !DistrType,
                        %3 as %arg4: !ConstDistrType)
                outputs(%2 as %arg5: !DistrType) on tile 0 -> !DistrType{
    VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !DistrType, !ConstDistrType, !DistrType
    }

    %5 = VPURT.AllocDistributed -> !DistrType
    %6 = VPURT.AllocDistributed -> !ConstDistrType
    %7 = VPURT.AllocDistributed -> !DistrType

    // the copy will be elimated because copy from the same const source
    %8 = VPUIP.Copy
        inputs(%cst : memref<1x18x1x1xf16, {order = #NHWC}>)
        outputs(%6 : !ConstDistrType) -> !ConstDistrType

    %9 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
                inputs(%5 as %arg3: !DistrType,
                        %8 as %arg4: !ConstDistrType)
                outputs(%7 as %arg5: !DistrType) on tile 0 -> !DistrType {
    VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !DistrType, !ConstDistrType, !DistrType
    }

    return %4, %9 : !DistrType, !DistrType


    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x18x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<18xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 18, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[DISTR_BUFFER0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CST]]
    // CHECK-SAME:          outputs([[DISTR_BUFFER1]]
    // CHECK:           [[SWKERNEL0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
    //CHECK-SAME:           inputs([[DISTR_BUFFER0]] as {{%[^:]+}}
    //CHECK-SAME:           [[COPY0]] as {{%[^:]+}}

    // CHECK:           [[DISTR_BUFFER3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SWKERNEL1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
    // CHECK-SAME:          inputs([[DISTR_BUFFER3]] as {{%[^:]+}}
    // CHECK-SAME:          [[COPY0]] as {{%[^:]+}}

    // CHECK:           return  [[SWKERNEL0]], [[SWKERNEL1]]
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_Multiply(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!FirstInputDistributed = !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!SecondInputDistributed = !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @OptimizeMultiplyParallelWithUserBetweenTwoSubview() -> (!FirstInputDistributed, !FirstInputDistributed, !SecondInputDistributed, memref<1x1x54x4xf16, @CMX_NN>, memref<1x1x54x4xf16, @CMX_NN>) {
    %input = memref.alloc() : memref<1x1x106x4xf16, @DDR>

    %subview1 = VPUIP.SubView %input [0, 0, 0, 0] [1, 1, 54, 4] : memref<1x1x106x4xf16, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !FirstInputDistributed
    %copy1 = VPUIP.Copy
        inputs(%subview1 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
        outputs(%alloc1 : !FirstInputDistributed) -> !FirstInputDistributed

    %subview1_alloc1 = memref.alloc() : memref<1x1x54x4xf16, @CMX_NN>
    %subview1_copy1 = VPUIP.Copy inputs(%subview1 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>) outputs(%subview1_alloc1 : memref<1x1x54x4xf16, @CMX_NN>) -> memref<1x1x54x4xf16, @CMX_NN>

    // This is sibling subView of %subview1 and will be removed.
    %subview2 = VPUIP.SubView %input [0, 0, 0, 0] [1, 1, 54, 4] : memref<1x1x106x4xf16, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    %alloc2 = VPURT.AllocDistributed -> !FirstInputDistributed
    // This is sibling copy of %copy1 and will be removed.
    %copy2 = VPUIP.Copy
        inputs(%subview2 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
        outputs(%alloc2 : !FirstInputDistributed) -> !FirstInputDistributed

    %alloc_out1 = VPURT.AllocDistributed -> !FirstInputDistributed
    %alloc_out2 = VPURT.AllocDistributed -> !SecondInputDistributed
    %extra_input = memref.alloc() : memref<1x1x52x4xf16, @CMX_NN>
    %sw:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
        inputs(%subview1_copy1 as %arg6: memref<1x1x54x4xf16, @CMX_NN>, %copy2 as %arg7: memref<1x1x54x4xf16, @CMX_NN>, %extra_input as %arg8: memref<1x1x52x4xf16, @CMX_NN>, %extra_input as %arg9: memref<1x1x52x4xf16, @CMX_NN>)
        outputs(%alloc_out1 as %arg10: !FirstInputDistributed, %alloc_out2 as %arg11: !SecondInputDistributed) on tile 0 -> (!FirstInputDistributed, !SecondInputDistributed){
    VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x1x54x4xf16, @CMX_NN>, memref<1x1x54x4xf16, @CMX_NN>, !FirstInputDistributed
    VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x1x52x4xf16, @CMX_NN>, memref<1x1x52x4xf16, @CMX_NN>, !SecondInputDistributed
    }

    %subview2_alloc1 = memref.alloc() : memref<1x1x54x4xf16, @CMX_NN>
    // This is sibling copy of %subview1_copy1 and will be removed.
    %subview2_copy1 = VPUIP.Copy inputs(%subview2 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>) outputs(%subview2_alloc1 : memref<1x1x54x4xf16, @CMX_NN>) -> memref<1x1x54x4xf16, @CMX_NN>

    %subview2_alloc2 = memref.alloc() : memref<1x1x54x4xf16, @CMX_NN>
    %subview2_subview2 = VPUIP.SubView %subview2 [0, 0, 0, 0] [1, 1, 54, 4] : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    // This is sibling copy of %subview1_copy1 and will be removed.
    %subview2_copy2 = VPUIP.Copy inputs(%subview2_subview2 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>) outputs(%subview2_alloc2 : memref<1x1x54x4xf16, @CMX_NN>) -> memref<1x1x54x4xf16, @CMX_NN>

    return %copy1, %sw#0, %sw#1, %subview2_copy1, %subview2_copy2 : !FirstInputDistributed, !FirstInputDistributed, !SecondInputDistributed, memref<1x1x54x4xf16, @CMX_NN>, memref<1x1x54x4xf16, @CMX_NN>

    // CHECK:       [[INPUT:%.+]] = memref.alloc() : memref<1x1x106x4xf16, @DDR>

    // CHECK:       [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 1, 54, 4] : memref<1x1x106x4xf16, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]]
    // CHECK-SAME:      outputs([[ALLOC1]]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[OUTPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[EXTRA_INPUT:%.+]] = memref.alloc() : memref<1x1x52x4xf16, @CMX_NN>

    // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x54x4xf16, @CMX_NN>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[ALLOC2]] : memref<1x1x54x4xf16, @CMX_NN>) -> memref<1x1x54x4xf16, @CMX_NN>

    // CHECK:       [[SW:%.+]]:2 = VPUIP.SW.Kernel
    // CHECK-SAME:      inputs([[COPY2:%.+]] as {{[^:]+}}: memref<1x1x54x4xf16, @CMX_NN>, [[COPY1]] as {{[^:]+}}: memref<1x1x54x4xf16, @CMX_NN>,
    // CHECK-SAME:             [[EXTRA_INPUT]] as {{[^:]+}}: memref<1x1x52x4xf16, @CMX_NN>, [[EXTRA_INPUT]] as {{[^:]+}}: memref<1x1x52x4xf16, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUTPUT1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          [[OUTPUT2]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       return [[COPY1]], [[SW]]#0, [[SW]]#1, [[COPY2]], [[COPY2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @OptimizeConstCopy(%arg0: memref<1x128x1x1xf16, @DDR>) -> memref<1x512x2x1xf16, @DDR> {
    %cst_0 = const.Declare memref<512x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x128xf32>, [#const.Reshape<[512, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<1x128x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128xf32>, [#const.Reshape<[1, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %alloc_0 = memref.alloc() : memref<1x128x1x1xf16, @CMX_NN>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x128x1x1xf16, @DDR>) outputs(%alloc_0 : memref<1x128x1x1xf16, @CMX_NN>) -> memref<1x128x1x1xf16, @CMX_NN>

    // %cst_1 is nce task input, will not be processed
    %alloc_1 = memref.alloc() : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>
    %1 = VPUIP.Copy inputs(%cst_1 : memref<1x128x1x1xf16, {order = #NHWC}>) outputs(%alloc_1 : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>

    // %cst_0 is nce task weight, will be processed
    %alloc_2 = memref.alloc() : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy inputs(%cst_0 : memref<512x128x1x1xf16, {order = #NHWC}>) outputs(%alloc_2 : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>

    %alloc_4 = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1710 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>) weights(%2 : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>) parent_input(%1 : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>) parent_output(%alloc_4 : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) outputs(%alloc_4 : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN> variants : {
      DPUTask {inEnd = [0, 0, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %5 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%4 : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x512x1x1xf16, @CMX_NN>

    // %6 is same weight copy as %2, will be removed.
    %alloc_6 = memref.alloc() : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.Copy inputs(%cst_0 : memref<512x128x1x1xf16, {order = #NHWC}>) outputs(%alloc_6 : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>

    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%0 : memref<1x128x1x1xf16, @CMX_NN>) -> memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>
    %alloc_8 = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1710 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%8 : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>) weights(%6 : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>) parent_input(%8 : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>) parent_output(%alloc_8 : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) outputs(%alloc_8 : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN> variants : {
      DPUTask {inEnd = [0, 0, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %10 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%9 : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x512x1x1xf16, @CMX_NN>

    %alloc_9 = memref.alloc() : memref<1x512x2x1xf16, @CMX_NN>
    %11 = VPUIP.ConcatView inputs(%5, %10 : memref<1x512x1x1xf16, @CMX_NN>, memref<1x512x1x1xf16, @CMX_NN>) outputs(%alloc_9 : memref<1x512x2x1xf16, @CMX_NN>) -> memref<1x512x2x1xf16, @CMX_NN>

    %alloc_10 = memref.alloc() : memref<1x512x2x1xf16, @DDR>
    %12 = VPUIP.Copy inputs(%11 : memref<1x512x2x1xf16, @CMX_NN>) outputs(%alloc_10 : memref<1x512x2x1xf16, @DDR>) -> memref<1x512x2x1xf16, @DDR>

    return %12 : memref<1x512x2x1xf16, @DDR>

    // CHECK: [[WEIGHT:%.+]] = const.Declare memref<512x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x128xf32>, [#const.Reshape<[512, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK: [[NCE_INPUT:%.+]] = const.Declare memref<1x128x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128xf32>, [#const.Reshape<[1, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK: [[BUFF:%.+]] = memref.alloc() : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[NCE_INPUT]] : memref<1x128x1x1xf16, {order = #NHWC}>) outputs([[BUFF]] : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[BUFF0:%.+]] = memref.alloc() : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[WEIGHT]] : memref<512x128x1x1xf16, {order = #NHWC}>) outputs([[BUFF0]] : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[BUFF2:%.+]] = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>
    // CHECK: [[RES0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1710 : i64}
    // CHECK-SAME: <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME: input([[COPY]] : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: weights([[COPY0]] : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: parent_input([[COPY]] : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: parent_output([[BUFF2]] : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: outputs([[BUFF2]] : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: -> memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[RES0]] : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x512x1x1xf16, @CMX_NN>

    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs({{%.+}} : memref<1x128x1x1xf16, @CMX_NN>) -> memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[BUFF4:%.+]] = memref.alloc() : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>
    // CHECK: VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1710 : i64}
    // CHECK-SAME: <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME: input([[PERMUTE]] : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: weights([[COPY0]] : memref<512x128x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: parent_input([[PERMUTE]] : memref<1x128x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: parent_output([[BUFF4]] : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: outputs([[BUFF4]] : memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME: -> memref<1x512x1x1xf16, {order = #NHWC}, @CMX_NN>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0095964997422461409:128>
!qElemType1 = !quant.uniform<u8:f16, 0.014437435187545478:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0075315213670917583:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!buffDistrType = !VPUIP.DistributedBuffer<1x64x250x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                        compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                        memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!buffDistrType1 = !VPUIP.DistributedBuffer<1x64x250x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                    memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!buffDistrType2 = !VPUIP.DistributedBuffer<1x64x250x32x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                    memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>

func.func @OptimizeCopiesWithinDistance() -> !buffDistrType {
    %alloc = memref.alloc() : memref<1x64x250x250x!qElemType1, {order = #NHWC}, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250x!qElemType1, {order = #NHWC}, @DDR> to memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %1 = VPURT.AllocDistributed -> !buffDistrType1
    %2 = VPUIP.ViewOp %1 : !buffDistrType1 to !buffDistrType
    %3 = VPURT.AllocDistributed -> !buffDistrType1
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%3 : !buffDistrType1) weights(%5 : !buffDistrType1) parent_input(%3 : !buffDistrType1) parent_output(%2 : !buffDistrType) outputs(%2 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %7 = VPURT.AllocDistributed -> !buffDistrType1
    %8 = VPUIP.ViewOp %7 : !buffDistrType1 to !buffDistrType
    %9 = VPURT.AllocDistributed -> !buffDistrType1
    %10 = VPURT.AllocDistributed -> !buffDistrType1
    // %11 will be fused to %5, since it is within cost distance
    %11 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%10 : !buffDistrType1) -> !buffDistrType1
    %12 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %12 : !buffDistrType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true,
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<ELTWISE>}>
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true,
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<ELTWISE>}>
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0095964997422461409:128>
!qElemType1 = !quant.uniform<u8:f16, 0.014437435187545478:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0075315213670917583:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!buffDistrType = !VPUIP.DistributedBuffer<1x64x250x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                        compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                        memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!buffDistrType1 = !VPUIP.DistributedBuffer<1x64x250x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                    memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!buffDistrType2 = !VPUIP.DistributedBuffer<1x64x250x32x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                    memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>


func.func @OptimizeCopiesConsiderDistanceForEltwise() -> !buffDistrType {
    %alloc = memref.alloc() : memref<1x64x250x250x!qElemType1, {order = #NHWC}, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250x!qElemType1, {order = #NHWC}, @DDR> to memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>
    %1 = VPURT.AllocDistributed -> !buffDistrType1
    %2 = VPUIP.ViewOp %1 : !buffDistrType1 to !buffDistrType
    %3 = VPURT.AllocDistributed -> !buffDistrType1
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%3 : !buffDistrType1) weights(%5 : !buffDistrType1) parent_input(%3 : !buffDistrType1) parent_output(%2 : !buffDistrType) outputs(%2 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %7 = VPURT.AllocDistributed -> !buffDistrType1
    %8 = VPUIP.ViewOp %7 : !buffDistrType1 to !buffDistrType
    %9 = VPURT.AllocDistributed -> !buffDistrType1
    %10 = VPURT.AllocDistributed -> !buffDistrType1
    // %11 will be fused to %5, since it is within cost distance
    %11 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%10 : !buffDistrType1) -> !buffDistrType1
    %12 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %13 = VPURT.AllocDistributed -> !buffDistrType1
    %14 = VPUIP.ViewOp %13 : !buffDistrType1 to !buffDistrType
    %15 = VPURT.AllocDistributed -> !buffDistrType1
    %16 = VPURT.AllocDistributed -> !buffDistrType1
    // %17 will be fused to %5, since it is within cost distance
    %17 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%16 : !buffDistrType1) -> !buffDistrType1
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%15 : !buffDistrType1) weights(%17 : !buffDistrType1) parent_input(%15 : !buffDistrType1) parent_output(%14 : !buffDistrType) outputs(%14 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %19 = VPURT.AllocDistributed -> !buffDistrType1
    %20 = VPUIP.ViewOp %19 : !buffDistrType1 to !buffDistrType
    %21 = VPURT.AllocDistributed -> !buffDistrType1
    %22 = VPURT.AllocDistributed -> !buffDistrType1
    // %23 will not be fused, since it is beyond cost distance
    %23 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%22 : !buffDistrType1) -> !buffDistrType1
    %24 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%21 : !buffDistrType1) weights(%23 : !buffDistrType1) parent_input(%21 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %25 = VPURT.AllocDistributed -> !buffDistrType1
    %26 = VPUIP.ViewOp %25 : !buffDistrType1 to !buffDistrType
    %27 = VPURT.AllocDistributed -> !buffDistrType1
    %28 = VPURT.AllocDistributed -> !buffDistrType1
    // %29 will be fused to %23, since it is within cost distance
    %29 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%28 : !buffDistrType1) -> !buffDistrType1
    %30 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%27 : !buffDistrType1) weights(%29 : !buffDistrType1) parent_input(%27 : !buffDistrType1) parent_output(%26 : !buffDistrType) outputs(%26 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %30 : !buffDistrType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!buffDistrType = !VPUIP.DistributedBuffer<1x64x250x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                        compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                        memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                        memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!buffDistrType1 = !VPUIP.DistributedBuffer<1x64x250x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]],
                    memory_shapes = [[1, 64, 63, 32], [1, 64, 63, 32], [1, 64, 62, 32], [1, 64, 62, 32]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0], [0, 0, 126, 0], [0, 0, 188, 0]]}>
!Weights_CMX = memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>

func.func @OptimizeCopiesConsiderDistanceForConv() -> !buffDistrType {
    %weights = memref.alloc() : !Weights_CMX

    %0 = memref.alloc() : memref<1x64x250x250xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250xf16, {order = #NHWC}, @DDR> to memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>
    %2 = VPURT.AllocDistributed -> !buffDistrType1
    %3 = VPUIP.ViewOp %2 : !buffDistrType1 to !buffDistrType
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1

    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%5 : !buffDistrType1) weights(%weights : !Weights_CMX) parent_input(%5 : !buffDistrType1) parent_output(%3 : !buffDistrType) outputs(%3 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    %7 = VPURT.AllocDistributed -> !buffDistrType1
    %8 = VPUIP.ViewOp %7 : !buffDistrType1 to !buffDistrType
    %9 = VPURT.AllocDistributed -> !buffDistrType1
    %10 = VPURT.AllocDistributed -> !buffDistrType1
    // %11 will be fused to %5, since it is within cost distance
    %11 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%10 : !buffDistrType1) -> !buffDistrType1
    %12 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %13 = VPURT.AllocDistributed -> !buffDistrType1
    %14 = VPUIP.ViewOp %13 : !buffDistrType1 to !buffDistrType
    %15 = VPURT.AllocDistributed -> !buffDistrType1
    %16 = VPURT.AllocDistributed -> !buffDistrType1
    // %17 will be fused to %5, since it is within cost distance
    %17 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%16 : !buffDistrType1) -> !buffDistrType1
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%15 : !buffDistrType1) weights(%17 : !buffDistrType1) parent_input(%15 : !buffDistrType1) parent_output(%14 : !buffDistrType) outputs(%14 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %19 = VPURT.AllocDistributed -> !buffDistrType1
    %20 = VPUIP.ViewOp %19 : !buffDistrType1 to !buffDistrType
    %21 = VPURT.AllocDistributed -> !buffDistrType1
    %22 = VPURT.AllocDistributed -> !buffDistrType1
    // %23 will not be fused, since it is beyond cost distance
    %23 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%22 : !buffDistrType1) -> !buffDistrType1
    %24 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%21 : !buffDistrType1) weights(%23 : !buffDistrType1) parent_input(%21 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    %25 = VPURT.AllocDistributed -> !buffDistrType1
    %26 = VPUIP.ViewOp %25 : !buffDistrType1 to !buffDistrType
    %27 = VPURT.AllocDistributed -> !buffDistrType1
    %28 = VPURT.AllocDistributed -> !buffDistrType1
    // %29 will be fused to %23, since it is within cost distance
    %29 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%28 : !buffDistrType1) -> !buffDistrType1
    %30 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%27 : !buffDistrType1) weights(%29 : !buffDistrType1) parent_input(%27 : !buffDistrType1) parent_output(%26 : !buffDistrType) outputs(%26 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %30 : !buffDistrType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250xf16, {order = #NHWC}, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK:     input([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3865 : i64}
    // CHECK:     <{is_inplace = true
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer
}


// -----

// CHECK-LABEL: @OnlyOptimizeSameTypeCopies
// CHECK-SAME:      [[INPUT:%.+]]: memref<128x1x1x1xf16, @DDR>
func.func @OnlyOptimizeSameTypeCopies(%arg0: memref<128x1x1x1xf16, @DDR>) -> (memref<1x128x1x1xf16, @DDR>, memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>) {
    %reshape = VPUIP.GenericReshape inputs(%arg0: memref<128x1x1x1xf16, @DDR>) -> memref<1x128x1x1xf16, @DDR>
    %alloc0 = memref.alloc() : memref<1x128x1x1xf16, @DDR>
    %alloc1 = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %alloc2 = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %copy0 = VPUIP.Copy inputs(%reshape : memref<1x128x1x1xf16, @DDR>) outputs(%alloc0 : memref<1x128x1x1xf16, @DDR>) -> memref<1x128x1x1xf16, @DDR>
    %copy1 = VPUIP.Copy inputs(%reshape : memref<1x128x1x1xf16, @DDR>) outputs(%alloc1 : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %copy2 = VPUIP.Copy inputs(%reshape : memref<1x128x1x1xf16, @DDR>) outputs(%alloc2 : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>

    return %copy0, %copy1, %copy2 : memref<1x128x1x1xf16, @DDR>, memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<128x1x1x1xf16, @DDR>) -> memref<1x128x1x1xf16, @DDR>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<1x128x1x1xf16, @DDR>
    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x128x1x1xf16, @DDR>) outputs([[ALLOC_0]] : memref<1x128x1x1xf16, @DDR>) -> memref<1x128x1x1xf16, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x128x1x1xf16, @DDR>) outputs([[ALLOC_1]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK: return [[COPY_0]], [[COPY_1]], [[COPY_1]] : memref<1x128x1x1xf16, @DDR>, memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func nested @builtin_DynamicReshape(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_reshape.cpp", VPU.kernel_entry = "dynamic_reshape"}
}

// CHECK-LABEL: @NotOptimizeConstCopyForDynamicReshape
// CHECK-SAME:      ([[INPUT_0:%.+]]: memref<100xsi32, [@CMX_NN, 0]>, [[INPUT_1:%.+]]: memref<2xsi32, [@CMX_NN, 0]>)
func.func @NotOptimizeConstCopyForDynamicReshape(%arg0: memref<100xsi32, [@CMX_NN, 0]>, %arg1: memref<2xsi32, [@CMX_NN, 0]>) -> (memref<1x100xsi32, [@CMX_NN, 0]>, memref<1x2xsi32, [@CMX_NN, 0]>) {
    %cst = const.Declare memref<2xsi32> = dense<[1, -1]> : tensor<2xsi32>

    %alloc = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%cst : memref<2xsi32>) outputs(%alloc : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x100xsi32, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    %1, %dynamicOutputShapes_0 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: -1, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_DynamicReshape
    inputs(%arg0 as %arg2: memref<100xsi32, [@CMX_NN, 0]>, %0 as %arg3: memref<2xsi32, [@CMX_NN, 0]>)
    outputs(%alloc_0 as %arg4: memref<1x100xsi32, [@CMX_NN, 0]>)
    dynamicOutputShapes(%alloc_1 : memref<2xsi32, [@CMX_NN, 0]>) on tile 0 ->
    (memref<1x100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : memref<100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x100xsi32, [@CMX_NN, 0]>
    }

    %alloc_2 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    %2 = VPUIP.Copy inputs(%cst : memref<2xsi32>) outputs(%alloc_2 : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<1x2xsi32, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    %3, %dynamicOutputShapes_1 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: -1, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_DynamicReshape
    inputs(%arg1 as %arg2: memref<2xsi32, [@CMX_NN, 0]>, %2 as %arg3: memref<2xsi32, [@CMX_NN, 0]>)
    outputs(%alloc_3 as %arg4: memref<1x2xsi32, [@CMX_NN, 0]>)
    dynamicOutputShapes(%alloc_4 : memref<2xsi32, [@CMX_NN, 0]>) on tile 0 ->
    (memref<1x2xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : memref<2xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x2xsi32, [@CMX_NN, 0]>
    }

    return %1, %3 : memref<1x100xsi32, [@CMX_NN, 0]>, memref<1x2xsi32, [@CMX_NN, 0]>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare memref<2xsi32> = dense<[1, -1]> : tensor<2xsi32>
    // CHECK:        [[ALLOC:%.+]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[COPY_0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<2xsi32>) outputs([[ALLOC]] : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[ALLOC_0:%.+]] = memref.alloc() : memref<1x100xsi32, [@CMX_NN, 0]>
    // CHECK:        [[ALLOC_1:%.+]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[RESULTS_0:%.+]], [[DYNAMIC_OUTPUT_SHAPES_0:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: -1, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_DynamicReshape
    // CHECK:        inputs([[INPUT_0]] as {{[^:]+}}: memref<100xsi32, [@CMX_NN, 0]>, [[COPY_0]] as {{[^:]+}}: memref<2xsi32, [@CMX_NN, 0]>)
    // CHECK:        outputs([[ALLOC_0]] as {{[^:]+}}: memref<1x100xsi32, [@CMX_NN, 0]>)
    // CHECK:        dynamicOutputShapes([[ALLOC_1]] : memref<2xsi32, [@CMX_NN, 0]>)
    // CHECK:        on tile 0 -> (memref<1x100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>){
    // CHECK:            VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<100xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x100xsi32, [@CMX_NN, 0]>
    // CHECK:        }
    // CHECK:        [[ALLOC_2:%.+]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[COPY_1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<2xsi32>) outputs([[ALLOC_2]] : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[ALLOC_3:%.+]] = memref.alloc() : memref<1x2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[ALLOC_4:%.+]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    // CHECK:        [[RESULTS_1:%.+]], [[DYNAMIC_OUTPUT_SHAPES_1:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: -1, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_DynamicReshape
    // CHECK:        inputs([[INPUT_1]] as {{[^:]+}}: memref<2xsi32, [@CMX_NN, 0]>, [[COPY_1]] as {{[^:]+}}: memref<2xsi32, [@CMX_NN, 0]>)
    // CHECK:        outputs([[ALLOC_3]] as {{[^:]+}}: memref<1x2xsi32, [@CMX_NN, 0]>)
    // CHECK:        dynamicOutputShapes([[ALLOC_4]] : memref<2xsi32, [@CMX_NN, 0]>)
    // CHECK:        on tile 0 -> (memref<1x2xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>){
    // CHECK:            VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<2xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x2xsi32, [@CMX_NN, 0]>
    // CHECK:        }

    // CHECK:        return [[RESULTS_0]], [[RESULTS_1]] : memref<1x100xsi32, [@CMX_NN, 0]>, memref<1x2xsi32, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @TwoAxisTilingConsiderDistanceSiblingSubview(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR> {
    %cst_0 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_1 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_1 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%5 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%4 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%5 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_3 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_3 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %7 will be fused to %5, since it is within cost distance
    %9 = VPUIP.Copy inputs(%7 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%9 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%8 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%9 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_5 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_5 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %11 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_6 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %12 will not be fused, since it is beyond cost distance
    %12 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_6 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %13 = VPUIP.Copy inputs(%11 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_7 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %14 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%13 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%12 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%13 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %15 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_9 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %16 will not be fused, since it is beyond cost distance
    %16 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_9 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_10 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %17 will be fused to %13, since it is within cost distance
    %17 = VPUIP.Copy inputs(%15 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_10 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_11 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%17 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%16 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%17 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %alloc_12 = memref.alloc() : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %19 = VPUIP.SubView %alloc_12 [0, 0, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %20 = VPUIP.Copy inputs(%6 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%19 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %21 = VPUIP.SubView %alloc_12 [0, 128, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %22 = VPUIP.Copy inputs(%10 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%21 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %23 = VPUIP.SubView %alloc_12 [0, 0, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %24 = VPUIP.Copy inputs(%14 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%23 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %25 = VPUIP.SubView %alloc_12 [0, 128, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %26 = VPUIP.Copy inputs(%18 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%25 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %27 = VPUIP.ConcatView inputs(%20, %22, %24, %26 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_12 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %28 = VPUIP.GenericReshape inputs(%27 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %29 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} inputs(%28 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %30 = VPUIP.GenericReshape inputs(%29 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    %31 = VPUIP.Copy inputs(%30 : memref<1x1024x256xf16, @DDR>) outputs(%arg1 : memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    return %31 : memref<1x1024x256xf16, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_1:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_2:%.+]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_2]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_1]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_2]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_3]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_6:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_7:%.+]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_5]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_4]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_6:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_5]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_6]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @TwoAxisTilingConsiderDistanceSiblingCopy(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR> {
    %cst_0 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_1 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_1 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%5 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_3 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %8 will be fused to %4, since it is within cost distance
    %8 = VPUIP.Copy inputs(%7 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_3 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %9 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_4 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%8 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%9 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%8 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_5 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_5 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %11 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_6 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %12 = VPUIP.Copy inputs(%11 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_6 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %13 will not be fused, since it is beyond cost distance
    %13 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_7 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %14 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%12 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%13 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%12 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %15 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_9 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %16 will be fused to %12, since it is within cost distance
    %16 = VPUIP.Copy inputs(%15 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_9 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_10 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %17 will not be fused, since it is beyond cost distance
    %17 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_10 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_11 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%16 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%17 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%16 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %alloc_12 = memref.alloc() : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %19 = VPUIP.SubView %alloc_12 [0, 0, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %20 = VPUIP.Copy inputs(%6 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%19 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %21 = VPUIP.SubView %alloc_12 [0, 128, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %22 = VPUIP.Copy inputs(%10 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%21 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %23 = VPUIP.SubView %alloc_12 [0, 0, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %24 = VPUIP.Copy inputs(%14 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%23 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %25 = VPUIP.SubView %alloc_12 [0, 128, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %26 = VPUIP.Copy inputs(%18 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%25 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %27 = VPUIP.ConcatView inputs(%20, %22, %24, %26 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_12 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %28 = VPUIP.GenericReshape inputs(%27 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %29 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} inputs(%28 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %30 = VPUIP.GenericReshape inputs(%29 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    %31 = VPUIP.Copy inputs(%30 : memref<1x1024x256xf16, @DDR>) outputs(%arg1 : memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    return %31 : memref<1x1024x256xf16, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_1:%.+]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_2:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_2]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_3]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_6:%.+]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_7:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_4]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_5]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_6:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_4]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_6]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotOptimizeParallelCopiesNoUsers
// CHECK-SAME:    ([[INPUT:%.+]]: memref<1x32x1x12544xf16, {order = #NHWC}, @DDR>)
func.func @DoNotOptimizeParallelCopiesNoUsers(%input: memref<1x32x1x12544xf16, {order = #NHWC}, @DDR>)  -> memref<1x32x112x112xf16, {order = #NHWC}, @DDR> {
    %reshape = VPUIP.GenericReshape inputs(%input: memref<1x32x1x12544xf16, {order = #NHWC}, @DDR>) -> memref<1x32x112x112xf16, {order = #NHWC}, @DDR>

    %subview1 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc1 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %copy1 = VPUIP.Copy inputs(%subview1 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
                           outputs(%alloc1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

    %subview2 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %copy2 = VPUIP.Copy inputs(%subview2 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
                           outputs(%alloc2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

    return %reshape : memref<1x32x112x112xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]]
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[RESHAPE]]
    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]]
    // CHECK-SAME:                             outputs([[ALLOC1]]
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[RESHAPE]]
    // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]]
    // CHECK-SAME:                             outputs([[ALLOC2]]
    // CHECK:       return [[RESHAPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotOptimizeParallelCopiesForTwoAxisTilingAndBlockArgumentInput
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x1024x256xf16, @DDR>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x1024x256xf16, @DDR>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<128x256x1x1xf16, @DDR>)
func.func @DoNotOptimizeParallelCopiesForTwoAxisTilingAndBlockArgumentInput(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>, %arg2: memref<128x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR> {
    %cst_1 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%arg2 : memref<128x256x1x1xf16, @DDR>) outputs(%alloc : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_1 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%5 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%4 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%5 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_3 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_3 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %7 will be fused to %5, since it is within cost distance
    %9 = VPUIP.Copy inputs(%7 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%9 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%8 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%9 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_5 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_5 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %alloc_6 = memref.alloc() : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %11 = VPUIP.SubView %alloc_6 [0, 0, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %12 = VPUIP.Copy inputs(%6 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%11 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %13 = VPUIP.SubView %alloc_6 [0, 128, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %14 = VPUIP.Copy inputs(%10 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%13 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %15 = VPUIP.ConcatView inputs(%12, %14 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_6 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %16 = VPUIP.GenericReshape inputs(%15 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %17 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} inputs(%16 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %18 = VPUIP.GenericReshape inputs(%17 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    %19 = VPUIP.Copy inputs(%18 : memref<1x1024x256xf16, @DDR>) outputs(%arg1 : memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>

    return %19 : memref<1x1024x256xf16, @DDR>

    // CHECK: [[CST:%.+]] = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    // CHECK: [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #map} inputs([[GENERICRESHAPE]] : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    // CHECK: [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTECAST]] : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[GENERICRESHAPE_1]] [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy inputs([[ARG_2]] : memref<128x256x1x1xf16, @DDR>) outputs([[ALLOC_0]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_2:%.+]] = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCEClusterTask_0:%.+]] = VPUIP.NCEClusterTask <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1]
    // CHECK-SAME:   task_type = #VPUIP.nce_task_type<CONV>}> input([[COPY_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[COPY_0]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input([[COPY_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output([[ALLOC_2]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[ALLOC_2]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
    // CHECK: DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK: } PPE : {
    // CHECK: }
    // CHECK: [[ALLOC_3:%.+]] = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_3]]  : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_4:%.+]] = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCEClusterTask_1:%.+]] = VPUIP.NCEClusterTask <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1]
    // CHECK-SAME:   task_type = #VPUIP.nce_task_type<CONV>}> input([[COPY_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[COPY_2]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input([[COPY_1]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output([[ALLOC_4]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[ALLOC_4]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
    // CHECK: DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK: } PPE : {
    // CHECK: }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoAxisTilingConsiderDistanceSiblingCopyInVF
func.func @TwoAxisTilingConsiderDistanceSiblingCopyInVF(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR> {
    %cst_0 = const.Declare memref<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>
    %cst_1 = const.Declare memref<256x128x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<256x128x1x1xf16, {order = #NHWC}>
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>

    // For VF subgraph {Conv0 -> Conv1}, with tiling strategy = [1, 2, 2, 1] and channel first unrolling
    //  IR order:
    //     Tile0 Conv0 -> Tile0 Conv1 -> Tile1 Conv0 -> Tile1 Conv1  -> Tile2 Conv0 -> Tile2 Conv1 -> Tile3 Conv0 -> Tile3 Conv1
    //  Tile0 and Tile1 will share the same copy for weights. So does for Tile2 and Tile3.


    // tile0 conv0
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_1 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.SubView %cst_0 [0, 0, 0, 0] [128, 256, 1, 1] : memref<256x256x1x1xf16, {order = #NHWC}> to memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>
    %6 = VPUIP.Copy inputs(%5 : memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>) outputs(%alloc_1 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %7 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // tile0 conv1
    %alloc_3 = memref.alloc() : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %8 = VPUIP.SubView %cst_1 [0, 0, 0, 0] [128, 128, 1, 1] : memref<256x128x1x1xf16, {order = #NHWC}> to memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>
    %9 = VPUIP.Copy inputs(%8 : memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>) outputs(%alloc_3 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_out_0 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%7 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%9 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%7 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_out_0 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_out_0 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // tile1 conv0
    %11 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_4 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %12 = VPUIP.Copy inputs(%11 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_4 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_5 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %13 = VPUIP.SubView %cst_0 [0, 0, 0, 0] [128, 256, 1, 1] : memref<256x256x1x1xf16, {order = #NHWC}> to memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>
    %14 = VPUIP.Copy inputs(%13 : memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>) outputs(%alloc_5 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_6 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %15 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%12 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%14 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%12 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_6 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_6 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // tile1 conv1
    %alloc_7 = memref.alloc() : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %16 = VPUIP.SubView %cst_1 [0, 0, 0, 0] [128, 128, 1, 1] : memref<256x128x1x1xf16, {order = #NHWC}> to memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>
    %17 = VPUIP.Copy inputs(%16 : memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>) outputs(%alloc_7 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_out_1 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %18 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%15 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%17 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%15 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_out_1 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_out_1 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // tile2 conv0
    %19 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_8 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %20 = VPUIP.Copy inputs(%19 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_8 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_9 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %21 = VPUIP.SubView %cst_0 [128, 0, 0, 0] [128, 256, 1, 1] : memref<256x256x1x1xf16, {order = #NHWC}> to memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>
    %22 = VPUIP.Copy inputs(%21 : memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>) outputs(%alloc_9 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_10 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %23 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%20 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%22 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%20 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_10 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_10 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // tile2 conv1
    %alloc_11 = memref.alloc() : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %24 = VPUIP.SubView %cst_1 [128, 0, 0, 0] [128, 128, 1, 1] : memref<256x128x1x1xf16, {order = #NHWC}> to memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>
    %25 = VPUIP.Copy inputs(%24 : memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>) outputs(%alloc_11 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_out_2 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %26 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%23 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%25 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%23 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_out_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_out_2 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }


    // tile3 conv0
    %27 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_12 = memref.alloc() : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %28 = VPUIP.Copy inputs(%27 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_12 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_13 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %29 = VPUIP.SubView %cst_0 [128, 0, 0, 0] [128, 256, 1, 1] : memref<256x256x1x1xf16, {order = #NHWC}> to memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>
    %30 = VPUIP.Copy inputs(%29 : memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}>) outputs(%alloc_13 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %alloc_14 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %31 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%28 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%30 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%28 : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_14 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_14 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // tile3 conv1
    %alloc_15 = memref.alloc() : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %32 = VPUIP.SubView %cst_1 [128, 0, 0, 0] [128, 128, 1, 1] : memref<256x128x1x1xf16, {order = #NHWC}> to memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>
    %33 = VPUIP.Copy inputs(%32 : memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}>) outputs(%alloc_15 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_out_3 = memref.alloc() : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %34 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
    input(%31 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%33 : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%31 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_out_3 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%alloc_out_3 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    %alloc_16 = memref.alloc() : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %35 = VPUIP.SubView %alloc_16 [0, 0, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %36 = VPUIP.Copy inputs(%10 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%35 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %37 = VPUIP.SubView %alloc_16 [0, 128, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %38 = VPUIP.Copy inputs(%18 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%37 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %39 = VPUIP.SubView %alloc_16 [0, 0, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %40 = VPUIP.Copy inputs(%26 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%39 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %41 = VPUIP.SubView %alloc_16 [0, 128, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, {order = #NHWC}, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %42 = VPUIP.Copy inputs(%34 : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%41 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %43 = VPUIP.ConcatView inputs(%36, %38, %40, %42 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_16 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x256x4xf16, {order = #NHWC}, @DDR>
    %44 = VPUIP.GenericReshape inputs(%43 : memref<1x256x256x4xf16, {order = #NHWC}, @DDR>) -> memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>
    %45 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} inputs(%44 : memref<1x256x1024x1xf16, {order = #NHWC}, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %46 = VPUIP.GenericReshape inputs(%45 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    %47 = VPUIP.Copy inputs(%46 : memref<1x1024x256xf16, @DDR>) outputs(%arg1 : memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    return %47 : memref<1x1024x256xf16, @DDR>


    // CHECK: [[CONV1_WEIGHTS_1:%.+]] = const.Declare memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}> = dense<2.000000e+00> : tensor<256x128x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 128, 1, 1]>]
    // CHECK: [[CONV0_WEIGHTS_1:%.+]] = const.Declare memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}> = dense<1.000000e+00> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>]
    // CHECK: [[CONV1_WEIGHTS_0:%.+]] = const.Declare memref<128x128x1x1xf16, {order = #NHWC, strides = [128, 1, 128, 128]}> = dense<2.000000e+00> : tensor<256x128x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 128, 1, 1]>]
    // CHECK: [[CONV0_WEIGHTS_0:%.+]] = const.Declare memref<128x256x1x1xf16, {order = #NHWC, strides = [256, 1, 256, 256]}> = dense<1.000000e+00> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 256, 1, 1]>]
    // CHECK: [[RESHAPE0:%.+]] = VPUIP.GenericReshape
    // CHECK: [[RESHAPE1:%.+]] = VPUIP.GenericReshape

    // CHECK: [[TILE0_CONV0_SUBVIEW:%.+]] = VPUIP.SubView [[RESHAPE1]] [0, 0, 0, 0] [1, 256, 128, 4]
    // CHECK: [[TILE0_CONV0_INPUT:%.+]] = VPUIP.Copy inputs([[TILE0_CONV0_SUBVIEW]]
    // CHECK: [[TILE0_CONV0_WEIGHTS:%.+]] = VPUIP.Copy inputs([[CONV0_WEIGHTS_0]]
    // CHECK: [[TILE0_CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[TILE0_CONV0_INPUT]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE0_CONV0_WEIGHTS]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE0_CONV1_WEIGHTS:%.+]] = VPUIP.Copy inputs([[CONV1_WEIGHTS_0]]
    // CHECK: [[TILE0_CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[TILE0_CONV0]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE0_CONV1_WEIGHTS]] : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE1_CONV0_SUBVIEW:%.+]] = VPUIP.SubView [[RESHAPE1]] [0, 0, 128, 0] [1, 256, 128, 4]
    // CHECK: [[TILE1_CONV0_INPUT:%.+]] = VPUIP.Copy inputs([[TILE1_CONV0_SUBVIEW]]
    // CHECK: [[TILE1_CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:         input([[TILE1_CONV0_INPUT]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE0_CONV0_WEIGHTS]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE1_CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:         input([[TILE1_CONV0]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE0_CONV1_WEIGHTS]] : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE2_CONV0_WEIGHTS:%.+]] = VPUIP.Copy inputs([[CONV0_WEIGHTS_1]]
    // CHECK: [[TILE2_CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[TILE0_CONV0_INPUT]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE2_CONV0_WEIGHTS]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE2_CONV1_WEIGHTS:%.+]] = VPUIP.Copy inputs([[CONV1_WEIGHTS_1]]
    // CHECK: [[TILE2_CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:         input([[TILE2_CONV0]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE2_CONV1_WEIGHTS]] : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE3_CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:         input([[TILE1_CONV0_INPUT]] : memref<1x256x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE2_CONV0_WEIGHTS]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[TILE3_CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:         input([[TILE3_CONV0]] : memref<1x128x128x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights([[TILE2_CONV1_WEIGHTS]] : memref<128x128x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DQRootInputDDR = memref<1x4096x128x8xf16, {order = #NHWC}, @DDR>
!DQInputSliceDDR = memref<1x4096x32x8xf16, {order = #NHWC, strides = [4194304, 1, 32768, 4096]}, @DDR>
!DQParamRootDDR = memref<128x4096x1x1xf16, {order = #NHWC}, @DDR>
!DQParamTileDDR = memref<16x4096x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>
!DQConvOutDDR = memref<1x32x32x8xf16, {order = #NHWC}, @DDR>
!DQConcatOutDDR = memref<1x256x32x8xf16, {order = #NHWC}, @DDR>

!DQInputCMX = !VPUIP.DistributedBuffer<
  1x4096x32x8xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 4096, 32, 8], [1, 4096, 32, 8], [1, 4096, 32, 8], [1, 4096, 32, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4096, 32, 8], [1, 4096, 32, 8], [1, 4096, 32, 8], [1, 4096, 32, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  }
>

!DQParamTileCMX = !VPUIP.DistributedBuffer<
  16x4096x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [4, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[4, 4096, 1, 1], [4, 4096, 1, 1], [4, 4096, 1, 1], [4, 4096, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [12, 0, 0, 0]],
    memory_shapes = [[4, 4096, 1, 1], [4, 4096, 1, 1], [4, 4096, 1, 1], [4, 4096, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [12, 0, 0, 0]]
  }
>

!DQWeightsCMX = !VPUIP.DistributedBuffer<
  32x4096x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]],
    memory_shapes = [[8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]]
  }
>

!DQWeightsReshapeCMX = !VPUIP.DistributedBuffer<
  32x1x4096x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[8, 1, 4096, 1], [8, 1, 4096, 1], [8, 1, 4096, 1], [8, 1, 4096, 1]],
    compute_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]],
    memory_shapes = [[8, 1, 4096, 1], [8, 1, 4096, 1], [8, 1, 4096, 1], [8, 1, 4096, 1]],
    memory_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]]
  }
>

!DQWeightsCastCMX = !VPUIP.DistributedBuffer<
  32x4096x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [8, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]],
    memory_shapes = [[8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1], [8, 4096, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]]
  }
>

!DQConvOutCMX = !VPUIP.DistributedBuffer<
  1x32x32x8xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 8, 32, 8], [1, 8, 32, 8], [1, 8, 32, 8], [1, 8, 32, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 32, 32, 8], [1, 32, 32, 8], [1, 32, 32, 8], [1, 32, 32, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  }
>

module @VPU.SW {
  func.func nested @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @NotOptimizeParallelCopiesIfBufferIsLarge
// CHECK-SAME: [[INPUT:%[^:]+]]: memref<1x4096x128x8xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[OUTPUT:%[^:]+]]: memref<1x256x32x8xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[DQ_BUFFER:%[^:]+]]: memref<128x4096x1x1xf16, {order = #NHWC}, @DDR>)
func.func @NotOptimizeParallelCopiesIfBufferIsLarge(%input: !DQRootInputDDR, %output: !DQConcatOutDDR, %dequantParams: !DQParamRootDDR) -> !DQConcatOutDDR {
    %w0Tile0Slice = VPUIP.SubView %dequantParams [0, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %w0Tile0Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %w0Tile0 = VPUIP.Copy inputs(%w0Tile0Slice : !DQParamTileDDR) outputs(%w0Tile0Buffer : !DQParamTileCMX) -> !DQParamTileCMX
    %s0Tile0Slice = VPUIP.SubView %dequantParams [16, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %s0Tile0Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %s0Tile0 = VPUIP.Copy inputs(%s0Tile0Slice : !DQParamTileDDR) outputs(%s0Tile0Buffer : !DQParamTileCMX) -> !DQParamTileCMX
    %w0Tile1Slice = VPUIP.SubView %dequantParams [32, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %w0Tile1Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %w0Tile1 = VPUIP.Copy inputs(%w0Tile1Slice : !DQParamTileDDR) outputs(%w0Tile1Buffer : !DQParamTileCMX) -> !DQParamTileCMX
    %s0Tile1Slice = VPUIP.SubView %dequantParams [48, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %s0Tile1Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %s0Tile1 = VPUIP.Copy inputs(%s0Tile1Slice : !DQParamTileDDR) outputs(%s0Tile1Buffer : !DQParamTileCMX) -> !DQParamTileCMX

    %dq0Tile0Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %dq0Tile1Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %dq0Tiled:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
      inputs(%w0Tile0 as %dq0Input0: !DQParamTileCMX, %s0Tile0 as %dq0Scale0: !DQParamTileCMX,
           %w0Tile1 as %dq0Input1: !DQParamTileCMX, %s0Tile1 as %dq0Scale1: !DQParamTileCMX)
      outputs(%dq0Tile0Buffer as %dq0Output0: !DQParamTileCMX,
          %dq0Tile1Buffer as %dq0Output1: !DQParamTileCMX) on tile 0 -> (!DQParamTileCMX, !DQParamTileCMX) {
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%dq0Input0, %dq0Scale0, %dq0Output0) : !DQParamTileCMX, !DQParamTileCMX, !DQParamTileCMX
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%dq0Input1, %dq0Scale1, %dq0Output1) : !DQParamTileCMX, !DQParamTileCMX, !DQParamTileCMX
    }
    %dq0Buffer = VPURT.AllocDistributed -> !DQWeightsCMX
    %dq0 = VPUIP.ConcatView inputs(%dq0Tiled#0, %dq0Tiled#1 : !DQParamTileCMX, !DQParamTileCMX) outputs(%dq0Buffer : !DQWeightsCMX) -> !DQWeightsCMX
    %dq0Reshape = VPUIP.GenericReshape inputs(%dq0 : !DQWeightsCMX) -> !DQWeightsReshapeCMX
    %dq0Permute = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%dq0Reshape : !DQWeightsReshapeCMX) -> !DQWeightsCMX
    %dq0CastC0S0 = VPUIP.DistributedCast inputs(%dq0Permute : !DQWeightsCMX) -> !DQWeightsCastCMX
    %dq0CastC0S1 = VPUIP.DistributedCast inputs(%dq0Permute : !DQWeightsCMX) -> !DQWeightsCastCMX
    %dq0CastC1S0 = VPUIP.DistributedCast inputs(%dq0Permute : !DQWeightsCMX) -> !DQWeightsCastCMX
    %dq0CastC1S1 = VPUIP.DistributedCast inputs(%dq0Permute : !DQWeightsCMX) -> !DQWeightsCastCMX

    %inputSlice0C0S0 = VPUIP.SubView %input [0, 0, 0, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer0C0S0 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy0C0S0 = VPUIP.Copy inputs(%inputSlice0C0S0 : !DQInputSliceDDR) outputs(%inputBuffer0C0S0 : !DQInputCMX) -> !DQInputCMX
    %out0C0S0Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv0C0S0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy0C0S0 : !DQInputCMX)
      weights(%dq0CastC0S0 : !DQWeightsCastCMX)
      parent_input(%inputCopy0C0S0 : !DQInputCMX)
      parent_output(%out0C0S0Buffer : !DQConvOutCMX)
      outputs(%out0C0S0Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out0C0S0DDR = memref.alloc() : !DQConvOutDDR
    %copyOut0C0S0 = VPUIP.Copy inputs(%conv0C0S0 : !DQConvOutCMX) outputs(%out0C0S0DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %inputSlice0C0S1 = VPUIP.SubView %input [0, 0, 32, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer0C0S1 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy0C0S1 = VPUIP.Copy inputs(%inputSlice0C0S1 : !DQInputSliceDDR) outputs(%inputBuffer0C0S1 : !DQInputCMX) -> !DQInputCMX
    %out0C0S1Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv0C0S1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy0C0S1 : !DQInputCMX)
      weights(%dq0CastC0S1 : !DQWeightsCastCMX)
      parent_input(%inputCopy0C0S1 : !DQInputCMX)
      parent_output(%out0C0S1Buffer : !DQConvOutCMX)
      outputs(%out0C0S1Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out0C0S1DDR = memref.alloc() : !DQConvOutDDR
    %copyOut0C0S1 = VPUIP.Copy inputs(%conv0C0S1 : !DQConvOutCMX) outputs(%out0C0S1DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %inputSlice0C1S0 = VPUIP.SubView %input [0, 0, 64, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer0C1S0 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy0C1S0 = VPUIP.Copy inputs(%inputSlice0C1S0 : !DQInputSliceDDR) outputs(%inputBuffer0C1S0 : !DQInputCMX) -> !DQInputCMX
    %out0C1S0Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv0C1S0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy0C1S0 : !DQInputCMX)
      weights(%dq0CastC1S0 : !DQWeightsCastCMX)
      parent_input(%inputCopy0C1S0 : !DQInputCMX)
      parent_output(%out0C1S0Buffer : !DQConvOutCMX)
      outputs(%out0C1S0Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out0C1S0DDR = memref.alloc() : !DQConvOutDDR
    %copyOut0C1S0 = VPUIP.Copy inputs(%conv0C1S0 : !DQConvOutCMX) outputs(%out0C1S0DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %inputSlice0C1S1 = VPUIP.SubView %input [0, 0, 96, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer0C1S1 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy0C1S1 = VPUIP.Copy inputs(%inputSlice0C1S1 : !DQInputSliceDDR) outputs(%inputBuffer0C1S1 : !DQInputCMX) -> !DQInputCMX
    %out0C1S1Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv0C1S1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy0C1S1 : !DQInputCMX)
      weights(%dq0CastC1S1 : !DQWeightsCastCMX)
      parent_input(%inputCopy0C1S1 : !DQInputCMX)
      parent_output(%out0C1S1Buffer : !DQConvOutCMX)
      outputs(%out0C1S1Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out0C1S1DDR = memref.alloc() : !DQConvOutDDR
    %copyOut0C1S1 = VPUIP.Copy inputs(%conv0C1S1 : !DQConvOutCMX) outputs(%out0C1S1DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %w1Tile0Slice = VPUIP.SubView %dequantParams [64, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %w1Tile0Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %w1Tile0 = VPUIP.Copy inputs(%w1Tile0Slice : !DQParamTileDDR) outputs(%w1Tile0Buffer : !DQParamTileCMX) -> !DQParamTileCMX
    %s1Tile0Slice = VPUIP.SubView %dequantParams [80, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %s1Tile0Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %s1Tile0 = VPUIP.Copy inputs(%s1Tile0Slice : !DQParamTileDDR) outputs(%s1Tile0Buffer : !DQParamTileCMX) -> !DQParamTileCMX
    %w1Tile1Slice = VPUIP.SubView %dequantParams [96, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %w1Tile1Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %w1Tile1 = VPUIP.Copy inputs(%w1Tile1Slice : !DQParamTileDDR) outputs(%w1Tile1Buffer : !DQParamTileCMX) -> !DQParamTileCMX
    %s1Tile1Slice = VPUIP.SubView %dequantParams [112, 0, 0, 0] [16, 4096, 1, 1] : !DQParamRootDDR to !DQParamTileDDR
    %s1Tile1Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %s1Tile1 = VPUIP.Copy inputs(%s1Tile1Slice : !DQParamTileDDR) outputs(%s1Tile1Buffer : !DQParamTileCMX) -> !DQParamTileCMX

    %dq1Tile0Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %dq1Tile1Buffer = VPURT.AllocDistributed -> !DQParamTileCMX
    %dq1Tiled:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
      inputs(%w1Tile0 as %dq1Input0: !DQParamTileCMX, %s1Tile0 as %dq1Scale0: !DQParamTileCMX,
           %w1Tile1 as %dq1Input1: !DQParamTileCMX, %s1Tile1 as %dq1Scale1: !DQParamTileCMX)
      outputs(%dq1Tile0Buffer as %dq1Output0: !DQParamTileCMX,
          %dq1Tile1Buffer as %dq1Output1: !DQParamTileCMX) on tile 0 -> (!DQParamTileCMX, !DQParamTileCMX) {
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%dq1Input0, %dq1Scale0, %dq1Output0) : !DQParamTileCMX, !DQParamTileCMX, !DQParamTileCMX
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%dq1Input1, %dq1Scale1, %dq1Output1) : !DQParamTileCMX, !DQParamTileCMX, !DQParamTileCMX
    }
    %dq1Buffer = VPURT.AllocDistributed -> !DQWeightsCMX
    %dq1 = VPUIP.ConcatView inputs(%dq1Tiled#0, %dq1Tiled#1 : !DQParamTileCMX, !DQParamTileCMX) outputs(%dq1Buffer : !DQWeightsCMX) -> !DQWeightsCMX
    %dq1Reshape = VPUIP.GenericReshape inputs(%dq1 : !DQWeightsCMX) -> !DQWeightsReshapeCMX
    %dq1Permute = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%dq1Reshape : !DQWeightsReshapeCMX) -> !DQWeightsCMX
    %dq1CastC0S0 = VPUIP.DistributedCast inputs(%dq1Permute : !DQWeightsCMX) -> !DQWeightsCastCMX
    %dq1CastC0S1 = VPUIP.DistributedCast inputs(%dq1Permute : !DQWeightsCMX) -> !DQWeightsCastCMX
    %dq1CastC1S0 = VPUIP.DistributedCast inputs(%dq1Permute : !DQWeightsCMX) -> !DQWeightsCastCMX
    %dq1CastC1S1 = VPUIP.DistributedCast inputs(%dq1Permute : !DQWeightsCMX) -> !DQWeightsCastCMX

    %inputSlice1C0S0 = VPUIP.SubView %input [0, 0, 0, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer1C0S0 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy1C0S0 = VPUIP.Copy inputs(%inputSlice1C0S0 : !DQInputSliceDDR) outputs(%inputBuffer1C0S0 : !DQInputCMX) -> !DQInputCMX
    %out1C0S0Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv1C0S0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy1C0S0 : !DQInputCMX)
      weights(%dq1CastC0S0 : !DQWeightsCastCMX)
      parent_input(%inputCopy1C0S0 : !DQInputCMX)
      parent_output(%out1C0S0Buffer : !DQConvOutCMX)
      outputs(%out1C0S0Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out1C0S0DDR = memref.alloc() : !DQConvOutDDR
    %copyOut1C0S0 = VPUIP.Copy inputs(%conv1C0S0 : !DQConvOutCMX) outputs(%out1C0S0DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %inputSlice1C0S1 = VPUIP.SubView %input [0, 0, 32, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer1C0S1 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy1C0S1 = VPUIP.Copy inputs(%inputSlice1C0S1 : !DQInputSliceDDR) outputs(%inputBuffer1C0S1 : !DQInputCMX) -> !DQInputCMX
    %out1C0S1Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv1C0S1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy1C0S1 : !DQInputCMX)
      weights(%dq1CastC0S1 : !DQWeightsCastCMX)
      parent_input(%inputCopy1C0S1 : !DQInputCMX)
      parent_output(%out1C0S1Buffer : !DQConvOutCMX)
      outputs(%out1C0S1Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out1C0S1DDR = memref.alloc() : !DQConvOutDDR
    %copyOut1C0S1 = VPUIP.Copy inputs(%conv1C0S1 : !DQConvOutCMX) outputs(%out1C0S1DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %inputSlice1C1S0 = VPUIP.SubView %input [0, 0, 64, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer1C1S0 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy1C1S0 = VPUIP.Copy inputs(%inputSlice1C1S0 : !DQInputSliceDDR) outputs(%inputBuffer1C1S0 : !DQInputCMX) -> !DQInputCMX
    %out1C1S0Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv1C1S0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy1C1S0 : !DQInputCMX)
      weights(%dq1CastC1S0 : !DQWeightsCastCMX)
      parent_input(%inputCopy1C1S0 : !DQInputCMX)
      parent_output(%out1C1S0Buffer : !DQConvOutCMX)
      outputs(%out1C1S0Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out1C1S0DDR = memref.alloc() : !DQConvOutDDR
    %copyOut1C1S0 = VPUIP.Copy inputs(%conv1C1S0 : !DQConvOutCMX) outputs(%out1C1S0DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %inputSlice1C1S1 = VPUIP.SubView %input [0, 0, 96, 0] [1, 4096, 32, 8] : !DQRootInputDDR to !DQInputSliceDDR
    %inputBuffer1C1S1 = VPURT.AllocDistributed -> !DQInputCMX
    %inputCopy1C1S1 = VPUIP.Copy inputs(%inputSlice1C1S1 : !DQInputSliceDDR) outputs(%inputBuffer1C1S1 : !DQInputCMX) -> !DQInputCMX
    %out1C1S1Buffer = VPURT.AllocDistributed -> !DQConvOutCMX
    %conv1C1S1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 12479 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
    }>
      input(%inputCopy1C1S1 : !DQInputCMX)
      weights(%dq1CastC1S1 : !DQWeightsCastCMX)
      parent_input(%inputCopy1C1S1 : !DQInputCMX)
      parent_output(%out1C1S1Buffer : !DQConvOutCMX)
      outputs(%out1C1S1Buffer : !DQConvOutCMX) -> !DQConvOutCMX variants : {
      DPUTask {inEnd = [7, 31, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %out1C1S1DDR = memref.alloc() : !DQConvOutDDR
    %copyOut1C1S1 = VPUIP.Copy inputs(%conv1C1S1 : !DQConvOutCMX) outputs(%out1C1S1DDR : !DQConvOutDDR) -> !DQConvOutDDR

    %concat = VPUIP.ConcatView inputs(%copyOut0C0S0, %copyOut0C0S1, %copyOut0C1S0, %copyOut0C1S1,
                      %copyOut1C0S0, %copyOut1C0S1, %copyOut1C1S0, %copyOut1C1S1 :
                      !DQConvOutDDR, !DQConvOutDDR, !DQConvOutDDR, !DQConvOutDDR,
                      !DQConvOutDDR, !DQConvOutDDR, !DQConvOutDDR, !DQConvOutDDR)
      outputs(%output : !DQConcatOutDDR) -> !DQConcatOutDDR
    return %concat : !DQConcatOutDDR

    // CHECK:       VPUIP.SubView [[DQ_BUFFER]] [0, 0, 0, 0] [16, 4096, 1, 1]
    // CHECK:       [[DQ0_TILED:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK:       [[DQ0:%.+]] = VPUIP.ConcatView inputs([[DQ0_TILED]]#0, [[DQ0_TILED]]#1
    // CHECK:       [[DQ0_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[DQ0]]
    // CHECK:       [[DQ0_PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[DQ0_RESHAPE]]
    // CHECK:       [[DQ0_CAST_C0_S0:%.+]] = VPUIP.DistributedCast inputs([[DQ0_PERMUTE]]
    // CHECK:       [[DQ0_CAST_C0_S1:%.+]] = VPUIP.DistributedCast inputs([[DQ0_PERMUTE]]
    // CHECK:       [[DQ0_CAST_C1_S0:%.+]] = VPUIP.DistributedCast inputs([[DQ0_PERMUTE]]
    // CHECK:       [[DQ0_CAST_C1_S1:%.+]] = VPUIP.DistributedCast inputs([[DQ0_PERMUTE]]

    // CHECK:       [[INPUT_SLICE_0_C0_S0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_0_C0_S0:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_0_C0_S0]]
    // CHECK:       [[CONV_0_C0_S0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_0_C0_S0]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ0_CAST_C0_S0]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       [[INPUT_SLICE_0_C0_S1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 32, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_0_C0_S1:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_0_C0_S1]]
    // CHECK:       [[CONV_0_C0_S1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_0_C0_S1]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ0_CAST_C0_S1]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       [[INPUT_SLICE_0_C1_S0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 64, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_0_C1_S0:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_0_C1_S0]]
    // CHECK:       [[CONV_0_C1_S0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_0_C1_S0]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ0_CAST_C1_S0]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       [[INPUT_SLICE_0_C1_S1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 96, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_0_C1_S1:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_0_C1_S1]]
    // CHECK:       [[CONV_0_C1_S1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_0_C1_S1]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ0_CAST_C1_S1]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       VPUIP.SubView [[DQ_BUFFER]] [64, 0, 0, 0] [16, 4096, 1, 1]
    // CHECK:       [[DQ1_TILED:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK:       [[DQ1:%.+]] = VPUIP.ConcatView inputs([[DQ1_TILED]]#0, [[DQ1_TILED]]#1
    // CHECK:       [[DQ1_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[DQ1]]
    // CHECK:       [[DQ1_PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[DQ1_RESHAPE]]
    // CHECK:       [[DQ1_CAST_C0_S0:%.+]] = VPUIP.DistributedCast inputs([[DQ1_PERMUTE]]
    // CHECK:       [[DQ1_CAST_C0_S1:%.+]] = VPUIP.DistributedCast inputs([[DQ1_PERMUTE]]
    // CHECK:       [[DQ1_CAST_C1_S0:%.+]] = VPUIP.DistributedCast inputs([[DQ1_PERMUTE]]
    // CHECK:       [[DQ1_CAST_C1_S1:%.+]] = VPUIP.DistributedCast inputs([[DQ1_PERMUTE]]

    // CHECK:       [[INPUT_SLICE_1_C0_S0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_1_C0_S0:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_1_C0_S0]]
    // CHECK:       [[CONV_1_C0_S0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_1_C0_S0]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ1_CAST_C0_S0]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       [[INPUT_SLICE_1_C0_S1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 32, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_1_C0_S1:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_1_C0_S1]]
    // CHECK:       [[CONV_1_C0_S1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_1_C0_S1]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ1_CAST_C0_S1]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       [[INPUT_SLICE_1_C1_S0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 64, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_1_C1_S0:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_1_C1_S0]]
    // CHECK:       [[CONV_1_C1_S0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_1_C1_S0]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ1_CAST_C1_S0]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16

    // CHECK:       [[INPUT_SLICE_1_C1_S1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 96, 0] [1, 4096, 32, 8]
    // CHECK:       [[INPUT_COPY_1_C1_S1:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE_1_C1_S1]]
    // CHECK:       [[CONV_1_C1_S1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[INPUT_COPY_1_C1_S1]] : !VPUIP.DistributedBuffer<1x4096x32x8xf16
    // CHECK-SAME:      weights([[DQ1_CAST_C1_S1]] : !VPUIP.DistributedBuffer<32x4096x1x1xf16
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView

    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistType = !VPUIP.DistributedBuffer<
  1x1376x256x4xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1376, 86, 4], [1, 1376, 85, 4], [1, 1376, 85, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]],
    memory_shapes = [[1, 1376, 86, 4], [1, 1376, 85, 4], [1, 1376, 85, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]]
  }
>

!WeightsDistType = !VPUIP.DistributedBuffer<
  64x1376x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[64, 1376, 1, 1], [64, 1376, 1, 1], [64, 1376, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 1376, 1, 1], [64, 1376, 1, 1], [64, 1376, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  }
>

!WeightTableDistType = !VPUIP.DistributedBuffer<
  64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  }
>

!OutputDistType = !VPUIP.DistributedBuffer<
  1x64x256x4xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 64, 86, 4], [1, 64, 85, 4], [1, 64, 85, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]],
    memory_shapes = [[1, 64, 86, 4], [1, 64, 85, 4], [1, 64, 85, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]]
  }
>

// CHECK-LABEL: @OptimizeParallelCopiesMultiUserSubView
// CHECK-SAME: [[ARG_INPUT:%[^:]+]]: memref<1x4096x256x4xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_WEIGHTS_0:%[^:]+]]: memref<64x1376x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>,
// CHECK-SAME: [[ARG_WEIGHTS_1:%[^:]+]]: memref<64x1376x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>
func.func @OptimizeParallelCopiesMultiUserSubView(
  %input: memref<1x4096x256x4xf16, {order = #NHWC}, @DDR>,
  %weights0: memref<64x1376x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>,
  %weights1: memref<64x1376x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>) -> (!OutputDistType, !OutputDistType) {
    %inputSlice = VPUIP.SubView %input [0, 1376, 0, 0] [1, 1376, 256, 4] : memref<1x4096x256x4xf16, {order = #NHWC}, @DDR> to memref<1x1376x256x4xf16, {order = #NHWC, strides = [4194304, 1, 16384, 4096]}, @DDR>

    %inputBuffer0 = VPURT.AllocDistributed -> !InputDistType
    %inputCopy0 = VPUIP.Copy inputs(%inputSlice : memref<1x1376x256x4xf16, {order = #NHWC, strides = [4194304, 1, 16384, 4096]}, @DDR>) outputs(%inputBuffer0 : !InputDistType) -> !InputDistType
    %weightsBuffer0 = VPURT.AllocDistributed -> !WeightsDistType
    %weightsCopy0 = VPUIP.Copy inputs(%weights0 : memref<64x1376x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>) outputs(%weightsBuffer0 : !WeightsDistType) -> !WeightsDistType
    %outputBuffer0 = VPURT.AllocDistributed -> !OutputDistType
    %conv0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 18781 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, tiling_loop_index = 4 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%inputCopy0 : !InputDistType) weights(%weightsCopy0 : !WeightsDistType) parent_input(%inputCopy0 : !InputDistType) parent_output(%outputBuffer0 : !OutputDistType) outputs(%outputBuffer0 : !OutputDistType) -> !OutputDistType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 85, 1375], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 85, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [3, 84, 1375], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 84, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [3, 84, 1375], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 84, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    %inputBuffer1 = VPURT.AllocDistributed -> !InputDistType
    %inputCopy1 = VPUIP.Copy inputs(%inputSlice : memref<1x1376x256x4xf16, {order = #NHWC, strides = [4194304, 1, 16384, 4096]}, @DDR>) outputs(%inputBuffer1 : !InputDistType) -> !InputDistType
    %weightsBuffer1 = VPURT.AllocDistributed -> !WeightsDistType
    %weightsCopy1 = VPUIP.Copy inputs(%weights1 : memref<64x1376x1x1xf16, {order = #NHWC, strides = [4096, 1, 4096, 4096]}, @DDR>) outputs(%weightsBuffer1 : !WeightsDistType) -> !WeightsDistType
    %outputBuffer1 = VPURT.AllocDistributed -> !OutputDistType
    %conv1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 18781 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, tiling_loop_index = 4 : i64} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%inputCopy1 : !InputDistType) weights(%weightsCopy1 : !WeightsDistType) parent_input(%inputCopy1 : !InputDistType) parent_output(%outputBuffer1 : !OutputDistType) outputs(%outputBuffer1 : !OutputDistType) -> !OutputDistType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 85, 1375], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 85, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [3, 84, 1375], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 84, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [3, 84, 1375], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 84, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    return %conv0, %conv1 : !OutputDistType, !OutputDistType

    // CHECK:      [[INPUT_SLICE:%.+]] = VPUIP.SubView [[ARG_INPUT]] [0, 1376, 0, 0] [1, 1376, 256, 4]
    // CHECK:      [[INPUT_BUFFER:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT_SLICE]]
    // CHECK-SAME:     outputs([[INPUT_BUFFER]]

    // CHECK:      [[WEIGHTS_BUFFER_0:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[WEIGHTS_COPY_0:%.+]] = VPUIP.Copy inputs([[ARG_WEIGHTS_0]]
    // CHECK-SAME:     outputs([[WEIGHTS_BUFFER_0]]
    // CHECK:      [[OUTPUT_BUFFER_0:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[CONV_0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:     input([[INPUT_COPY]]
    // CHECK-SAME:     weights([[WEIGHTS_COPY_0]]
    // CHECK-SAME:     outputs([[OUTPUT_BUFFER_0]]

    // CHECK-NOT:  VPUIP.Copy inputs([[INPUT_SLICE]]
    // CHECK:      [[WEIGHTS_BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[WEIGHTS_COPY_1:%.+]] = VPUIP.Copy inputs([[ARG_WEIGHTS_1]]
    // CHECK-SAME:     outputs([[WEIGHTS_BUFFER_1]]
    // CHECK:      [[OUTPUT_BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[CONV_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:     input([[INPUT_COPY]]
    // CHECK-SAME:     weights([[WEIGHTS_COPY_1]]
    // CHECK-SAME:     outputs([[OUTPUT_BUFFER_1]]

    // CHECK:      return [[CONV_0]], [[CONV_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#QKV_IDENTITY = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#QKV_ACT_PERM = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
#QKV_WEIGHTS_PERM = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!QKVLayerNormRootDDR = memref<1x1x256x384xf16, @DDR>
!QKVLayerNormTileDDR = memref<1x1x64x384xf16, {order = #QKV_IDENTITY, strides = [98304, 98304, 384, 1]}, @DDR>
!QKVRmsInputDDR = memref<1x1x1024x384xf16, @DDR>
!QKVRmsInputTileDDR = memref<1x1x256x384xf16, {order = #QKV_IDENTITY, strides = [393216, 393216, 384, 1]}, @DDR>
!QKVRmsInputHalfDDR = memref<1x1x128x384xf16, {order = #QKV_IDENTITY, strides = [393216, 393216, 384, 1]}, @DDR>
!QKVRmsOutputHalfDDR = memref<1x1x128x384xf16, {order = #QKV_IDENTITY, strides = [98304, 98304, 384, 1]}, @DDR>
!QKVRmsHalfCMX = memref<1x1x128x384xf16, @CMX_NN>
!QKVActFlatDDR = memref<64x384x1x1xf16, @DDR>
!QKVActPermuteDDR = memref<1x384x64x1xf16, {order = #NHWC}, @DDR>
!QKVActDDR = memref<1x384x8x8xf16, {order = #NHWC}, @DDR>
!QKVActCMX = memref<1x384x8x8xf16, {order = #NHWC}, @CMX_NN>
!QKVWeightsParamDDR = memref<6x192x64xf16, @DDR>
!QKVWeightsRootDDR = memref<1x6x192x64xf16, @DDR>
!QKVWeightsTileDDR = memref<1x6x64x64xf16, {order = #QKV_IDENTITY, strides = [73728, 12288, 64, 1]}, @DDR>
!QKVWeightsTileCMX = memref<1x6x64x64xf16, {order = #QKV_IDENTITY, strides = [73728, 12288, 64, 1]}, @CMX_NN>
!QKVWeightsPermuteCMX = memref<1x64x6x64xf16, @CMX_NN>
!QKVWeightsPermuteHalfCMX = memref<1x32x6x64xf16, {order = #QKV_IDENTITY, strides = [24576, 384, 64, 1]}, @CMX_NN>
!QKVScalesDDR = memref<1x64x6x1xf16, @DDR>
!QKVScalesCMX = memref<1x64x6x1xf16, @CMX_NN>
!QKVScaleHalfCMX = memref<1x32x6x1xf16, {order = #QKV_IDENTITY, strides = [384, 6, 1, 1]}, @CMX_NN>
!QKVWeightsDequantCMX = memref<1x64x6x64xf16, @CMX_NN>
!QKVWeightsDequantHalfCMX = memref<1x32x6x64xf16, {order = #QKV_IDENTITY, strides = [24576, 384, 64, 1]}, @CMX_NN>
!QKVWeightsCMXRaw = memref<64x384x1x1xf16, @CMX_NN>
!QKVWeightsCMX = memref<64x384x1x1xf16, {order = #NHWC}, @CMX_NN>
!QKVOutCMX = memref<1x64x8x8xf16, {order = #NHWC}, @CMX_NN>
!QKVOutDDR = memref<1x64x8x8xf16, {order = #NHWC}, @DDR>

module @VPU.SW {
  func.func nested @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
  func.func nested @builtin_RMS(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "rms.cpp", VPU.kernel_entry = "rms"}
}

// CHECK-LABEL: @KeepInputCopyDynamicDequantConvVFPatternTilingOnTwoAxis
// CHECK-SAME: [[RMS_INPUT:%[^:]+]]: memref<1x1x1024x384xf16, @DDR>,
// CHECK-SAME: [[OUT0:%[^:]+]]: memref<1x64x8x8xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[OUT1:%[^:]+]]: memref<1x64x8x8xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[OUT2:%[^:]+]]: memref<1x64x8x8xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[WEIGHTS_PARAM:%[^:]+]]: memref<6x192x64xf16, @DDR>,
// CHECK-SAME: [[SCALES:%[^:]+]]: memref<1x64x6x1xf16, @DDR>)
func.func @KeepInputCopyDynamicDequantConvVFPatternTilingOnTwoAxis(%rmsInput: !QKVRmsInputDDR, %out0: !QKVOutDDR, %out1: !QKVOutDDR, %out2: !QKVOutDDR, %weightsParam: !QKVWeightsParamDDR, %scales: !QKVScalesDDR) -> (!QKVOutDDR, !QKVOutDDR, !QKVOutDDR) {
    %rmsInputTile = VPUIP.SubView %rmsInput [0, 0, 0, 0] [1, 1, 256, 384] : !QKVRmsInputDDR to !QKVRmsInputTileDDR
    %rmsInputHiSlice = VPUIP.SubView %rmsInputTile [0, 0, 128, 0] [1, 1, 128, 384] : !QKVRmsInputTileDDR to !QKVRmsInputHalfDDR
    %rmsInputHiBuffer = memref.alloc() : !QKVRmsHalfCMX
    %rmsInputHiCopy = VPUIP.Copy inputs(%rmsInputHiSlice : !QKVRmsInputHalfDDR) outputs(%rmsInputHiBuffer : !QKVRmsHalfCMX) -> !QKVRmsHalfCMX
    %rmsInputLoSlice = VPUIP.SubView %rmsInputTile [0, 0, 0, 0] [1, 1, 128, 384] : !QKVRmsInputTileDDR to !QKVRmsInputHalfDDR
    %rmsInputLoBuffer = memref.alloc() : !QKVRmsHalfCMX
    %rmsInputLoCopy = VPUIP.Copy inputs(%rmsInputLoSlice : !QKVRmsInputHalfDDR) outputs(%rmsInputLoBuffer : !QKVRmsHalfCMX) -> !QKVRmsHalfCMX
    %rmsLoBuffer = memref.alloc() : !QKVRmsHalfCMX
    %rmsHiBuffer = memref.alloc() : !QKVRmsHalfCMX
    %rms:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RMS
      inputs(%rmsInputLoCopy as %rmsKernelInputLo: !QKVRmsHalfCMX, %rmsInputHiCopy as %rmsKernelInputHi: !QKVRmsHalfCMX)
      outputs(%rmsLoBuffer as %rmsKernelOutputLo: !QKVRmsHalfCMX, %rmsHiBuffer as %rmsKernelOutputHi: !QKVRmsHalfCMX) on tile 0 -> (!QKVRmsHalfCMX, !QKVRmsHalfCMX) {
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%rmsKernelInputLo, %rmsKernelOutputLo) : !QKVRmsHalfCMX, !QKVRmsHalfCMX
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%rmsKernelInputHi, %rmsKernelOutputHi) : !QKVRmsHalfCMX, !QKVRmsHalfCMX
    }
    %rmsDDR = memref.alloc() : !QKVLayerNormRootDDR
    %rmsOutputLoSlice = VPUIP.SubView %rmsDDR [0, 0, 0, 0] [1, 1, 128, 384] : !QKVLayerNormRootDDR to !QKVRmsOutputHalfDDR
    %rmsDDRLo = VPUIP.Copy inputs(%rms#0 : !QKVRmsHalfCMX) outputs(%rmsOutputLoSlice : !QKVRmsOutputHalfDDR) -> !QKVRmsOutputHalfDDR
    %rmsOutputHiSlice = VPUIP.SubView %rmsDDR [0, 0, 128, 0] [1, 1, 128, 384] : !QKVLayerNormRootDDR to !QKVRmsOutputHalfDDR
    %rmsDDRHi = VPUIP.Copy inputs(%rms#1 : !QKVRmsHalfCMX) outputs(%rmsOutputHiSlice : !QKVRmsOutputHalfDDR) -> !QKVRmsOutputHalfDDR
    %rmsDDRConcat = VPUIP.ConcatView inputs(%rmsDDRLo, %rmsDDRHi : !QKVRmsOutputHalfDDR, !QKVRmsOutputHalfDDR) outputs(%rmsDDR : !QKVLayerNormRootDDR) -> !QKVLayerNormRootDDR

    %weights = VPUIP.GenericReshape inputs(%weightsParam : !QKVWeightsParamDDR) -> !QKVWeightsRootDDR
    %wRootSlice = VPUIP.SubView %weights [0, 0, 0, 0] [1, 6, 64, 64] : !QKVWeightsRootDDR to !QKVWeightsTileDDR
    %wSiblingRootSlice = VPUIP.SubView %weights [0, 0, 64, 0] [1, 6, 64, 64] : !QKVWeightsRootDDR to !QKVWeightsTileDDR
    %wRootCopyBuffer = memref.alloc() : !QKVWeightsTileCMX
    %wRootCopy = VPUIP.Copy inputs(%wRootSlice : !QKVWeightsTileDDR) outputs(%wRootCopyBuffer : !QKVWeightsTileCMX) -> !QKVWeightsTileCMX
    %wPermuteBuffer = memref.alloc() : !QKVWeightsPermuteCMX
    %wPermute = VPUIP.PermuteDMA <{mem_perm = #QKV_WEIGHTS_PERM}> inputs(%wRootCopy : !QKVWeightsTileCMX) outputs(%wPermuteBuffer : !QKVWeightsPermuteCMX) -> !QKVWeightsPermuteCMX
    %wPermuteHi = VPUIP.SubView %wPermute [0, 32, 0, 0] [1, 32, 6, 64] : !QKVWeightsPermuteCMX to !QKVWeightsPermuteHalfCMX
    %wPermuteLo = VPUIP.SubView %wPermute [0, 0, 0, 0] [1, 32, 6, 64] : !QKVWeightsPermuteCMX to !QKVWeightsPermuteHalfCMX
    %scalesBuffer = memref.alloc() : !QKVScalesCMX
    %scalesCopy = VPUIP.Copy inputs(%scales : !QKVScalesDDR) outputs(%scalesBuffer : !QKVScalesCMX) -> !QKVScalesCMX
    %scalesHi = VPUIP.SubView %scalesCopy [0, 32, 0, 0] [1, 32, 6, 1] : !QKVScalesCMX to !QKVScaleHalfCMX
    %scalesLo = VPUIP.SubView %scalesCopy [0, 0, 0, 0] [1, 32, 6, 1] : !QKVScalesCMX to !QKVScaleHalfCMX
    %wDqBuffer = memref.alloc() : !QKVWeightsDequantCMX
    %wDqHiBuffer = VPUIP.SubView %wDqBuffer [0, 32, 0, 0] [1, 32, 6, 64] : !QKVWeightsDequantCMX to !QKVWeightsDequantHalfCMX
    %wDqLoBuffer = VPUIP.SubView %wDqBuffer [0, 0, 0, 0] [1, 32, 6, 64] : !QKVWeightsDequantCMX to !QKVWeightsDequantHalfCMX
    %wDq:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
      inputs(%wPermuteLo as %wDqInputLo: !QKVWeightsPermuteHalfCMX, %scalesLo as %wDqScaleLo: !QKVScaleHalfCMX, %wPermuteHi as %wDqInputHi: !QKVWeightsPermuteHalfCMX, %scalesHi as %wDqScaleHi: !QKVScaleHalfCMX)
      outputs(%wDqLoBuffer as %wDqOutputLo: !QKVWeightsDequantHalfCMX, %wDqHiBuffer as %wDqOutputHi: !QKVWeightsDequantHalfCMX) on tile 0 -> (!QKVWeightsDequantHalfCMX, !QKVWeightsDequantHalfCMX) {
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%wDqInputLo, %wDqScaleLo, %wDqOutputLo) : !QKVWeightsPermuteHalfCMX, !QKVScaleHalfCMX, !QKVWeightsDequantHalfCMX
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%wDqInputHi, %wDqScaleHi, %wDqOutputHi) : !QKVWeightsPermuteHalfCMX, !QKVScaleHalfCMX, !QKVWeightsDequantHalfCMX
    }
    %wConcat = VPUIP.ConcatView inputs(%wDq#0, %wDq#1 : !QKVWeightsDequantHalfCMX, !QKVWeightsDequantHalfCMX) outputs(%wDqBuffer : !QKVWeightsDequantCMX) -> !QKVWeightsDequantCMX
    %wReshape = VPUIP.GenericReshape inputs(%wConcat : !QKVWeightsDequantCMX) -> !QKVWeightsCMXRaw
    %w = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%wReshape : !QKVWeightsCMXRaw) -> !QKVWeightsCMX
    %wSiblingRootCopyBuffer = memref.alloc() : !QKVWeightsTileCMX
    %wSiblingRootCopy = VPUIP.Copy inputs(%wSiblingRootSlice : !QKVWeightsTileDDR) outputs(%wSiblingRootCopyBuffer : !QKVWeightsTileCMX) -> !QKVWeightsTileCMX
    %wSiblingPermuteBuffer = memref.alloc() : !QKVWeightsPermuteCMX
    %wSiblingPermute = VPUIP.PermuteDMA <{mem_perm = #QKV_WEIGHTS_PERM}> inputs(%wSiblingRootCopy : !QKVWeightsTileCMX) outputs(%wSiblingPermuteBuffer : !QKVWeightsPermuteCMX) -> !QKVWeightsPermuteCMX
    %wSiblingReshape = VPUIP.GenericReshape inputs(%wSiblingPermute : !QKVWeightsPermuteCMX) -> !QKVWeightsCMXRaw
    %wSibling = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%wSiblingReshape : !QKVWeightsCMXRaw) -> !QKVWeightsCMX
    %actSlice0 = VPUIP.SubView %rmsDDRConcat [0, 0, 0, 0] [1, 1, 64, 384] : !QKVLayerNormRootDDR to !QKVLayerNormTileDDR
    %actFlat0 = VPUIP.GenericReshape inputs(%actSlice0 : !QKVLayerNormTileDDR) -> !QKVActFlatDDR
    %actPermute0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #QKV_ACT_PERM} inputs(%actFlat0 : !QKVActFlatDDR) -> !QKVActPermuteDDR
    %actDDR0 = VPUIP.GenericReshape inputs(%actPermute0 : !QKVActPermuteDDR) -> !QKVActDDR
    %act0Buffer = memref.alloc() : !QKVActCMX
    %act0 = VPUIP.Copy inputs(%actDDR0 : !QKVActDDR) outputs(%act0Buffer : !QKVActCMX) -> !QKVActCMX
    %conv0Buffer = memref.alloc() : !QKVOutCMX
    %conv0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%act0 : !QKVActCMX) weights(%w : !QKVWeightsCMX) parent_input(%act0 : !QKVActCMX) parent_output(%conv0Buffer : !QKVOutCMX) outputs(%conv0Buffer : !QKVOutCMX) -> !QKVOutCMX variants : {
      DPUTask {inEnd = [7, 7, 383], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 7, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %copyOut0 = VPUIP.Copy inputs(%conv0 : !QKVOutCMX) outputs(%out0 : !QKVOutDDR) -> !QKVOutDDR

    %actAnchorBuffer = memref.alloc() : !QKVActCMX
    %actAnchor = VPUIP.Copy inputs(%actDDR0 : !QKVActDDR) outputs(%actAnchorBuffer : !QKVActCMX) -> !QKVActCMX
    %convAnchorBuffer = memref.alloc() : !QKVOutCMX
    %convAnchor = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%actAnchor : !QKVActCMX) weights(%w : !QKVWeightsCMX) parent_input(%actAnchor : !QKVActCMX) parent_output(%convAnchorBuffer : !QKVOutCMX) outputs(%convAnchorBuffer : !QKVOutCMX) -> !QKVOutCMX variants : {
      DPUTask {inEnd = [7, 7, 383], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 7, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %tmpOutAnchor = memref.alloc() : !QKVOutDDR
    %copyOutAnchor = VPUIP.Copy inputs(%convAnchor : !QKVOutCMX) outputs(%tmpOutAnchor : !QKVOutDDR) -> !QKVOutDDR

    %actSliceNext = VPUIP.SubView %rmsDDRConcat [0, 0, 64, 0] [1, 1, 64, 384] : !QKVLayerNormRootDDR to !QKVLayerNormTileDDR
    %actFlatNext = VPUIP.GenericReshape inputs(%actSliceNext : !QKVLayerNormTileDDR) -> !QKVActFlatDDR
    %actPermuteNext = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #QKV_ACT_PERM} inputs(%actFlatNext : !QKVActFlatDDR) -> !QKVActPermuteDDR
    %actDDRNext = VPUIP.GenericReshape inputs(%actPermuteNext : !QKVActPermuteDDR) -> !QKVActDDR
    %actNextBuffer = memref.alloc() : !QKVActCMX
    %actNext = VPUIP.Copy inputs(%actDDRNext : !QKVActDDR) outputs(%actNextBuffer : !QKVActCMX) -> !QKVActCMX
    %convNextBuffer = memref.alloc() : !QKVOutCMX
    %convNext = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%actNext : !QKVActCMX) weights(%wSibling : !QKVWeightsCMX) parent_input(%actNext : !QKVActCMX) parent_output(%convNextBuffer : !QKVOutCMX) outputs(%convNextBuffer : !QKVOutCMX) -> !QKVOutCMX variants : {
      DPUTask {inEnd = [7, 7, 383], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 7, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %tmpOut = memref.alloc() : !QKVOutDDR
    %copyOutNext = VPUIP.Copy inputs(%convNext : !QKVOutCMX) outputs(%tmpOut : !QKVOutDDR) -> !QKVOutDDR

    %act1Buffer = memref.alloc() : !QKVActCMX
    %act1 = VPUIP.Copy inputs(%actDDR0 : !QKVActDDR) outputs(%act1Buffer : !QKVActCMX) -> !QKVActCMX
    %conv1Buffer = memref.alloc() : !QKVOutCMX
    %conv1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%act1 : !QKVActCMX) weights(%w : !QKVWeightsCMX) parent_input(%act1 : !QKVActCMX) parent_output(%conv1Buffer : !QKVOutCMX) outputs(%conv1Buffer : !QKVOutCMX) -> !QKVOutCMX variants : {
      DPUTask {inEnd = [7, 7, 383], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 7, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %copyOut1 = VPUIP.Copy inputs(%conv1 : !QKVOutCMX) outputs(%out1 : !QKVOutDDR) -> !QKVOutDDR

    %actSliceNext2 = VPUIP.SubView %rmsDDRConcat [0, 0, 128, 0] [1, 1, 64, 384] : !QKVLayerNormRootDDR to !QKVLayerNormTileDDR
    %actFlatNext2 = VPUIP.GenericReshape inputs(%actSliceNext2 : !QKVLayerNormTileDDR) -> !QKVActFlatDDR
    %actPermuteNext2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #QKV_ACT_PERM} inputs(%actFlatNext2 : !QKVActFlatDDR) -> !QKVActPermuteDDR
    %actDDRNext2 = VPUIP.GenericReshape inputs(%actPermuteNext2 : !QKVActPermuteDDR) -> !QKVActDDR
    %actNext2Buffer = memref.alloc() : !QKVActCMX
    %actNext2 = VPUIP.Copy inputs(%actDDRNext2 : !QKVActDDR) outputs(%actNext2Buffer : !QKVActCMX) -> !QKVActCMX
    %convNext2Buffer = memref.alloc() : !QKVOutCMX
    %convNext2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%actNext2 : !QKVActCMX) weights(%w : !QKVWeightsCMX) parent_input(%actNext2 : !QKVActCMX) parent_output(%convNext2Buffer : !QKVOutCMX) outputs(%convNext2Buffer : !QKVOutCMX) -> !QKVOutCMX variants : {
      DPUTask {inEnd = [7, 7, 383], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 7, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %tmpOut2 = memref.alloc() : !QKVOutDDR
    %copyOutNext2 = VPUIP.Copy inputs(%convNext2 : !QKVOutCMX) outputs(%tmpOut2 : !QKVOutDDR) -> !QKVOutDDR

    %act2Buffer = memref.alloc() : !QKVActCMX
    %act2 = VPUIP.Copy inputs(%actDDR0 : !QKVActDDR) outputs(%act2Buffer : !QKVActCMX) -> !QKVActCMX
    %conv2Buffer = memref.alloc() : !QKVOutCMX
    %conv2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%act2 : !QKVActCMX) weights(%w : !QKVWeightsCMX) parent_input(%act2 : !QKVActCMX) parent_output(%conv2Buffer : !QKVOutCMX) outputs(%conv2Buffer : !QKVOutCMX) -> !QKVOutCMX variants : {
      DPUTask {inEnd = [7, 7, 383], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 7, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %copyOut2 = VPUIP.Copy inputs(%conv2 : !QKVOutCMX) outputs(%out2 : !QKVOutDDR) -> !QKVOutDDR

    return %copyOut0, %copyOut1, %copyOut2 : !QKVOutDDR, !QKVOutDDR, !QKVOutDDR

    // CHECK:       [[RMS_INPUT_TILE:%.+]] = VPUIP.SubView [[RMS_INPUT]] [0, 0, 0, 0] [1, 1, 256, 384]
    // CHECK:       [[RMS_INPUT_HI:%.+]] = VPUIP.SubView [[RMS_INPUT_TILE]] [0, 0, 128, 0] [1, 1, 128, 384]
    // CHECK:       [[RMS_INPUT_HI_COPY:%.+]] = VPUIP.Copy inputs([[RMS_INPUT_HI]]
    // CHECK:       [[RMS_INPUT_LO:%.+]] = VPUIP.SubView [[RMS_INPUT_TILE]] [0, 0, 0, 0] [1, 1, 128, 384]
    // CHECK:       [[RMS_INPUT_LO_COPY:%.+]] = VPUIP.Copy inputs([[RMS_INPUT_LO]]
    // CHECK:       [[RMS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RMS
    // CHECK-SAME:      inputs([[RMS_INPUT_LO_COPY]] as
    // CHECK-SAME:      [[RMS_INPUT_HI_COPY]] as
    // CHECK:       [[RMS_DDR:%.+]] = memref.alloc() : memref<1x1x256x384xf16, @DDR>
    // CHECK:       [[RMS_OUTPUT_LO:%.+]] = VPUIP.SubView [[RMS_DDR]] [0, 0, 0, 0] [1, 1, 128, 384]
    // CHECK:       [[RMS_DDR_LO:%.+]] = VPUIP.Copy inputs([[RMS]]#0
    // CHECK:       [[RMS_OUTPUT_HI:%.+]] = VPUIP.SubView [[RMS_DDR]] [0, 0, 128, 0] [1, 1, 128, 384]
    // CHECK:       [[RMS_DDR_HI:%.+]] = VPUIP.Copy inputs([[RMS]]#1
    // CHECK:       [[RMS_DDR_CONCAT:%.+]] = VPUIP.ConcatView inputs([[RMS_DDR_LO]], [[RMS_DDR_HI]]
    // CHECK-SAME:      outputs([[RMS_DDR]]

    // CHECK:       [[WEIGHTS:%.+]] = VPUIP.GenericReshape inputs([[WEIGHTS_PARAM]]
    // CHECK:       [[W_ROOT_SLICE:%.+]] = VPUIP.SubView [[WEIGHTS]] [0, 0, 0, 0] [1, 6, 64, 64]
    // CHECK:       [[W_SIBLING_ROOT_SLICE:%.+]] = VPUIP.SubView [[WEIGHTS]] [0, 0, 64, 0] [1, 6, 64, 64]
    // CHECK:       [[W_ROOT_COPY:%.+]] = VPUIP.Copy inputs([[W_ROOT_SLICE]]
    // CHECK:       [[W_PERMUTE:%.+]] = VPUIP.PermuteDMA
    // CHECK-SAME:      inputs([[W_ROOT_COPY]]
    // CHECK:       [[W_PERMUTE_LO:%.+]] = VPUIP.SubView [[W_PERMUTE]] [0, 0, 0, 0] [1, 32, 6, 64]
    // CHECK:       [[SCALES_COPY:%.+]] = VPUIP.Copy inputs([[SCALES]]
    // CHECK:       [[W_DQ:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK-SAME:      inputs([[W_PERMUTE_LO]] as
    // CHECK:       [[W_CONCAT:%.+]] = VPUIP.ConcatView inputs([[W_DQ]]#0, [[W_DQ]]#1
    // CHECK:       [[W_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[W_CONCAT]]
    // CHECK:       [[W:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[W_RESHAPE]]
    // CHECK:       [[W_SIBLING_ROOT_COPY:%.+]] = VPUIP.Copy inputs([[W_SIBLING_ROOT_SLICE]]
    // CHECK:       [[W_SIBLING_PERMUTE:%.+]] = VPUIP.PermuteDMA
    // CHECK-SAME:      inputs([[W_SIBLING_ROOT_COPY]]
    // CHECK:       [[W_SIBLING_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[W_SIBLING_PERMUTE]]
    // CHECK:       [[W_SIBLING:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[W_SIBLING_RESHAPE]]

    // CHECK:       [[ACT_SLICE0:%.+]] = VPUIP.SubView [[RMS_DDR_CONCAT]] [0, 0, 0, 0] [1, 1, 64, 384]
    // CHECK:       [[ACT_FLAT0:%.+]] = VPUIP.GenericReshape inputs([[ACT_SLICE0]]
    // CHECK:       [[ACT_PERMUTE0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #map} inputs([[ACT_FLAT0]]
    // CHECK:       [[ACT_DDR0:%.+]] = VPUIP.GenericReshape inputs([[ACT_PERMUTE0]]
    // CHECK:       [[ACT0:%.+]] = VPUIP.Copy inputs([[ACT_DDR0]]
    // CHECK:       [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ACT0]]
    // CHECK-SAME:      weights([[W]]
    // CHECK:       [[COPY_OUT0:%.+]] = VPUIP.Copy inputs([[CONV0]]
    // CHECK:       [[CONV_ANCHOR:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ACT0]]
    // CHECK-SAME:      weights([[W]]

    // CHECK:       [[ACT_SLICE_NEXT:%.+]] = VPUIP.SubView [[RMS_DDR_CONCAT]] [0, 0, 64, 0] [1, 1, 64, 384]
    // CHECK:       [[ACT_FLAT_NEXT:%.+]] = VPUIP.GenericReshape inputs([[ACT_SLICE_NEXT]]
    // CHECK:       [[ACT_PERMUTE_NEXT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #map} inputs([[ACT_FLAT_NEXT]]
    // CHECK:       [[ACT_DDR_NEXT:%.+]] = VPUIP.GenericReshape inputs([[ACT_PERMUTE_NEXT]]
    // CHECK:       [[ACT_NEXT:%.+]] = VPUIP.Copy inputs([[ACT_DDR_NEXT]]
    // CHECK:       [[CONV_NEXT:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ACT_NEXT]]
    // CHECK-SAME:      weights([[W_SIBLING]]
    // CHECK:       [[COPY_OUT_NEXT:%.+]] = VPUIP.Copy inputs([[CONV_NEXT]]
    // CHECK:       [[ACT1:%.+]] = VPUIP.Copy inputs([[ACT_DDR0]]
    // CHECK:       [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ACT1]]
    // CHECK-SAME:      weights([[W]]

    // CHECK:       [[COPY_OUT1:%.+]] = VPUIP.Copy inputs([[CONV1]]
    // CHECK:       [[ACT_SLICE_NEXT2:%.+]] = VPUIP.SubView [[RMS_DDR_CONCAT]] [0, 0, 128, 0] [1, 1, 64, 384]
    // CHECK:       [[ACT_FLAT_NEXT2:%.+]] = VPUIP.GenericReshape inputs([[ACT_SLICE_NEXT2]]
    // CHECK:       [[ACT_PERMUTE_NEXT2:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #map} inputs([[ACT_FLAT_NEXT2]]
    // CHECK:       [[ACT_DDR_NEXT2:%.+]] = VPUIP.GenericReshape inputs([[ACT_PERMUTE_NEXT2]]
    // CHECK:       [[ACT_NEXT2:%.+]] = VPUIP.Copy inputs([[ACT_DDR_NEXT2]]
    // CHECK:       [[CONV_NEXT2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ACT_NEXT2]]
    // CHECK-SAME:      weights([[W]]
    // CHECK:       [[COPY_OUT_NEXT2:%.+]] = VPUIP.Copy inputs([[CONV_NEXT2]]
    // CHECK:       [[ACT2:%.+]] = VPUIP.Copy inputs([[ACT_DDR0]]
    // CHECK:       [[CONV2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ACT2]]
    // CHECK-SAME:      weights([[W]]
    // CHECK:       [[COPY_OUT2:%.+]] = VPUIP.Copy inputs([[CONV2]]

    // CHECK:       return [[COPY_OUT0]], [[COPY_OUT1]], [[COPY_OUT2]]
}
