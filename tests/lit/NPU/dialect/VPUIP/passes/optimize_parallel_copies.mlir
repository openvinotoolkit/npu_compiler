//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelConstCopies(
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = const.Declare memref<1x16x112x112xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<1x16x112x112xf16>, [#const.Reorder<#NHWC>]
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %6 = VPUIP.Copy
            inputs(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %8 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %10 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%9 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%9 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %11 = VPUIP.Copy
            inputs(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

}

// CHECK-LABEL: func.func @OptimizeParallelConstCopies

// CHECK:       [[VAR0:%.+]] =  const.Declare memref<1x16x112x112xf16, #NHWC, @DDR>
// CHECK:       [[VAR1:%.+]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR3:%.+]] =  VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK:       [[VAR4:%.+]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR5:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR4]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR6:%.+]] =  VPUIP.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelActCopies(
        %input: memref<1x32x1x12544xf16, #NHWC, @DDR>,
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %weights_table = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>

    %reshape = VPUIP.GenericReshape inputs(%input: memref<1x32x1x12544xf16, #NHWC, @DDR>) -> memref<1x32x112x112xf16, #NHWC, @DDR>
    %subview1 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc_in1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %copy_in1 = VPUIP.Copy
            inputs(%subview1 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
            outputs(%alloc_in1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %alloc_out1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %nce1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%copy_in1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%copy_in1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%alloc_out1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%alloc_out1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants : {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %copy_out1 = VPUIP.Copy
            inputs(%nce1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %subview2 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc_in2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %copy_in2 = VPUIP.Copy
            inputs(%subview2 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
            outputs(%alloc_in2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %alloc_out2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %nce2 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%copy_in2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%copy_in2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%alloc_out2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%alloc_out2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants : {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %copy_out2 = VPUIP.Copy
            inputs(%nce2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output2 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %copy_out1, %copy_out1 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

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
func.func @OptimizeParallelSubViewInputCopies(
        %input: memref<1x16x112x113xf16, #NHWC, @DDR>,
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>

    %2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.Copy
            inputs(%6 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %8 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %10 = VPUIP.Copy
            inputs(%9 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
            outputs(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %11 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %12 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%11 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%11 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %13 = VPUIP.Copy
            inputs(%12 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %7, %13 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>
}

// CHECK-LABEL: func.func @OptimizeParallelSubViewInputCopies

// CHECK:       %[[ALLOC_CMX:.+]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[SUBVIEW:.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 112, 112] :
// CHECK-SAME:        memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>

// CHECK:       %[[DDR2CMX:.+]] = VPUIP.Copy inputs(%[[SUBVIEW]] :
// CHECK-SAME:        memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[ALLOC_NCE_CMX_1:.+]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[NCE_1:.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input(%[[DDR2CMX]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%[[ALLOC_NCE_CMX_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[CMX2DDR_1:.+]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[NCE_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK:       %[[ALLOC_NCE_CMX_2:.+]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[NCE_2:.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input(%[[DDR2CMX]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%[[ALLOC_NCE_CMX_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[CMX2DDR_2:.+]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[NCE_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK:       return %[[CMX2DDR_1]], %[[CMX2DDR_2]] : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x32x28x28xf16, #NHWC, @DDR>
!Output_CMX = memref<1x32x28x28xf16, #NHWC, @CMX_NN>
!Output = memref<1x32x28x28xf16,  #NHWC>
!Weights_CMX = memref<128x32x1x1xf16, #NHWC, @CMX_NN>
!Output_CONV = memref<1x128x28x28xf16, #NHWC, @CMX_NN>
!Weights_table_CMX = memref<128x1x1x4xsi32, @CMX_NN>

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
    %8 = memref.alloc() : !Weights_table_CMX
    %9 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
           }
        input(%5 : !CopyOutput_Distributed)
        weights(%7 : !Weights_CMX)
        weight_table(%8 : !Weights_table_CMX)
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
    %12 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
           }
        input(%10 : !CopyOutput_Distributed)
        weights(%7 : !Weights_CMX)
        weight_table(%8 : !Weights_table_CMX)
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
//CHECK-SAME:  inputs([[COMMON_ROOT]] : memref<1x32x28x28xf16, #NHWC, @DDR>)
//CHECK-SAME:  outputs([[DISTR_BUFFER0]] : !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

//CHECK: [[BRANCH1_CONSUMER:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
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
    func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
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
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
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
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) outputs(%alloc_3 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
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

func.func @OptimizeParallelSubViewWithClusterTilingCopies(
        %input: memref<1x144x128x128xf16, #NHWC, @DDR>,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x144x128x128xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
        -> memref<1x144x128x128xf16, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%3 : !OutputDistributed) -> !OutputDistributed
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV> }
        input(%4 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.Copy
        inputs(%7 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%8 : !OutputDistributed) -> !OutputDistributed
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV> }
        input(%9 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%9 : !OutputDistributed)
        parent_output(%10 : !OutputDistributed)
        outputs(%10 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

    return %6, %11 : !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.+]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.+]] = IERT.Convert inputs(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>) outputs([[IN_BUFFER]] : memref<1x144x128x128xf16, #NHWC, @DDR>) -> memref<1x144x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]]
    // CHECK-SAME:      weights(%arg1
    // CHECK-SAME:      weight_table(%arg2
    // CHECK-SAME:      outputs([[BUFFER_2]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUFFER_3:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]]
    // CHECK-SAME:      weights(%arg1
    // CHECK-SAME:      weight_table(%arg2
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

func.func @NotOptimizeParallelClusterTilingCopiesWithSubviewHasDiffOffset(
        %arg0: memref<1x1x512x1xf16, @DDR>) -> !OutputDistributed {
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x1x512x1xf16, @DDR>) -> memref<1x1x512x1xf16, #NHWC, @DDR>
    %2 = VPURT.AllocDistributed -> !OutputDistributed

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 1, 512, 1] : !OutputDistributed to !SubViewDistributed
    %4 = VPUIP.Copy
        inputs(%1 : memref<1x1x512x1xf16, #NHWC, @DDR>)
        outputs(%3 : !SubViewDistributed) -> !SubViewDistributed

    %5 = VPUIP.SubView %2 [0, 1, 0, 0] [1, 1, 512, 1] : !OutputDistributed to !SubViewDistributed
    %6 = VPUIP.Copy
        inputs(%1 : memref<1x1x512x1xf16, #NHWC, @DDR>)
        outputs(%5 : !SubViewDistributed) -> !SubViewDistributed

    %7 = VPUIP.ConcatView
            inputs(%4, %6 : !SubViewDistributed, !SubViewDistributed)
            outputs(%2 : !OutputDistributed) -> !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x512x1xf16, @DDR>) -> memref<1x1x512x1xf16, #NHWC, @DDR>
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

!IODataDDRType = memref<1x32x28x28xf16, #NHWC, @DDR>
!IOSMDDRType = memref<1x32x28x28xi1, #NHWC, @DDR>

!IODataCMXType = memref<1x32x28x28xf16, #NHWC, @CMX_NN>
!IOSMCMXType = memref<1x32x28x28xi1, #NHWC, @CMX_NN>

!Weights_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!Weights_table_CMX = memref<32x1x1x4xsi32, @CMX_NN>

!IODataDistrType = !VPUIP.DistributedBuffer<
  1x32x28x28xf16, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!IOSMDistrType = !VPUIP.DistributedBuffer<
  1x32x28x28xi1, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

// CHECK-LABEL: @OptimizeParallelMulticlusterCopiesSparse
func.func @OptimizeParallelMulticlusterCopiesSparse()
        -> (!IODataDistrType, !IOSMDistrType, !IODataDistrType, !IOSMDistrType) {
    %0 = memref.alloc() : !IODataCMXType
    %1 = memref.alloc() : !IOSMCMXType

    %3 = memref.alloc() : !IODataDDRType
    %4 = memref.alloc() : !IOSMDDRType
    %6 = VPUIP.Copy
        inputs(%0 : !IODataCMXType)
        outputs(%3 : !IODataDDRType) -> !IODataDDRType
    %sm6 = VPUIP.Copy
        inputs(%1 : !IOSMCMXType)
        outputs(%4 : !IOSMDDRType) -> !IOSMDDRType

    %7 = VPURT.AllocDistributed -> !IODataDistrType
    %8 = VPURT.AllocDistributed -> !IOSMDistrType
    %in_data_0 = VPUIP.Copy
        inputs(%6 : !IODataDDRType)
        outputs(%7 : !IODataDistrType) -> !IODataDistrType
    %in_sm_0 = VPUIP.Copy
        inputs(%sm6 : !IOSMDDRType)
        outputs(%8 : !IOSMDistrType) -> !IOSMDistrType

    %out_data_0 = VPURT.AllocDistributed -> !IODataDistrType
    %out_sm_0 = VPURT.AllocDistributed -> !IOSMDistrType

    %14 = memref.alloc() : !Weights_CMX
    %15 = memref.alloc() : !Weights_table_CMX

    %16:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          }
          input(%in_data_0 : !IODataDistrType)
          input_sparsity_map(%in_sm_0 : !IOSMDistrType)
          weights(%14 : !Weights_CMX)
          weight_table(%15 : !Weights_table_CMX)
          parent_input(%in_data_0 : !IODataDistrType)
          parent_input_sparsity_map(%in_sm_0 : !IOSMDistrType)
          parent_output(%out_data_0 : !IODataDistrType)
          parent_output_sparsity_map(%out_sm_0 : !IOSMDistrType)
          outputs(%out_data_0 : !IODataDistrType)
          output_sparsity_map(%out_sm_0 : !IOSMDistrType)
            -> !IODataDistrType, !IOSMDistrType variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
       } PPE :  { }

    %17 = VPURT.AllocDistributed -> !IODataDistrType
    %18 =  VPURT.AllocDistributed -> !IOSMDistrType
    %in_data_1 = VPUIP.Copy
        inputs(%6 : !IODataDDRType)
        outputs(%17 : !IODataDistrType) -> !IODataDistrType
    %in_sm_1 = VPUIP.Copy
        inputs(%sm6 : !IOSMDDRType)
        outputs(%18 : !IOSMDistrType) -> !IOSMDistrType

    %out_data_1 = VPURT.AllocDistributed -> !IODataDistrType
    %out_sm_1 =  VPURT.AllocDistributed -> !IOSMDistrType

      %24:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          }
          input(%in_data_1 : !IODataDistrType)
          input_sparsity_map(%in_sm_1 : !IOSMDistrType)
          weights(%14 : !Weights_CMX)
          weight_table(%15 : !Weights_table_CMX)
          parent_input(%in_data_1 : !IODataDistrType)
          parent_input_sparsity_map(%in_sm_1 : !IOSMDistrType)
          parent_output(%out_data_1 : !IODataDistrType)
          parent_output_sparsity_map(%out_sm_1 : !IOSMDistrType)
          outputs(%out_data_1 : !IODataDistrType)
          output_sparsity_map(%out_sm_1 : !IOSMDistrType)
            -> !IODataDistrType, !IOSMDistrType variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  { }

    return %16#0, %16#1, %24#0, %24#1: !IODataDistrType, !IOSMDistrType, !IODataDistrType, !IOSMDistrType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x32x28x28xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x32x28x28xi1, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x32x28x28xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x32x28x28xi1, #NHWC, @DDR>

    // CHECK:       [[COMMON_ROOT:%.+]] = VPUIP.Copy inputs([[BUFF_0_DATA]]
    // CHECK-SAME:      outputs([[BUFF_1_DATA]]

    // CHECK:       [[COMMON_ROOT_SM:%.+]] = VPUIP.Copy inputs([[BUFF_0_SM]]
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_0:%.+]] = VPUIP.Copy inputs([[COMMON_ROOT]]
    // CHECK-SAME:      outputs([[BUFF_2_DATA]]

    // CHECK:       [[SM_0:%.+]] = VPUIP.Copy inputs([[COMMON_ROOT_SM]]
    // CHECK-SAME:      outputs([[BUFF_2_SM]]

    // CHECK:       [[DATA_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE0_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]]
    // CHECK-SAME:      outputs([[DATA_1]]

    // CHECK:       [[DATA_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE1_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]]
    // CHECK-SAME:      outputs([[DATA_3]]

    // CHECK: return [[NCE0_U]]#0, [[NCE0_U]]#1, [[NCE1_U]]#0, [[NCE1_U]]#1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataDDRType = memref<1x144x128x128xf16, #NHWC, @DDR>
!ISMDDRType = memref<1x144x128x128xi1, #NHWC, @DDR>

!IDataHalfCMXType = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!ISMHalfCMXType = memref<1x144x64x128xi1, #NHWC, @CMX_NN>

!ODistrDataType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSMType = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

// CHECK-LABEL: @OptimizeParallelSubViewWithClusterTilingCopiesSparse
func.func @OptimizeParallelSubViewWithClusterTilingCopiesSparse(
        %input: !IDataDDRType,
        %input_sm: !ISMDDRType,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType) {

    %0 = memref.alloc() : !IDataDDRType
    %1 = memref.alloc() : !ISMDDRType

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%0 : !IDataDDRType) -> !IDataDDRType
    %sm3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%1 : !ISMDDRType) -> !ISMDDRType

    %4 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm4 = VPUIP.SubView %sm3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %5 = VPURT.AllocDistributed -> !ODistrDataType
    %6 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_0 = VPUIP.Copy
        inputs(%4 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%5 : !ODistrDataType) -> !ODistrDataType
    %in_sm_0 = VPUIP.Copy
        inputs(%sm4 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%6 : !ODistrSMType) -> !ODistrSMType

    %out_data_0 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_0 = VPURT.AllocDistributed -> !ODistrSMType

    %12:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in_data_0 : !ODistrDataType)
        input_sparsity_map(%in_sm_0 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data_0 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_0 : !ODistrSMType)
        parent_output(%out_data_0 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_0 : !ODistrSMType)
        outputs(%out_data_0 : !ODistrDataType)
        output_sparsity_map(%out_sm_0 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    %13 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm13 = VPUIP.SubView %sm3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %14 = VPURT.AllocDistributed -> !ODistrDataType
    %15 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_1 = VPUIP.Copy
        inputs(%13 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%14 : !ODistrDataType) -> !ODistrDataType
    %in_sm_1 = VPUIP.Copy
        inputs(%sm13 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%15 : !ODistrSMType) -> !ODistrSMType

    %out_data_1 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_1 = VPURT.AllocDistributed -> !ODistrSMType

    %21:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in_data_1 : !ODistrDataType)
        input_sparsity_map(%in_sm_1 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data_1 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_1 : !ODistrSMType)
        parent_output(%out_data_1 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_1 : !ODistrSMType)
        outputs(%out_data_1 : !ODistrDataType)
        output_sparsity_map(%out_sm_1 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    return %12#0, %12#1, %21#0, %21#1 : !ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x144x128x128xi1, #NHWC, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_DATA]]
    // CHECK-SAME:      -> memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[PERMUTE_SM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_SM]]
    // CHECK-SAME:      -> memref<1x144x128x128xi1, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[PERMUTE_SM]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]]
    // CHECK-SAME:      outputs([[BUFF_1_DATA]]

    // CHECK:       [[SM_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_SM]]
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[DATA_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE0_U:%.+]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[DATA_0]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_output([[DATA_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-SAME:      outputs([[DATA_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       [[DATA_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE1_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]]
    // CHECK-SAME:      outputs([[DATA_3]]
    // CHECK:       return [[NCE0_U]]#0, [[NCE0_U]]#1, [[NCE1_U]]#0, [[NCE1_U]]#1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataDDRType = memref<1x144x128x128xf16, #NHWC, @DDR>
!ISMDDRType = memref<1x144x128x128xi1, #NHWC, @DDR>

!IDataHalfCMXType = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!ISMHalfCMXType = memref<1x144x64x128xi1, #NHWC, @CMX_NN>

!ODistrDataType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSMType = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

// CHECK-LABEL: @NotOptimizeParallelClusterTilingCopiesWithSubviewHasDiffOffsetSparse
func.func @NotOptimizeParallelClusterTilingCopiesWithSubviewHasDiffOffsetSparse(
        %input: !IDataDDRType,
        %input_sm: !ISMDDRType,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType) {

    %0 = memref.alloc() : !IDataDDRType
    %1 = memref.alloc() : !ISMDDRType

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%0 : !IDataDDRType) -> !IDataDDRType
    %sm3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%1 : !ISMDDRType) -> !ISMDDRType

    %4 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm4 = VPUIP.SubView %sm3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %5 = VPURT.AllocDistributed -> !ODistrDataType
    %6 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_0 = VPUIP.Copy
        inputs(%4 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%5 : !ODistrDataType) -> !ODistrDataType
    %in_sm_0 = VPUIP.Copy
        inputs(%sm4 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%6 : !ODistrSMType) -> !ODistrSMType

    %out_data_0 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_0 = VPURT.AllocDistributed -> !ODistrSMType

    %12:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in_data_0 : !ODistrDataType)
        input_sparsity_map(%in_sm_0 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data_0 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_0 : !ODistrSMType)
        parent_output(%out_data_0 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_0 : !ODistrSMType)
        outputs(%out_data_0 : !ODistrDataType)
        output_sparsity_map(%out_sm_0 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    %13 = VPUIP.SubView %3 [0, 1, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm13 = VPUIP.SubView %sm3 [0, 1, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %14 = VPURT.AllocDistributed -> !ODistrDataType
    %15 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_1 = VPUIP.Copy
        inputs(%13 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%14 : !ODistrDataType) -> !ODistrDataType
    %in_sm_1 = VPUIP.Copy
        inputs(%sm13 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%15 : !ODistrSMType) -> !ODistrSMType

    %out_data_1 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_1 = VPURT.AllocDistributed -> !ODistrSMType

    %21:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in_data_1 : !ODistrDataType)
        input_sparsity_map(%in_sm_1 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%in_data_1 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_1 : !ODistrSMType)
        parent_output(%out_data_1 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_1 : !ODistrSMType)
        outputs(%out_data_1 : !ODistrDataType)
        output_sparsity_map(%out_sm_1 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    return %12#0, %12#1, %21#0, %21#1 : !ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x144x128x128xi1, #NHWC, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_DATA]]
    // CHECK-SAME:      -> memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[PERMUTE_SM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_SM]]
    // CHECK-SAME:      -> memref<1x144x128x128xi1, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[PERMUTE_SM]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]]
    // CHECK-SAME:      outputs([[BUFF_1_DATA]]
    // CHECK:       [[SM_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_SM]]
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[DATA_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE0_U:%.+]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[DATA_0]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_input_sparsity_map([[SM_0]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      output([[DATA_1]]
    // CHECK-SAME:      output_sparsity_map([[SM_1]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 1, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SUBVIEW_1_SM:%.+]] = VPUIP.SubView [[PERMUTE_SM]] [0, 1, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFF_3_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1]]
    // CHECK-SAME:      outputs([[BUFF_3_DATA]]
    // CHECK:       [[SM_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_SM]]
    // CHECK-SAME:      outputs([[BUFF_3_SM]]

    // CHECK:       [[DATA_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE1_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      input_sparsity_map([[SM_2]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights(%arg2 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg3 : memref<144x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[DATA_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_input_sparsity_map([[SM_2]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_output([[DATA_3]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_output_sparsity_map(%18 : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      outputs([[DATA_3]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      output_sparsity_map(%18 : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       return [[NCE0_U]]#0, [[NCE0_U]]#1, [[NCE1_U]]#0, [[NCE1_U]]#1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

func.func @OptimizeParallelSubViewWithParallelClusterTilingCopies(
        %input: memref<1x144x128x128xf16, #NHWC, @DDR>,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x144x128x128xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
        -> memref<1x144x128x128xf16, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
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
    %6 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV> }
        input(%4 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

	    %102 = VPURT.AllocDistributed -> !OutputDistributed
    %103 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV> }
        input(%101 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%101 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }


    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.Copy
        inputs(%7 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%8 : !OutputDistributed) -> !OutputDistributed
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV> }
        input(%9 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%9 : !OutputDistributed)
        parent_output(%10 : !OutputDistributed)
        outputs(%10 : !OutputDistributed)
            -> !OutputDistributed variants : {
            DPUTask { cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
    } PPE : {
    }

    return %6, %11, %103 : !OutputDistributed, !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.+]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.+]] = IERT.Convert inputs(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>) outputs([[IN_BUFFER]] : memref<1x144x128x128xf16, #NHWC, @DDR>) -> memref<1x144x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights(%arg1 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg2 : memref<144x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_4]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights(%arg1 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg2 : memref<144x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_4]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_6:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights(%arg1 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg2 : memref<144x1x1x4xsi32, @CMX_NN>)
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

// CHECK-LABEL: @OptimizeParallelClusterTilingCopiesWithInPlaceNCEEltwise
func.func @OptimizeParallelClusterTilingCopiesWithInPlaceNCEEltwise(%arg0:
memref<1x128x104x104xf32, #NHWC, @DDR>) -> (!CopyOutputDistributed, !OutputDistributed) {
    %0 = memref.alloc() : memref<1x128x104x104xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%arg0 : memref<1x128x104x104xf32, #NHWC, @DDR>)
        outputs(%0 : memref<1x128x104x104xf16, #NHWC, @DDR>)
        -> memref<1x128x104x104xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 128, 52, 104] :
                memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
        outputs(%3 : !OutputDistributed) -> !OutputDistributed
    %5 = VPURT.AllocDistributed -> !CopyOutputDistributed
    %6 = VPUIP.Copy
        inputs(%4 : !OutputDistributed)
        outputs(%5 : !CopyOutputDistributed) -> !CopyOutputDistributed
    %7 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 128, 52, 104] :
        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %8 = VPUIP.SubView %1 [0, 0, 52, 0] [1, 128, 52, 104] :
        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %9 = VPURT.AllocDistributed -> !OutputDistributed
    %10 = VPUIP.Copy
        inputs(%7 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
        outputs(%9 : !OutputDistributed) -> !OutputDistributed
    %11 = VPURT.AllocDistributed -> !OutputDistributed
    %12 = VPUIP.Copy
        inputs(%8 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
        outputs(%11 : !OutputDistributed) -> !OutputDistributed
    %13 = VPUIP.NCEClusterTask {
                is_inplace = true,
                minimumHardwareExecutionCost = 4294967400 : i64,
                task_type = #VPUIP.nce_task_type<ELTWISE>
             }
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
    // CHECK:       [[MEM_REF:%.+]] = memref.alloc() : memref<1x128x104x104xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.+]] = IERT.Convert inputs(%arg0 : memref<1x128x104x104xf32, #NHWC, @DDR>) outputs([[MEM_REF]] : memref<1x128x104x104xf16, #NHWC, @DDR>) -> memref<1x128x104x104xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 0, 0] [1, 128, 52, 104] :
    // CHECK-SAME:        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
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
    // CHECK-SAME:        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    // CHECK:       [[DISTBUFF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:        inputs([[SUBVIEW_0]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>)
    // CHECK-SAME:        outputs([[DISTBUFF_2]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_3:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 4294967400 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
  func.func private @builtin_PRelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "prelu_fp16.cpp", VPU.kernel_entry = "prelu_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OptimizeConstCopyToSoftKernel() -> (!DistrType, !DistrType)  {
    %cst = const.Declare memref<1x18x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<18xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 18, 1, 1]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %0 = VPURT.AllocDistributed -> !DistrType
    %1 = VPURT.AllocDistributed -> !ConstDistrType
    %2 = VPURT.AllocDistributed -> !DistrType

    %3 = VPUIP.Copy
        inputs(%cst : memref<1x18x1x1xf16, #NHWC>)
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
        inputs(%cst : memref<1x18x1x1xf16, #NHWC>)
        outputs(%6 : !ConstDistrType) -> !ConstDistrType

    %9 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
                inputs(%5 as %arg3: !DistrType,
                        %8 as %arg4: !ConstDistrType)
                outputs(%7 as %arg5: !DistrType) on tile 0 -> !DistrType {
    VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : !DistrType, !ConstDistrType, !DistrType
    }

    return %4, %9 : !DistrType, !DistrType


    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x18x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<18xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 18, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[DISTR_BUFFER0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CST]]
    // CHECK-SAME:          outputs([[DISTR_BUFFER1]]
    // CHECK:           [[SWKERNEL0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
    //CHECK-SAME:           inputs([[DISTR_BUFFER0]] as %arg0
    //CHECK-SAME:           [[COPY0]] as %arg1

    // CHECK:           [[DISTR_BUFFER3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SWKERNEL1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
    // CHECK-SAME:          inputs([[DISTR_BUFFER3]] as %arg0
    // CHECK-SAME:          [[COPY0]] as %arg1

    // CHECK:           return  [[SWKERNEL0]], [[SWKERNEL1]]
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Multiply(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
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
    %cst = const.Declare memref<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    %cst_0 = const.Declare memref<512x128x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<512x128xf32>, [#const.Reshape<[512, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<1x128x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<1x128xf32>, [#const.Reshape<[1, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %alloc_0 = memref.alloc() : memref<1x128x1x1xf16, @CMX_NN>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x128x1x1xf16, @DDR>) outputs(%alloc_0 : memref<1x128x1x1xf16, @CMX_NN>) -> memref<1x128x1x1xf16, @CMX_NN>

    // %cst_1 is nce task input, will not be processed
    %alloc_1 = memref.alloc() : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.Copy inputs(%cst_1 : memref<1x128x1x1xf16, #NHWC>) outputs(%alloc_1 : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x128x1x1xf16, #NHWC, @CMX_NN>

    // %cst_0 is nce task weight, will be processed
    %alloc_2 = memref.alloc() : memref<512x128x1x1xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy inputs(%cst_0 : memref<512x128x1x1xf16, #NHWC>) outputs(%alloc_2 : memref<512x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<512x128x1x1xf16, #NHWC, @CMX_NN>

    // %cst is nce task weight_table, will not be processed
    %alloc_3 = memref.alloc() : memref<512x1x1x4xsi32, @CMX_NN>
    %3 = VPUIP.Copy inputs(%cst : memref<512x1x1x4xsi32>) outputs(%alloc_3 : memref<512x1x1x4xsi32, @CMX_NN>) -> memref<512x1x1x4xsi32, @CMX_NN>

    %alloc_4 = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @CMX_NN>
    %4 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1710 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%1 : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) weights(%2 : memref<512x128x1x1xf16, #NHWC, @CMX_NN>) weight_table(%3 : memref<512x1x1x4xsi32, @CMX_NN>) parent_input(%1 : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) parent_output(%alloc_4 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) outputs(%alloc_4 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x1xf16, #NHWC, @CMX_NN> variants : {
      DPUTask {inEnd = [0, 0, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %5 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%4 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x1xf16, @CMX_NN>

    // %6 is same weight copy as %2, will be removed.
    %alloc_6 = memref.alloc() : memref<512x128x1x1xf16, #NHWC, @CMX_NN>
    %6 = VPUIP.Copy inputs(%cst_0 : memref<512x128x1x1xf16, #NHWC>) outputs(%alloc_6 : memref<512x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<512x128x1x1xf16, #NHWC, @CMX_NN>

    %alloc_7 = memref.alloc() : memref<512x1x1x4xsi32, @CMX_NN>
    %7 = VPUIP.Copy inputs(%cst : memref<512x1x1x4xsi32>) outputs(%alloc_7 : memref<512x1x1x4xsi32, @CMX_NN>) -> memref<512x1x1x4xsi32, @CMX_NN>

    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%0 : memref<1x128x1x1xf16, @CMX_NN>) -> memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    %alloc_8 = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1710 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%8 : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) weights(%6 : memref<512x128x1x1xf16, #NHWC, @CMX_NN>) weight_table(%7 : memref<512x1x1x4xsi32, @CMX_NN>) parent_input(%8 : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) parent_output(%alloc_8 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) outputs(%alloc_8 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x1xf16, #NHWC, @CMX_NN> variants : {
      DPUTask {inEnd = [0, 0, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    %10 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%9 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x1xf16, @CMX_NN>

    %alloc_9 = memref.alloc() : memref<1x512x2x1xf16, @CMX_NN>
    %11 = VPUIP.ConcatView inputs(%5, %10 : memref<1x512x1x1xf16, @CMX_NN>, memref<1x512x1x1xf16, @CMX_NN>) outputs(%alloc_9 : memref<1x512x2x1xf16, @CMX_NN>) -> memref<1x512x2x1xf16, @CMX_NN>

    %alloc_10 = memref.alloc() : memref<1x512x2x1xf16, @DDR>
    %12 = VPUIP.Copy inputs(%11 : memref<1x512x2x1xf16, @CMX_NN>) outputs(%alloc_10 : memref<1x512x2x1xf16, @DDR>) -> memref<1x512x2x1xf16, @DDR>

    return %12 : memref<1x512x2x1xf16, @DDR>

    // CHECK: [[WEIGHT_TABLE:%.+]] = const.Declare memref<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    // CHECK: [[WEIGHT:%.+]] = const.Declare memref<512x128x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<512x128xf32>, [#const.Reshape<[512, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK: [[NCE_INPUT:%.+]] = const.Declare memref<1x128x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<1x128xf32>, [#const.Reshape<[1, 128, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK: [[BUFF:%.+]] = memref.alloc() : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[NCE_INPUT]] : memref<1x128x1x1xf16, #NHWC>) outputs([[BUFF]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x128x1x1xf16, #NHWC, @CMX_NN>

    // CHECK: [[BUFF0:%.+]] = memref.alloc() : memref<512x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[WEIGHT]] : memref<512x128x1x1xf16, #NHWC>) outputs([[BUFF0]] : memref<512x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<512x128x1x1xf16, #NHWC, @CMX_NN>

    // CHECK: [[BUFF1:%.+]] = memref.alloc() : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<512x1x1x4xsi32>) outputs([[BUFF1]] : memref<512x1x1x4xsi32, @CMX_NN>) -> memref<512x1x1x4xsi32, @CMX_NN>

    // CHECK: [[BUFF2:%.+]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[RES0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1710 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME: input([[COPY]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: weights([[COPY0]] : memref<512x128x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: weight_table([[COPY1]] : memref<512x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME: parent_input([[COPY]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: parent_output([[BUFF2]] : memref<1x512x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[BUFF2]] : memref<1x512x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: -> memref<1x512x1x1xf16, #NHWC, @CMX_NN>

    // CHECK: VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[RES0]] : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x1xf16, @CMX_NN>

    // CHECK: [[BUFF3:%.+]] = memref.alloc() : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<512x1x1x4xsi32>) outputs([[BUFF3]] : memref<512x1x1x4xsi32, @CMX_NN>) -> memref<512x1x1x4xsi32, @CMX_NN>

    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs({{%.+}} : memref<1x128x1x1xf16, @CMX_NN>) -> memref<1x128x1x1xf16, #NHWC, @CMX_NN>

    // CHECK: [[BUFF4:%.+]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: VPUIP.NCEClusterTask
    // CHECK-SAME: {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1710 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME: input([[PERMUTE]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: weights([[COPY0]] : memref<512x128x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: weight_table([[COPY2]] : memref<512x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME: parent_input([[PERMUTE]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: parent_output([[BUFF4]] : memref<1x512x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[BUFF4]] : memref<1x512x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: -> memref<1x512x1x1xf16, #NHWC, @CMX_NN>

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
    %alloc = memref.alloc() : memref<1x64x250x250x!qElemType1, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250x!qElemType1, #NHWC, @DDR> to memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>

    %1 = VPURT.AllocDistributed -> !buffDistrType1
    %2 = VPUIP.ViewOp %1 : !buffDistrType1 to !buffDistrType
    %3 = VPURT.AllocDistributed -> !buffDistrType1
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1
    %6 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%3 : !buffDistrType1) weights(%5 : !buffDistrType1) parent_input(%3 : !buffDistrType1) parent_output(%2 : !buffDistrType) outputs(%2 : !buffDistrType) -> !buffDistrType variants : {
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
    %12 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %12 : !buffDistrType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250x!qElemType1, #NHWC, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
    %alloc = memref.alloc() : memref<1x64x250x250x!qElemType1, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250x!qElemType1, #NHWC, @DDR> to memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>
    %1 = VPURT.AllocDistributed -> !buffDistrType1
    %2 = VPUIP.ViewOp %1 : !buffDistrType1 to !buffDistrType
    %3 = VPURT.AllocDistributed -> !buffDistrType1
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%0 : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1
    %6 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%3 : !buffDistrType1) weights(%5 : !buffDistrType1) parent_input(%3 : !buffDistrType1) parent_output(%2 : !buffDistrType) outputs(%2 : !buffDistrType) -> !buffDistrType variants : {
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
    %12 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
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
    %18 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%15 : !buffDistrType1) weights(%17 : !buffDistrType1) parent_input(%15 : !buffDistrType1) parent_output(%14 : !buffDistrType) outputs(%14 : !buffDistrType) -> !buffDistrType variants : {
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
    %24 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%21 : !buffDistrType1) weights(%23 : !buffDistrType1) parent_input(%21 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
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
    %30 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%27 : !buffDistrType1) weights(%29 : !buffDistrType1) parent_input(%27 : !buffDistrType1) parent_output(%26 : !buffDistrType) outputs(%26 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %30 : !buffDistrType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250x!qElemType1, #NHWC, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32x!qElemType1, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
!Weights_CMX = memref<64x64x1x1xf16, #NHWC, @CMX_NN>
!Weights_table_CMX = memref<64x1x1x4xsi32, @CMX_NN>

func.func @OptimizeCopiesConsiderDistanceForConv() -> !buffDistrType {
    %weights = memref.alloc() : !Weights_CMX
    %table = memref.alloc() : !Weights_table_CMX

    %0 = memref.alloc() : memref<1x64x250x250xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 250, 32] : memref<1x64x250x250xf16, #NHWC, @DDR> to memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>
    %2 = VPURT.AllocDistributed -> !buffDistrType1
    %3 = VPUIP.ViewOp %2 : !buffDistrType1 to !buffDistrType
    %4 = VPURT.AllocDistributed -> !buffDistrType1
    %5 = VPUIP.Copy inputs(%1 : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>) outputs(%4 : !buffDistrType1) -> !buffDistrType1

    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%5 : !buffDistrType1) weights(%weights : !Weights_CMX) weight_table(%table : !Weights_table_CMX) parent_input(%5 : !buffDistrType1) parent_output(%3 : !buffDistrType) outputs(%3 : !buffDistrType) -> !buffDistrType variants : {
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
    %12 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%9 : !buffDistrType1) weights(%11 : !buffDistrType1) parent_input(%9 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
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
    %18 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%15 : !buffDistrType1) weights(%17 : !buffDistrType1) parent_input(%15 : !buffDistrType1) parent_output(%14 : !buffDistrType) outputs(%14 : !buffDistrType) -> !buffDistrType variants : {
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
    %24 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%21 : !buffDistrType1) weights(%23 : !buffDistrType1) parent_input(%21 : !buffDistrType1) parent_output(%8 : !buffDistrType) outputs(%8 : !buffDistrType) -> !buffDistrType variants : {
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
    %30 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%27 : !buffDistrType1) weights(%29 : !buffDistrType1) parent_input(%27 : !buffDistrType1) parent_output(%26 : !buffDistrType) outputs(%26 : !buffDistrType) -> !buffDistrType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [31, 62, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 62, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [31, 61, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 61, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %30 : !buffDistrType

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x64x250x250xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 64, 250, 32]

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)

    // CHECK: [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK:     input([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY]] : !VPUIP.DistributedBuffer

    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x250x32xf16, {order = #NHWC, strides = [4000000, 1, 16000, 64]}, @DDR>)
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:     weights([[COPY_1]] : !VPUIP.DistributedBuffer

    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     {is_inplace = true, minimumHardwareExecutionCost = 3865 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
!Weights_table_CMX = memref<128x1x1x4xsi32, @CMX_NN>
func.func @TwoAxisTilingConsiderDistanceSiblingSubview(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR> {
    %cst_0 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_1 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %table = memref.alloc() : !Weights_table_CMX
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, #NHWC, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, #NHWC, @DDR>) -> memref<1x256x256x4xf16, #NHWC, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_1 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%5 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%4 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%5 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_3 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_3 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // %7 will be fused to %5, since it is within cost distance
    %9 = VPUIP.Copy inputs(%7 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_4 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%9 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%8 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%9 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_5 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_5 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %11 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_6 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %12 will not be fused, since it is beyond cost distance
    %12 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_6 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %13 = VPUIP.Copy inputs(%11 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_7 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %14 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%13 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%12 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%13 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %15 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_9 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %16 will not be fused, since it is beyond cost distance
    %16 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_9 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_10 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // %17 will be fused to %13, since it is within cost distance
    %17 = VPUIP.Copy inputs(%15 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_10 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_11 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %18 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%17 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%16 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%17 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %alloc_12 = memref.alloc() : memref<1x256x256x4xf16, #NHWC, @DDR>
    %19 = VPUIP.SubView %alloc_12 [0, 0, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %20 = VPUIP.Copy inputs(%6 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%19 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %21 = VPUIP.SubView %alloc_12 [0, 128, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %22 = VPUIP.Copy inputs(%10 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%21 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %23 = VPUIP.SubView %alloc_12 [0, 0, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %24 = VPUIP.Copy inputs(%14 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%23 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %25 = VPUIP.SubView %alloc_12 [0, 128, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %26 = VPUIP.Copy inputs(%18 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%25 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %27 = VPUIP.ConcatView inputs(%20, %22, %24, %26 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_12 : memref<1x256x256x4xf16, #NHWC, @DDR>) -> memref<1x256x256x4xf16, #NHWC, @DDR>
    %28 = VPUIP.GenericReshape inputs(%27 : memref<1x256x256x4xf16, #NHWC, @DDR>) -> memref<1x256x1024x1xf16, #NHWC, @DDR>
    %29 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} inputs(%28 : memref<1x256x1024x1xf16, #NHWC, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %30 = VPUIP.GenericReshape inputs(%29 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    %31 = VPUIP.Copy inputs(%30 : memref<1x1024x256xf16, @DDR>) outputs(%arg1 : memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    return %31 : memref<1x1024x256xf16, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_1:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_2:%.+]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_1]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_3]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_6:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_7:%.+]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_5]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_4]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_6:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_5]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_6]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}
// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Weights_table_CMX = memref<128x1x1x4xsi32, @CMX_NN>
func.func @TwoAxisTilingConsiderDistanceSiblingCopy(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR> {
    %cst_0 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_1 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %table = memref.alloc() : !Weights_table_CMX
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, #NHWC, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, #NHWC, @DDR>) -> memref<1x256x256x4xf16, #NHWC, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_1 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%4 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%5 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%4 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_3 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // %8 will be fused to %4, since it is within cost distance
    %8 = VPUIP.Copy inputs(%7 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_3 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %9 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_4 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%8 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%8 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_5 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_5 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %11 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_6 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPUIP.Copy inputs(%11 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_6 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %13 will not be fused, since it is beyond cost distance
    %13 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_7 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %14 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%12 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%13 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%12 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %15 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc_9 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // %16 will be fused to %12, since it is within cost distance
    %16 = VPUIP.Copy inputs(%15 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_9 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_10 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %17 will not be fused, since it is beyond cost distance
    %17 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_10 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_11 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %18 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%16 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%17 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%table : !Weights_table_CMX) parent_input(%16 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %alloc_12 = memref.alloc() : memref<1x256x256x4xf16, #NHWC, @DDR>
    %19 = VPUIP.SubView %alloc_12 [0, 0, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %20 = VPUIP.Copy inputs(%6 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%19 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %21 = VPUIP.SubView %alloc_12 [0, 128, 0, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %22 = VPUIP.Copy inputs(%10 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%21 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %23 = VPUIP.SubView %alloc_12 [0, 0, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %24 = VPUIP.Copy inputs(%14 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%23 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %25 = VPUIP.SubView %alloc_12 [0, 128, 128, 0] [1, 128, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %26 = VPUIP.Copy inputs(%18 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%25 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) -> memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %27 = VPUIP.ConcatView inputs(%20, %22, %24, %26 : memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>, memref<1x128x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_12 : memref<1x256x256x4xf16, #NHWC, @DDR>) -> memref<1x256x256x4xf16, #NHWC, @DDR>
    %28 = VPUIP.GenericReshape inputs(%27 : memref<1x256x256x4xf16, #NHWC, @DDR>) -> memref<1x256x1024x1xf16, #NHWC, @DDR>
    %29 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} inputs(%28 : memref<1x256x1024x1xf16, #NHWC, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %30 = VPUIP.GenericReshape inputs(%29 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    %31 = VPUIP.Copy inputs(%30 : memref<1x1024x256xf16, @DDR>) outputs(%arg1 : memref<1x1024x256xf16, @DDR>) -> memref<1x1024x256xf16, @DDR>
    return %31 : memref<1x1024x256xf16, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_1:%.+]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_2:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_2]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_3]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_6:%.+]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[CST:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_7:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_4]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_5]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_6:%.+]] = VPUIP.Copy inputs([[CST_0:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_4]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_6]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotOptimizeParallelCopiesNoUsers
// CHECK-SAME:    ([[INPUT:%.+]]: memref<1x32x1x12544xf16, #NHWC, @DDR>)
func.func @DoNotOptimizeParallelCopiesNoUsers(%input: memref<1x32x1x12544xf16, #NHWC, @DDR>)  -> memref<1x32x112x112xf16, #NHWC, @DDR> {
    %reshape = VPUIP.GenericReshape inputs(%input: memref<1x32x1x12544xf16, #NHWC, @DDR>) -> memref<1x32x112x112xf16, #NHWC, @DDR>

    %subview1 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %copy1 = VPUIP.Copy inputs(%subview1 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
                           outputs(%alloc1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

    %subview2 = VPUIP.SubView %reshape [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>
    %alloc2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %copy2 = VPUIP.Copy inputs(%subview2 : memref<1x16x112x112xf16, { order = #NHWC, strides = [401408, 1, 3584, 32] }, @DDR>)
                           outputs(%alloc2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

    return %reshape : memref<1x32x112x112xf16, #NHWC, @DDR>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]]
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[RESHAPE]]
    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]]
    // CHECK-SAME:                             outputs([[ALLOC1]]
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[RESHAPE]]
    // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]]
    // CHECK-SAME:                             outputs([[ALLOC2]]
    // CHECK:       return [[RESHAPE]]
}
