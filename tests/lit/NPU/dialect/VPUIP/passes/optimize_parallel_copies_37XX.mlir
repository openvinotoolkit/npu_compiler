//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeParallelNonConstCopies
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
func.func @OptimizeParallelNonConstCopies(
        %input: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output1: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output2: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
         -> (memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %3 = VPUIP.Copy
            inputs(%input : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%3 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%3 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
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
            inputs(%input : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
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

// CHECK:       [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
// CHECK:       [[VAR2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// CHECK:       [[VAR4:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
