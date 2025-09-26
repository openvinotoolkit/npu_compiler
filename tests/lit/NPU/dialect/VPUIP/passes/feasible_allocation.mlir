//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SimpleGraph
module @SimpleGraph {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x4x4xf16>
    }
// CHECK:   config.Resources {{[0-9]+}} of @NCE

func.func @main(%in: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %3 = async.await %f2 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
    return %out : memref<1x16x4x4xf16, #NHWC>

    // CHECK-DAG:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SimpleGraphWithReservedMem
module @SimpleGraphWithReservedMem {

config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
        module @DmaProfilingReservedMemory {
            config.MemoryResource 512 bytes of @CMX_NN offset 0
        }
    }
}

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x4x4xf16>
    }

// CHECK:   config.Resources {{[0-9]+}} of @NCE

func.func @main(%in: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %3 = async.await %f2 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
    return %out : memref<1x16x4x4xf16, #NHWC>

    // CHECK-DAG:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoOutputs
module @TwoOutputs {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x16x4x4xf16>
        DataInfo "prob2" : tensor<1x16x4x4xf16>
    }

// CHECK:   config.Resources {{[0-9]+}} of @NCE

func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x4x4xf16, #NHWC>, %arg2: memref<1x16x4x4xf16, #NHWC>)
        -> (memref<1x16x4x4xf16, #NHWC>, memref<1x16x4x4xf16, #NHWC>) {
    %cst = const.Declare memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]> = dense<1.000000e+00> : tensor<1x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %token, %results = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64}  {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%arg0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %token_0, %results_1 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64}  {
        %1 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%cst : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%cst : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %4 = async.await %results : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
    %5 = async.await %results_1 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
    return %arg1, %arg2 : memref<1x16x4x4xf16, #NHWC>, memref<1x16x4x4xf16, #NHWC>

    // CHECK-DAG:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DeclareBuffersInMiddle
module @DeclareBuffersInMiddle {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x4x4xf16>
    }
// CHECK:   config.Resources {{[0-9]+}} of @NCE

func.func @main(%in: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %buf2 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %3 = async.await %f2 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
    return %out : memref<1x16x4x4xf16, #NHWC>

    // CHECK-DAG:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
}

}
