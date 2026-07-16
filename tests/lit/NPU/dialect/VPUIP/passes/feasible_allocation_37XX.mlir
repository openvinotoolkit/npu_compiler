//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Spilling
module @Spilling {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x96x96xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x32x96x96xf16>
    }

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x32x96x96xf16, {order = #NHWC}>
func.func @main(%in: memref<1x32x96x96xf16, {order = #NHWC}>, %out: memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x32x96x96xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x96x96xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf_in = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t_in, %r_in = async.execute -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf_in : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r0 as %1 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r1 as %1 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %1 = VPUIP.NNDMA inputs(%0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}>
        async.yield %1 : memref<1x32x96x96xf16, {order = #NHWC}>
    }

    %6 = async.await %r6 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}>>
    return %6 : memref<1x32x96x96xf16, {order = #NHWC}>

    // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589824> -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1179648> -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[BUF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[BUF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1179648> -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[BUF5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589824> -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG_0]] : memref<1x32x96x96xf16, {order = #NHWC}>) outputs([[BUF0:%.+]] :

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NNDMA

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = #VPUIP.nce_task_type<MAXPOOL>

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:      attributes {VPUIP.executor = @DPU
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>

    // COM:         spill_write is optimized because %arg0 is only used as input, only re-read DMA is inserted
    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      attributes {VPUIP.executor = @DMA_NN
    // CHECK:       VPUIP.NNDMA <{port = 0 : i64}> inputs([[ARG_0]] : memref<1x32x96x96xf16, {order = #NHWC}>) outputs([[BUF5]] :

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      attributes {VPUIP.executor = @DPU
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK-SAME:      attributes {VPUIP.executor = @DMA_NN
    // CHECK:       VPUIP.NNDMA <{port = 0 : i64}> inputs({{%[^:]+}}: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs({{%[^:]+}} :
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SpillingOpWith2Outputs
module @SpillingOpWith2Outputs {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x96x96xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x32x96x96xf16>
    }

// CHECK-LABEL: func.func @main
// CHECK-SAME:  ([[IN:%.+]]: memref<1x32x96x96xf16, {order = #NHWC}>, [[OUT:%.+]]: memref<1x32x96x96xf16, {order = #NHWC}>)
func.func @main(%in: memref<1x32x96x96xf16, {order = #NHWC}>, %out: memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x32x96x96xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x96x96xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf3 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // Below operation has two outputs and one of them would need to be spilled when scheduled
    %t2, %r2:2 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> (!async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1, %2 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t1, %t2] (%r2#0 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r1 as %arg1 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf4 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf4 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t1, %t3] (%r2#1 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r3 as %arg1 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf5 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf5 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t4] (%r4 as %arg0 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x96x96xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}>
        async.yield %0 : memref<1x32x96x96xf16, {order = #NHWC}>
    }

    %3 = async.await %r5 : !async.value<memref<1x32x96x96xf16, {order = #NHWC}>>
    return %3 : memref<1x32x96x96xf16, {order = #NHWC}>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK:       [[T1:%.+]], [[R1:%.+]]:2 = async.execute {{.+}} ([[R0]] as [[INNER_ARG:[^:]+]]: !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:       input([[INNER_ARG]] : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[BUF1:[^:]+]]
    // CHECK:           VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:       input([[INNER_ARG]] : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[BUF2:[^:]+]]
    // CHECK:       [[T_SPILL_WRITE:%.+]], [[R_SPILL_WRITE:%.+]] = async.execute
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:       inputs([[BUF2]]: memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      -> memref<1x32x96x96xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute {{.+}} ([[R1]]#0 as [[INNER_ARG:[^:]+]]: !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:        task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:        input([[INNER_ARG]] : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       [[T_SPILL_READ:%.+]], [[R_SPILL_READ:%.+]] = async.execute {{.+}} ([[R_SPILL_WRITE]] as [[INNER_ARG:[^:]+]]: !async.value<memref<1x32x96x96xf16, {order = #NHWC}, @DDR>>)
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:       inputs([[INNER_ARG]] : memref<1x32x96x96xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      -> memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute {{.+}} ([[R_SPILL_READ]] as [[INNER_ARG:[^:]+]]: !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:        task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:        input([[INNER_ARG]] : memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute {{.+}} ([[R6]] as [[INNER_ARG:[^:]+]]: !async.value<memref<1x32x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    // CHECK-NEXT:      VPUIP.NNDMA
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#strides = [262144, 1, 4096, 64]

// CHECK-LABEL: @SpillingOfSubViewBuffer
module @SpillingOfSubViewBuffer {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x64x64xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x32x64x64xf16>
    }

// CHECK-LABEL: @main
func.func @main(%in: memref<1x32x64x64xf16, {order = #NHWC}>, %out: memref<1x32x64x64xf16, {order = #NHWC}>) -> memref<1x32x64x64xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x32x64x64xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x64x64xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    // master buffer that will get spilled
    %buf_master = memref.alloc() : memref<1x64x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf6 = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>

    %t_dma_in, %r_dma_in = async.execute -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x32x64x64xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // Operation that is using master buffer which will not be directly identified for spilling but for
    // which dependant operations still need to be updated as it uses spilled master buffer
    %t0, %r0 = async.execute [%t_dma_in] (%r_dma_in as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 32, 0, 0][1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // Operation that is using master buffer and will be identified as necessary for spilling
    // Dependant operations will need to be updated to refer to spillRead result
    %t1, %r1 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 0, 0, 0][1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t2, %r2 = async.execute -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x32x64x64xf16, {order = #NHWC}>) outputs(%buf1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t1] (%r1 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t3, %t2] (%r3 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>, %r2 as %arg1 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // operation that is using buffer that will be spilled through result of async exec op
    %t5, %r5 = async.execute [%t1, %t4] (%r1 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>, %r4 as %arg1 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 6 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf4 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf4 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // operation that is using directly master buffer that will be spilled
    %t6, %r6 = async.execute [%t5] (%r5 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 0, 0, 0][1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf5 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf5 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // operation that is a user of other op that is also using master buffer which got spilled
    %t7, %r7 = async.execute [%t6] (%r0 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>, %r6 as %arg1 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf6 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf6 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 64, 64], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t_tma_out, %r_dma_out = async.execute [%t7] (%r7 as %arg0 : !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x64x64xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 9 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) outputs(%out : memref<1x32x64x64xf16, {order = #NHWC}>) -> memref<1x32x64x64xf16, {order = #NHWC}>
        async.yield %0 : memref<1x32x64x64xf16, {order = #NHWC}>
    }

    %result = async.await %r_dma_out : !async.value<memref<1x32x64x64xf16, {order = #NHWC}>>
    return %result : memref<1x32x64x64xf16, {order = #NHWC}>

    // CHECK:       [[BUF_MASTER:%.+]] = VPURT.DeclareBuffer
    // CHECK-SAME:      > -> memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
    // CHECK:       [[BUF_SPILL_WRITE:%.+]] = memref.alloc() : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.+]] = VPURT.DeclareBuffer
    // CHECK-SAME:      > -> memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK:       VPUIP.NNDMA

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK:       VPUIP.SubView
    // CHECK-SAME:      [0, 32, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<MAXPOOL>

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       VPUIP.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<MAXPOOL>

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<MAXPOOL>

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      spillId
    // CHECK-SAME:      inputs([[BUF_MASTER]] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>) outputs([[BUF_SPILL_WRITE]] : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>)

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-SAME:      ([[R4]] as [[ARG0:%.+]]: !async.value<memref<1x64x64x64xf16, {order = #NHWC}, @DDR>>
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      spillId
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>) outputs([[BUF_SPILL_READ]] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>)

    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK-SAME:      ([[R7]] as [[ARG1:%.+]]: !async.value<memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ARG1]] [0, 0, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[SUBVIEW_0]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>

    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUF_SPILL_READ]] [0, 0, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[SUBVIEW_1]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK-SAME:      ([[R7]] as [[ARG2:%.+]]: !async.value<memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ARG2]] [0, 32, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[SUBVIEW_2]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, [@CMX_NN, 0]>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SpillWriteOptimize
module @SpillWriteOptimize {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x48x75x75xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x48x75x75xf16>
    }

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x48x75x75xf16, {order = #NHWC}>
func.func @main(%in: memref<1x48x75x75xf16, {order = #NHWC}>, %out: memref<1x48x75x75xf16, {order = #NHWC}>) -> memref<1x48x75x75xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x48x75x75xf16, {order = #NHWC}> = dense<2.0> : tensor<1x48x75x75xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<48x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<48x1x1x4xsi32>

    %buf_in = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf6 = memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t_in, %r_in = async.execute -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x48x75x75xf16, {order = #NHWC}>) outputs(%buf_in : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x48x75x75xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [48, 75, 75], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t2, %r2 = async.execute [%t0, %t1] (%r0 as %0 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r1 as %1 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [48, 75, 75], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t_in, %t2] (%r_in as %0 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r2 as %1 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [48, 75, 75], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t3] -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x48x75x75xf16, {order = #NHWC}>) outputs(%buf4 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t3, %t4] (%r3 as %0 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r4 as %1 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 6 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf5 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf5 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [48, 75, 75], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t6, %r6 = async.execute [%t_in, %t5] (%r_in as %0 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r5 as %1 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf6 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf6 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [48, 75, 75], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t7, %r7 = async.execute [%t6] (%r6 as %0 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x48x75x75xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %1 = VPUIP.NNDMA inputs(%0 : memref<1x48x75x75xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x48x75x75xf16, {order = #NHWC}>) -> memref<1x48x75x75xf16, {order = #NHWC}>
        async.yield %1 : memref<1x48x75x75xf16, {order = #NHWC}>
    }

    %result = async.await %r7 : !async.value<memref<1x48x75x75xf16, {order = #NHWC}>>
    return %result : memref<1x48x75x75xf16, {order = #NHWC}>

    // COM:         spill_write ops are all optimized away
    // CHECK-NOT:   memref.alloc() : memref<1x48x75x75xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NNDMA

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<MAXPOOL>

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>

    // COM:         Re-read optimization: read input again instead of spill-write/spill-read
    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]]

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>

    // COM:         Re-read optimization again: read input a third time
    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]]

    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ControlEdgeOverlapMemory
module @ControlEdgeOverlapMemory {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x112x112xf16>
    }
    outputsInfo : {
        DataInfo "prob0" : tensor<1x32x112x112xf16>
        DataInfo "prob1" : tensor<1x32x112x112xf16>
    }

// CHECK-LABEL: @main
func.func @main(%in: memref<1x32x112x112xf16, {order = #NHWC}>, %out0: memref<1x32x112x112xf16, {order = #NHWC}>, %out1: memref<1x32x112x112xf16, {order = #NHWC}>) -> (memref<1x32x112x112xf16, {order = #NHWC}>, memref<1x32x112x112xf16, {order = #NHWC}>) {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // Task 0
    %t0, %f0 = async.execute -> !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x32x112x112xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // Task 1
    %t1, %f1 = async.execute (%f0 as %arg0 : !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // Task 2
    %t2, %f2 = async.execute (%f1 as %arg0 : !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x112x112xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out0 : memref<1x32x112x112xf16, {order = #NHWC}>) -> memref<1x32x112x112xf16, {order = #NHWC}>
        async.yield %0 : memref<1x32x112x112xf16, {order = #NHWC}>
    }

    // Task 3
    %t3, %f3 = async.execute (%f0 as %arg0 : !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [32, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // Task 4
    %t4, %f4 = async.execute (%f3 as %arg0 : !async.value<memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x112x112xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x32x112x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out1 : memref<1x32x112x112xf16, {order = #NHWC}>) -> memref<1x32x112x112xf16, {order = #NHWC}>
        async.yield %0 : memref<1x32x112x112xf16, {order = #NHWC}>
    }

    %r0 = async.await %f2 : !async.value<memref<1x32x112x112xf16, {order = #NHWC}>>
    %r1 = async.await %f4 : !async.value<memref<1x32x112x112xf16, {order = #NHWC}>>
    return %r0, %r1 : memref<1x32x112x112xf16, {order = #NHWC}>, memref<1x32x112x112xf16, {order = #NHWC}>

    // Token dependencies will match data flow by default:
    //  Task0 -> Task1 -> Task2
    //  Task0 -> Task3 -> Task4
    // besides that due to overlapping memory ranges of Task3 and Task1
    // additional control edge will be inserted:
    //  Task2 -> Task3
    // Optimization of token dependencies (transitive reduction) is beyond
    // this pass and done as a separate step

    // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0>
    // CHECK:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <802816>
    // CHECK:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <802816>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:      outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]]

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:      VPUIP.NNDMA

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:      [[T0]], [[T1]], [[T2]]
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      [[T3]]
    // CHECK-NEXT:      VPUIP.NNDMA
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ControlEdgeOverlapMemoryCheckProdCons
module @ControlEdgeOverlapMemoryCheckProdCons {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x80x60x60xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x80x60x60xf16>
    }

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x80x60x60xf16, {order = #NHWC}>
func.func @main(%in: memref<1x80x60x60xf16, {order = #NHWC}>, %out: memref<1x80x60x60xf16, {order = #NHWC}>) -> memref<1x80x60x60xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x80x60x60xf16, {order = #NHWC}> = dense<2.0> : tensor<1x80x60x60xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<32x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<32x1x1x4xsi32>

    %buf_in = memref.alloc() : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t_in, %r_in = async.execute -> !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x80x60x60xf16, {order = #NHWC}>) outputs(%buf_in : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [80, 60, 60], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x80x60x60xf16, {order = #NHWC}>) outputs(%buf1 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r0 as %1 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [80, 60, 60], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r1 as %1 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%1 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [80, 60, 60], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %2 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x80x60x60xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %1 = VPUIP.NNDMA inputs(%0 : memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x80x60x60xf16, {order = #NHWC}>) -> memref<1x80x60x60xf16, {order = #NHWC}>
        async.yield %1 : memref<1x80x60x60xf16, {order = #NHWC}>
    }

    %6 = async.await %r6 : !async.value<memref<1x80x60x60xf16, {order = #NHWC}>>
    return %6 : memref<1x80x60x60xf16, {order = #NHWC}>

    // Token dependencies will match data flow by default:
    //  Task0 -> Task1 -> Task3 -> Task4
    //  Task2 -> Task3
    //  Task0 -> Task_SW -> Task_SR -> Task4 -> Task5
    // besides that due to overlapping memory ranges additional control edge will be inserted.
    // Important is relation between Task0, Task1, Task_SW.
    // Execution order is following:
    //  t0: Task0 produces BUF0
    //  tX: Task1 reads BUF0
    //  tY: Task_SW reads BUF0
    // Resulting dependencies from just looking at memory intervals and their users throughout execution time
    // is following: Task0 -> Task1, Task0 -> Task_SW
    // If there would be no differentiation between resource producer and consumer unnecessary dependency
    // would be inserted from Task1 -> Task_SW
    //
    // Optimization of token dependencies (transitive reduction) is beyond
    // this pass and done as a separate step

    // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0>
    // CHECK:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576000>
    // CHECK:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1152000>
    // CHECK:       [[BUF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0>
    // CHECK:       [[BUF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1152000>
    // CHECK:       [[BUF_REREAD:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576000> -> memref<1x80x60x60xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       VPUIP.NNDMA
    // CHECK-SAME:       outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NNDMA
    // CHECK-SAME:       outputs([[BUF2]]

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-SAME:       [[T0]]
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:       outputs([[BUF1]]

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:       [[T1]], [[T2]]
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:       outputs([[BUF3]]

    // COM:         Re-read optimization: read input again from DDR instead of spilling
    // CHECK:       [[T_REREAD:%.+]], [[R_REREAD:%.+]] = async.execute
    // CHECK-SAME:       [[T0]], [[T3]]
    // CHECK-NEXT:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG_0]]
    // CHECK-SAME:       outputs([[BUF_REREAD]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:       [[T0]], [[T3]], [[T_REREAD]]
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:       outputs([[BUF4]]

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:       [[T4]]
    // CHECK-NEXT:       VPUIP.NNDMA
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    num_clusters = 4
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<64x32x3x3xf16, {order = #NHWC}, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x64x16x16xf16, {order = #NHWC}, @DDR>

!WeightsTableStub = memref<64x1x1x4xsi32>
!InputStub_CMX = memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsTableStub_CMX = memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>

// CHECK-LABEL: @SingleConvWithClusteringAndDmaPortDistribution
module @SingleConvWithClusteringAndDmaPortDistribution {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x16x16xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64x16x16xf16>
    }

// CHECK-LABEL: func.func @main
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
func.func @main(%input: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare memref<64x1x1x4xsi32, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NCHW>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

        async.yield
    }

    %t1 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NNDMA { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

        async.yield
    }

    %t2 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NNDMA { out_mem_space = @CMX_NN } inputs(%weights_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

        async.yield
    }

    %t3 = async.execute [%t0, %t1, %t2]
                attributes {VPUIP.executor = @DPU, VPUIP.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
            %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%input_cmx : !InputDistributed)
                    weights(%weights_cmx : !WeightsDistributed)
                    weight_table(%weights_table_cmx : !WeightsTableDistributed)
                    parent_input(%input_cmx : !InputDistributed)
                    parent_output(%output_buff_cmx : !OutputDistributed)
                    outputs(%output_buff_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 15, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    } PPE :  {
                    }

            async.yield
    }

    %t4 = async.execute [%t3]
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NNDMA { out_mem_space = @DDR } inputs(%output_buff_cmx: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @DDR>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NCHW>]
    // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <45056> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> <49152> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> <36864> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF4:%.+]] = memref.alloc() : memref<1x64x16x16xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[T0:%.+]] = async.execute
    // CHECK-SAME:      VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0, 1]
    // CHECK-SAME:      cycleBegin = 0 : i64, cycleEnd = 1 : i64
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:          inputs([[ARG_0]] : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:          outputs([[BUF0]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, num_clusters = 4 : i64}>)

    // CHECK:       [[T1:%.+]] = async.execute
    // CHECK-SAME:      VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0]
    // CHECK-SAME:      cycleBegin = 1 : i64, cycleEnd = 2 : i64
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST_WEIGHTS]] : memref<64x32x3x3xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[BUF1]] : !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T2:%.+]] = async.execute
    // CHECK-SAME:      VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1]
    // CHECK-SAME:      cycleBegin = 1 : i64, cycleEnd = 2 : i64
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST_WEIGHTS_TABLE]] : memref<64x1x1x4xsi32, @DDR>)
    // CHECK-SAME:      outputs([[BUF2]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T3:%.+]] = async.execute
    // CHECK-SAME:      [[T0]], [[T1]], [[T2]]
    // CHECK-SAME:      VPUIP.executor = @DPU
    // CHECK-SAME:      cycleBegin = 2 : i64, cycleEnd = 3 : i64
    // CHECK:         VPUIP.NCEClusterTask
    // CHECK-SAME:        input([[BUF0]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, num_clusters = 4 : i64}>)
    // CHECK-SAME:        weights([[BUF1]] : !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:        weight_table([[BUF2]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:        parent_input([[BUF0]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, num_clusters = 4 : i64}>)
    // CHECK-SAME:        parent_output([[BUF3]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:        outputs([[BUF3]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[T4:%.+]] = async.execute
    // CHECK-SAME:      [[T3]]
    // CHECK-SAME:      VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0, 1]
    // CHECK-SAME:      cycleBegin = 3 : i64, cycleEnd = 4 : i64
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF3]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:      outputs([[BUF4]] : memref<1x64x16x16xf16, {order = #NHWC}, @DDR>)

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufDistributed = !VPUIP.DistributedBuffer<
    1x64x64x64xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WtDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!BufMemrefDDR = memref<1x64x64x64xf16, {order = #NHWC}, @DDR>
!BufMemrefCMX = memref<1x64x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>

!WtMemrefDDR = memref<64x1x1x4xsi32, {order = #NHWC}, @DDR>
!WtMemrefCMX = memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>

// CHECK-LABEL: @SpillingWithClustering
module @SpillingWithClustering {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x64x64x64xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64x64x64xf16>
    }

// CHECK-LABEL: func.func @main
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x64x64x64xf16, {order = #NHWC}, @DDR>
func.func @main(%input: !BufMemrefDDR) -> !BufMemrefDDR {
    %cst0 = const.Declare memref<1x64x64x64xf16, {order = #NHWC}, @DDR> = dense<2.0> : tensor<1x64x64x64xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare !WtMemrefDDR = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %buf_in = VPURT.AllocDistributed -> !BufDistributed
    %buf_wt = VPURT.AllocDistributed -> !WtDistributed
    %buf0 = VPURT.AllocDistributed -> !BufDistributed
    %buf1 = VPURT.AllocDistributed -> !BufDistributed
    %buf2 = VPURT.AllocDistributed -> !BufDistributed
    %buf3 = VPURT.AllocDistributed -> !BufDistributed
    %output = memref.alloc() : !BufMemrefDDR

    %t_in, %r_in = async.execute -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%input : !BufMemrefDDR) outputs(%buf_in : !BufDistributed) -> !BufDistributed
        async.yield %0: !BufDistributed
    }

    %t0, %r0 = async.execute -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>) outputs(%buf0 : !BufDistributed) -> !BufDistributed
        async.yield %0: !BufDistributed
    }

    %t10, %r10 = async.execute -> !async.value<!WtDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst1 : !WtMemrefDDR) outputs(%buf_wt : !WtDistributed) -> !WtDistributed
        async.yield %0: !WtDistributed
    }

    %t3, %r3 = async.execute [%t_in, %t10] (%r_in as %async_arg0 : !async.value<!BufDistributed>,
                                                  %r10 as %async_arg1 : !async.value<!WtDistributed>)
                -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 4 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%async_arg0 : !BufDistributed)
            weight_table(%async_arg1 : !WtDistributed)
            parent_input(%async_arg0 : !BufDistributed)
            parent_output(%buf1 : !BufDistributed)
            outputs(%buf1 : !BufDistributed) -> !BufDistributed
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
        }
        async.yield %0: !BufDistributed
    }

    %t1, %r1 = async.execute [%t0, %t10, %t3] (%r0 as %async_arg0 : !async.value<!BufDistributed>, %r3 as %async_arg1 : !async.value<!BufDistributed>)
                -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 4 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%async_arg0 : !BufDistributed)
            weights(%async_arg1 : !BufDistributed)
            parent_input(%async_arg0 : !BufDistributed)
            parent_output(%buf2 : !BufDistributed)
            outputs(%buf2 : !BufDistributed) -> !BufDistributed
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0: !BufDistributed
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %async_arg0 : !async.value<!BufDistributed>, %r1 as %async_arg1 : !async.value<!BufDistributed>)
                -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 4 : i64, "async-deps-index" = 6 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }>
            input(%async_arg0 : !BufDistributed)
            weights(%async_arg1 : !BufDistributed)
            parent_input(%async_arg0 : !BufDistributed)
            parent_output(%buf3 : !BufDistributed)
            outputs(%buf3 : !BufDistributed) -> !BufDistributed
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
                }
        async.yield %0: !BufDistributed
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %async_arg0 : !async.value<!BufDistributed>) -> !async.value<!BufMemrefDDR>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
            %0 = VPUIP.NNDMA { out_mem_space = @DDR } inputs(%async_arg0: !BufDistributed) outputs(%output: !BufMemrefDDR) -> !BufMemrefDDR
        async.yield %0: !BufMemrefDDR
    }

    %6 = async.await %r6 : !async.value<!BufMemrefDDR>
    return %6 : !BufMemrefDDR

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare memref<1x64x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK-DAG:       [[CST1:%.+]] = const.Declare memref<64x1x1x4xsi32, {order = #NHWC}, @DDR>

    // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> <524288> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> <525312> -> !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1049600> -> !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF5:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1048576> -> !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF6:%.+]] = memref.alloc() : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BUF_REREAD:%.+]] = VPURT.DeclareBuffer <CMX_NN> <524288> -> !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]] : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[BUF0]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST1]] : memref<64x1x1x4xsi32, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[BUF1]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST0]] : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[BUF2]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:    [[T0]], [[T1]]
    // CHECK-SAME:    ([[R0]] as [[ARG1:%.+]]: !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK-SAME:     [[R1]] as [[ARG2:%.+]]: !async.value<!VPUIP.DistributedBuffer<64x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK-SAME:        -> !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:         input([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         weight_table([[ARG2]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         parent_input([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         parent_output([[BUF3]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         outputs([[BUF3]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:    [[T1]], [[T2]], [[T3]]
    // CHECK-SAME:    ([[R2]] as [[ARG1:%.+]]: !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>,
    // CHECK-SAME:     [[R3]] as [[ARG2:%.+]]: !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK-SAME:      -> !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:         input([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         weights([[ARG2]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         parent_input([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         parent_output([[BUF4]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         outputs([[BUF4]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // COM:         Re-read optimization: read input again from DDR instead of spilling
    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:    [[T0]], [[T3]], [[T4]]
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:         inputs([[ARG_0]] : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:          outputs([[BUF_REREAD]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK-SAME:    [[T0]], [[T4]], [[T5]]
    // CHECK-SAME:    ([[R5]] as [[ARG1:%.+]]: !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>,
    // CHECK-SAME:     [[R4]] as [[ARG2:%.+]]: !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK-SAME:      -> !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:         input([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         weights([[ARG2]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         parent_input([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         parent_output([[BUF5]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:         outputs([[BUF5]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-SAME:    [[T6]]
    // CHECK-SAME:    ([[R6]] as [[ARG1:%.+]]: !async.value<!VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK:        VPUIP.NNDMA
    // CHECK-SAME:     inputs([[ARG1]] : !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUF6]] : memref<1x64x64x64xf16, {order = #NHWC}, @DDR>)
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Prefetching
module @Prefetching {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x16x16xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x128x4x4xf16>
    }

// CHECK-LABEL: @main
func.func @main(%in: memref<1x32x16x16xf16, {order = #NHWC}>, %out: memref<1x128x4x4xf16, {order = #NHWC}>) -> memref<1x128x4x4xf16, {order = #NHWC}> {

    %cst_4 = const.Declare memref<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>
    %cst_10 = const.Declare memref<64x32x3x3xf16, {order = #NHWC}> = dense<2.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare memref<128x1x1x4xsi32> = dense<2> : tensor<128x1x1x4xsi32>
    %cst_11 = const.Declare memref<128x64x3x3xf16, {order = #NHWC}> = dense<2.0> : tensor<128x64x3x3xf16>, [#const.Reorder<#NHWC>]

    %buf0 = memref.alloc() : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %11 = memref.alloc() : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %12 = memref.alloc() : memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %13 = memref.alloc() : memref<128x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = memref.alloc() : memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %token_30, %results_31 = async.execute -> !async.value<memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, "cycleCost" = 683 : i64} {
      %32 = VPUIP.NNDMA inputs(%in : memref<1x32x16x16xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
      async.yield %32 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %token_32, %results_33 = async.execute -> !async.value<memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, "cycleCost" = 2172 : i64} {
      %32 = VPUIP.NNDMA inputs(%cst_10 : memref<64x32x3x3xf16, {order = #NHWC}>) outputs(%9 : memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
      async.yield %32 : memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %token_34, %results_35 = async.execute -> !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, "cycleCost" = 22 : i64} {
      %32 = VPUIP.NNDMA inputs(%cst_4 : memref<64x1x1x4xsi32>) outputs(%10 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
      async.yield %32 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    }
    %token_36, %results_37 = async.execute [%token_30, %token_32, %token_34] (%results_31 as %arg2:
        !async.value<memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %results_33 as %arg3:
        !async.value<memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %results_35 as %arg4:
        !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, "cycleCost" = 734 : i64} {
      %32 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 734 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
            kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
            input(%arg2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%arg3 : memref<64x32x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%arg4 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg2 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%11 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%11 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [3, 3, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        DPUTask {outEnd = [7, 3, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [4, 0, 0]}
        DPUTask {outEnd = [3, 7, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, outStart = [0, 4, 0]}
        DPUTask {outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, outStart = [4, 4, 0]}
      } PPE :  {
      }
      async.yield %32 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %token_38, %results_39 = async.execute -> !async.value<memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, "cycleCost" = 6144 : i64} {
      %32 = VPUIP.NNDMA inputs(%cst_11 : memref<128x64x3x3xf16, {order = #NHWC}>) outputs(%12 : memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
      async.yield %32 : memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %token_40, %results_41 = async.execute -> !async.value<memref<128x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, "cycleCost" = 61 : i64} {
      %32 = VPUIP.NNDMA inputs(%cst_3 : memref<128x1x1x4xsi32>) outputs(%13 : memref<128x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<128x1x1x4xsi32, [@CMX_NN, 0]>
      async.yield %32 : memref<128x1x1x4xsi32, [@CMX_NN, 0]>
    }
    %token_42, %results_43 = async.execute [%token_36, %token_38, %token_40] (%results_37 as %arg2:
        !async.value<memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %results_39 as %arg3:
        !async.value<memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %results_41 as %arg4:
        !async.value<memref<128x1x1x4xsi32, [@CMX_NN, 0]>>) ->
        !async.value<memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 6 : i64, "cycleCost" = 686 : i64} {
      %32 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 686 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, kernel_size = [3, 3],
            kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
            input(%arg2 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weights(%arg3 : memref<128x64x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%arg4 : memref<128x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg2 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%14 : memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%14 : memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                -> memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [3, 3, 31], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
        DPUTask {outEnd = [3, 3, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, outStart = [0, 0, 32]}
        DPUTask {outEnd = [3, 3, 95], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
        DPUTask {outEnd = [3, 3, 127], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, outStart = [0, 0, 96]}
      } PPE :  {
      }
      async.yield %32 : memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %token_43, %result_44 = async.execute [%token_42] (%results_43 as %0 : !async.value<memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x128x4x4xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, "cycleCost" = 171 : i64} {
        %1 = VPUIP.NNDMA inputs(%0 : memref<1x128x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x128x4x4xf16, {order = #NHWC}>) -> memref<1x128x4x4xf16, {order = #NHWC}>
        async.yield %1 : memref<1x128x4x4xf16, {order = #NHWC}>
    }

    %44 = async.await %result_44 : !async.value<memref<1x128x4x4xf16, {order = #NHWC}>>
    return %44 : memref<1x128x4x4xf16, {order = #NHWC}>

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 683 : i64, cycleEnd = 683 : i64}
    // CHECK:           VPUIP.NNDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2172 : i64, cycleEnd = 2172 : i64}
    // CHECK:           VPUIP.NNDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 683 : i64, cycleCost = 22 : i64, cycleEnd = 705 : i64}
    // CHECK:           VPUIP.NNDMA

    // Prefetched Copy executing in parallel cycles to NCEClusterTask

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64, cycleBegin = 705 : i64, cycleCost = 6144 : i64, cycleEnd = 6849 : i64}
    // CHECK:           VPUIP.NNDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 4 : i64, cycleBegin = 2172 : i64, cycleCost = 61 : i64, cycleEnd = 2233 : i64}
    // CHECK:           VPUIP.NNDMA

    // CHECK:       {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 2172 : i64, cycleCost = 734 : i64, cycleEnd = 2906 : i64}
    // CHECK:           VPUIP.NCEClusterTask

    // CHECK:       {VPUIP.executor = @DPU, "async-deps-index" = 6 : i64, cycleBegin = 6849 : i64, cycleCost = 686 : i64, cycleEnd = 7535 : i64}
    // CHECK:           VPUIP.NCEClusterTask

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 7 : i64, cycleBegin = 7535 : i64, cycleCost = 171 : i64, cycleEnd = 7706 : i64}
    // CHECK:           VPUIP.NNDMA
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PipelineShaveAct
module @PipelineShaveAct {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "in0" : tensor<1x32x48x48xf16>
        DataInfo "in1" : tensor<1x32x48x48xf16>
    }
    outputsInfo : {
        DataInfo "vf0" : tensor<1x32x48x48xf16>
        DataInfo "vf1" : tensor<1x32x48x48xf16>
    }

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @main
func.func @main(%in0: memref<1x32x48x48xf16, {order = #NHWC}>, %in1: memref<1x32x48x48xf16, {order = #NHWC}>, %out0: memref<1x32x48x48xf16, {order = #NHWC}>, %out1: memref<1x32x48x48xf16, {order = #NHWC}>) -> (memref<1x32x48x48xf16, {order = #NHWC}>, memref<1x32x48x48xf16, {order = #NHWC}>) {
    %wt = const.Declare memref<32x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<32x1x1x4xsi32>

    %buf_in0 = memref.alloc() : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_in1 = memref.alloc() : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %buf2 = memref.alloc() : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // vertical path 1

    %t_in_vp1, %r_in_vp1 = async.execute
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64, "cycleCost" = 10 : i64} {
        %0 = VPUIP.NNDMA inputs(%in0 : memref<1x32x48x48xf16, {order = #NHWC}>) outputs(%buf_in0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t_nce_vp1, %r_nce_vp1 = async.execute [%t_in_vp1] (%r_in_vp1 as %0 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64, "cycleCost" = 40 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [80, 60, 60], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t_sw_vp1, %r_sw_vp1 = async.execute [%t_nce_vp1] (%r_nce_vp1 as %0 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64, "cycleCost" = 30 : i64} {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%0 as %arg3: memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf1 as %arg4: memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
            }
        async.yield %1 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t_out_vp1, %r_out_vp1 = async.execute [%t_sw_vp1] (%r_sw_vp1 as %0 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}>>
                attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64, "cycleCost" = 10 : i64} {
        %1 = VPUIP.NNDMA inputs(%0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out0 : memref<1x32x48x48xf16, {order = #NHWC}>) -> memref<1x32x48x48xf16, {order = #NHWC}>
        async.yield %1 : memref<1x32x48x48xf16, {order = #NHWC}>
    }

    // vertical path 2

    %t_in_vp2, %r_in_vp2 = async.execute
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64, "cycleCost" = 10 : i64} {
        %0 = VPUIP.NNDMA inputs(%in1 : memref<1x32x48x48xf16, {order = #NHWC}>) outputs(%buf_in1 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t_nce_vp2, %r_nce_vp2 = async.execute [%t_in_vp2] (%r_in_vp2 as %0 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64, "cycleCost" = 40 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [80, 60, 60], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t_sw_vp2, %r_sw_vp2 = async.execute [%t_nce_vp2] (%r_nce_vp2 as %0 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 6 : i64, "cycleCost" = 30 : i64} {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%0 as %arg3: memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3 as %arg4: memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
            }
        async.yield %1 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t_out_vp2, %r_out_vp2 = async.execute [%t_sw_vp2] (%r_sw_vp2 as %0 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x48x48xf16, {order = #NHWC}>>
                attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64, "cycleCost" = 10 : i64} {
        %1 = VPUIP.NNDMA inputs(%0 : memref<1x32x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out1 : memref<1x32x48x48xf16, {order = #NHWC}>) -> memref<1x32x48x48xf16, {order = #NHWC}>
        async.yield %1 : memref<1x32x48x48xf16, {order = #NHWC}>
    }

    %end_vp1 = async.await %r_out_vp1 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}>>
    %end_vp2 = async.await %r_out_vp2 : !async.value<memref<1x32x48x48xf16, {order = #NHWC}>>
    return %end_vp1, %end_vp2 : memref<1x32x48x48xf16, {order = #NHWC}>, memref<1x32x48x48xf16, {order = #NHWC}>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute
    // CHECK-SAME:      attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64,
    // CHECK-SAME:      cycleBegin = 0 : i64, cycleCost = 10 : i64, cycleEnd = 10 : i64}

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:      attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64,
    // CHECK-SAME:      cycleBegin = 0 : i64, cycleCost = 10 : i64, cycleEnd = 10 : i64}

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK-SAME:      [[R0]]
    // CHECK-SAME:      attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64,
    // CHECK-SAME:      cycleBegin = 10 : i64, cycleCost = 40 : i64, cycleEnd = 50 : i64}

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:      [[T2]]
    // CHECK-SAME:      [[R2]]
    // CHECK-SAME:      attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64,
    // CHECK-SAME:      cycleBegin = 50 : i64, cycleCost = 30 : i64, cycleEnd = 80 : i64}

    // Note: SHAVE_ACT above and NCE below execute during the same cycles

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      [[T1]], [[T2]]
    // CHECK-SAME:      [[R1]]
    // CHECK-SAME:      attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64,
    // CHECK-SAME:      cycleBegin = 50 : i64, cycleCost = 40 : i64, cycleEnd = 90 : i64}

    // Note: DPU above and DMA below execute during the same cycles

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      [[T3]]
    // CHECK-SAME:      [[R3]]
    // CHECK-SAME:      attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64,
    // CHECK-SAME:      cycleBegin = 80 : i64, cycleCost = 10 : i64, cycleEnd = 90 : i64}

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK-SAME:      [[T3]], [[T4]]
    // CHECK-SAME:      [[R4]]
    // CHECK-SAME:      attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 6 : i64,
    // CHECK-SAME:      cycleBegin = 90 : i64, cycleCost = 30 : i64, cycleEnd = 120 : i64}

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-SAME:      [[T6]]
    // CHECK-SAME:      [[R6]]
    // CHECK-SAME:      attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64,
    // CHECK-SAME:      cycleBegin = 120 : i64, cycleCost = 10 : i64, cycleEnd = 130 : i64}
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PrefetchNoActSpillAtEndAndWrongOrder
module @PrefetchNoActSpillAtEndAndWrongOrder {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x1x1x1000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1x1x1000xf16>
    }

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @main
func.func @main(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %0 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %1 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %5 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %6 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %7 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %8 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %token, %results = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
      async.yield %10 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_0, %results_1 = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%0 : memref<1x1x1x1000xf16, @DDR>) outputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
      async.yield %10 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_2, %results_3 = async.execute [%token_0] (%results_1 as %arg2: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%arg2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
      async.yield %10 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_4, %results_5 = async.execute [%token, %token_0] (%results as %arg2: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>, %results_1 as %arg3: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleCost = 2 : i64} {
      %results_18 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg2 as %arg4: memref< 1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg3 as %arg5: memref< 1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref< 1x1x1x1000xf16, [@CMX_NN, 0]>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref< 1x1x1x1000xf16, [@CMX_NN, 0]>, memref< 1x1x1x1000xf16, [@CMX_NN, 0]>
            }
      async.yield %results_18 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_6, %results_7 = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%4 : memref<1x1x1x1000xf16, @DDR>) outputs(%5 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
      async.yield %10 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_8, %results_9 = async.execute [%token_6] (%results_7 as %arg2: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%arg2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
      async.yield %10 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_10, %results_11 = async.execute [%token_2, %token_6] (%results_3 as %arg2: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>, %results_7 as %arg3: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 6 : i64, cycleCost = 2 : i64} {
      %results_18 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg2 as %arg4: memref< 1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg3 as %arg5: memref< 1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref< 1x1x1x1000xf16, [@CMX_NN, 0]>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref< 1x1x1x1000xf16, [@CMX_NN, 0]>, memref< 1x1x1x1000xf16, [@CMX_NN, 0]>
            }
      async.yield %results_18 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_12, %results_13 = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%7 : memref<1x1x1x1000xf16, @DDR>) outputs(%8 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
      async.yield %10 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %token_14, %results_15 = async.execute [%token_12] (%results_13 as %arg2: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 1047 : i64} {
      %10 = VPUIP.NNDMA inputs(%arg2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
      async.yield %10 : memref<1x1x1x1000xf16, @DDR>
    }
    %token_16, %results_17 = async.execute [%token_8, %token_12] (%results_9 as %arg2: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>, %results_13 as %arg3: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 9 : i64, cycleCost = 2 : i64} {
      %results_18 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg2 as %arg4: memref< 1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg3 as %arg5: memref< 1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref< 1x1x1x1000xf16, [@CMX_NN, 0]>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref< 1x1x1x1000xf16, [@CMX_NN, 0]>, memref< 1x1x1x1000xf16, [@CMX_NN, 0]>
            }
      async.yield %results_18 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %9 = async.await %results_15 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    return %9 : memref<1x1x1x1000xf16, @DDR>

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1047 : i64, cycleEnd = 1047 : i64}
    // CHECK:           VPUIP.NNDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 1047 : i64, cycleEnd = 1047 : i64}
    // CHECK:           VPUIP.NNDMA

    // CHECK:       {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 4 : i64, cycleBegin = 1047 : i64, cycleCost = 2 : i64, cycleEnd = 1049 : i64}
    // CHECK:           VPUIP.SW.Kernel

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 1047 : i64, cycleCost = 1047 : i64, cycleEnd = 2094 : i64}
    // CHECK:           VPUIP.NNDMA

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 3 : i64, cycleBegin = 1047 : i64, cycleCost = 1047 : i64, cycleEnd = 2094 : i64}
    // CHECK:           VPUIP.NNDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 5 : i64, cycleBegin = 2094 : i64, cycleCost = 1047 : i64, cycleEnd = 3141 : i64}
    // CHECK:           VPUIP.NNDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 6 : i64, cycleBegin = 2094 : i64, cycleCost = 1047 : i64, cycleEnd = 3141 : i64}
    // CHECK:           VPUIP.NNDMA

    // CHECK:       {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 7 : i64, cycleBegin = 3141 : i64, cycleCost = 2 : i64, cycleEnd = 3143 : i64}
    // CHECK:           VPUIP.SW.Kernel

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 8 : i64, cycleBegin = 3141 : i64, cycleCost = 1047 : i64, cycleEnd = 4188 : i64}
    // CHECK:           VPUIP.NNDMA

    // CHECK:       {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 9 : i64, cycleBegin = 4188 : i64, cycleCost = 2 : i64, cycleEnd = 4190 : i64}
    // CHECK:           VPUIP.SW.Kernel
}

}
