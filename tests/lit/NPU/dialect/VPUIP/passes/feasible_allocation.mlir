//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#strides = [196608, 1, 4096, 64]

!act_type_DDR = memref<1x32x48x64xf16, {order = #NHWC}>
!act_type_CMX = memref<1x32x48x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
!act_type_CMX_2 = memref<1x1x1x98304xf16, {order = #NHWC}, [@CMX_NN, 0]>
!act_master_type_CMX = memref<1x64x48x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
!act_type = tensor<1x32x48x64xf16>
!wt_type = tensor<16x1x1x4xsi32>
!wt_type_CMX = memref<16x1x1x4xsi32, [@CMX_NN, 0]>

// CHECK-LABEL: @RereadLimitStridedSubview
module @RereadLimitStridedSubview {
config.ExecutorResource 2 of @DMA_NN
config.Resources 6 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : !act_type
    }
    outputsInfo : {
        DataInfo "prob" : !act_type
    }

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @main
func.func @main(%in: !act_type_DDR, %out: !act_type_DDR) -> !act_type_DDR {
    %cst0 = const.Declare !act_type_DDR = dense<2.0> : !act_type, [#const.Reorder<#NHWC>]
    %buf_master = memref.alloc() : !act_master_type_CMX
    %buf2 = memref.alloc() : !act_type_CMX
    %buf3 = memref.alloc() : !act_type_CMX
    %buf4 = memref.alloc() : !act_type_CMX
    %buf5 = memref.alloc() : !act_type_CMX
    %buf6 = memref.alloc() : !act_type_CMX

    // DATA_IN DMA writing into a SubView of a shared root buffer (candidate for re-read optimization).
    %t_dma_in0, %r_dma_in0 = async.execute -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 0, 0, 0][1, 32, 48, 64] : !act_master_type_CMX to !act_type_CMX
        %1 = VPUIP.NNDMA inputs(%in : !act_type_DDR) outputs(%0 : !act_type_CMX) -> !act_type_CMX
        async.yield %1 : !act_type_CMX
    }

    %t_dma_in1, %r_dma_in1 = async.execute -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 32, 0, 0][1, 32, 48, 64] : !act_master_type_CMX to !act_type_CMX
        %1 = VPUIP.NNDMA inputs(%cst0 : !act_type_DDR) outputs(%0 : !act_type_CMX) -> !act_type_CMX
        async.yield %1 : !act_type_CMX
    }

    %t0, %r0 = async.execute [%t_dma_in0, %t_dma_in1] (%r_dma_in0 as %arg0 : !async.value<!act_type_CMX>, %r_dma_in1 as %arg1 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !act_type_CMX, %arg1 as %arg3: !act_type_CMX) outputs(%buf2 as %arg4: !act_type_CMX) on tile 0 -> !act_type_CMX  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3, %arg4) : !act_type_CMX, !act_type_CMX, !act_type_CMX
            }
        async.yield %0 : !act_type_CMX
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %arg0 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !act_type_CMX) outputs(%buf2 as %arg3: !act_type_CMX) on tile 0 -> !act_type_CMX  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : !act_type_CMX, !act_type_CMX
            }
        async.yield %0 : !act_type_CMX
    }

    %t_hold, %r_hold = async.execute [%t_dma_in1] (%r_dma_in1 as %arg0 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !act_type_CMX) outputs(%buf5 as %arg3: !act_type_CMX) on tile 0 -> !act_type_CMX  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : !act_type_CMX, !act_type_CMX
            }
        async.yield %0 : !act_type_CMX
    }

    %t2, %r2 = async.execute [%t1] (%r1 as %arg0 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !act_type_CMX) outputs(%buf3 as %arg3: !act_type_CMX) on tile 0 -> !act_type_CMX  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : !act_type_CMX, !act_type_CMX
            }
        async.yield %0 : !act_type_CMX
    }

    %t3, %r3 = async.execute [%t2, %t_dma_in0] (%r2 as %arg0 : !async.value<!act_type_CMX>, %r_dma_in0 as %arg1 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 6 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !act_type_CMX, %arg1 as %arg3: !act_type_CMX) outputs(%buf4 as %arg4: !act_type_CMX) on tile 0 -> !act_type_CMX  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3, %arg4) : !act_type_CMX, !act_type_CMX, !act_type_CMX
            }
        async.yield %0 : !act_type_CMX
    }

    %t4, %r4 = async.execute [%t3, %t_hold] (%r3 as %arg0 : !async.value<!act_type_CMX>, %r_hold as %arg1 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_CMX>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 7 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !act_type_CMX, %arg1 as %arg3: !act_type_CMX) outputs(%buf6 as %arg4: !act_type_CMX) on tile 0 -> !act_type_CMX  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3, %arg4) : !act_type_CMX, !act_type_CMX, !act_type_CMX
            }
        async.yield %0 : !act_type_CMX
    }

    %t_dma_out, %r_dma_out = async.execute [%t4] (%r4 as %arg0 : !async.value<!act_type_CMX>)
            -> !async.value<!act_type_DDR> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : !act_type_CMX) outputs(%out : !act_type_DDR) -> !act_type_DDR
        async.yield %0 : !act_type_DDR
    }

    %result = async.await %r_dma_out : !async.value<!act_type_DDR>
    return %result : !act_type_DDR

    // Expect spill path to remain (no re-read clone for shared-root SubView users).
    // CHECK:       [[BUF_MASTER:%.+]] = VPURT.DeclareBuffer
    // CHECK-SAME:      -> memref<1x64x48x64xf16, {order = #NHWC, strides = [196608, 1, 4096, 64]}, [@CMX_NN, 0]>

    // CHECK:       [[T_SRC0:%.+]], [[R_SRC0:%.+]] = async.execute
    // CHECK:       VPUIP.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 32, 48, 64]
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_IN:%.+]] : memref<1x32x48x64xf16, {order = #NHWC}>)

    // CHECK:       [[T_SRC1:%.+]], [[R_SRC1:%.+]] = async.execute
    // CHECK:       VPUIP.SubView
    // CHECK-SAME:      [0, 32, 0, 0] [1, 32, 48, 64]

    // If limitation is triggered, pass keeps spill write/read instead of creating a re-read from [[ARG_IN]].
    // CHECK:       [[T_SPILL_WRITE:%.+]], [[R_SPILL_WRITE:%.+]] = async.execute {{.+}} -> !async.value<memref<1x64x48x64xf16, {order = #NHWC}, @DDR>>
    // CHECK:       VPUIP.NNDMA {{.+}} spillId = 0
    // CHECK-SAME:      inputs([[BUF_MASTER]] : memref<1x64x48x64xf16, {order = #NHWC, strides = [196608, 1, 4096, 64]}, [@CMX_NN, 0]>)
    // CHECK-NOT:   inputs([[ARG_IN]] : memref<1x32x48x64xf16, {order = #NHWC}>)
    // CHECK:       [[T_SPILL_READ:%.+]], [[R_SPILL_READ:%.+]] = async.execute {{.+}} ([[R_SPILL_WRITE]] as [[SPILL_ARG:%.+]]: !async.value<memref<1x64x48x64xf16, {order = #NHWC}, @DDR>>)
    // CHECK:       VPUIP.NNDMA {{.+}} spillId = 0

    // Final consumer must read from spill-read value.
    // CHECK:       [[T_CONSUMER:%.+]], [[R_CONSUMER:%.+]] = async.execute
    // CHECK-SAME:      [[R_SPILL_READ]] as [[READ_MASTER:%.+]]: !async.value<memref<1x64x48x64xf16, {order = #NHWC, strides = [196608, 1, 4096, 64]}, [@CMX_NN, 0]>>
    // CHECK:       VPUIP.SubView [[READ_MASTER]] [0, 0, 0, 0] [1, 32, 48, 64]
}

}


// -----


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

func.func @main(%in: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, {order = #NHWC}>) -> memref<1x16x4x4xf16, {order = #NHWC}> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%in : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %3 = async.await %f2 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    return %out : memref<1x16x4x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
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

func.func @main(%in: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, {order = #NHWC}>) -> memref<1x16x4x4xf16, {order = #NHWC}> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%in : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %3 = async.await %f2 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    return %out : memref<1x16x4x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
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

func.func @main(%arg0: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x16x4x4xf16, {order = #NHWC}>, %arg2: memref<1x16x4x4xf16, {order = #NHWC}>)
        -> (memref<1x16x4x4xf16, {order = #NHWC}>, memref<1x16x4x4xf16, {order = #NHWC}>) {
    %cst = const.Declare memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]> = dense<1.000000e+00> : tensor<1x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %token, %results = async.execute -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64}  {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%arg0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %token_0, %results_1 = async.execute -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64}  {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%cst : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%cst : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %4 = async.await %results : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    %5 = async.await %results_1 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    return %arg1, %arg2 : memref<1x16x4x4xf16, {order = #NHWC}>, memref<1x16x4x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
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

func.func @main(%in: memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, {order = #NHWC}>) -> memref<1x16x4x4xf16, {order = #NHWC}> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%in : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %buf1 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %buf2 = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }>
            input(%1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %3 = async.await %f2 : !async.value<memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>>
    return %out : memref<1x16x4x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}

}
