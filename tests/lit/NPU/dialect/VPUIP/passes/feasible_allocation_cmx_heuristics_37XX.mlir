//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s --check-prefix=DEFAULT
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR cmx-small-buffer-top-threshold=1024" %s | FileCheck %s --check-prefix=SMALL-TOP
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR weights-ping-pong=true" %s | FileCheck %s --check-prefix=PING-PONG
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Chain of three Eltwise ops with DMA-loaded inputs and a small long-lived weight table used only by the last op.
//  - by default the weight table (256 bytes) is placed right after the big buffers, in the middle of CMX,
//    and the output of the second Eltwise reuses the input range freed by the first Eltwise;
//  - cmx-small-buffer-top-threshold moves the weight table to the top of CMX;
//  - weights-ping-pong keeps the output of the second Eltwise out of the CMX ranges of DMA inputs
//    which were just freed by the first Eltwise.

// DEFAULT-LABEL: @CmxAllocationHeuristics
// SMALL-TOP-LABEL: @CmxAllocationHeuristics
// PING-PONG-LABEL: @CmxAllocationHeuristics
module @CmxAllocationHeuristics {
net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x96x96xf16>
} outputsInfo : {
    DataInfo "prob" : tensor<1x16x96x96xf16>
}

// DEFAULT-LABEL: func.func @main
// SMALL-TOP-LABEL: func.func @main
// PING-PONG-LABEL: func.func @main
func.func @main(%in: memref<1x16x96x96xf16, {order = #NHWC}>, %out: memref<1x16x96x96xf16, {order = #NHWC}>) -> memref<1x16x96x96xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x16x96x96xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x96x96xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<1x16x96x96xf16, {order = #NHWC}> = dense<2.0> : tensor<1x16x96x96xf16>, [#const.Reorder<#NHWC>]
    %cst2 = const.Declare memref<1x16x96x96xf16, {order = #NHWC}> = dense<3.0> : tensor<1x16x96x96xf16>, [#const.Reorder<#NHWC>]
    %cst_wt = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %buf_in = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_wt = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf_w0 = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_o0 = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_w1 = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_o1 = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_w2 = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_o2 = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf_last = memref.alloc() : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %t_in, %r_in = async.execute -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NNDMA inputs(%in : memref<1x16x96x96xf16, {order = #NHWC}>) outputs(%buf_in : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_wt, %r_wt = async.execute -> !async.value<memref<16x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst_wt : memref<16x1x1x4xsi32>) outputs(%buf_wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
        async.yield %0 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }
    %t_w0, %r_w0 = async.execute -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst0 : memref<1x16x96x96xf16, {order = #NHWC}>) outputs(%buf_w0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_e0, %r_e0 = async.execute [%t_in, %t_w0] (%r_in as %a : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r_w0 as %b : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%b : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf_o0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf_o0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask { outEnd = [96, 96, 16], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            } PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_w1, %r_w1 = async.execute -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst1 : memref<1x16x96x96xf16, {order = #NHWC}>) outputs(%buf_w1 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_e1, %r_e1 = async.execute [%t_e0, %t_w1] (%r_e0 as %a : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r_w1 as %b : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%b : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf_o1 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf_o1 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask { outEnd = [96, 96, 16], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            } PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_w2, %r_w2 = async.execute -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 6 : i64} {
        %0 = VPUIP.NNDMA inputs(%cst2 : memref<1x16x96x96xf16, {order = #NHWC}>) outputs(%buf_w2 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_e2, %r_e2 = async.execute [%t_e1, %t_w2] (%r_e1 as %a : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r_w2 as %b : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%b : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf_o2 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf_o2 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask { outEnd = [96, 96, 16], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            } PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_mp, %r_mp = async.execute [%t_e2, %t_wt] (%r_e2 as %a : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %r_wt as %b : !async.value<memref<16x1x1x4xsi32, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %0 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<MAXPOOL>}>
            input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%b : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf_last : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf_last : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {
                DPUTask { outEnd = [96, 96, 16], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            } PPE : {
            }
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    %t_out, %r_out = async.execute [%t_mp] (%r_mp as %a : !async.value<memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x96x96xf16, {order = #NHWC}>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 9 : i64} {
        %0 = VPUIP.NNDMA inputs(%a : memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x16x96x96xf16, {order = #NHWC}>) -> memref<1x16x96x96xf16, {order = #NHWC}>
        async.yield %0 : memref<1x16x96x96xf16, {order = #NHWC}>
    }
    // DEFAULT:  [[IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // DEFAULT-NEXT:  [[WT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1474560> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // DEFAULT-NEXT:  [[W0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <294912> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // DEFAULT-NEXT:  [[O0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589824> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // DEFAULT-NEXT:  [[W1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <884736> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // DEFAULT-NEXT:  [[O1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // DEFAULT-NEXT:  [[W2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1179648> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // SMALL-TOP:  [[IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // SMALL-TOP-NEXT:  [[WT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1982208> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // SMALL-TOP-NEXT:  [[W0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <294912> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // SMALL-TOP-NEXT:  [[O0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589824> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // SMALL-TOP-NEXT:  [[W1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <884736> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // SMALL-TOP-NEXT:  [[O1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // SMALL-TOP-NEXT:  [[W2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1179648> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // PING-PONG:  [[IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // PING-PONG-NEXT:  [[WT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1474560> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // PING-PONG-NEXT:  [[W0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <294912> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // PING-PONG-NEXT:  [[O0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589824> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // PING-PONG-NEXT:  [[W1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <884736> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // PING-PONG-NEXT:  [[O1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1474816> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // PING-PONG-NEXT:  [[W2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1179648> -> memref<1x16x96x96xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %res = async.await %r_out : !async.value<memref<1x16x96x96xf16, {order = #NHWC}>>
    return %res : memref<1x16x96x96xf16, {order = #NHWC}>
}
}
