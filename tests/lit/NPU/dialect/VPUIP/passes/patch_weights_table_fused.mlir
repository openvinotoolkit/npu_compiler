//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --patch-weight-table %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PatchFusedConstantWithSpill
func.func @PatchFusedConstantWithSpill() ->  memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]> {
    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <556416> -> memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]>

    %fused_const = const.Declare memref<1x1x1x784xui8> = dense<1> : tensor<16x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x784xui8>,
            weightsTable=<dense<1> : tensor<16x1x1x4xsi32>>,
            weights=<dense<1.0> : tensor<16x16x1x1xf16>>>
        ]
    %fused_constant_1 = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1x1x1x784xui8, [@CMX_NN, 0]>
    %fused_constant_DDR = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x784xui8, @DDR>
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <1837056> -> memref<1x1x1x784xui8, [@CMX_NN, 0]>

    %1 = VPUIP.NNDMA inputs(%fused_const : memref<1x1x1x784xui8>) outputs(%fused_constant_1 : memref<1x1x1x784xui8, [@CMX_NN, 0]>) -> memref<1x1x1x784xui8, [@CMX_NN, 0]>
    %2 = VPUIP.NNDMA inputs(%1 : memref<1x1x1x784xui8, [@CMX_NN, 0]>) outputs(%fused_constant_DDR : memref<1x1x1x784xui8, @DDR>) -> memref<1x1x1x784xui8, @DDR>
    %3 = VPUIP.NNDMA inputs(%2 : memref<1x1x1x784xui8, @DDR>) outputs(%0 : memref<1x1x1x784xui8, [@CMX_NN, 0]>) -> memref<1x1x1x784xui8, [@CMX_NN, 0]>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 1, 1, 256] : memref<1x1x1x784xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, [@CMX_NN, 0]>
    %5 = VPUIP.SubView %3 [0, 0, 0, 256] [1, 1, 1, 512] : memref<1x1x1x784xui8, #NCHW, [@CMX_NN, 0]> to memref<1x1x1x512xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, [@CMX_NN, 0]>
    %6 = VPUIP.SubView %3 [0, 0, 0, 768] [1, 1, 1, 16] : memref<1x1x1x784xui8, [@CMX_NN, 0]> to memref<1x1x1x16xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, [@CMX_NN, 0]>

    %7 = VPUIP.ViewOp %4 : memref<1x1x1x256xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %8 = VPUIP.ViewOp %5 : memref<1x1x1x512xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, [@CMX_NN, 0]> to memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %9 = VPUIP.ViewOp %6 : memref<1x1x1x16xui8, {order = #NCHW, strides = [784, 784, 784, 1]}, [@CMX_NN, 0]> to memref<1x1x1x16xui8, [@CMX_NN, 0]>

    %10 = VPUIP.NCEClusterTask {
        constantsFused = true,
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [7, 7],
        kernel_strides = [7, 7],
        task_type = #VPUIP.nce_task_type<DWCONV>
        }
        input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>)

        weights(%8 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%7 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)

        parent_input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out : memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [1, 0, 1007], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %10 : memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <556416> -> memref<1x126x2x2xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:   [[FUSED_CONSTANT:%.*]] = const.Declare memref<1x1x1x784xui8> =
    // CHECK-SAME:  #const.RelocateWeightsTable<weightsPtr=[1837312], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=256 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>

    // CHECK:   [[FUSED_CONSTANT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1x1x1x784xui8, [@CMX_NN, 0]>
    // CHECK:   [[FUSED_CONSTANT_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x784xui8, @DDR>
    // CHECK:   [[FUSED_CONSTANT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1837056> -> memref<1x1x1x784xui8, [@CMX_NN, 0]>

    // CHECK:   [[COPY_OP_1:.*]] = VPUIP.NNDMA inputs([[FUSED_CONSTANT]] : memref<1x1x1x784xui8>) outputs([[FUSED_CONSTANT_1]] : memref<1x1x1x784xui8, [@CMX_NN, 0]>) -> memref<1x1x1x784xui8, [@CMX_NN, 0]>
    // CHECK:   [[COPY_OP_2:.*]] = VPUIP.NNDMA
    // CHECK:   [[COPY_OP_3:.*]] = VPUIP.NNDMA

    // CHECK:   [[SUBVIEW_1:.*]] = VPUIP.SubView
    // CHECK:   [[SUBVIEW_2:.*]] = VPUIP.SubView
    // CHECK:   [[SUBVIEW_3:.*]] = VPUIP.SubView

    // CHECK:   [[VIEW_1:.*]] = VPUIP.ViewOp
    // CHECK:   [[VIEW_2:.*]] = VPUIP.ViewOp
    // CHECK:   [[VIEW_3:.*]] = VPUIP.ViewOp
    // CHECK:   [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!IpOp_Stub = memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @PatchFusedConstantWithSpillAsyncConstruct
func.func @PatchFusedConstantWithSpillAsyncConstruct() -> !IpOp_Stub {

    %cst_26 = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<64x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x5120xui8>,
            weightsTable=<dense<1> : tensor<64x1x1x4xsi32>>,
            weights=<dense<1.0> : tensor<64x64x1x1xf16>>>
        ]

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <173184> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>

    %buf_ddr = VPUIP.StaticAlloc<2076800> -> memref<1x1x1x5120xui8, @DDR>
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <1404928> -> memref<1x1x1x5120xui8, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x5120xui8, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 147 : i64} {
        %1 = VPUIP.NNDMA inputs(%cst_26 : memref<1x1x1x5120xui8>) outputs(%0 : memref<1x1x1x5120xui8, [@CMX_NN, 0]>) -> memref<1x1x1x5120xui8, [@CMX_NN, 0]>
    async.yield %1 : memref<1x1x1x5120xui8, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x5120xui8, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 150 : i64} {
        %2 = VPUIP.NNDMA inputs(%0 : memref<1x1x1x5120xui8, [@CMX_NN, 0]>) outputs(%buf_ddr : memref<1x1x1x5120xui8, @DDR>) -> memref<1x1x1x5120xui8, @DDR>
    async.yield %2 : memref<1x1x1x5120xui8, @DDR>
    }

    %t2, %r2 = async.execute (%r1 as %arg6: !async.value<memref<1x1x1x5120xui8, @DDR>>) -> !async.value<memref<1x1x1x5120xui8, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 152 : i64} {
        %3 = VPUIP.NNDMA inputs(%arg6 : memref<1x1x1x5120xui8, @DDR>) outputs(%0 : memref<1x1x1x5120xui8, [@CMX_NN, 0]>) -> memref<1x1x1x5120xui8, [@CMX_NN, 0]>
    async.yield %3 : memref<1x1x1x5120xui8, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute (%r2 as %arg7: !async.value<memref<1x1x1x5120xui8, [@CMX_NN, 0]>>) -> !async.value<!IpOp_Stub> attributes
        {VPUIP.executor = @DPU, "async-deps-index" = 155 : i64} {
        %4= VPUIP.SubView %arg7 [0, 0, 0, 0] [1, 1, 1, 1024] : memref<1x1x1x5120xui8, [@CMX_NN, 0]> to memref<1x1x1x1024xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, [@CMX_NN, 0]>
        %5 = VPUIP.SubView %arg7 [0, 0, 0, 1024] [1, 1, 1, 4096] : memref<1x1x1x5120xui8, [@CMX_NN, 0]> to memref<1x1x1x4096xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, [@CMX_NN, 0]>
        %6 = VPUIP.ViewOp %4: memref<1x1x1x1024xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, [@CMX_NN, 0]> to memref<64x1x1x4xsi32, [@CMX_NN, 0]>
        %7 = VPUIP.ViewOp %5 : memref<1x1x1x4096xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, [@CMX_NN, 0]> to memref<64x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
        %8 = VPUIP.NCEClusterTask {constantsFused = true, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 56892 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%7 : memref<64x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%6 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%in : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%out : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
        variants : {
            DPUTask {outEnd = [51, 51, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        }
        PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }
    async.yield %out : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    }

    %5 = async.await %r3 : !async.value<memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>>
    return %5 : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK-DAG:   [[FUSED_CONSTANT:%.*]] = const.Declare memref<1x1x1x5120xui8> =
    // CHECK-SAME:  #const.RelocateWeightsTable<weightsPtr=[1405952], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=1024 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>
    // CHECK:   [[INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <173184> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[DDR_FUSED_BUF:%.*]] = VPUIP.StaticAlloc<2076800> -> memref<1x1x1x5120xui8, @DDR>
    // CHECK:   [[FUSED_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1404928> -> memref<1x1x1x5120xui8, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x1x1x5120xui8, [@CMX_NN, 0]>>
    // CHECK:           [[VAR0:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs([[FUSED_CONSTANT]] : memref<1x1x1x5120xui8>)
    // CHECK-SAME:          outputs([[FUSED_BUF]] : memref<1x1x1x5120xui8, [@CMX_NN, 0]>)
    // CHECK:           async.yield [[VAR0]] : memref<1x1x1x5120xui8, [@CMX_NN, 0]>


    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x1x1x5120xui8, @DDR>>
    // CHECK:           [[VAR1:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs([[FUSED_BUF]] : memref<1x1x1x5120xui8, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[DDR_FUSED_BUF]] : memref<1x1x1x5120xui8, @DDR>
    // CHECK:           async.yield [[VAR1]] : memref<1x1x1x5120xui8, @DDR>

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       ([[R1]] as [[ARG1:%.*]]: !async.value<memref<1x1x1x5120xui8, @DDR>>)
    // CHECK-SAME:          -> !async.value<memref<1x1x1x5120xui8, [@CMX_NN, 0]>>
    // CHECK:           [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs([[ARG1]] : memref<1x1x1x5120xui8, @DDR>)
    // CHECK-SAME:          outputs([[FUSED_BUF]] : memref<1x1x1x5120xui8, [@CMX_NN, 0]>)
    // CHECK:           async.yield [[VAR2]] : memref<1x1x1x5120xui8, [@CMX_NN, 0]>

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>>
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView
    // CHECK:       [[VIEW_1:%.*]] = VPUIP.ViewOp
    // CHECK:       [[VIEW_2:%.*]] = VPUIP.ViewOp
    // CHECK:   [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
}
