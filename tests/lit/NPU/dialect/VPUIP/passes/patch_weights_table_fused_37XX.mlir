//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --patch-weight-table %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!IpOp_Stub = memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
!FusedConstantType_DDR = memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!FusedConstantType_CMX = memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

!FusedConstantType_CMX_SubView1 = memref<1x1x1x1024xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
!FusedConstantType_CMX_SubView2 = memref<1x1x1x4096xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1], swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

!FusedConstantType_CMX_View1 = memref<64x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
!FusedConstantType_CMX_View2 = memref<64x64x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

// CHECK-LABEL: @PatchFusedConstantWithSwizzling
func.func @PatchFusedConstantWithSwizzling() -> !IpOp_Stub {

    %cst_26 = const.Declare !FusedConstantType_DDR = dense<1> : tensor<64x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x5120xui8>,
            weightsTable=<dense<1> : tensor<64x1x1x4xsi32>>,
            weights=<dense<1.0> : tensor<64x64x1x1xf16>>>,
        #const.SwizzleConstant<5 : i64, 3 : i64>
        ]

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <173184> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <1404928> -> !FusedConstantType_CMX

    %t0, %r0 = async.execute -> !async.value<!FusedConstantType_CMX> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 147 : i64} {
        %1 = VPUIP.NNDMA inputs(%cst_26 : !FusedConstantType_DDR) outputs(%0 : !FusedConstantType_CMX) -> !FusedConstantType_CMX
    async.yield %1 : !FusedConstantType_CMX
    }

    %t1, %r1 = async.execute (%r0 as %arg7: !async.value<!FusedConstantType_CMX>) -> !async.value<!IpOp_Stub> attributes
        {VPUIP.executor = @DPU, "async-deps-index" = 155 : i64} {
        %4= VPUIP.SubView %arg7 [0, 0, 0, 0] [1, 1, 1, 1024] : !FusedConstantType_CMX to !FusedConstantType_CMX_SubView1
        %5 = VPUIP.SubView %arg7 [0, 0, 0, 1024] [1, 1, 1, 4096] : !FusedConstantType_CMX to !FusedConstantType_CMX_SubView2
        %6 = VPUIP.ViewOp %4: !FusedConstantType_CMX_SubView1 to !FusedConstantType_CMX_View1
        %7 = VPUIP.ViewOp %5: !FusedConstantType_CMX_SubView2 to !FusedConstantType_CMX_View2
        %8 = VPUIP.NCEClusterTask {constantsFused = true, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 56892 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%in : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>) weights(%7 : !FusedConstantType_CMX_View2) weight_table(%6 : !FusedConstantType_CMX_View1) parent_input(%in : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%out : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
        variants : {
            DPUTask {outEnd = [51, 51, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        }
        PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }
    async.yield %out : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    }

    %5 = async.await %r1 : !async.value<memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>>
    return %5 : memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:   [[FUSED_CONSTANT:%.*]] = const.Declare memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<
    // CHECK-SAME: #const.RelocateWeightsTable<weightsPtr=[1405952], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=1024 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64
    // CHECK-SAME: #const.SwizzleConstant<5 : i64, 3 : i64>
    // CHECK:   [[INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <173184> -> memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[FUSED_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1404928> -> memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>>
    // CHECK:           [[VAR0:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs([[FUSED_CONSTANT]] : memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>)
    // CHECK-SAME:          outputs([[FUSED_BUF]] : memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
    // CHECK:           async.yield [[VAR0]] : memref<1x1x1x5120xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x64x52x52xf16, #NHWC, [@CMX_NN, 0]>>
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView
    // CHECK:       [[VIEW_1:%.*]] = VPUIP.ViewOp
    // CHECK:       [[VIEW_2:%.*]] = VPUIP.ViewOp
    // CHECK:   [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
}
