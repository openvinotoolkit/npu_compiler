//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-eltwise-to-in-place --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseSameType
func.func @InplaceEltwiseSameType(%in: memref<1x32x96x96xf16, {order = #NHWC}>, %out: memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}> {
    %cst0 = const.Declare memref<1x32x96x96xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x96x96xf16>, [#const.Reorder<#NHWC>]

    %buf_in = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>
    %buf0 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>

    %0 = VPUIP.Copy inputs(%in : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf_in : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>

    %1 = VPUIP.Copy inputs(%cst0 : memref<1x32x96x96xf16, {order = #NHWC}>) outputs(%buf0 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>

    %2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                task_type = #VPUIP.nce_task_type<ELTWISE>,
                is_inplace = true
            }>
            input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
            weights(%1 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
            parent_input(%0 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
            parent_output(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%buf1 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
            }

    %3 = VPUIP.Copy inputs(%2 : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>) outputs(%out : memref<1x32x96x96xf16, {order = #NHWC}>) -> memref<1x32x96x96xf16, {order = #NHWC}>

    return %3 : memref<1x32x96x96xf16, {order = #NHWC}>

    // CHECK:       [[BUF0:%.+]] = memref.alloc()
    // CHECK:       [[BUF1:%.+]] = memref.alloc()
    // CHECK-NOT:   [[BUF3:%.+]] = memref.alloc()

    // CHECK:       [[VAL0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK:       [[VAL1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)

    // CHECK:       [[VAL2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[VAL0]] : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      weights([[VAL1]] : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)


    // CHECK:       [[VAL3:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAL2]] : memref<1x32x96x96xf16, {order = #NHWC}, @CMX_NN>)

    //CHECK:        return [[VAL3]] : memref<1x32x96x96xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseFpDistributedOp
// CHECK-SAME:    ([[ARG0:%.+]]: memref<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:    [[ARG1:%.+]]: memref<1x256x56x56xf16, {order = #NHWC}>)
// CHECK-SAME:    -> memref<1x256x56x56xf16, {order = #NHWC}> {
func.func @InplaceEltwiseFpDistributedOp(%in: memref<1x256x56x56xf16, {order = #NHWC}>, %out: memref<1x256x56x56xf16, {order = #NHWC}>) -> memref<1x256x56x56xf16, {order = #NHWC}> {

    %cst0 = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]

    %buf_0  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %buf_in = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %output_buf = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %0 = VPUIP.Copy
        inputs(%cst0 : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_0 : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%in : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_in : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%0 : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights(%1 : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_input(%0 : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output(%output_buf : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%output_buf : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %3 = VPUIP.Copy
        inputs(%2 : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%out : memref<1x256x56x56xf16, {order = #NHWC}>)  ->  memref<1x256x56x56xf16, {order = #NHWC}>

    return %3 : memref<1x256x56x56xf16, {order = #NHWC}>

    //CHECK-DAG:    [[CST:%.+]] = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[BUF_IN0:%.+]] = VPURT.AllocDistributed
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        [[BUF_IN1:%.+]] = VPURT.AllocDistributed
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-NOT:    VPURT.AllocDistributed
    // CHECK:    [[COPY_IN0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<1x256x56x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF_IN0]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[COPY_IN1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG0]]
    // CHECK-SAME:     outputs([[BUF_IN1]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    //CHECK:        [[ELTW_RES:%.+]] = VPUIP.NCEClusterTask
    //CHECK-SAME:       input([[COPY_IN0]]
    //CHECK-SAME:       output([[BUF_IN0]]
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ELTW_RES]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[ARG1]]
    // CHECK-SAME:     -> memref<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_COPY]] : memref<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.011588541666666667:128>
!qElemType1 = !quant.uniform<u8:f16, 0.020557598039215686:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0088848039215686271>

!qType0 = memref<1x256x56x56x!qElemType, {order = #NHWC}>
!qType1 = memref<1x256x56x56x!qElemType1, {order = #NHWC}>
!qType2 = memref<1x256x56x56x!qElemType2, {order = #NHWC}>

!qType0CMX = memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>
!qType1CMX = memref<1x256x56x56x!qElemType1, {order = #NHWC}, @CMX_NN>
!qType2CMX = memref<1x256x56x56x!qElemType2, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @InplaceEltwiseQuantizedView
// CHECK-SAME:    ([[ARG0:%.+]]: memref<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:    [[ARG1:%.+]]: memref<1x256x56x56x!qElemType1, {order = #NHWC}>,
// CHECK-SAME:    [[ARG2:%.+]]: memref<1x256x56x56x!qElemType2, {order = #NHWC}>)
// CHECK-SAME:    -> memref<1x256x56x56x!qElemType2, {order = #NHWC}> {
func.func @InplaceEltwiseQuantizedView(%in: !qType0, %in2: !qType1, %out: !qType2) -> !qType2 {
    %buf_in = memref.alloc() : !qType0CMX
    %buf0 = memref.alloc() : !qType1CMX
    %buf1 = memref.alloc() : !qType2CMX

    %0 = VPUIP.Copy inputs(%in : !qType0) outputs(%buf_in : !qType0CMX) -> !qType0CMX
    %1 = VPUIP.Copy inputs(%in2 : !qType1) outputs(%buf0 : !qType1CMX) -> !qType1CMX

    %2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 11669 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%0 : !qType0CMX)
        weights(%1 : !qType1CMX)
        parent_input(%0 : !qType0CMX)
        parent_output(%buf1 : !qType2CMX)
        outputs(%buf1 : !qType2CMX)
        -> !qType2CMX variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
                PPETask {ppe = #VPU.PPEStub<>}
        }

    %3 = VPUIP.Copy inputs(%2 : !qType2CMX) outputs(%out : !qType2) -> !qType2

    return %3 : !qType2

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[VIEW0:%.+]] = VPUIP.ViewOp
    // CHECK-SAME:       [[BUF0]] : memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK-SAME:       to memref<1x256x56x56x!qElemType2, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUF2:%.+]] = memref.alloc() : memref<1x256x56x56x!qElemType1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[INP1:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x256x56x56x!qElemType, {order = #NHWC}>) outputs([[BUF0]] : memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[INP2:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<1x256x56x56x!qElemType1, {order = #NHWC}>) outputs([[BUF2]] : memref<1x256x56x56x!qElemType1, {order = #NHWC}, @CMX_NN>) -> memref<1x256x56x56x!qElemType1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[ELTWISE_OUT:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 11669 : i64} <{is_inplace = true
    // CHECK-SAME:      input([[INP1]] : memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      weights([[INP2]] : memref<1x256x56x56x!qElemType1, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INP1]] : memref<1x256x56x56x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_output([[VIEW0]] : memref<1x256x56x56x!qElemType2, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[VIEW0]] : memref<1x256x56x56x!qElemType2, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x256x56x56x!qElemType2, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ELTWISE_OUT]] : memref<1x256x56x56x!qElemType2, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x256x56x56x!qElemType2, {order = #NHWC}>)
    // CHECK-SAME:      -> memref<1x256x56x56x!qElemType2, {order = #NHWC}>

    // CHECK:    return [[COPY_OUT]] : memref<1x256x56x56x!qElemType2, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistType0 = !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!InputDistType1 = !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!EltwiseDistType = !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// CHECK-LABEL: @InplaceEltwiseNeedsCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x512x28x28xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x512x28x28xf16, {order = #NHWC}>)
func.func @InplaceEltwiseNeedsCast(%input : memref<1x512x28x28xf16, {order = #NHWC}>, %out: memref<1x512x28x28xf16, {order = #NHWC}>) -> memref<1x512x28x28xf16, {order = #NHWC}> {

    %inputBuf0 = VPURT.AllocDistributed -> !InputDistType0
    %inputBuf1 = VPURT.AllocDistributed -> !InputDistType1
    %copyInput0 = VPUIP.Copy
        inputs(%input : memref<1x512x28x28xf16, {order = #NHWC}>)
        outputs(%inputBuf0 : !InputDistType0)  ->  !InputDistType0
    %copyInput1 = VPUIP.Copy
        inputs(%input : memref<1x512x28x28xf16, {order = #NHWC}>)
        outputs(%inputBuf1 : !InputDistType1)  ->  !InputDistType1

    %eltwiseIn0 = VPUIP.DistributedCast inputs(%copyInput0 : !InputDistType0) -> !EltwiseDistType
    %eltwiseIn1 = VPUIP.DistributedCast inputs(%copyInput1 : !InputDistType1) -> !EltwiseDistType
    // This buffer will be eliminated, input 0 will be used insted
    %eltwiseOutBuf = VPURT.AllocDistributed -> !EltwiseDistType
    %eltwise_output = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 32317 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%eltwiseIn0 : !EltwiseDistType)
        weights(%eltwiseIn1 : !EltwiseDistType)
        parent_input(%eltwiseIn0 : !EltwiseDistType)
        parent_output(%eltwiseOutBuf : !EltwiseDistType)
        outputs(%eltwiseOutBuf : !EltwiseDistType)
    ->  !EltwiseDistType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 27, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 27, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %copyOut = VPUIP.Copy
        inputs(%eltwise_output : !EltwiseDistType)
        outputs(%out : memref<1x512x28x28xf16, {order = #NHWC}>)  ->  memref<1x512x28x28xf16, {order = #NHWC}>

    return %copyOut : memref<1x512x28x28xf16, {order = #NHWC}>

    // output of Eltwise has been redirected to this buffer which is the first input
    // CHECK:       [[BUF0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // since the first input has different distribution mode it need distribution cast operation
    // CHECK:       [[ELTW_OUT_BUF:%.+]] = VPUIP.DistributedCast
    // CHECK-SAME:       inputs([[BUF0]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[BUF1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]] : memref<1x512x28x28xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF0]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}

    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]] : memref<1x512x28x28xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF1]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}

    // CHECK:       [[ELTW_IN0:%.+]] = VPUIP.DistributedCast
    // CHECK-SAME:      inputs([[COPY0]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[ELTW_IN1:%.+]] = VPUIP.DistributedCast
    // CHECK-SAME:      inputs([[COPY1]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[ELTW_OUT:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ELTW_IN0]]
    // CHECK-SAME:      output([[ELTW_OUT_BUF]]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}

    // CHECK:    [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ELTW_OUT]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[ARG_1]] : memref<1x512x28x28xf16, {order = #NHWC}>)  -> memref<1x512x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT]] : memref<1x512x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:    func @InplaceEltwiseFirstInputHas2Consumers([[ARG_0:%[^:]+]]: memref<1x256x56x56xf16, {order = #NHWC}>)
func.func @InplaceEltwiseFirstInputHas2Consumers(%in: memref<1x256x56x56xf16, {order = #NHWC}>) -> (!DistributedType, !DistributedType) {

    %cst0 = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<1.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    %buf_0  = VPURT.AllocDistributed -> !DistributedType
    %buf_1  = VPURT.AllocDistributed -> !DistributedType
    %buf_in = VPURT.AllocDistributed -> !DistributedType
    %buf_in_1 = VPURT.AllocDistributed -> !DistributedType
    %output_buf = VPURT.AllocDistributed -> !DistributedType
    %output_buf_1 = VPURT.AllocDistributed -> !DistributedType
    %0 = VPUIP.Copy
        inputs(%cst0 : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_0 : !DistributedType)  ->  !DistributedType
    %1 = VPUIP.Copy
        inputs(%in : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_in : !DistributedType)  ->  !DistributedType
    %2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%0 : !DistributedType)
        weights(%1 : !DistributedType)
        parent_input(%0 : !DistributedType)
        parent_output(%output_buf : !DistributedType)
        outputs(%output_buf : !DistributedType)
    ->  !DistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %4 = VPUIP.Copy
        inputs(%cst1 : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_1 : !DistributedType)  ->  !DistributedType
    %5 = VPUIP.Copy
        inputs(%in : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_in_1 : !DistributedType)  ->  !DistributedType
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%5 : !DistributedType)
        weights(%4 : !DistributedType)
        parent_input(%5 : !DistributedType)
        parent_output(%output_buf_1 : !DistributedType)
        outputs(%output_buf_1 : !DistributedType)
    ->  !DistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %2, %6 : !DistributedType, !DistributedType

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[BUF_0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_IN0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_IN1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:       VPURT.AllocDistributed
    // CHECK:    [[COPY_IN0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST_0]] : memref<1x256x56x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[COPY_IN1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]] : memref<1x256x56x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF_IN0]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:        [[ELTW_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:        input([[COPY_IN0]]
    // CHECK-SAME:        output([[OUTPUT_BUF]]
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[COPY_IN2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<1x256x56x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF_1]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    // CHECK:    [[COPY_IN3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]] : memref<1x256x56x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF_IN1]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:        [[ELTW_RES1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:        input([[COPY_IN3]]
    // CHECK-SAME:        weights([[COPY_IN2]]
    // CHECK-SAME:        outputs([[BUF_1]]
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:        return [[ELTW_RES]], [[ELTW_RES1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!PermuteInDistributedType = !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64
}>

!PermuteOutDistributedType = !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64
}>

!EltwiseDistributedType = !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:    func @InplaceEltwiseFirstInput2ConsumersSecondInputFromNCEPermute
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x256x56x56xf16, {order = #NHWC}>)
func.func @InplaceEltwiseFirstInput2ConsumersSecondInputFromNCEPermute(%in: memref<1x256x56x56xf16, {order = #NHWC}>) -> (!EltwiseDistributedType, !EltwiseDistributedType) {

    %cst0 = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<1x56x256x56xf16, {order = #NHWC}> = dense<1.0> : tensor<1x56x256x56xf16>, [#const.Reorder<#NHWC>]
    %buf_0  = VPURT.AllocDistributed -> !EltwiseDistributedType
    %buf_1  = VPURT.AllocDistributed -> !PermuteInDistributedType
    %permute_out_buf  = VPURT.AllocDistributed -> !PermuteOutDistributedType
    %buf_in = VPURT.AllocDistributed -> !EltwiseDistributedType
    %buf_in_1 = VPURT.AllocDistributed -> !EltwiseDistributedType
    %output_buf = VPURT.AllocDistributed -> !EltwiseDistributedType
    %elt_in_place_out_buf = VPURT.AllocDistributed -> !EltwiseDistributedType
    %0 = VPUIP.Copy
        inputs(%cst0 : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_0 : !EltwiseDistributedType)  ->  !EltwiseDistributedType
    %1 = VPUIP.Copy
        inputs(%in : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_in : !EltwiseDistributedType)  ->  !EltwiseDistributedType
    %2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%0 : !EltwiseDistributedType)
        weights(%1 : !EltwiseDistributedType)
        parent_input(%0 : !EltwiseDistributedType)
        parent_output(%output_buf : !EltwiseDistributedType)
        outputs(%output_buf : !EltwiseDistributedType)
    ->  !EltwiseDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %4 = VPUIP.Copy
        inputs(%cst1 : memref<1x56x256x56xf16, {order = #NHWC}>)
        outputs(%buf_1 : !PermuteInDistributedType)  ->  !PermuteInDistributedType
    %5 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 17144 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_permute_quantize, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%4 : !PermuteInDistributedType)
        weights(%4 : !PermuteInDistributedType)
        parent_input(%4 : !PermuteInDistributedType)
        parent_output(%permute_out_buf : !PermuteOutDistributedType)
        outputs(%permute_out_buf : !PermuteOutDistributedType)
    ->  !PermuteOutDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 255, 55], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 255, 55], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %6 = VPUIP.Copy
        inputs(%in : memref<1x256x56x56xf16, {order = #NHWC}>)
        outputs(%buf_in_1 : !EltwiseDistributedType)  ->  !EltwiseDistributedType

    %7 = VPUIP.ViewOp %5 : !PermuteOutDistributedType to !EltwiseDistributedType

    %8 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%6 : !EltwiseDistributedType)
        weights(%7 : !EltwiseDistributedType)
        parent_input(%6 : !EltwiseDistributedType)
        parent_output(%elt_in_place_out_buf : !EltwiseDistributedType)
        outputs(%elt_in_place_out_buf : !EltwiseDistributedType)
            -> !EltwiseDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %2, %8 : !EltwiseDistributedType, !EltwiseDistributedType

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x256x56x56xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare memref<1x56x256x56xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x56x256x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[BUF_0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERM_QUANT_IN:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK:       [[PERM_QUANT_OUT:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>

    // CHECK:       [[VIEW0:%.+]] = VPUIP.ViewOp [[PERM_QUANT_OUT]] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[BUF_IN1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_IN2:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:       VPURT.AllocDistributed
    // CHECK:    [[COPY_IN0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<1x256x56x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[COPY_IN1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG0]]
    // CHECK-SAME:     outputs([[BUF_IN1]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:        [[ELTW_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:        input([[COPY_IN0]]
    // CHECK-SAME:        weights([[COPY_IN1]]
    // CHECK-SAME:        outputs([[OUTPUT_BUF]]
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    // CHECK:    [[COPY_IN_PERM:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST_0]] : memref<1x56x256x56xf16, {order = #NHWC}>)
    // CHECK-SAME:     outputs([[PERM_QUANT_IN]] : !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}


    // CHECK:        [[PERM_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:        is_permute_quantize
    // CHECK-SAME:        input([[COPY_IN_PERM]]
    // CHECK-SAME:        output([[PERM_QUANT_OUT]]
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}

    // CHECK:    [[COPY_IN3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG0]]
    // CHECK-SAME:     outputs([[BUF_IN2]] : !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:       [[VIEW1:%.+]] = VPUIP.ViewOp [[PERM_RES]] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[ELTW_RES_IN_PLACE:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:        is_inplace = true
    // CHECK-SAME:        input([[COPY_IN3]]
    // CHECK-SAME:        weights([[VIEW1]]
    // CHECK-SAME:        output([[VIEW0]]
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:        return [[ELTW_RES]], [[ELTW_RES_IN_PLACE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x128x52x104xf16, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x128x104x208xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
    func.func nested @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK:    func @InplaceEltwiseSubViewInterp([[ARG_0:%[^:]+]]: memref<1x128x104x104xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: memref<1x128x104x104xf16, {order = #NHWC}>)
func.func @InplaceEltwiseSubViewInterp(%in1: memref<1x128x104x104xf16, {order = #NHWC}>, %in2: memref<1x128x104x104xf16, {order = #NHWC}>) -> (!DistributedType2, !DistributedType2, !DistributedType1) {
    %buf_0  = VPURT.AllocDistributed -> !DistributedType1
    %buf_2  = VPURT.AllocDistributed -> !DistributedType1
    %buf_in = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_1 = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_2 = VPURT.AllocDistributed -> !DistributedType1
    %output_buf = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_1 = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_2 = VPURT.AllocDistributed -> !DistributedType1

    %0 = VPUIP.SubView %in1 [0, 0, 0, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    %1 = VPUIP.SubView %in1 [0, 0, 0, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>
    %3 = VPUIP.Copy
        inputs(%0 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
        outputs(%buf_in : !DistributedType1)  ->  !DistributedType1

    %4 = VPUIP.SW.Kernel
        {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate
            inputs(%3 as %arg6: memref<1x128x52x104xf16, {order = #NHWC}, @CMX_NN>) outputs(%output_buf as %arg7: memref<1x128x104x208xf16, {order = #NHWC}, @CMX_NN>) on tile 0 -> !DistributedType2 {
        VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [128, 52, 104, 1], [128, 104, 208, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg6, %arg7) : memref<1x128x52x104xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x104x208xf16, {order = #NHWC}, @CMX_NN>
    }

    %6 = VPUIP.SubView %in2 [0, 0, 0, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    %7 = VPUIP.SubView %in2 [0, 0, 52, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>
    %9 = VPUIP.Copy
        inputs(%7 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
        outputs(%buf_in_1 : !DistributedType1)  ->  !DistributedType1

    %10 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%9 as %arg6: memref<1x128x52x104xf16, {order = #NHWC}, @CMX_NN>) outputs(%output_buf_1 as %arg7: memref<1x128x104x208xf16, {order = #NHWC}, @CMX_NN>) on tile 0 -> !DistributedType2 {
        VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [128, 52, 104, 1], [128, 104, 208, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg6, %arg7) : memref<1x128x52x104xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x104x208xf16, {order = #NHWC}, @CMX_NN>
        }

    %12 = VPUIP.Copy
        inputs(%1 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
        outputs(%buf_2 : !DistributedType1)  ->  !DistributedType1
    %13 = VPUIP.Copy
        inputs(%6 : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
        outputs(%buf_in_2 : !DistributedType1)  ->  !DistributedType1
    %14 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%12 : !DistributedType1)
        weights(%13 : !DistributedType1)
        parent_input(%12 : !DistributedType1)
        parent_output(%output_buf_2 : !DistributedType1)
        outputs(%output_buf_2 : !DistributedType1)
    ->  !DistributedType1 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 51, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 103, 127], outStart = [0, 52, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    return %4, %10, %14 : !DistributedType2, !DistributedType2, !DistributedType1

    // CHECK:         [[BUF_2:%.+]]  = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN2:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         [[OUTPUT_BUF1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-NOT:         VPURT.AllocDistributed

    // CHECK:        [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    // CHECK:        [[SUBVIEW1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>
    // CHECK:    [[COPY_IN1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW0]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
    // CHECK-SAME:     outputs([[BUF_IN0]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:        [[INTERP_0:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:        inputs([[COPY_IN1]] as {{%[^:]+}}: memref<1x128x52x104xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:        outputs([[OUTPUT_BUF]] as {{%[^:]+}}: memref<1x128x104x208xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:        [[SUBVIEW2:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>
    // CHECK:        [[SUBVIEW3:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 52, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, {order = #NHWC}> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>
    // CHECK:    [[COPY_IN3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW3]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
    // CHECK-SAME:     outputs([[BUF_IN1]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>



    // CHECK:        [[INTERP_1:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:        inputs([[COPY_IN3]] as {{%[^:]+}}: memref<1x128x52x104xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:        outputs([[OUTPUT_BUF1]] as {{%[^:]+}}: memref<1x128x104x208xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[COPY_IN4:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW1]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
    // CHECK-SAME:     outputs([[BUF_2]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[COPY_IN5:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>)
    // CHECK-SAME:     outputs([[BUF_IN2]] : !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:        [[ELTW_RES2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:        input([[COPY_IN4]]
    // CHECK-SAME:        output([[BUF_IN2]]
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:        return [[INTERP_0]], [[INTERP_1]], [[ELTW_RES2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType2 = !quant.uniform<u8:f16, 2.0000000000000000E-1>


!DistributedType1 = !VPUIP.DistributedBuffer<
    1x64x26x52x!qElemType2,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x64x13x26x!qElemType2,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK:    func @InplaceEltwisePermQuantSubView([[ARG_0:%[^:]+]]: memref<1x64x52x52x!qElemType>, [[ARG_1:%[^:]+]]: memref<1x64x52x52x!qElemType>)
func.func @InplaceEltwisePermQuantSubView(%in1: memref<1x64x52x52x!qElemType1>, %in2: memref<1x64x52x52x!qElemType1>) -> (!DistributedType2, !DistributedType2, !DistributedType1) {
    %wt = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x1x1x4xsi32>
    %buf  = VPURT.AllocDistributed -> !DistributedType1
    %buf_in = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_1 = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_2 = VPURT.AllocDistributed -> !DistributedType1
    %output_buf = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_1 = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_2 = VPURT.AllocDistributed -> !DistributedType1

    %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
        inputs(%in1 : memref<1x64x52x52x!qElemType1>)
        -> memref<1x64x52x52x!qElemType1, {order = #NHWC}>

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x64x52x52x!qElemType1, {order = #NHWC}>) -> memref<1x64x52x52x!qElemType2, {order = #NHWC}>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, {order = #NHWC}> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, {order = #NHWC}> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
        outputs(%buf_in : !DistributedType1)  ->  !DistributedType1
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10325 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}>
        input(%4 : !DistributedType1)
        weight_table(%wt : memref<64x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : !DistributedType1)
        parent_output(%output_buf : !DistributedType2)
        outputs(%output_buf : !DistributedType2)
    ->  !DistributedType2 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 13, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
        inputs(%in2 : memref<1x64x52x52x!qElemType1>) -> memref<1x64x52x52x!qElemType1, {order = #NHWC}>

    %8 = VPUIP.QuantizeCast inputs(%7 : memref<1x64x52x52x!qElemType1, {order = #NHWC}>) -> memref<1x64x52x52x!qElemType2, {order = #NHWC}>

    %10 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, {order = #NHWC}> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    %11 = VPUIP.SubView %8 [0, 0, 26, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, {order = #NHWC}> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>
    %12 = VPUIP.Copy
        inputs(%10 : memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
        outputs(%buf_in_1 : !DistributedType1)  ->  !DistributedType1
    %13 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 10325 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}>
        input(%12 : !DistributedType1)
        weight_table(%wt : memref<64x1x1x4xsi32, @CMX_NN>)
        parent_input(%12 : !DistributedType1)
        parent_output(%output_buf_1 : !DistributedType2)
        outputs(%output_buf_1 : !DistributedType2)
    ->  !DistributedType2 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 13, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %14 = VPUIP.Copy
        inputs(%3 : memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
        outputs(%buf : !DistributedType1)  ->  !DistributedType1
    %15 = VPUIP.Copy
        inputs(%11 : memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
        outputs(%buf_in_2 : !DistributedType1)  ->  !DistributedType1
    %16 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 31170 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%14 : !DistributedType1)
        weights(%15 : !DistributedType1)
        parent_input(%14 : !DistributedType1)
        parent_output(%output_buf_2 : !DistributedType1)
        outputs(%output_buf_2 : !DistributedType1)
    ->  !DistributedType1 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 51, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 103, 127], outStart = [0, 52, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %6, %13, %16 : !DistributedType2, !DistributedType2, !DistributedType1

    // CHECK-DAG:     [[WT:%.+]] = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:         [[BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN_2:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_OUT:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_OUT_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:       VPURT.AllocDistributed

    // CHECK:         [[PERM_CAST:%.+]] = VPUIP.PermuteCast
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:       inputs([[ARG_0]] : memref<1x64x52x52x!qElemType>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType, {order = #NHWC}>

    // CHECK:         [[QUANT_CAST:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:       inputs([[PERM_CAST]] : memref<1x64x52x52x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType1, {order = #NHWC}>

    // CHECK:         [[SUB_VIEW:%.+]] = VPUIP.SubView [[QUANT_CAST]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    // CHECK:         [[SUB_VIEW_1:%.+]] = VPUIP.SubView [[QUANT_CAST]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUB_VIEW]] : memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
    // CHECK-SAME:     outputs([[BUF_IN]] : !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:         [[MAXPOOL:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:       input([[COPY]]
    // CHECK-SAME:       weight_table([[WT]]
    // CHECK-SAME:       output([[BUF_OUT]]
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:         [[PERM_CAST_1:%.+]] = VPUIP.PermuteCast
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:       inputs([[ARG_1]] : memref<1x64x52x52x!qElemType>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType, {order = #NHWC}>

    // CHECK:         [[QUANT_CAST_1:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:       inputs([[PERM_CAST_1]] : memref<1x64x52x52x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType1, {order = #NHWC}>

    // CHECK:         [[SUB_VIEW_2:%.+]] = VPUIP.SubView [[QUANT_CAST_1]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    // CHECK:         [[SUB_VIEW_3:%.+]] = VPUIP.SubView [[QUANT_CAST_1]]
    // CHECK-SAME:       [0, 0, 26, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUB_VIEW_2]] : memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
    // CHECK-SAME:     outputs([[BUF_IN_1]] : !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:         [[MAXPOOL_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:       input([[COPY_1]]
    // CHECK-SAME:       output([[BUF_OUT_1:]]
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    // CHECK:    [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUB_VIEW_1]] : memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
    // CHECK-SAME:     outputs([[BUF]] : !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:    [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUB_VIEW_3]] : memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>)
    // CHECK-SAME:     outputs([[BUF_IN_2]] : !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}


    // CHECK:         [[INPLACE_ELTWISE:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:       input([[COPY_2]]
    // CHECK-SAME:       output([[BUF_IN_2]]
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}

    // CHECK:         return [[MAXPOOL]], [[MAXPOOL_1]], [[INPLACE_ELTWISE]]
}
