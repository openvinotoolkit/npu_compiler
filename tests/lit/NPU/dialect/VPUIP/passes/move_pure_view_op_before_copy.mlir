//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MovePureViewOpBeforeCopyMultipleConsumers(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX>,
        %arg1: memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> (memref<1x16x56x224xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NCHW, @DDR>) {
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NCHW, @DDR>

    %3 = VPUIP.GenericReshape inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x56x224xf16, #NHWC, @DDR>

    %4 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        outputs(%arg1 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> memref<1x16x112x112xf16, #NCHW, @DDR>

    return %3, %4 : memref<1x16x56x224xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NCHW, @DDR>


    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, @CMX>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[PERMUTECAST]] : memref<1x16x112x112xf16, @CMX>) outputs([[ALLOC0]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    // CHECK: [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x56x224xf16, #NHWC, @CMX>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x56x224xf16, #NHWC, @DDR>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[GENERICRESHAPE]] : memref<1x16x56x224xf16, #NHWC, @CMX>) outputs([[ALLOC1]] : memref<1x16x56x224xf16, #NHWC, @DDR>) -> memref<1x16x56x224xf16, #NHWC, @DDR>

    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x16x112x112xf16, @DDR>) outputs(%arg1 : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    // CHECK: return [[COPY1]], [[COPY2]] : memref<1x16x56x224xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MovePureViewOpBeforeCopy(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX>,
        %arg1: memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> memref<1x16x112x112xf16, #NCHW, @DDR> {
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NCHW, @DDR>

    %3 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        outputs(%arg1 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> memref<1x16x112x112xf16, #NCHW, @DDR>

    return %3 : memref<1x16x112x112xf16, #NCHW, @DDR>

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK-SAME:                   inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, @CMX>

    //CHECK:        [[VAR1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    //CHECK:        [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, @CMX>)
    //CHECK-SAME:                   outputs([[VAR1]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    //CHECK:        [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, @DDR>)
    //CHECK-SAME:                   outputs(%arg1 : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    //CHECK:        return [[VAR3]] : memref<1x16x112x112xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveSeveralPureViewOpsBeforeCopy(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX>,
        %arg1: memref<1x16x392x32xf16, @DDR>)
        -> memref<1x16x392x32xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x12544xf16, @DDR>

    %4 = memref.alloc() : memref<1x16x12544xf16, @DDR>
    %5 = VPUIP.Copy inputs(%3 : memref<1x16x12544xf16, @DDR>)
        outputs(%4 : memref<1x16x12544xf16, @DDR>)
        -> memref<1x16x12544xf16, @DDR>

    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x16x12544xf16, @DDR>) -> memref<1x16x392x32xf16, @DDR>

    %7 = VPUIP.Copy inputs(%6 : memref<1x16x392x32xf16, @DDR>)
        outputs(%arg1 : memref<1x16x392x32xf16, @DDR>)
        -> memref<1x16x392x32xf16, @DDR>

    return %7 : memref<1x16x392x32xf16, @DDR>

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK-SAME:                   inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, @CMX>
    //CHECK:        [[VAR1:%.+]] = VPUIP.GenericReshape inputs([[VAR0]] : memref<1x16x112x112xf16, @CMX>) -> memref<1x16x12544xf16, @CMX>
    //CHECK:        [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x16x12544xf16, @CMX>) -> memref<1x16x392x32xf16, @CMX>

    //CHECK:        [[VAR3:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x392x32xf16, @CMX>) outputs([[VAR3]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        [[VAR5:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x392x32xf16, @DDR>) outputs([[VAR5]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x392x32xf16, @DDR>) outputs(%arg1 : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        return [[VAR7]] : memref<1x16x392x32xf16, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @QuantizeCastBeforeDistributedCopy(%arg0: memref<1x128x8x8x!qElemType, #NHWC, @CMX_NN>) -> memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN> {
    %buf0 = memref.alloc() : memref<1x128x8x8x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x8x8x!qElemType, #NHWC, @CMX_NN>)
        outputs(%buf0 : memref<1x128x8x8x!qElemType, #NHWC, @DDR>)  ->  memref<1x128x8x8x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x128x8x8x!qElemType, #NHWC, @DDR>) -> memref<1x128x8x8x!qElemType1, #NHWC, @DDR>

    %buf1 = memref.alloc() : memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy
        inputs(%1 : memref<1x128x8x8x!qElemType1, #NHWC, @DDR>)
        outputs(%buf1 : memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>)  ->  memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>

    return %2 : memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.+]] = VPUIP.QuantizeCast inputs(%arg0 :
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[VAR0]]
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[VAR1]]
    // CHECK:       return [[VAR2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @MoveSubviewToTheFrontOfCopy(%arg0: memref<1x16x2x2xf16, @DDR>, %arg1: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %1 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %2 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%1 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 8, 2, 2] : memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>) outputs(%arg1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    return %4 : memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 8, 2, 2] :
    // CHECK-SAME:                      memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>

    // CHECK:       [[VAR1:%.+]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:                      outputs([[VAR1]] : memref<1x8x2x2xf16, @DDR>)

    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x8x2x2xf16, @DDR>)
    // CHECK-SAME:                      outputs(%arg1 : memref<1x8x2x2xf16, @DDR>)

    // CHECK:       return [[VAR3]] : memref<1x8x2x2xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0069489371542837105>

func.func @DoNotMoveSubviewToTheFrontOfSparseCopy(%arg0: !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>>)
-> !VPUIP.SparseBuffer<data=memref<1x64x17x56x!qElemType, {order = #NHWC, strides = [200704, 1, 3584, 64]}>, sparsity_map=memref<1x64x17x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}>> {
    %alloc_0 = memref.alloc() : memref<1x64x56x56x!qElemType, #NHWC>
    %alloc_1 = memref.alloc() : memref<1x64x56x56xi1, #NHWC>
    %0 = VPUIP.GroupSparseBuffer(%alloc_0, %alloc_1) -> !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, #NHWC>, sparsity_map=memref<1x64x56x56xi1, #NHWC>>
    %1 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>>) outputs(%0 : !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, #NHWC>, sparsity_map=memref<1x64x56x56xi1, #NHWC>>) -> !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, #NHWC>, sparsity_map=memref<1x64x56x56xi1, #NHWC>>
    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 17, 56] : !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, #NHWC>, sparsity_map=memref<1x64x56x56xi1, #NHWC>> to !VPUIP.SparseBuffer<data=memref<1x64x17x56x!qElemType, {order = #NHWC, strides = [200704, 1, 3584, 64]}>, sparsity_map=memref<1x64x17x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}>>

    return %2 : !VPUIP.SparseBuffer<data=memref<1x64x17x56x!qElemType, {order = #NHWC, strides = [200704, 1, 3584, 64]}>, sparsity_map=memref<1x64x17x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}>>

    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x56x56x!qElemType, #NHWC>
    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x64x56x56xi1, #NHWC>
    // CHECK:       [[SPARSEBUFFER:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC0]], [[ALLOC1]])
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs(%arg0
    // CHECK-SAME:  outputs([[SPARSEBUFFER]]
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[COPY]] [0, 0, 0, 0] [1, 64, 17, 56]
    // CHECK: return [[SUBVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

 func.func @MoveSubviewToTheFrontOfTillingCopy(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x32x48x88x!qElemType, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        weights(%in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        parent_input(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        outputs(%2 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)  ->  memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 48, 88] : memref<1x64x48x88x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> to memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [270336, 1, 5632, 64]}, @DDR>
    %5 = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [270336, 1, 5632, 64]}, @DDR>) outputs(%5 : memref<1x32x48x88x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x48x88x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    return %6 : memref<1x32x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK:     ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> variants : {
    // CHECK:     DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<MATRIX>, outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
    // CHECK:     PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ADD_0]] [0, 0, 0, 0] [1, 32, 48, 88] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:    [[Tilling_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>)  ->  memref<1x32x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[Tilling_COPY]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_2]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK:       return [[COPY]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @DoNotMoveSubviewToTheFrontOfTillingCopy(%arg0: memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR> {

    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x136x240xf16, #NCHW, @CMX_NN,
        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    %alloc = memref.alloc() : memref<1x16x136x240xf16, @DDR>
    %1 = VPUIP.Copy
        inputs(%0 : !VPUIP.DistributedBuffer<1x16x136x240xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>)
        outputs(%alloc : memref<1x16x136x240xf16, @DDR>)  ->  memref<1x16x136x240xf16, @DDR>
    %2 = VPUIP.SubView %1 [0, 1, 0, 0] [1, 1, 136, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
    outputs(%arg0 : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR>
    return %3 : memref<1x1x136x240xf16, @DDR>


    // CHECK:       [[ALLOCDISTRIBUTED:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x136x240xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x16x136x240xf16, @DDR>
    // CHECK:       [[CLUSTERTILLING:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ALLOCDISTRIBUTED]] : !VPUIP.DistributedBuffer<1x16x136x240xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>)
    // CHECK-SAME:     outputs([[ALLOC]] : memref<1x16x136x240xf16, @DDR>)  ->  memref<1x16x136x240xf16, @DDR>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[CLUSTERTILLING]] [0, 1, 0, 0] [1, 1, 136, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW:%.+]] : memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
    // CHECK-SAME:  outputs({{[^:]+}} : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR>
    // CHECK:       return [[COPY0]] : memref<1x1x136x240xf16, @DDR>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NoChangesForStridedCopy(
        %arg0: memref<1x2x16x4xf16, @DDR>)
        -> memref<16x4xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 16, 4] : memref<1x2x16x4xf16, @DDR> to memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>

    %1 = memref.alloc() : memref<1x1x16x4xf16, @DDR>
    %2 = VPUIP.Copy inputs(%0 : memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>)
        outputs(%1 : memref<1x1x16x4xf16, @DDR>)
        -> memref<1x1x16x4xf16, @DDR>

    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x1x16x4xf16, @DDR>) -> memref<16x4xf16, @DDR>

    return %3 : memref<16x4xf16, @DDR>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 16, 4] : memref<1x2x16x4xf16, @DDR> to memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x1x16x4xf16, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x1x16x4xf16, @DDR>)
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x1x16x4xf16, @DDR>) -> memref<16x4xf16, @DDR>

    // CHECK:       return [[RESHAPE]] : memref<16x4xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = memref<1x16x32x32x!qElemType , #NHWC, @CMX_NN>
!Output_DDR = memref<1x16x32x32x!qElemType, #NHWC, @DDR>

func.func @MoveQuantizeCastBeforeTilingCopy(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : !Output_DDR
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : !Output_DDR)  ->  !Output_DDR
    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_0:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:       return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16,#NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = memref<1x16x32x32xf16, @CMX_NN>
!Output_DDR = memref<1x16x32x32xf16, @DDR>

func.func @MoveGenericReshapeBeforeTilingCopyRankShrinks(
        %arg0: !InputDistributed)
        -> memref<1x16x1024xf16, @DDR> {
    %out = memref.alloc() : !Output_DDR
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : !Output_DDR)  ->  !Output_DDR
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, @DDR>) -> memref<1x16x1024xf16, @DDR>

    return %1 : memref<1x16x1024xf16, @DDR>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x1024xf16, #CHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_0:%.+]] = memref.alloc() : memref<1x16x1024xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x16x1024xf16, #CHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x1024xf16, @DDR>)  ->  memref<1x16x1024xf16, @DDR>
    // CHECK:       return [[COPY]] : memref<1x16x1024xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!Output_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>

func.func @MovePermuteCastBeforeTilingCopy(
        %arg0: !InputDistributed)
        -> memref<1x16x32x32xf16, @DDR> {

    %out = memref.alloc() : !Output_DDR
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : !Output_DDR)  ->  !Output_DDR
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, @DDR>

    return %1 : memref<1x16x32x32xf16, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x16x32x32xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : memref<1x16x32x32xf16, @DDR>)  ->  memref<1x16x32x32xf16, @DDR>
    // CHECK:       return [[COPY]] : memref<1x16x32x32xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @DoNotMovePermuteCastBeforeTilingCopySegmentedForNonTrivialReorder(%arg0: !InputDistributed) -> memref<1x16x32x32xf16, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, @DDR>

    return %1 : memref<1x16x32x32xf16, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[COPY]] : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, @DDR>
    // CHECK:    return [[PERMUTE]] : memref<1x16x32x32xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @DoNotMovePermuteCastBeforeTilingCopySegmentedForUnmatchedOutShape(%arg0: !InputDistributed) -> memref<1x32x16x32xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x32x16x32xf16, #NHWC, @DDR>

    return %1 : memref<1x32x16x32xf16, #NHWC, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[COPY]] : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x32x16x32xf16, #NHWC, @DDR>
    // CHECK:    return [[PERMUTE]] : memref<1x32x16x32xf16, #NHWC, @DDR>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @MoveShapeCastBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x4x64x64xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 4, 64, 64]} inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR>

    return %1 : memref<1x4x64x64xf16, #NHWC, @DDR>

    //CHECK:    [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 4, 64, 64]} inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64x64xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64x64xf16, #NHWC, @DDR>)  ->  memref<1x4x64x64xf16, #NHWC, @DDR>
    //CHECK:    return [[COPY]] : memref<1x4x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotMoveShapeCastBeforeTilingCopySegmented(%arg0: memref<1x16x9x3xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x9x3xf16, #NHWC, @CMX_NN>) -> memref<1x16x3x9xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 234 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%arg0 : memref<1x16x9x3xf16, #NHWC, @CMX_NN>)
        weights(%arg1 : memref<1x16x9x3xf16, #NHWC, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x9x3xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [2, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [2, 8, 15], outStart = [0, 5, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %2 = memref.alloc() : memref<1x16x9x3xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x16x9x3xf16, #NHWC, @DDR>)  ->  memref<1x16x9x3xf16, #NHWC, @DDR>
    %4 = VPUIP.ShapeCast {shape = [1, 16, 3, 9]} inputs(%3 : memref<1x16x9x3xf16, #NHWC, @DDR>) -> memref<1x16x3x9xf16, #NHWC, @DDR>

    return %4 : memref<1x16x3x9xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_0:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CLUSTER_TILING_0:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc()
    // CHECK:       [[CLUSTER_TILING_1:%.+]] = VPUIP.Copy
    // CHECK:       [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 3, 9]}
    // CHECK:       return [[SHAPE_CAST]] : memref<1x16x3x9xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @MoveGenericReshapeBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x4x64x64xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR>

    return %1 : memref<1x4x64x64xf16, #NHWC, @DDR>

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64x64xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64x64xf16, #NHWC, @DDR>)  ->  memref<1x4x64x64xf16, #NHWC, @DDR>
    //CHECK:    return [[COPY]] : memref<1x4x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @DoNotMoveGenericReshapeBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x16x1024xf16, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x1024xf16, @DDR>

    return %1 : memref<1x16x1024xf16, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x16x32x32xf16, #NHWC, @DDR>)  ->  memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x1024xf16, @DDR>
    // CHECK:    return [[RESHAPE]] : memref<1x16x1024xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @DoNotMoveGenericReshapeWithDifferentOrderBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x1x32x1024xf16, #NCWH, @DDR> {
    %out = memref.alloc() : memref<1x32x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x32x32x32xf16, #NHWC, @DDR>)  ->  memref<1x32x32x32xf16, #NHWC, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x32x32x32xf16, #NHWC, @DDR>) -> memref<1x1x32x1024xf16, #NCWH, @DDR>

    return %1 : memref<1x1x32x1024xf16, #NCWH, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x32x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x32x32x32xf16, #NHWC, @DDR>)  ->  memref<1x32x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x32x32x32xf16, #NHWC, @DDR>) -> memref<1x1x32x1024xf16, #NCWH, @DDR>
    // CHECK:    return [[RESHAPE]] : memref<1x1x32x1024xf16, #NCWH, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @MoveQuantizeCastBeforeTilingCopySegmentedOverH(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    //CHECK:    [[QUANTIZE:%.+]] = VPUIP.QuantizeCast inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    //CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

//CHECK-LABEL: @MoveQuantizeCastBeforeTilingCopySegmentedOverK
//CHECK-SAME:  ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN
func.func @MoveQuantizeCastBeforeTilingCopySegmentedOverK(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    //CHECK:      [[QUANTIZE:%.+]] = VPUIP.QuantizeCast
    //CHECK-SAME:   inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    //CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveQuantizeCastBeforeTilingCopyOverlapped
func.func @MoveQuantizeCastBeforeTilingCopyOverlapped(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)
        -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:    [[QUANTIZE:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveQuantizeCastBeforeTilingCopyOverlappedMixedTypes
func.func @MoveQuantizeCastBeforeTilingCopyOverlappedMixedTypes(%arg0: !InputDistributed) -> memref<1x16x32x32xui8, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)
        -> memref<1x16x32x32xui8, #NHWC, @DDR>

    return %1 : memref<1x16x32x32xui8, #NHWC, @DDR>

    // CHECK:    [[QUANTIZE:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x32x32xui8, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32xui8, #NHWC, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32xui8, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32xui8, #NHWC, @DDR>)  ->  memref<1x16x32x32xui8, #NHWC, @DDR>
    // CHECK:    return [[COPY]] : memref<1x16x32x32xui8, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:1, {1.25, 1.5, 1.75, 2.0,
                                             1.25, 1.5, 1.75, 2.0,
                                             1.25, 1.5, 1.75, 2.0,
                                             1.25, 1.5, 1.75, 2.0}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.25, 0.5, 0.75, 1.0,
                                             0.25, 0.5, 0.75, 1.0,
                                             0.25, 0.5, 0.75, 1.0,
                                             0.25, 0.5, 0.75, 1.0}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @SkipPerChannelOverlappedQuantizeCast
func.func @SkipPerChannelOverlappedQuantizeCast(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)  ->  memref<1x16x32x32x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)
        -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:   [[ALLOCATE:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType, #NHWC, @DDR>
    // CHECK:    [[TILING:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN
    // CHECK-SAME:     outputs([[ALLOCATE]] : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:     ->  memref<1x16x32x32x!qElemType, #NHWC, @DDR>

    // CHECK:   [[QUANTIZE_CAST:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs([[TILING]] : memref<1x16x32x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:   return [[QUANTIZE_CAST]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!Distributed0 = !VPUIP.DistributedBuffer<
    1x16x3x9xf16, #NWCH, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    equal_memory_and_compute_view
}>

!Distributed1 = !VPUIP.DistributedBuffer<
    1x3x9x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed2 = !VPUIP.DistributedBuffer<
    1x48x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 4, 1]}>

// CHECK-LABEL: @DoNotMoveShapeCastWhenDistributedNotCompatibleAfterShapeChange
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x16x3x9xf16, #NWCH, @CMX_NN
func.func @DoNotMoveShapeCastWhenDistributedNotCompatibleAfterShapeChange(
        %arg0: !Distributed0)
        -> !Distributed2 {
    %0 = VPUIP.ViewOp %arg0 : !Distributed0 to !Distributed1
    %1 = memref.alloc() : memref<1x3x9x16xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy
        inputs(%0 : !Distributed1)
        outputs(%1 : memref<1x3x9x16xf16, #NHWC, @DDR>)  ->  memref<1x3x9x16xf16, #NHWC, @DDR>
    %3 = VPUIP.ShapeCast {shape = [1, 48, 3, 3]} inputs(%2 : memref<1x3x9x16xf16, #NHWC, @DDR>) -> memref<1x48x3x3xf16, #NHWC, @DDR>
    %4 = VPURT.AllocDistributed -> !Distributed2
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x48x3x3xf16, #NHWC, @DDR>)
        outputs(%4 : !Distributed2)  ->  !Distributed2

    return %5 : !Distributed2

    // CHECK:       [[VIEWOP:%.+]] = VPUIP.ViewOp [[ARG0]] : !VPUIP.DistributedBuffer<1x16x3x9xf16, #NWCH, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    // CHECK-SAME:   to !VPUIP.DistributedBuffer<1x3x9x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[MEMREF_ALLOC:%.+]] = memref.alloc() : memref<1x3x9x16xf16, #NHWC, @DDR>

    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[VIEWOP]] : !VPUIP.DistributedBuffer<1x3x9x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[MEMREF_ALLOC]] : memref<1x3x9x16xf16, #NHWC, @DDR>)  ->  memref<1x3x9x16xf16, #NHWC, @DDR>

    // CHECK:       [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 48, 3, 3]} inputs([[COPY0]] : memref<1x3x9x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:            -> memref<1x48x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_DISTRIBUTED:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x48x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:     {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>

    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[SHAPE_CAST]] : memref<1x48x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC_DISTRIBUTED]] : !VPUIP.DistributedBuffer<1x48x3x3xf16, #NHWC, @CMX_NN
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x48x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>

    // CHECK:       return [[COPY1]] : !VPUIP.DistributedBuffer<1x48x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                   {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>
!qElemType2 = !quant.uniform<u8:f16, 0.1009497549019607928>
!qElemType3 = !quant.uniform<u8:f16, 0.0147441789215686223>
!qElemType4 = !quant.uniform<u8:f16, 0.0113204656862745024>
!qElemType5 = !quant.uniform<u8:f16, 0.0030503216911764706>
!qElemType6 = !quant.uniform<u8:f16, 0.0030503216911345706>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x128x32x32x!qElemType2, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputDistributed1 = !VPUIP.DistributedBuffer<
    64x128x1x1x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!InputDistributed2 = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!InputDistributed3 = !VPUIP.DistributedBuffer<
    64x16x1x1x!qElemType5, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!InputDistributed4 = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!InputDistributed5 = !VPUIP.DistributedBuffer<
    1x1x1x16xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>


!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

func.func @MoveQuantizeCastBeforeTilingCopyMultipleConsumers(%in0: !InputDistributed0, %in1: !InputDistributed1, %in2: !InputDistributed2, %in3: !InputDistributed3, %in4: !InputDistributed4, %in5: !InputDistributed5) -> (memref<1x64x32x32x!qElemType3, #NHWC, @DDR>, !OutputDistributed) {

    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 3030 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in0 : !InputDistributed0)
        weights(%in1 : !InputDistributed1)
        weight_table(%in2 : !InputDistributed2)
        parent_input(%in0 : !InputDistributed0)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 63], outStart = [0, 16, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %4 = memref.alloc() : memref<1x64x32x32x!qElemType1, #NHWC, @DDR>
    %5 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x32x32x!qElemType1, #NHWC, @DDR>)  ->  memref<1x64x32x32x!qElemType1, #NHWC, @DDR>
    %7 = VPUIP.QuantizeCast inputs(%5 : memref<1x64x32x32x!qElemType1, #NHWC, @DDR>) -> memref<1x64x32x32x!qElemType3, #NHWC, @DDR>
    %8 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %9 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1180 : i64, task_type = #VPUIP.nce_task_type<DWCONV>}
        input(%1 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights(%in3 : !InputDistributed3)
        weight_table(%in4 : !InputDistributed4)
        parent_input(%1 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output(%8 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%8 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 63], outStart = [0, 16, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %7, %9 : memref<1x64x32x32x!qElemType3, #NHWC, @DDR>, !OutputDistributed
    // CHECK:       [[NCE_OUT:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[NCE_OUT]] : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType5, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x32x32x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF:%.+]] = memref.alloc() : memref<1x64x32x32x!qElemType3, #NHWC, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUF]] : memref<1x64x32x32x!qElemType3, #NHWC, @DDR>)  ->  memref<1x64x32x32x!qElemType3, #NHWC, @DDR>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[NCE_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16:1, {1.0, 1.75, 1.5, 1.25, 1.0, 1.5, 1.75, 1.25}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {1.0, 1.75, 1.5, 1.25, 1.0, 1.5}>
// CHECK-DAG: [[QUANT_8_CHAN:.*]] = !quant.uniform<u8<0:254>:f16:1, {
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME:	1.5
// CHECK-DAG-SAME:	1.25
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME:	1.5
// CHECK-DAG-SAME:	1.25
// CHECK-DAG-SAME: }>

// CHECK-DAG: [[QUANT_6_CHAN:.*]] = !quant.uniform<u8<0:254>:f16:1, {
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME:	1.5
// CHECK-DAG-SAME:	1.25
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME: }>

func.func @MoveSubViewWithPerAxisQuantization(%arg0: memref<1x8x2x2x!qElemType, @DDR>,
                                              %arg1: memref<1x6x2x2x!qElemType1, @DDR>)
                                              -> memref<1x6x2x2x!qElemType1, @DDR> {
    %ALLOC = memref.alloc() : memref<1x8x2x2x!qElemType, @DDR>

    %IN_COPY = VPUIP.Copy
        inputs(%arg0: memref<1x8x2x2x!qElemType, @DDR>)
        outputs(%ALLOC : memref<1x8x2x2x!qElemType, @DDR>)
            -> memref<1x8x2x2x!qElemType, @DDR>

    %SUBVIEW = VPUIP.SubView %IN_COPY [0, 0, 0, 0] [1, 6, 2, 2] :
        memref<1x8x2x2x!qElemType, @DDR>
        to memref<1x6x2x2x!qElemType1, {order = #NCHW, strides = [32, 4, 2, 1]}, @DDR>

    %OUT_COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x6x2x2x!qElemType1, {order = #NCHW, strides = [32, 4, 2, 1]}, @DDR>)
        outputs(%arg1 : memref<1x6x2x2x!qElemType1, @DDR>) -> memref<1x6x2x2x!qElemType1, @DDR>

    return %OUT_COPY : memref<1x6x2x2x!qElemType1, @DDR>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 6, 2, 2] :
    // CHECK-SAME:  memref<1x8x2x2x[[QUANT_8_CHAN]], @DDR>
    // CHECK-SAME:  to memref<1x6x2x2x[[QUANT_6_CHAN]], {order = #NCHW, strides = [32, 4, 2, 1]}, @DDR>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>

    // CHECK: [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[SUBVIEW]] : memref<1x6x2x2x[[QUANT_6_CHAN]], {order = #NCHW, strides = [32, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>)
    // CHECK-SAME:      -> memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>

    // CHECK: [[OUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[IN_COPY]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>) -> memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>

    // CHECK:   return [[OUT_COPY]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DoNotMoveShapeCastWhenCompressConv(
        %arg0: memref<1x4x104x208xf16, #NHWC, [@CMX_NN, 0]>,
        %arg1: memref<32x1x1x32xf16, #NHWC>,
        %arg2: memref<32x1x1x4xsi32>)
        -> memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]> {

    %0 = memref.alloc() : memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg1 : memref<32x1x1x32xf16, #NHWC>) outputs(%0 : memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg2 : memref<32x1x1x4xsi32>) outputs(%2 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %4 = VPUIP.ShapeCast {shape = [32, 16, 3, 3]} inputs(%1 : memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<32x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.ShapeCast {shape = [1, 16, 104, 208]} inputs(%arg0 : memref<1x4x104x208xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x104x208xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPUIP.NCEClusterTask {cm_sp_pattern = 15 : i64, input_channels_compression,
                                kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                                kernel_size = [3, 3], kernel_strides = [2, 2],
                                minimumHardwareExecutionCost = 4294967398 : i64, task_type = #VPUIP.nce_task_type<CONV>}
                input(%6 : memref<1x16x104x208xf16, #NHWC, [@CMX_NN, 0]>)
                weights(%4 : memref<32x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
                weight_table(%3 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
                parent_input(%6 : memref<1x16x104x208xf16, #NHWC, [@CMX_NN, 0]>)
                parent_output(%5 : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%5 : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>
    variants : {
      DPUTask {inEnd = [207, 103, 3], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [103, 51, 31], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    return %7 : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_WEIGHTS:%.+]] = memref.alloc() : memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[W_CMX:%.+]] = VPUIP.Copy inputs(%arg1 : memref<32x1x1x32xf16, #NHWC>) outputs([[ALLOC_WEIGHTS]] : memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_WEIGHTSTABLE:%.+]] = memref.alloc() : memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WT_CMX:%.+]] = VPUIP.Copy inputs(%arg2 : memref<32x1x1x4xsi32>) outputs([[ALLOC_WEIGHTSTABLE]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[SHAPECAST_WEIGHTS:%.+]] = VPUIP.ShapeCast {shape = [32, 16, 3, 3]} inputs([[W_CMX]] : memref<32x1x1x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<32x16x3x3xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[SHAPECAST_INPUT:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 104, 208]} inputs(%arg0 : memref<1x4x104x208xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x104x208xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[COMPRESS_CONV:%.+]] = VPUIP.NCEClusterTask {cm_sp_pattern = 15 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 4294967398 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:  input([[SHAPECAST_INPUT]] : memref<1x16x104x208xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weights([[SHAPECAST_WEIGHTS]] : memref<32x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WT_CMX]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_input([[SHAPECAST_INPUT]] : memref<1x16x104x208xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_output([[ALLOC_OUTPUT]] : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[ALLOC_OUTPUT]] : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]> variants : {
    // CHECK:        DPUTask {inEnd = [207, 103, 3], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [103, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:       PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:       }

    // CHECK:  return [[COMPRESS_CONV]] : memref<1x32x52x104xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @MoveSubviewToTheFrontOfCopyMultipleConsumers(%arg0: memref<1x16x2x2xf16, @DDR>, %arg1: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %1 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %2 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %3 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%1 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %4 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%2 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %5 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 8, 2, 2] : memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>) outputs(%arg1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    return %6 : memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 8, 2, 2] :
    // CHECK-SAME:                      memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>

    // CHECK:       [[BUF_0:%.+]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:                      outputs([[BUF_0]] : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[BUF_1:%.+]] = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16, @DDR>) outputs([[BUF_1]] : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[COPY_0]] : memref<1x8x2x2xf16, @DDR>) outputs(%arg1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    // CHECK:       return [[COPY_2]] : memref<1x8x2x2xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveSubviewToTheFrontOfTillingCopyMultipleConsumers(%in0 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>,
                                %in1 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
                                    -> (memref<1x24x128x128xf16, #NHWC, @DDR>, memref<1x32x128x128xf16, #NHWC, @DDR>) {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 20758 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%in0 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
        weights(%in1 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
        parent_input(%in0 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x32x128x128xf16, #NHWC, @DDR>)  ->  memref<1x32x128x128xf16, #NHWC, @DDR>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 24, 128, 128] : memref<1x32x128x128xf16, #NHWC, @DDR> to memref<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    %5 = memref.alloc() : memref<1x24x128x128xf16, #NHWC, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>) outputs(%5 : memref<1x24x128x128xf16, #NHWC, @DDR>) -> memref<1x24x128x128xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>
    %8 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%7 : memref<1x32x128x128xf16, #NHWC, @DDR>)  ->  memref<1x32x128x128xf16, #NHWC, @DDR>

    return %6, %8 : memref<1x24x128x128xf16, #NHWC, @DDR>, memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 20758 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input(%arg0 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights(%arg1 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input(%arg0 : memref<1x32x128x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:     ->  !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        // CHECK:     DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        // CHECK:     DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
        // CHECK:     PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ADD_0]] [0, 0, 0, 0] [1, 24, 128, 128] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x24x128x128xf16, #NHWC, @DDR>
    // CHECK:    [[Tilling_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x24x128x128xf16, #NHWC, @DDR>)  ->  memref<1x24x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x24x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs([[Tilling_COPY_0]] : memref<1x24x128x128xf16, #NHWC, @DDR>) outputs([[BUFF_2]] : memref<1x24x128x128xf16, #NHWC, @DDR>) -> memref<1x24x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_3:%.+]] = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>
    // CHECK:    [[Tilling_COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_3]] : memref<1x32x128x128xf16, #NHWC, @DDR>)  ->  memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[COPY]], [[Tilling_COPY_1:%.+]] : memref<1x24x128x128xf16, #NHWC, @DDR>, memref<1x32x128x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotMoveSubviewToTheFrontOfTillingCopyForIncompatibleDistributedBuffer(%in0 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>,
                                %in1 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
                                    -> memref<1x32x128x128xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 20758 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%in0 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
        weights(%in1 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
        parent_input(%in0 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x32x129x128xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x32x129x128xf16, #NHWC, @DDR>)  ->  memref<1x32x129x128xf16, #NHWC, @DDR>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 128, 128] : memref<1x32x129x128xf16, #NHWC, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>
    %5 = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>) outputs(%5 : memref<1x32x128x128xf16, #NHWC, @DDR>) -> memref<1x32x128x128xf16, #NHWC, @DDR>

    return %6 : memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 20758 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input(%arg0 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights(%arg1 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input(%arg0 : memref<1x32x129x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:     ->  !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        // CHECK:     DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        // CHECK:     DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
        // CHECK:     PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x32x129x128xf16, #NHWC, @DDR>
    // CHECK:       [[Tilling_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x32x129x128xf16, #NHWC, @DDR>)
    // CHECK-SAME: ->  memref<1x32x129x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[Tilling_COPY_0]] [0, 0, 0, 0] [1, 32, 128, 128] :
    // CHECK-SAME:      memref<1x32x129x128xf16, #NHWC, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>) outputs([[BUFF_2]] : memref<1x32x128x128xf16, #NHWC, @DDR>) -> memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[COPY]] : memref<1x32x128x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    compute_shapes = [[1, 32, 32, 32], [1, 16, 32, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 48, 32, 32], [1, 48, 32, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x96x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 4, 96, 128], [1, 4, 96, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4, 96, 128], [1, 4, 96, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveShapeCastBeforeTilingCopyExplicitSegDuplicated
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x48x32x32xf16
func.func @MoveShapeCastBeforeTilingCopyExplicitSegDuplicated(%arg0: !InputDistributed) -> !OutputDistributed {
    %out = memref.alloc() : memref<1x48x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x48x32x32xf16, #NHWC, @DDR>)  ->  memref<1x48x32x32xf16, #NHWC, @DDR>

    %1 = VPUIP.ShapeCast {shape = [1, 4, 96, 128]} inputs(%0 : memref<1x48x32x32xf16, #NHWC, @DDR>)
        -> memref<1x4x96x128xf16, #NHWC, @DDR>

    %cmxBuff = VPURT.AllocDistributed -> !OutputDistributed
    %2 = VPUIP.Copy {out_mem_space = @CMX_NN}
        inputs(%1 : memref<1x4x96x128xf16, #NHWC, @DDR>)
        outputs(%cmxBuff : !OutputDistributed)  ->  !OutputDistributed

    return %2 : !OutputDistributed

    //CHECK:    [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 4, 96, 128]}
    //CHECK-SAME:   inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x48x32x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                      {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:                       num_clusters = 2 : i64, alignment = [1, 16, 1, 1],
    //CHECK-SAME{LITERAL}:              compute_shapes = [[1, 32, 32, 32], [1, 16, 32, 32]],
    //CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:              memory_shapes = [[1, 48, 32, 32], [1, 48, 32, 32]],
    //CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x4x96x128xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 96, 128], [1, 4, 96, 128]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 96, 128], [1, 4, 96, 128]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x96x128xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:         inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x4x96x128xf16, #NHWC, @CMX_NN
    //CHECK-SAME:         outputs([[OUTBUFF]] : memref<1x4x96x128xf16, #NHWC, @DDR>)
    //CHECK-SAME:     -> memref<1x4x96x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 48, 16, 32], [1, 48, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 48, 32, 32], [1, 48, 32, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x384x1x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 384, 1, 128], [1, 384, 1, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 384, 1, 128], [1, 384, 1, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyExplicitSegMulticasted
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x48x32x32xf16
func.func @MoveGenericReshapeBeforeTilingCopyExplicitSegMulticasted(%arg0: !InputDistributed) -> !OutputDistributed {
    %out = memref.alloc() : memref<1x48x32x32xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x48x32x32xf16, #NHWC, @DDR>)  ->  memref<1x48x32x32xf16, #NHWC, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x48x32x32xf16, #NHWC, @DDR>)
        -> memref<1x384x1x128xf16, #NHWC, @DDR>

    %cmxBuff = VPURT.AllocDistributed -> !OutputDistributed
    %2 = VPUIP.Copy {out_mem_space = @CMX_NN}
        inputs(%1 : memref<1x384x1x128xf16, #NHWC, @DDR>)
        outputs(%cmxBuff : !OutputDistributed)  ->  !OutputDistributed

    return %2 : !OutputDistributed

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:   inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x48x32x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                      {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:              compute_shapes = [[1, 48, 16, 32], [1, 48, 16, 32]],
    //CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    //CHECK-SAME{LITERAL}:              memory_shapes = [[1, 48, 32, 32], [1, 48, 32, 32]],
    //CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x384x1x128xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 384, 1, 128], [1, 384, 1, 128]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 384, 1, 128], [1, 384, 1, 128]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x384x1x128xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:         inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x384x1x128xf16, #NHWC, @CMX_NN
    //CHECK-SAME:         outputs([[OUTBUFF]] : memref<1x384x1x128xf16, #NHWC, @DDR>)
    //CHECK-SAME:     -> memref<1x384x1x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x48x1x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    compute_shapes = [[1, 32, 1, 16], [1, 16, 1, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 48, 1, 16], [1, 48, 1, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x16x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 16, 16, 3], [1, 16, 16, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 16, 3], [1, 16, 16, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveMultipleViewOpsBeforeTilingCopyExplicitSegDuplicated
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x48x1x16xf16
func.func @MoveMultipleViewOpsBeforeTilingCopyExplicitSegDuplicated(%arg0: !InputDistributed) -> !OutputDistributed {
    %out = memref.alloc() : memref<1x48x1x16xf16, #NHWC, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x48x1x16xf16, #NHWC, @DDR>)  ->  memref<1x48x1x16xf16, #NHWC, @DDR>

    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHCW} inputs(%0 : memref<1x48x1x16xf16, #NHWC, @DDR>)
            -> memref<1x48x16x1xf16, #NHWC, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x48x16x1xf16, #NHWC, @DDR>)
        -> memref<1x16x16x3xf16, #NHWC, @DDR>

    %cmxBuff = VPURT.AllocDistributed -> !OutputDistributed
    %3 = VPUIP.Copy {out_mem_space = @CMX_NN}
        inputs(%2 : memref<1x16x16x3xf16, #NHWC, @DDR>)
        outputs(%cmxBuff : !OutputDistributed)  ->  !OutputDistributed

    return %3 : !OutputDistributed

    //CHECK:    [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHCW}
    //CHECK-SAME:   inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x48x1x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                      {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:                       num_clusters = 2 : i64, alignment = [1, 16, 1, 1],
    //CHECK-SAME{LITERAL}:              compute_shapes = [[1, 32, 1, 16], [1, 16, 1, 16]],
    //CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:              memory_shapes = [[1, 48, 1, 16], [1, 48, 1, 16]],
    //CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x48x16x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:          num_clusters = 2 : i64, alignment = [1, 16, 1, 1],
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 16, 1], [1, 16, 16, 1]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 48, 16, 1], [1, 48, 16, 1]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:   inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<1x48x16x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                      {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:                       num_clusters = 2 : i64, alignment = [1, 16, 1, 1],
    //CHECK-SAME{LITERAL}:              compute_shapes = [[1, 32, 16, 1], [1, 16, 16, 1]],
    //CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    //CHECK-SAME{LITERAL}:              memory_shapes = [[1, 48, 16, 1], [1, 48, 16, 1]],
    //CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x16x16x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 3], [1, 16, 16, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 3], [1, 16, 16, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x16x3xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:         inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x16x16x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         outputs([[OUTBUFF]] : memref<1x16x16x3xf16, #NHWC, @DDR>)
    //CHECK-SAME:     -> memref<1x16x16x3xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    4x64x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

func.func @MoveGenericReshapeBeforeTilingCopyRankChanged(%arg0: !InputDistributed) -> memref<1x4x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<1x4x64xf16, @DDR>

    return %1 : memref<1x4x64xf16, @DDR>

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:       inputs(%arg0 : !VPUIP.DistributedBuffer<
    //CHECK-SAME:           4x64x1x1xf16, #NCHW, @CMX_NN, {
    //CHECK-SAME:           mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:           num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:           num_clusters = 2 : i64,
    //CHECK-SAME:           alignment = [1, 16, 1, 1],
    //CHECK-SAME:           uniform_distributed_segments
    //CHECK-SAME:           }>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<
    //CHECK-SAME:           1x4x64xf16, #CHW, @CMX_NN, {
    //CHECK-SAME:           mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:           num_tiles = [1, 1, 2],
    //CHECK-SAME:           num_clusters = 2 : i64,
    //CHECK-SAME:           alignment = [1, 1, 16],
    //CHECK-SAME:           uniform_distributed_segments
    //CHECK-SAME:           }>

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x4x64xf16, #CHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2], num_clusters = 2 : i64, alignment = [1, 1, 16], uniform_distributed_segments}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64xf16, @DDR>)  ->  memref<1x4x64xf16, @DDR>

    //CHECK:    return [[COPY]] : memref<1x4x64xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    4x64x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[4, 48, 1, 1], [4, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
    memory_shapes = [[4, 64, 1, 1], [4, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @MoveGenericReshapeBeforeTilingCopyRankChangedExplicitDistribution(%arg0: !InputDistributed) -> memref<1x4x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<1x4x64xf16, @DDR>

    return %1 : memref<1x4x64xf16, @DDR>

    //CHECK:        [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:           inputs(%arg0 : !VPUIP.DistributedBuffer<
    //CHECK-SAME:               4x64x1x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:               num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:               num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:      compute_shapes = [[4, 48, 1, 1], [4, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:      memory_shapes = [[4, 64, 1, 1], [4, 64, 1, 1]],
    //CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<
    //CHECK-SAME:               1x4x64xf16, #CHW, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED",
    //CHECK-SAME:               num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:      compute_shapes = [[1, 4, 64], [1, 4, 64]],
    //CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0], [0, 0, 0]],
    //CHECK-SAME{LITERAL}:      memory_shapes = [[1, 4, 64], [1, 4, 64]],
    //CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0], [0, 0, 0]]
    //CHECK-SAME:               }>

    //CHECK:        [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64xf16, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x4x64xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64xf16, @DDR>)  ->  memref<1x4x64xf16, @DDR>

    //CHECK:        return [[COPY]] : memref<1x4x64xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    4x64x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

func.func @MoveGenericReshapeBeforeTilingCopyRankChangedAndSplitHigherDim(%arg0: !InputDistributed) -> memref<2x2x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<2x2x64xf16, @DDR>

    return %1 : memref<2x2x64xf16, @DDR>

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:       inputs(%arg0 : !VPUIP.DistributedBuffer<
    //CHECK-SAME:           4x64x1x1xf16, #NCHW, @CMX_NN, {
    //CHECK-SAME:           mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:           num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:           num_clusters = 2 : i64,
    //CHECK-SAME:           alignment = [1, 16, 1, 1],
    //CHECK-SAME:           uniform_distributed_segments
    //CHECK-SAME:           }>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<
    //CHECK-SAME:           2x2x64xf16, #CHW, @CMX_NN, {
    //CHECK-SAME:           mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:           num_tiles = [1, 1, 2],
    //CHECK-SAME:           num_clusters = 2 : i64,
    //CHECK-SAME:           alignment = [1, 1, 16],
    //CHECK-SAME:           uniform_distributed_segments
    //CHECK-SAME:           }>

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<2x2x64xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<2x2x64xf16, #CHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2], num_clusters = 2 : i64, alignment = [1, 1, 16], uniform_distributed_segments}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<2x2x64xf16, @DDR>)  ->  memref<2x2x64xf16, @DDR>

    //CHECK:    return [[COPY]] : memref<2x2x64xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    4x64x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[4, 48, 1, 1], [4, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
    memory_shapes = [[4, 64, 1, 1], [4, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @MoveGenericReshapeBeforeTilingCopyRankChangedAndSplitHigherDimExplicitDistribution(%arg0: !InputDistributed) -> memref<2x2x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<2x2x64xf16, @DDR>

    return %1 : memref<2x2x64xf16, @DDR>

    //CHECK:        [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:           inputs(%arg0 : !VPUIP.DistributedBuffer<
    //CHECK-SAME:               4x64x1x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:               num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:               num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:      compute_shapes = [[4, 48, 1, 1], [4, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:      memory_shapes = [[4, 64, 1, 1], [4, 64, 1, 1]],
    //CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<
    //CHECK-SAME:               2x2x64xf16, #CHW, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED",
    //CHECK-SAME:               num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:      compute_shapes = [[2, 2, 64], [2, 2, 64]],
    //CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0], [0, 0, 0]],
    //CHECK-SAME{LITERAL}:      memory_shapes = [[2, 2, 64], [2, 2, 64]],
    //CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0], [0, 0, 0]]
    //CHECK-SAME:               }>

    //CHECK:        [[OUTBUFF:%.+]] = memref.alloc() : memref<2x2x64xf16, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<2x2x64xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<2x2x64xf16, @DDR>)  ->  memref<2x2x64xf16, @DDR>

    //CHECK:        return [[COPY]] : memref<2x2x64xf16, @DDR>
}
