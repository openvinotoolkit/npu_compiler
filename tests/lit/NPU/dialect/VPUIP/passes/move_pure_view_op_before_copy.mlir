//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePureViewOpBeforeCopyMultipleConsumers
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x16x112x112xf16, @DDR>
func.func @MovePureViewOpBeforeCopyMultipleConsumers(
        %arg0: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>,
        %arg1: memref<1x16x112x112xf16, @DDR>)
        -> (memref<1x16x56x224xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, @DDR>) {
    %0 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    %3 = VPUIP.GenericReshape inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) -> memref<1x16x56x224xf16, {order = #NHWC}, @DDR>

    %4 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, @DDR>)
        outputs(%arg1 : memref<1x16x112x112xf16, @DDR>)
        -> memref<1x16x112x112xf16, @DDR>

    return %3, %4 : memref<1x16x56x224xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, @DDR>


    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[ARG_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>) -> memref<1x16x112x112xf16, @CMX>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[PERMUTECAST]] : memref<1x16x112x112xf16, @CMX>) outputs([[ALLOC0]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    // CHECK: [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>) -> memref<1x16x56x224xf16, {order = #NHWC}, @CMX>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x56x224xf16, {order = #NHWC}, @DDR>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[GENERICRESHAPE]] : memref<1x16x56x224xf16, {order = #NHWC}, @CMX>) outputs([[ALLOC1]] : memref<1x16x56x224xf16, {order = #NHWC}, @DDR>) -> memref<1x16x56x224xf16, {order = #NHWC}, @DDR>

    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x16x112x112xf16, @DDR>) outputs([[ARG_1]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    // CHECK: return [[COPY1]], [[COPY2]] : memref<1x16x56x224xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePureViewOpBeforeCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x16x112x112xf16, @DDR>
func.func @MovePureViewOpBeforeCopy(
        %arg0: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>,
        %arg1: memref<1x16x112x112xf16, @DDR>)
        -> memref<1x16x112x112xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    %3 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, @DDR>)
        outputs(%arg1 : memref<1x16x112x112xf16, @DDR>)
        -> memref<1x16x112x112xf16, @DDR>

    return %3 : memref<1x16x112x112xf16, @DDR>

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK-SAME:                   inputs([[ARG_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>) -> memref<1x16x112x112xf16, @CMX>

    //CHECK:        [[VAR1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    //CHECK:        [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, @CMX>)
    //CHECK-SAME:                   outputs([[VAR1]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    //CHECK:        [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, @DDR>)
    //CHECK-SAME:                   outputs([[ARG_1]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    //CHECK:        return [[VAR3]] : memref<1x16x112x112xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveSeveralPureViewOpsBeforeCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x16x392x32xf16, @DDR>
func.func @MoveSeveralPureViewOpsBeforeCopy(
        %arg0: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>,
        %arg1: memref<1x16x392x32xf16, @DDR>)
        -> memref<1x16x392x32xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) -> memref<1x16x112x112xf16, @DDR>

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
    //CHECK-SAME:                   inputs([[ARG_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>) -> memref<1x16x112x112xf16, @CMX>
    //CHECK:        [[VAR1:%.+]] = VPUIP.GenericReshape inputs([[VAR0]] : memref<1x16x112x112xf16, @CMX>) -> memref<1x16x12544xf16, @CMX>
    //CHECK:        [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x16x12544xf16, @CMX>) -> memref<1x16x392x32xf16, @CMX>

    //CHECK:        [[VAR3:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x392x32xf16, @CMX>) outputs([[VAR3]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        [[VAR5:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x392x32xf16, @DDR>) outputs([[VAR5]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x392x32xf16, @DDR>) outputs([[ARG_1]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        return [[VAR7]] : memref<1x16x392x32xf16, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @QuantizeCastBeforeDistributedCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x128x8x8x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @QuantizeCastBeforeDistributedCopy(%arg0: memref<1x128x8x8x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x128x8x8x!qElemType1, {order = #NHWC}, @CMX_NN> {
    %buf0 = memref.alloc() : memref<1x128x8x8x!qElemType, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x8x8x!qElemType, {order = #NHWC}, @CMX_NN>)
        outputs(%buf0 : memref<1x128x8x8x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x128x8x8x!qElemType, {order = #NHWC}, @DDR>

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x128x8x8x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x128x8x8x!qElemType1, {order = #NHWC}, @DDR>

    %buf1 = memref.alloc() : memref<1x128x8x8x!qElemType1, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy
        inputs(%1 : memref<1x128x8x8x!qElemType1, {order = #NHWC}, @DDR>)
        outputs(%buf1 : memref<1x128x8x8x!qElemType1, {order = #NHWC}, @CMX_NN>)  ->  memref<1x128x8x8x!qElemType1, {order = #NHWC}, @CMX_NN>

    return %2 : memref<1x128x8x8x!qElemType1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[VAR0:%.+]] = VPUIP.QuantizeCast inputs([[ARG_0]] :
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[VAR0]]
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[VAR1]]
    // CHECK:       return [[VAR2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MoveSubviewToTheFrontOfCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x16x2x2xf16, @DDR>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x8x2x2xf16, @DDR>
func.func @MoveSubviewToTheFrontOfCopy(%arg0: memref<1x16x2x2xf16, @DDR>, %arg1: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %1 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %2 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%1 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 8, 2, 2] : memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>) outputs(%arg1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    return %4 : memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 8, 2, 2] :
    // CHECK-SAME:                      memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>

    // CHECK:       [[VAR1:%.+]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:                      outputs([[VAR1]] : memref<1x8x2x2xf16, @DDR>)

    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x8x2x2xf16, @DDR>)
    // CHECK-SAME:                      outputs([[ARG_1]] : memref<1x8x2x2xf16, @DDR>)

    // CHECK:       return [[VAR3]] : memref<1x8x2x2xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0069489371542837105>

// CHECK-LABEL: @DoNotMoveSubviewToTheFrontOfSparseCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}, [@CMX_NN, 0]>>
func.func @DoNotMoveSubviewToTheFrontOfSparseCopy(%arg0: !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}, [@CMX_NN, 0]>>)
-> !VPUIP.SparseBuffer<data=memref<1x64x17x56x!qElemType, {order = #NHWC, strides = [200704, 1, 3584, 64]}>, sparsity_map=memref<1x64x17x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}>> {
    %alloc_0 = memref.alloc() : memref<1x64x56x56x!qElemType, {order = #NHWC}>
    %alloc_1 = memref.alloc() : memref<1x64x56x56xi1, {order = #NHWC}>
    %0 = VPUIP.GroupSparseBuffer(%alloc_0, %alloc_1) -> !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}>>
    %1 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}, [@CMX_NN, 0]>>) outputs(%0 : !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}>>) -> !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}>>
    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 17, 56] : !VPUIP.SparseBuffer<data=memref<1x64x56x56x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x64x56x56xi1, {order = #NHWC}>> to !VPUIP.SparseBuffer<data=memref<1x64x17x56x!qElemType, {order = #NHWC, strides = [200704, 1, 3584, 64]}>, sparsity_map=memref<1x64x17x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}>>

    return %2 : !VPUIP.SparseBuffer<data=memref<1x64x17x56x!qElemType, {order = #NHWC, strides = [200704, 1, 3584, 64]}>, sparsity_map=memref<1x64x17x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}>>

    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x56x56x!qElemType, {order = #NHWC}>
    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x64x56x56xi1, {order = #NHWC}>
    // CHECK:       [[SPARSEBUFFER:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC0]], [[ALLOC1]])
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs([[ARG_0]]
    // CHECK-SAME:  outputs([[SPARSEBUFFER]]
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[COPY]] [0, 0, 0, 0] [1, 64, 17, 56]
    // CHECK: return [[SUBVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

// CHECK-LABEL: @MoveSubviewToTheFrontOfTillingCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
 func.func @MoveSubviewToTheFrontOfTillingCopy(%in0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                                    -> memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%in0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
        weights(%in1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_input(%in0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        outputs(%2 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x64x48x88x!qElemType, {order = #NHWC}, @DDR>
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 48, 88] : memref<1x64x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, @DDR> to memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [270336, 1, 5632, 64]}, @DDR>
    %5 = memref.alloc() : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [270336, 1, 5632, 64]}, @DDR>) outputs(%5 : memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, @DDR>) -> memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, @DDR>
    return %6 : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     task_type = #VPUIP.nce_task_type<ELTWISE>}>
    // CHECK-SAME:     input([[ARG_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     weights([[ARG_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_input([[ARG_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK:     ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> variants : {
    // CHECK:     DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<MATRIX>, outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
    // CHECK:     PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ADD_0]] [0, 0, 0, 0] [1, 32, 48, 88] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:    [[Tilling_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[Tilling_COPY]] : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_2]] : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>)
    // CHECK:       return [[COPY]] : memref<1x32x48x88x!qElemType, {order = #NHWC}, @DDR>
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = memref<1x16x32x32x!qElemType, {order = #NHWC}, @CMX_NN>
!Output_DDR = memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>

// CHECK-LABEL: @MoveQuantizeCastBeforeTilingCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
func.func @MoveQuantizeCastBeforeTilingCopy(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : !Output_DDR
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : !Output_DDR)  ->  !Output_DDR
    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    return %1 : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_0:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK:       return [[COPY]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = memref<1x16x32x32xf16, @CMX_NN>
!Output_DDR = memref<1x16x32x32xf16, @DDR>

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyRankShrinks
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN,
func.func @MoveGenericReshapeBeforeTilingCopyRankShrinks(
        %arg0: !InputDistributed)
        -> memref<1x16x1024xf16, @DDR> {
    %out = memref.alloc() : !Output_DDR
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : !Output_DDR)  ->  !Output_DDR
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, @DDR>) -> memref<1x16x1024xf16, @DDR>

    return %1 : memref<1x16x1024xf16, @DDR>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x1024xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_0:%.+]] = memref.alloc() : memref<1x16x1024xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x16x1024xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
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

!OutputStub_CMX = memref<1x16x32x32xf16, {order = #NHWC}, @CMX_NN>
!Output_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>

// CHECK-LABEL: @MovePermuteCastBeforeTilingCopy
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
func.func @MovePermuteCastBeforeTilingCopy(
        %arg0: !InputDistributed)
        -> memref<1x16x32x32xf16, @DDR> {

    %out = memref.alloc() : !Output_DDR
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : !Output_DDR)  ->  !Output_DDR
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%0 : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x16x32x32xf16, @DDR>

    return %1 : memref<1x16x32x32xf16, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x16x32x32xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : memref<1x16x32x32xf16, @DDR>)  ->  memref<1x16x32x32xf16, @DDR>
    // CHECK:       return [[COPY]] : memref<1x16x32x32xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1x64x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @MovePermuteCastBeforeTilingCopyTrivialPermute
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x1x64x64xf16, #NCHW, @CMX_NN,
func.func @MovePermuteCastBeforeTilingCopyTrivialPermute(%arg0: !InputDistributed) -> memref<1x1x64x64xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x1x64x64xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x1x64x64xf16, @DDR>) -> memref<1x1x64x64xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}
        inputs(%0 : memref<1x1x64x64xf16, @DDR>) -> memref<1x1x64x64xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x1x64x64xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x1x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x1x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUF:%.+]] = memref.alloc() : memref<1x1x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<1x1x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:      outputs([[BUF]] : memref<1x1x64x64xf16, {order = #NHWC}, @DDR>) -> memref<1x1x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK:       return [[COPY]] : memref<1x1x64x64xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1x77x768xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]],
    memory_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]]
}>

// CHECK-LABEL: @MovePermuteCastBeforeTilingCopyTrivialPermuteAndSameLogicShape
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x1x77x768xf16, #NHWC, @CMX_NN,
func.func @MovePermuteCastBeforeTilingCopyTrivialPermuteAndSameLogicShape(%arg0: !InputDistributed) -> memref<1x1x77x768xf16> {
    %out = memref.alloc() : memref<1x1x77x768xf16, {order = #NHWC}>
    %0 = VPUIP.Copy inputs(%arg0 : !InputDistributed) outputs(%out : memref<1x1x77x768xf16, {order = #NHWC}>) -> memref<1x1x77x768xf16, {order = #NHWC}>
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%0 : memref<1x1x77x768xf16, {order = #NHWC}>) -> memref<1x1x77x768xf16>
    return %1 : memref<1x1x77x768xf16>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
    // CHECK-SAME:      1x1x77x768xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:      1x1x77x768xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]]}>
    // CHECK:       [[BUF:%.+]] = memref.alloc() : memref<1x1x77x768xf16>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<
    // CHECK-SAME:      1x1x77x768xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 20, 768], [1, 1, 19, 768], [1, 1, 19, 768], [1, 1, 19, 768]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 39, 0], [0, 0, 58, 0]]}>)
    // CHECK-SAME:      outputs([[BUF]] : memref<1x1x77x768xf16>) -> memref<1x1x77x768xf16>
    // CHECK:       return [[COPY]] : memref<1x1x77x768xf16>
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

// CHECK-LABEL: @DoNotMovePermuteCastBeforeTilingCopySegmentedForNonTrivialReorder
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
func.func @DoNotMovePermuteCastBeforeTilingCopySegmentedForNonTrivialReorder(%arg0: !InputDistributed) -> memref<1x16x32x32xf16, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} inputs(%0 : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x16x32x32xf16, @DDR>

    return %1 : memref<1x16x32x32xf16, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]]
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[COPY]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x16x32x32xf16, @DDR>
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

// CHECK-LABEL: @DoNotMovePermuteCastBeforeTilingCopySegmentedForUnmatchedOutShape
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
func.func @DoNotMovePermuteCastBeforeTilingCopySegmentedForUnmatchedOutShape(%arg0: !InputDistributed) -> memref<1x32x16x32xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%0 : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x32xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x32x16x32xf16, {order = #NHWC}, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]]
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[COPY]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x32x16x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    return [[PERMUTE]] : memref<1x32x16x32xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveShapeCastBeforeTilingCopySegmented
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
func.func @MoveShapeCastBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x4x64x64xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 4, 64, 64]} inputs(%0 : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x4x64x64xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 4, 64, 64]} inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>)  ->  memref<1x4x64x64xf16, {order = #NHWC}, @DDR>
    //CHECK:    return [[COPY]] : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveShapeCastBeforeTilingCopySegmentedPrefixPreserved
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
func.func @MoveShapeCastBeforeTilingCopySegmentedPrefixPreserved(%arg0: !InputDistributed) -> memref<1x8x32x32xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 8, 32, 32]} inputs(%0 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x8x32x32xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x8x32x32xf16, {order = #NHWC}, @DDR>

    // CHECK:   [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 8, 32, 32]} inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x8x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:    outputs([[OUT_BUFF]] : memref<1x8x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x8x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:   return [[COPY]] : memref<1x8x32x32xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotMoveShapeCastBeforeTilingCopySegmented(%arg0: memref<1x16x9x3xf16, {order = #NHWC}, @CMX_NN>, %arg1: memref<1x16x9x3xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x3x9xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 234 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg0 : memref<1x16x9x3xf16, {order = #NHWC}, @CMX_NN>)
        weights(%arg1 : memref<1x16x9x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x9x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [2, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [2, 8, 15], outStart = [0, 5, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %2 = memref.alloc() : memref<1x16x9x3xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x16x9x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x16x9x3xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x9x3xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.ShapeCast {shape = [1, 16, 3, 9]} inputs(%3 : memref<1x16x9x3xf16, {order = #NHWC}, @DDR>) -> memref<1x16x3x9xf16, {order = #NHWC}, @DDR>

    return %4 : memref<1x16x3x9xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[ALLOC_0:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CLUSTER_TILING_0:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc()
    // CHECK:       [[CLUSTER_TILING_1:%.+]] = VPUIP.Copy
    // CHECK:       [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 3, 9]}
    // CHECK:       return [[SHAPE_CAST]] : memref<1x16x3x9xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopySegmented
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
func.func @MoveGenericReshapeBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x4x64x64xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x4x64x64xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>)  ->  memref<1x4x64x64xf16, {order = #NHWC}, @DDR>
    //CHECK:    return [[COPY]] : memref<1x4x64x64xf16, {order = #NHWC}, @DDR>
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
    %out = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1024xf16, @DDR>

    return %1 : memref<1x16x1024xf16, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]]
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1024xf16, @DDR>
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

// CHECK-LABEL: @DoNotMoveGenericReshapeWithDifferentOrderBeforeTilingCopySegmented
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN,
func.func @DoNotMoveGenericReshapeWithDifferentOrderBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x1x32x1024xf16, {order = #NCWH}, @DDR> {
    %out = memref.alloc() : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x32x32xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x1x32x1024xf16, {order = #NCWH}, @DDR>

    return %1 : memref<1x1x32x1024xf16, {order = #NCWH}, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]]
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x32x32xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>) -> memref<1x1x32x1024xf16, {order = #NCWH}, @DDR>
    // CHECK:    return [[RESHAPE]] : memref<1x1x32x1024xf16, {order = #NCWH}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1x512x2xf16, #NCWH, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 128, 2], [1, 1, 128, 2], [1, 1, 128, 2], [1, 1, 128, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 128, 0], [0, 0, 256, 0], [0, 0, 384, 0]],
    memory_shapes = [[1, 1, 128, 2], [1, 1, 128, 2], [1, 1, 128, 2], [1, 1, 128, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 128, 0], [0, 0, 256, 0], [0, 0, 384, 0]]
}>

// CHECK-LABEL: @DoNotMoveGenericReshapeWithDifferentLayout
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x1x512x2xf16, #NCWH, @CMX_NN,
func.func @DoNotMoveGenericReshapeWithDifferentLayout(%arg0: !InputDistributed) -> memref<1x512x2x1xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x1x512x2xf16, {order = #NCWH}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x1x512x2xf16, {order = #NCWH}, @DDR>) -> memref<1x1x512x2xf16, {order = #NCWH}, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x1x512x2xf16, {order = #NCWH}, @DDR>) -> memref<1x512x2x1xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x512x2x1xf16, {order = #NHWC}, @DDR>

    // CHECK:    [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x1x512x2xf16, {order = #NCWH}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]]
    // CHECK-SAME:     outputs([[OUT_BUFF]] : memref<1x1x512x2xf16, {order = #NCWH}, @DDR>) -> memref<1x1x512x2xf16, {order = #NCWH}, @DDR>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x1x512x2xf16, {order = #NCWH}, @DDR>) -> memref<1x512x2x1xf16, {order = #NHWC}, @DDR>
    // CHECK:    return [[RESHAPE]] : memref<1x512x2x1xf16, {order = #NHWC}, @DDR>
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

// CHECK-LABEL: @MoveQuantizeCastBeforeTilingCopySegmentedOverH
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
func.func @MoveQuantizeCastBeforeTilingCopySegmentedOverH(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    //CHECK:    [[QUANTIZE:%.+]] = VPUIP.QuantizeCast inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    //CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
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
func.func @MoveQuantizeCastBeforeTilingCopySegmentedOverK(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    //CHECK:      [[QUANTIZE:%.+]] = VPUIP.QuantizeCast
    //CHECK-SAME:   inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    //CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
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
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
func.func @MoveQuantizeCastBeforeTilingCopyOverlapped(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)
        -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    // CHECK:    [[QUANTIZE:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
    // CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
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
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
func.func @MoveQuantizeCastBeforeTilingCopyOverlappedMixedTypes(%arg0: !InputDistributed) -> memref<1x16x32x32xui8, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)
        -> memref<1x16x32x32xui8, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x32x32xui8, {order = #NHWC}, @DDR>

    // CHECK:    [[QUANTIZE:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x32x32xui8, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x32x32xui8, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTIZE]] : !VPUIP.DistributedBuffer<1x16x32x32xui8, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x32x32xui8, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32xui8, {order = #NHWC}, @DDR>
    // CHECK:    return [[COPY]] : memref<1x16x32x32xui8, {order = #NHWC}, @DDR>
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
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
func.func @SkipPerChannelOverlappedQuantizeCast(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)
        -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    // CHECK:   [[ALLOCATE:%.+]] = memref.alloc() : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:    [[TILING:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN
    // CHECK-SAME:     outputs([[ALLOCATE]] : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)
    // CHECK-SAME:     ->  memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>

    // CHECK:   [[QUANTIZE_CAST:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs([[TILING]] : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)
    // CHECK-SAME:  -> memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>

    // CHECK:   return [[QUANTIZE_CAST]] : memref<1x16x32x32x!qElemType1, {order = #NHWC}, @DDR>
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
    %1 = memref.alloc() : memref<1x3x9x16xf16, {order = #NHWC}, @DDR>
    %2 = VPUIP.Copy
        inputs(%0 : !Distributed1)
        outputs(%1 : memref<1x3x9x16xf16, {order = #NHWC}, @DDR>)  ->  memref<1x3x9x16xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.ShapeCast {shape = [1, 48, 3, 3]} inputs(%2 : memref<1x3x9x16xf16, {order = #NHWC}, @DDR>) -> memref<1x48x3x3xf16, {order = #NHWC}, @DDR>
    %4 = VPURT.AllocDistributed -> !Distributed2
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x48x3x3xf16, {order = #NHWC}, @DDR>)
        outputs(%4 : !Distributed2)  ->  !Distributed2

    return %5 : !Distributed2

    // CHECK:       [[VIEWOP:%.+]] = VPUIP.ViewOp [[ARG0]] : !VPUIP.DistributedBuffer<1x16x3x9xf16, #NWCH, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    // CHECK-SAME:   to !VPUIP.DistributedBuffer<1x3x9x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[MEMREF_ALLOC:%.+]] = memref.alloc() : memref<1x3x9x16xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[VIEWOP]] : !VPUIP.DistributedBuffer<1x3x9x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[MEMREF_ALLOC]] : memref<1x3x9x16xf16, {order = #NHWC}, @DDR>)  ->  memref<1x3x9x16xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 48, 3, 3]} inputs([[COPY0]] : memref<1x3x9x16xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:            -> memref<1x48x3x3xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[ALLOC_DISTRIBUTED:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x48x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:     {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>

    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[SHAPE_CAST]] : memref<1x48x3x3xf16, {order = #NHWC}, @DDR>)
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

!InputDistributed3 = !VPUIP.DistributedBuffer<
    64x16x1x1x!qElemType5, #NHWC, @CMX_NN, {
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

func.func @MoveQuantizeCastBeforeTilingCopyMultipleConsumers(%in0: !InputDistributed0, %in1: !InputDistributed1, %in3: !InputDistributed3, %in5: !InputDistributed5) -> (memref<1x64x32x32x!qElemType3, {order = #NHWC}, @DDR>, !OutputDistributed) {

    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 3030 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in0 : !InputDistributed0)
        weights(%in1 : !InputDistributed1)
        parent_input(%in0 : !InputDistributed0)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 63], outStart = [0, 16, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %4 = memref.alloc() : memref<1x64x32x32x!qElemType1, {order = #NHWC}, @DDR>
    %5 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x32x32x!qElemType1, {order = #NHWC}, @DDR>)  ->  memref<1x64x32x32x!qElemType1, {order = #NHWC}, @DDR>
    %7 = VPUIP.QuantizeCast inputs(%5 : memref<1x64x32x32x!qElemType1, {order = #NHWC}, @DDR>) -> memref<1x64x32x32x!qElemType3, {order = #NHWC}, @DDR>
    %8 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %9 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1180 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<DWCONV>}>
        input(%1 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights(%in3 : !InputDistributed3)
        parent_input(%1 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output(%8 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%8 : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x64x32x32x!qElemType4, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 15, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 63], outStart = [0, 16, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %7, %9 : memref<1x64x32x32x!qElemType3, {order = #NHWC}, @DDR>, !OutputDistributed
    // CHECK:       [[NCE_OUT:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[NCE_OUT]] : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType5, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x32x32x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF:%.+]] = memref.alloc() : memref<1x64x32x32x!qElemType3, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x32x32x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUF]] : memref<1x64x32x32x!qElemType3, {order = #NHWC}, @DDR>)  ->  memref<1x64x32x32x!qElemType3, {order = #NHWC}, @DDR>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[NCE_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16:1, {1.0, 1.75, 1.5, 1.25, 1.0, 1.5, 1.75, 1.25}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {1.0, 1.75, 1.5, 1.25, 1.0, 1.5}>
// CHECK-DAG: [[QUANT_8_CHAN:.+]] = !quant.uniform<u8<0:254>:f16:1, {
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME:	1.5
// CHECK-DAG-SAME:	1.25
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME:	1.5
// CHECK-DAG-SAME:	1.25
// CHECK-DAG-SAME: }>

// CHECK-DAG: [[QUANT_6_CHAN:.+]] = !quant.uniform<u8<0:254>:f16:1, {
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME:	1.5
// CHECK-DAG-SAME:	1.25
// CHECK-DAG-SAME:	1.0
// CHECK-DAG-SAME:	1.75
// CHECK-DAG-SAME: }>

// CHECK: @MoveSubViewWithPerAxisQuantization
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x8x2x2x!qElemType, @DDR>,
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x6x2x2x!qElemType1, @DDR>)
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

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 6, 2, 2] :
    // CHECK-SAME:  memref<1x8x2x2x[[QUANT_8_CHAN]], @DDR>
    // CHECK-SAME:  to memref<1x6x2x2x[[QUANT_6_CHAN]], {order = #NCHW, strides = [32, 4, 2, 1]}, @DDR>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>

    // CHECK: [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[SUBVIEW]] : memref<1x6x2x2x[[QUANT_6_CHAN]], {order = #NCHW, strides = [32, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:  outputs([[ALLOC]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>)
    // CHECK-SAME:      -> memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>

    // CHECK: [[OUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[IN_COPY]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>) -> memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>

    // CHECK:   return [[OUT_COPY]] : memref<1x6x2x2x[[QUANT_6_CHAN]], @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotMoveShapeCastWhenCompressConv
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x4x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>,
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<32x1x1x32xf16, {order = #NHWC}>)
func.func @DoNotMoveShapeCastWhenCompressConv(
        %arg0: memref<1x4x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>,
        %arg1: memref<32x1x1x32xf16, {order = #NHWC}>)
        -> memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]> {

    %0 = memref.alloc() : memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg1 : memref<32x1x1x32xf16, {order = #NHWC}>) outputs(%0 : memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.ShapeCast {shape = [32, 16, 3, 3]} inputs(%1 : memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<32x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPUIP.ShapeCast {shape = [1, 16, 104, 208]} inputs(%arg0 : memref<1x4x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %7 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4294967398 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{cm_sp_pattern = 15 : i64, input_channels_compression,
                                kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                                kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
                input(%6 : memref<1x16x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                weights(%4 : memref<32x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                parent_input(%6 : memref<1x16x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                parent_output(%5 : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                outputs(%5 : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                -> memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>
    variants : {
      DPUTask {inEnd = [207, 103, 3], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [103, 51, 31], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }

    return %7 : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_WEIGHTS:%.+]] = memref.alloc() : memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[W_CMX:%.+]] = VPUIP.Copy inputs([[ARG_1]] : memref<32x1x1x32xf16, {order = #NHWC}>) outputs([[ALLOC_WEIGHTS]] : memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>


    // CHECK:       [[SHAPECAST_WEIGHTS:%.+]] = VPUIP.ShapeCast {shape = [32, 16, 3, 3]} inputs([[W_CMX]] : memref<32x1x1x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<32x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:       [[SHAPECAST_INPUT:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 104, 208]} inputs([[ARG_0]] : memref<1x4x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[COMPRESS_CONV:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4294967398 : i64} <{cm_sp_pattern = 15 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     kernel_size = [3, 3], kernel_strides = [2, 2],
    // CHECK-SAME:     task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:  input([[SHAPECAST_INPUT]] : memref<1x16x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  weights([[SHAPECAST_WEIGHTS]] : memref<32x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_input([[SHAPECAST_INPUT]] : memref<1x16x104x208xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_output([[ALLOC_OUTPUT]] : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[ALLOC_OUTPUT]] : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
    // CHECK:        DPUTask {inEnd = [207, 103, 3], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [103, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:       PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:       }

    // CHECK:  return [[COMPRESS_CONV]] : memref<1x32x52x104xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MoveSubviewToTheFrontOfCopyMultipleConsumers
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x16x2x2xf16, @DDR>,
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x8x2x2xf16, @DDR>)
func.func @MoveSubviewToTheFrontOfCopyMultipleConsumers(%arg0: memref<1x16x2x2xf16, @DDR>, %arg1: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %1 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %2 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %3 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%1 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %4 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%2 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %5 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 8, 2, 2] : memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>) outputs(%arg1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    return %6 : memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 8, 2, 2] :
    // CHECK-SAME:                      memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>

    // CHECK:       [[BUF_0:%.+]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:                      outputs([[BUF_0]] : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[BUF_1:%.+]] = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x16x2x2xf16, @DDR>) outputs([[BUF_1]] : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[COPY_0]] : memref<1x8x2x2xf16, @DDR>) outputs([[ARG_1]] : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    // CHECK:       return [[COPY_2]] : memref<1x8x2x2xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveSubviewToTheFrontOfTillingCopyMultipleConsumers
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
func.func @MoveSubviewToTheFrontOfTillingCopyMultipleConsumers(%in0 : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>,
                                %in1 : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
                                    -> (memref<1x24x128x128xf16, {order = #NHWC}, @DDR>, memref<1x32x128x128xf16, {order = #NHWC}, @DDR>) {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 20758 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%in0 : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
        weights(%in1 : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in0 : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 24, 128, 128] : memref<1x32x128x128xf16, {order = #NHWC}, @DDR> to memref<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    %5 = memref.alloc() : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>) outputs(%5 : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x24x128x128xf16, {order = #NHWC}, @DDR>

    %7 = memref.alloc() : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
    %8 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%7 : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    return %6, %8 : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>, memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 20758 : i64} <{is_inplace = true,
    // CHECK-SAME:                   task_type = #VPUIP.nce_task_type<ELTWISE>}>
    // CHECK-SAME:     input([[ARG_0]] : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     weights([[ARG_1]] : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_input([[ARG_0]] : memref<1x32x128x128xf16, {order = #NHWC}, @CMX_NN>)
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

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[Tilling_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x24x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>)  ->  memref<1x24x128x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs([[Tilling_COPY_0]] : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>) outputs([[BUFF_2]] : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x24x128x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_3:%.+]] = memref.alloc() : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[Tilling_COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_3]] : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       return [[COPY]], [[Tilling_COPY_1:%.+]] : memref<1x24x128x128xf16, {order = #NHWC}, @DDR>, memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMoveSubviewToTheFrontOfTillingCopyForIncompatibleDistributedBuffer
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[ARG_1:%[^:]+]]: memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
func.func @NotMoveSubviewToTheFrontOfTillingCopyForIncompatibleDistributedBuffer(%in0 : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>,
                                %in1 : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
                                    -> memref<1x32x128x128xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 20758 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%in0 : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
        weights(%in1 : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in0 : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x32x129x128xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x32x129x128xf16, {order = #NHWC}, @DDR>)  ->  memref<1x32x129x128xf16, {order = #NHWC}, @DDR>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 128, 128] : memref<1x32x129x128xf16, {order = #NHWC}, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>
    %5 = memref.alloc() : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>) outputs(%5 : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    return %6 : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 20758 : i64} <{is_inplace = true,
    // CHECK-SAME:                   task_type = #VPUIP.nce_task_type<ELTWISE>}>
    // CHECK-SAME:     input([[ARG_0]] : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     weights([[ARG_1]] : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_input([[ARG_0]] : memref<1x32x129x128xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:     ->  !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        // CHECK:     DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        // CHECK:     DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
        // CHECK:     PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x32x129x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[Tilling_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : !VPUIP.DistributedBuffer<1x32x129x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x32x129x128xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME: ->  memref<1x32x129x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[Tilling_COPY_0]] [0, 0, 0, 0] [1, 32, 128, 128] :
    // CHECK-SAME:      memref<1x32x129x128xf16, {order = #NHWC}, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x32x128x128xf16, {order = #NHWC, strides = [528384, 1, 4096, 32]}, @DDR>) outputs([[BUFF_2]] : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x32x128x128xf16, {order = #NHWC}, @DDR>

    // CHECK:       return [[COPY]] : memref<1x32x128x128xf16, {order = #NHWC}, @DDR>
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
    %out = memref.alloc() : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x48x32x32xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.ShapeCast {shape = [1, 4, 96, 128]} inputs(%0 : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x96x128xf16, {order = #NHWC}, @DDR>

    %cmxBuff = VPURT.AllocDistributed -> !OutputDistributed
    %2 = VPUIP.Copy {out_mem_space = @CMX_NN}
        inputs(%1 : memref<1x4x96x128xf16, {order = #NHWC}, @DDR>)
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

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x96x128xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:         inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x4x96x128xf16, #NHWC, @CMX_NN
    //CHECK-SAME:         outputs([[OUTBUFF]] : memref<1x4x96x128xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:     -> memref<1x4x96x128xf16, {order = #NHWC}, @DDR>
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
    %out = memref.alloc() : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>)  ->  memref<1x48x32x32xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>)
        -> memref<1x384x1x128xf16, {order = #NHWC}, @DDR>

    %cmxBuff = VPURT.AllocDistributed -> !OutputDistributed
    %2 = VPUIP.Copy {out_mem_space = @CMX_NN}
        inputs(%1 : memref<1x384x1x128xf16, {order = #NHWC}, @DDR>)
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

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x384x1x128xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:         inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x384x1x128xf16, #NHWC, @CMX_NN
    //CHECK-SAME:         outputs([[OUTBUFF]] : memref<1x384x1x128xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:     -> memref<1x384x1x128xf16, {order = #NHWC}, @DDR>
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
    %out = memref.alloc() : memref<1x48x1x16xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x48x1x16xf16, {order = #NHWC}, @DDR>)  ->  memref<1x48x1x16xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHCW} inputs(%0 : memref<1x48x1x16xf16, {order = #NHWC}, @DDR>)
            -> memref<1x48x16x1xf16, {order = #NHWC}, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x48x16x1xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x16x3xf16, {order = #NHWC}, @DDR>

    %cmxBuff = VPURT.AllocDistributed -> !OutputDistributed
    %3 = VPUIP.Copy {out_mem_space = @CMX_NN}
        inputs(%2 : memref<1x16x16x3xf16, {order = #NHWC}, @DDR>)
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

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x16x3xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:         inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x16x16x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         outputs([[OUTBUFF]] : memref<1x16x16x3xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:     -> memref<1x16x16x3xf16, {order = #NHWC}, @DDR>
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

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyRankChanged
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<4x64x1x1xf16, #NCHW, @CMX_NN,
func.func @MoveGenericReshapeBeforeTilingCopyRankChanged(%arg0: !InputDistributed) -> memref<1x4x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<1x4x64xf16, @DDR>

    return %1 : memref<1x4x64xf16, @DDR>

    //  CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //  CHECK-SAME:       inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
    //  CHECK-SAME:           4x64x1x1xf16, #NCHW, @CMX_NN, {
    //  CHECK-SAME:           mode = "DUPLICATED|SEGMENTED",
    //  CHECK-SAME:           num_tiles = [1, 2, 1, 1],
    //  CHECK-SAME:           num_clusters = 2 : i64,
    //  CHECK-SAME:           alignment = [1, 16, 1, 1],
    //  CHECK-SAME:           uniform_distributed_segments
    //  CHECK-SAME:           }>)
    //  CHECK-SAME:       -> !VPUIP.DistributedBuffer<
    //  CHECK-SAME:           1x4x64xf16, #CHW, @CMX_NN, {
    //  CHECK-SAME:           mode = "DUPLICATED",
    //  CHECK-SAME:           num_clusters = 2 : i64,
    //  CHECK-SAME:           uniform_distributed_segments
    //  CHECK-SAME:           }>

    //  CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x4x64xf16, @DDR>
    //  CHECK:    [[COPY:%.+]] = VPUIP.Copy
    //  CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x4x64xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>)
    //  CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x4x64xf16, @DDR>)  ->  memref<1x4x64xf16, @DDR>

    //  CHECK:    return [[COPY]] : memref<1x4x64xf16, @DDR>
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

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyRankChanged
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<4x64x1x1xf16, #NCHW, @CMX_NN,
func.func @MoveGenericReshapeBeforeTilingCopyRankChangedExplicitDistribution(%arg0: !InputDistributed) -> memref<1x4x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<1x4x64xf16, @DDR>

    return %1 : memref<1x4x64xf16, @DDR>

    //CHECK:        [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:           inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
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

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyRankChangedAndSplitHigherDim
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<4x64x1x1xf16, #NCHW, @CMX_NN,
func.func @MoveGenericReshapeBeforeTilingCopyRankChangedAndSplitHigherDim(%arg0: !InputDistributed) -> memref<2x2x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<2x2x64xf16, @DDR>

    return %1 : memref<2x2x64xf16, @DDR>

    //CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:       inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
    //CHECK-SAME:           4x64x1x1xf16, #NCHW, @CMX_NN, {
    //CHECK-SAME:           mode = "DUPLICATED|SEGMENTED",
    //CHECK-SAME:           num_tiles = [1, 2, 1, 1],
    //CHECK-SAME:           num_clusters = 2 : i64,
    //CHECK-SAME:           alignment = [1, 16, 1, 1],
    //CHECK-SAME:           uniform_distributed_segments
    //CHECK-SAME:           }>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<
    //CHECK-SAME:           2x2x64xf16, #CHW, @CMX_NN, {
    //CHECK-SAME:           mode = "DUPLICATED",
    //CHECK-SAME:           num_clusters = 2 : i64,
    //CHECK-SAME:           uniform_distributed_segments
    //CHECK-SAME:           }>

    //CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<2x2x64xf16, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<2x2x64xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>)

    // CHECK:    return [[COPY]] : memref<2x2x64xf16, @DDR>
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

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyRankChangedAndSplitHigherDimExplicitDistribution
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<4x64x1x1xf16, #NCHW, @CMX_NN,
func.func @MoveGenericReshapeBeforeTilingCopyRankChangedAndSplitHigherDimExplicitDistribution(%arg0: !InputDistributed) -> memref<2x2x64xf16, @DDR> {
    %out = memref.alloc() : memref<4x64x1x1xf16, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<4x64x1x1xf16, @DDR>)  ->  memref<4x64x1x1xf16, @DDR>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<4x64x1x1xf16, @DDR>) -> memref<2x2x64xf16, @DDR>

    return %1 : memref<2x2x64xf16, @DDR>

    //CHECK:        [[RESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:           inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
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

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @NoChangeForStridedCopy
func.func @NoChangeForStridedCopy(
        %arg0: memref<1x2048x8192xui8, {order = #CHW, strides = [33554432, 16384, 1]}, @DDR>) -> memref<1x1x4096x4096xui8, @DDR> {

    %0 = memref.alloc() : memref<1x2048x8192xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x2048x8192xui8, {order = #CHW, strides = [33554432, 16384, 1]}, @DDR>)
        outputs(%0 : memref<1x2048x8192xui8, @DDR>)
        -> memref<1x2048x8192xui8, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x2048x8192xui8, @DDR>) -> memref<1x1x4096x4096xui8, @DDR>

    return %2 : memref<1x1x4096x4096xui8, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.GenericReshape
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @ChangeForStridedCopy
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x256x32xui8, {order = #CHW, strides = [16384, 32, 1]}, @DDR>)
func.func @ChangeForStridedCopy(
        %arg0: memref<1x256x32xui8, {order = #CHW, strides = [16384, 32, 1]}, @DDR>) -> memref<256x32xui8, @DDR> {

    %0 = memref.alloc() : memref<1x256x32xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x256x32xui8, {order = #CHW, strides = [16384, 32, 1]}, @DDR>)
        outputs(%0 : memref<1x256x32xui8, @DDR>)
        -> memref<1x256x32xui8, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x32xui8, @DDR>) -> memref<256x32xui8, @DDR>

    return %2 : memref<256x32xui8, @DDR>
    // CHECK:   [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x256x32xui8, {order = #CHW, strides = [16384, 32, 1]}, @DDR>) -> memref<256x32xui8, @DDR>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<256x32xui8, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy inputs([[GENERICRESHAPE]] : memref<256x32xui8, @DDR>) outputs([[ALLOC]] : memref<256x32xui8, @DDR>) -> memref<256x32xui8, @DDR>
    // CHECK:   return [[COPY]] : memref<256x32xui8, @DDR>
}

// -----

// CHECK-LABEL: @MoveTrivialPermuteCastBeforeTilingCopySegmented5D
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<17x2048x16x1x1xf16,
func.func @MoveTrivialPermuteCastBeforeTilingCopySegmented5D(
        %arg0: !VPUIP.DistributedBuffer<17x2048x16x1x1xf16, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
            compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [5, 2048, 16, 1, 1]],
            compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
            memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [5, 2048, 16, 1, 1]],
            memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>)
        -> memref<17x1x16x2048x1xf16, {order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>}> {
    %out = memref.alloc() : memref<17x2048x16x1x1xf16>
    %0 = VPUIP.Copy
        inputs(%arg0 : !VPUIP.DistributedBuffer<17x2048x16x1x1xf16, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, @CMX_NN,
            {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
            compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [5, 2048, 16, 1, 1]],
            compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
            memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [5, 2048, 16, 1, 1]],
            memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>)
        outputs(%out : memref<17x2048x16x1x1xf16>)
        -> memref<17x2048x16x1x1xf16>

    %1 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>}
        inputs(%0 : memref<17x2048x16x1x1xf16>)
        -> memref<17x1x16x2048x1xf16, {order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>}>

    return %1 : memref<17x1x16x2048x1xf16, {order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>}>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #map}
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<17x2048x16x1x1xf16, #NCDHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [5, 2048, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [5, 2048, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<17x1x16x2048x1xf16, #GNHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [5, 1, 16, 2048, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [5, 1, 16, 2048, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>
    // CHECK:       [[BUF:%.+]] = memref.alloc() : memref<17x1x16x2048x1xf16, {order = #GNHWC}>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<17x1x16x2048x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64
    // CHECK-SAME:      outputs([[BUF]] : memref<17x1x16x2048x1xf16, {order = #GNHWC}>) -> memref<17x1x16x2048x1xf16, {order = #GNHWC}>
    // CHECK:       return [[COPY]] : memref<17x1x16x2048x1xf16, {order = #GNHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedReshape = !VPUIP.DistributedBuffer<
    1x512x32x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0], [0, 256, 0, 0], [0, 384, 0, 0]],
    memory_shapes = [[1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0], [0, 256, 0, 0], [0, 384, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeCopySegmentedOverC
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x512x32x128xf16,
func.func @MoveGenericReshapeBeforeCopySegmentedOverC(
        %arg0: !InputDistributedReshape) -> memref<512x4096x1x1xf16> {
    %alloc = memref.alloc() : memref<1x512x32x128xf16>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributedReshape)
        outputs(%alloc : memref<1x512x32x128xf16>)
        -> memref<1x512x32x128xf16>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x512x32x128xf16>) -> memref<512x4096x1x1xf16>

    return %1 : memref<512x4096x1x1xf16>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
    // CHECK-SAME:          1x512x32x128xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 4, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0], [0, 256, 0, 0], [0, 384, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128], [1, 128, 32, 128]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0], [0, 256, 0, 0], [0, 384, 0, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:          512x4096x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [4, 1, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>
    // CHECK:       [[OUTBUFF:%.+]] = memref.alloc() : memref<512x4096x1x1xf16>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<512x4096x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:      outputs([[OUTBUFF]] : memref<512x4096x1x1xf16>) -> memref<512x4096x1x1xf16>
    // CHECK:       return [[COPY]] : memref<512x4096x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedPermute = !VPUIP.DistributedBuffer<
    512x4096x1x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    memory_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]
}>

// CHECK-LABEL: @MovePermuteCastBeforeCopySegmentedOverN
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<512x4096x1x1xf16,
func.func @MovePermuteCastBeforeCopySegmentedOverN(
        %arg0: !InputDistributedPermute) -> memref<512x4096x1x1xf16, {order = #NHWC}> {
    %alloc = memref.alloc() : memref<512x4096x1x1xf16>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributedPermute)
        outputs(%alloc : memref<512x4096x1x1xf16>)
        -> memref<512x4096x1x1xf16>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%0 : memref<512x4096x1x1xf16>) -> memref<512x4096x1x1xf16, {order = #NHWC}>

    return %1 : memref<512x4096x1x1xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<
    // CHECK-SAME:          512x4096x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [4, 1, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:          512x4096x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [4, 1, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1], [128, 4096, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]}>
    // CHECK:       [[OUTBUFF:%.+]] = memref.alloc() : memref<512x4096x1x1xf16, {order = #NHWC}>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<512x4096x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:      outputs([[OUTBUFF]] : memref<512x4096x1x1xf16, {order = #NHWC}>) -> memref<512x4096x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[COPY]] : memref<512x4096x1x1xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!InputDistributed = !VPUIP.DistributedBuffer<
    18x1x16x2048x1xf16, #GNHWC, @DDR, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    18x1x16x512x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

// CHECK-LABEL: @Move5DGenericReshapeBeforeDDRToCMXCopySEGMENTED
// CHECK-SAME: [[INPUT:%.+]]: !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @DDR
func.func @Move5DGenericReshapeBeforeDDRToCMXCopySEGMENTED(%arg0: !InputDistributed) -> !OutputDistributed {
    %0 = memref.alloc() : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>
    %3 = memref.alloc() : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %4 = VPUIP.Copy
        inputs(%2 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>)
        outputs(%3 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.Copy
        inputs(%4 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>)
        outputs(%5 : !OutputDistributed) -> !OutputDistributed
    return %6 : !OutputDistributed

    // CHECK:      [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @DDR
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @DDR
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK:      [[ALLOC1:%.+]] = memref.alloc() : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>
    // CHECK:      [[COPY1:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @DDR
    // CHECK-SAME: outputs([[ALLOC1]] : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>

    // CHECK:      [[ALLOC2:%.+]] = memref.alloc() : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    // CHECK:      [[COPY2:%.+]] = VPUIP.Copy inputs([[COPY1]] : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>)
    // CHECK-SAME: outputs([[ALLOC2]] : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    // CHECK:      [[ALLOC_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @CMX_NN
    // CHECK:      [[COPY3:%.+]] = VPUIP.Copy inputs([[COPY2]] : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>)
    // CHECK-SAME: outputs([[ALLOC_DIST]] : !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @CMX_NN
    // CHECK:      return [[COPY3]] : !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @CMX_NN
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!InputDistributed = !VPUIP.DistributedBuffer<
    18x1x16x512x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    18x1x16x2048x1xf16, #GNHWC, @DDR, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeCopyForMatMulBackprop
// CHECK-SAME: [[INPUT:%.+]]: !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @CMX_NN
func.func @MoveGenericReshapeBeforeCopyForMatMulBackprop(%arg0: !InputDistributed) -> !OutputDistributed {
    // Forward path: MatMul output (18x1x16x512x4) needs to be reshaped to (18x1x16x2048x1) for next layer
    %0 = memref.alloc() : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    %4 = VPUIP.Copy
        inputs(%2 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>)
        outputs(%3 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.Copy
        inputs(%4 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>)
        outputs(%5 : !OutputDistributed) -> !OutputDistributed
    return %6 : !OutputDistributed

    // CHECK:      [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @CMX_NN
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64
    // CHECK:      [[ALLOC0:%.+]] = memref.alloc() : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>
    // CHECK:      [[COPY0:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @CMX_NN
    // CHECK-SAME: outputs([[ALLOC0]] : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>
    // CHECK:      [[COPY_DDR:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>)
    // CHECK-SAME: outputs({{%.+}} : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    // CHECK:      [[ALLOC_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @DDR
    // CHECK:      [[COPY1:%.+]] = VPUIP.Copy inputs([[COPY_DDR]] : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>)
    // CHECK-SAME: outputs([[ALLOC_DIST]] : !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @DDR
    // CHECK:      return [[COPY1]] : !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @DDR
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!InDist4D = !VPUIP.DistributedBuffer<
    1x18x2048x16xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]],
    memory_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]]
}>

!OutDist5D = !VPUIP.DistributedBuffer<
    18x2048x16x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshape4DTo5DBeforeCopySEGMENTED
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x18x2048x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
func.func @MoveGenericReshape4DTo5DBeforeCopySEGMENTED(%arg0: !InDist4D) -> !OutDist5D {
    %0 = memref.alloc() : memref<1x18x2048x16xf16, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : !InDist4D) outputs(%0 : memref<1x18x2048x16xf16, @CMX_NN>) -> memref<1x18x2048x16xf16, @CMX_NN>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x18x2048x16xf16, @CMX_NN>) -> memref<18x2048x16x1x1xf16, {order = #GNHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @CMX_NN>) outputs(%3 : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @DDR>) -> memref<18x2048x16x1x1xf16, {order = #GNHWC}, @DDR>
    %5 = VPURT.AllocDistributed -> !OutDist5D
    %6 = VPUIP.Copy inputs(%4 : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @DDR>) outputs(%5 : !OutDist5D) -> !OutDist5D
    return %6 : !OutDist5D

    // CHECK:      [[RESH:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME: inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x18x2048x16xf16, #NCHW, @CMX_NN,
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]], compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]], memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]]}>)
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<18x2048x16x1x1xf16, #GNHWC, @CMX_NN,
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>
    // CHECK:      [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[RESH]] : !VPUIP.DistributedBuffer<18x2048x16x1x1xf16, #GNHWC, @CMX_NN,
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>)
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!InDist5D_A = !VPUIP.DistributedBuffer<
    18x1x16x2048x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

!OutDist5D_A = !VPUIP.DistributedBuffer<
    18x1x16x512x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshape5DTo5DBeforeCopySEGMENTED_Forward
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<18x1x16x2048x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED"
func.func @MoveGenericReshape5DTo5DBeforeCopySEGMENTED_Forward(%arg0: !InDist5D_A) -> !OutDist5D_A {
    %0 = memref.alloc() : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : !InDist5D_A) outputs(%0 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) outputs(%3 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>
    %5 = VPURT.AllocDistributed -> !OutDist5D_A
    %6 = VPUIP.Copy inputs(%4 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @DDR>) outputs(%5 : !OutDist5D_A) -> !OutDist5D_A
    return %6 : !OutDist5D_A

    // CHECK: [[RESH:%.+]] = VPUIP.GenericReshape inputs([[INPUT]]
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[RESH]]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!InDist5D_B = !VPUIP.DistributedBuffer<
    18x1x16x512x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 512, 4], [6, 1, 16, 512, 4], [6, 1, 16, 512, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

!OutDist5D_B = !VPUIP.DistributedBuffer<
    18x1x16x2048x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1], [6, 1, 16, 2048, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshape5DTo5DBeforeCopySEGMENTED_Backward
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<18x1x16x512x4xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED"
func.func @MoveGenericReshape5DTo5DBeforeCopySEGMENTED_Backward(%arg0: !InDist5D_B) -> !OutDist5D_B {
    %0 = memref.alloc() : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : !InDist5D_B) outputs(%0 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<18x1x16x512x4xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @CMX_NN>) outputs(%3 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>) -> memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>
    %5 = VPURT.AllocDistributed -> !OutDist5D_B
    %6 = VPUIP.Copy inputs(%4 : memref<18x1x16x2048x1xf16, {order = #GNHWC}, @DDR>) outputs(%5 : !OutDist5D_B) -> !OutDist5D_B
    return %6 : !OutDist5D_B

    // CHECK: [[RESH:%.+]] = VPUIP.GenericReshape inputs([[INPUT]]
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[RESH]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!InDist5D_C = !VPUIP.DistributedBuffer<
    18x2048x16x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
}>

!OutDist4D = !VPUIP.DistributedBuffer<
    1x18x2048x16xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]],
    memory_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]]
}>

// CHECK-LABEL: @MoveGenericReshape5DTo4DBeforeCopySEGMENTED
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<18x2048x16x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED"
func.func @MoveGenericReshape5DTo4DBeforeCopySEGMENTED(%arg0: !InDist5D_C) -> !OutDist4D {
    %0 = memref.alloc() : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : !InDist5D_C) outputs(%0 : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @CMX_NN>) -> memref<18x2048x16x1x1xf16, {order = #GNHWC}, @CMX_NN>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<18x2048x16x1x1xf16, {order = #GNHWC}, @CMX_NN>) -> memref<1x18x2048x16xf16, @CMX_NN>
    %3 = memref.alloc() : memref<1x18x2048x16xf16, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x18x2048x16xf16, @CMX_NN>) outputs(%3 : memref<1x18x2048x16xf16, @DDR>) -> memref<1x18x2048x16xf16, @DDR>
    %5 = VPURT.AllocDistributed -> !OutDist4D
    %6 = VPUIP.Copy inputs(%4 : memref<1x18x2048x16xf16, @DDR>) outputs(%5 : !OutDist4D) -> !OutDist4D
    return %6 : !OutDist4D

    // CHECK:      [[RESH:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME: inputs([[INPUT]] : !VPUIP.DistributedBuffer<18x2048x16x1x1xf16, #GNHWC, @CMX_NN,
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1], [6, 2048, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [6, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>)
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x18x2048x16xf16, #NCHW, @CMX_NN,
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]], compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]], memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]]}>
    // CHECK:      [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[RESH]] : !VPUIP.DistributedBuffer<1x18x2048x16xf16, #NCHW, @CMX_NN,
    // CHECK-SAME: {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]], compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 6, 2048, 16], [1, 6, 2048, 16], [1, 6, 2048, 16]], memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0]]}>)
}
