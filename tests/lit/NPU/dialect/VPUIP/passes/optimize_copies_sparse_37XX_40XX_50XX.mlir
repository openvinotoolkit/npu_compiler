//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-rewriters="rewriter=optimize-copies-set" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @CopyWithSubViewOp(%in : memref<1x16x113x113xf16, {order = #NHWC}, @DDR>,
                        %in_sm : memref<1x16x113x113xi1, {order = #NHWC}, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
                        -> (memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>,
                            memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>) {

    %buf0 = memref.alloc() : memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>
    %buf0sm = memref.alloc() : memref<1x16x113x113xi1, {order = #NHWC}, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>
    %buf1sm = memref.alloc() : memref<1x16x56x56xi1, {order = #NHWC}, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x32x56x56xf16, {order = #NHWC}, @CMX_NN>
    %buf2sm = memref.alloc() : memref<1x32x56x56xi1, {order = #NHWC}, @CMX_NN>

    // activation copy-in
    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x113x113xf16, {order = #NHWC}, @DDR>)
            outputs(%buf0 : memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>

    %sm0 = VPUIP.Copy
            inputs(%in_sm : memref<1x16x113x113xi1, {order = #NHWC}, @DDR>)
            outputs(%buf0sm : memref<1x16x113x113xi1, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x113x113xi1, {order = #NHWC}, @CMX_NN>

    %1:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%0 : memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%sm0 : memref<1x16x113x113xi1, {order = #NHWC}, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%0 : memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        parent_output_sparsity_map(%buf1sm : memref<1x16x56x56xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        output_sparsity_map(%buf1sm : memref<1x16x56x56xi1, {order = #NHWC}, @CMX_NN>)
        -> memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>, memref<1x16x56x56xi1, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 10, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    // slice of buffer where the NCE writes
    %2 = VPUIP.SubView %buf2 [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56xf16, {order = #NHWC}, @CMX_NN> to
        memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    %sm2 = VPUIP.SubView %buf2sm [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56xi1, {order = #NHWC}, @CMX_NN> to
        memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // copy of the output NCE from NNCMX->NNCMX
    %3 = VPUIP.Copy
        inputs(%1#0 : memref<1x16x56x56xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%2 : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>)
        -> memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    %sm3 = VPUIP.Copy
        inputs(%1#1 : memref<1x16x56x56xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%sm2 : memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>)
        -> memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    return %2, %sm2 :
        memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>,
        memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // verify that the SubView operation is not removed along with the copy operation

    // CHECK:       [[VAL1SM:%.+]] = memref.alloc() : memref<1x32x56x56xi1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[VAL1:%.+]] = memref.alloc() : memref<1x32x56x56xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[VAL0:%.+]] = memref.alloc() : memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[VAL0SM:%.+]] = memref.alloc() : memref<1x16x113x113xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[VAL2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs({{%.+}} : memref<1x16x113x113xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x16x113x113xf16, {order = #NHWC}, @CMX_NN>)

    // CHECK:       [[VAL2SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs({{%.+}} : memref<1x16x113x113xi1, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[VAL0SM]] : memref<1x16x113x113xi1, {order = #NHWC}, @CMX_NN>)

    // subView present
    // CHECK:       [[VAL3:%.+]] = VPUIP.SubView [[VAL1]] [0, 0, 0, 0] [1, 16, 56, 56]
    // CHECK-SAME:      to memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // CHECK:       [[VAL3SM:%.+]] = VPUIP.SubView [[VAL1SM]] [0, 0, 0, 0] [1, 16, 56, 56]
    // CHECK-SAME:      to memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // CHECK:       [[VAL4:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK:           input([[VAL2]]
    // CHECK:           input_sparsity_map([[VAL2SM]]
    // CHECK:           outputs([[VAL3]]
    // CHECK:           output_sparsity_map([[VAL3SM]]

    // copy optimized
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[VAL3]], [[VAL3SM]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingTilingCopyOptimization(
        %in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>,
        %in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>,
        %in1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        -> (!VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, {order = #NHWC}, @DDR>, memref<1x128x36x36xi1, {order = #NHWC}, @DDR>) {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %9 = memref.alloc() : memref<1x128x36x36xf16, {order = #NHWC}, @DDR>
    %sm9 = memref.alloc() : memref<1x128x36x36xi1, {order = #NHWC}, @DDR>
    %2:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>)
        weights(%in1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>)
        parent_input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output_sparsity_map(%sm0 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        output_sparsity_map(%sm0 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> ,  !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %3:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>)
        weights(%in1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>)
        parent_input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>)
        parent_output(%1 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output_sparsity_map(%sm1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%1 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        output_sparsity_map(%sm1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> ,  !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %6 = VPUIP.Copy
        inputs(%2#0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm5 = VPUIP.SubView %sm4 [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm6 = VPUIP.Copy
        inputs(%2#1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm5 : !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %7 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %8 = VPUIP.Copy
        inputs(%3#0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%7 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm7 = VPUIP.SubView %sm4 [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm8 = VPUIP.Copy
        inputs(%3#1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm7 : !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %10 = VPUIP.ConcatView
        inputs(%6, %8 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%4 : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm10 = VPUIP.ConcatView
        inputs(%sm6, %sm8 : !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm4 : !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %11 = VPUIP.Copy
        inputs(%2#0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%9 : memref<1x128x36x36xf16, {order = #NHWC}, @DDR>)  ->  memref<1x128x36x36xf16, {order = #NHWC}, @DDR>
    %sm11 = VPUIP.Copy
        inputs(%2#1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm9 : memref<1x128x36x36xi1, {order = #NHWC}, @DDR>)  ->  memref<1x128x36x36xi1, {order = #NHWC}, @DDR>

    return %10, %sm10, %11, %sm11: !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, {order = #NHWC}, @DDR>, memref<1x128x36x36xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[BUFF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x36x36xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x128x36x36xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[BUFF_1_SM]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[NCETASK_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          outputs([[SUBVIEW_0]]
    // CHECK-SAME:          output_sparsity_map([[SUBVIEW_0_SM]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[SUBVIEW_1_SM:%.+]] = VPUIP.SubView [[BUFF_1_SM]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[NCETASK_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          outputs([[SUBVIEW_1]]
    // CHECK-SAME:          output_sparsity_map([[SUBVIEW_1_SM]]

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCETASK_0:%.+]]#0, [[NCETASK_1:%.+]]#0
    // CHECK-SAME:      outputs([[BUFF_1]]

    // CHECK:       [[CONCAT_SM:%.+]] = VPUIP.ConcatView inputs([[NCETASK_0:%.+]]#1, [[NCETASK_1:%.+]]#1
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[TILING:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCETASK_0]]#0
    // CHECK-SAME:      outputs([[BUFF_0]]

    // CHECK:       [[TILING_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCETASK_0]]#1
    // CHECK-SAME:      outputs([[BUFF_0_SM]]

    // CHECK:       return [[CONCAT]], [[CONCAT_SM]], [[TILING]], [[TILING_SM]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingTilingCopyOptimizationSameParent(
    %in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>,
    %in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>,
    %in1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>) ->
    (!VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {

    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>)
        weights(%in1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, {order = #NHWC}, @CMX_NN>)
        parent_input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output_sparsity_map(%sm0 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        output_sparsity_map(%sm0 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> ,  !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
        DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm3 = VPUIP.SubView %sm2 [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.Copy
        inputs(%1#0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%3 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm4 = VPUIP.Copy
        inputs(%1#1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm3 : !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %5 = VPUIP.SubView %2 [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm5 = VPUIP.SubView %sm2 [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %6 = VPUIP.Copy
        inputs(%1#0 : !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm6 = VPUIP.Copy
        inputs(%1#1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm5 : !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %7 = VPUIP.ConcatView
        inputs(%4, %6 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%2 : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm7 = VPUIP.ConcatView
        inputs(%sm4, %sm6 : !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm2 : !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    return %7, %sm7 : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[CONCAT_OUT_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[CONCAT_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CONCAT_OUT]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[CONCAT_OUT_SM]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[CONV:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[SUBVIEW_0]]
    // CHECK-SAME:      output_sparsity_map([[SUBVIEW_0_SM]]

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONCAT_OUT]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[SUBVIEW_1_SM:%.+]] = VPUIP.SubView [[CONCAT_OUT_SM]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[SUBVIEW_1_COPY:%.+]] = VPUIP.Copy inputs([[CONV]]#0
    // CHECK-SAME:  outputs([[SUBVIEW_1]]

    // CHECK: [[SUBVIEW_1_COPY_SM:%.+]] = VPUIP.Copy inputs([[CONV]]#1
    // CHECK-SAME:  outputs([[SUBVIEW_1_SM]]

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CONV]]#0, [[SUBVIEW_1_COPY]]
    // CHECK-SAME:  outputs([[CONCAT_OUT]]
    // CHECK: [[CONCAT_SM:%.+]] = VPUIP.ConcatView inputs([[CONV]]#1, [[SUBVIEW_1_COPY_SM]]
    // CHECK-SAME:  outputs([[CONCAT_OUT_SM]]

    // CHECK:  return [[CONCAT]], [[CONCAT_SM]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvOut0 = !VPUIP.DistributedBuffer<
    1x48x14x14xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvSMOut0 = !VPUIP.DistributedBuffer<
    1x48x14x14xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistribCast0 = !VPUIP.DistributedBuffer<
    1x48x14x14xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistribCastSM0 = !VPUIP.DistributedBuffer<
    1x48x14x14xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConcatIn0 = !VPUIP.DistributedBuffer<
    1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConcatSMIn0 = !VPUIP.DistributedBuffer<
    1x48x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOut1 = !VPUIP.DistributedBuffer<
    1x96x14x14xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvSMOut1 = !VPUIP.DistributedBuffer<
    1x96x14x14xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistribCast1 = !VPUIP.DistributedBuffer<
    1x96x14x14xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistribCastSM1 = !VPUIP.DistributedBuffer<
    1x96x14x14xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConcatIn1 = !VPUIP.DistributedBuffer<
    1x96x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConcatSMIn1 = !VPUIP.DistributedBuffer<
    1x96x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConcatOut = !VPUIP.DistributedBuffer<
    1x144x14x14xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!ConcatSMOut = !VPUIP.DistributedBuffer<
    1x144x14x14xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @CMX2CMXCopyOptimizationWithDuplicatedExplicitDistributedAttr
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME:  [[WEIGHTS0:%.+]]: memref<48x16x1x1xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME:  [[WEIGHTS1:%.+]]: memref<96x16x1x1xf16, {order = #NHWC}, @CMX_NN>
func.func @CMX2CMXCopyOptimizationWithDuplicatedExplicitDistributedAttr(
  %input: memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>, %weights0: memref<48x16x1x1xf16, {order = #NHWC}, @CMX_NN>,
  %weights1: memref<96x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
      -> (!ConcatOut, !ConcatSMOut) {

  %concatBuff = VPURT.AllocDistributed -> !ConcatOut
  %concatSMBuff = VPURT.AllocDistributed -> !ConcatSMOut

  %outBuff0 = VPURT.AllocDistributed -> !ConvOut0
  %outSM0 = VPURT.AllocDistributed -> !ConvSMOut0
  %conv0:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%input : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
      weights(%weights0 : memref<48x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
      parent_input(%input : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
      parent_output(%outBuff0 : !ConvOut0)
      parent_output_sparsity_map(%outSM0 : !ConvSMOut0)
      outputs(%outBuff0 : !ConvOut0)
      output_sparsity_map(%outSM0 : !ConvSMOut0)
  ->  !ConvOut0 ,  !ConvSMOut0 variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [13, 13, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 13, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 0 : i64, inEnd = [13, 13, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 13, 31], outStart = [0, 0, 15], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 0 : i64, inEnd = [13, 13, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 13, 47], outStart = [0, 0, 32], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
  } PPE : {
  }

  %distributedCast0 = VPUIP.DistributedCast inputs(%conv0#0 : !ConvOut0) -> !DistribCast0

  %subview0 = VPUIP.SubView %concatBuff [0, 0, 0, 0] [1, 48, 14, 14] : !ConcatOut to !ConcatIn0
  %concatIn0 = VPUIP.Copy
      inputs(%distributedCast0 : !DistribCast0)
      outputs(%subview0 : !ConcatIn0)  ->  !ConcatIn0

  %distributedCastSM0 = VPUIP.DistributedCast inputs(%conv0#1 : !ConvSMOut0) -> !DistribCastSM0

  %subviewSM0 = VPUIP.SubView %concatSMBuff [0, 0, 0, 0] [1, 48, 14, 14] : !ConcatSMOut to !ConcatSMIn0
  %concatSMIn0 = VPUIP.Copy
      inputs(%distributedCastSM0 : !DistribCastSM0)
      outputs(%subviewSM0 : !ConcatSMIn0)  ->  !ConcatSMIn0

  %outBuff1 = VPURT.AllocDistributed -> !ConvOut1
  %outSM1 = VPURT.AllocDistributed -> !ConvSMOut1
  %conv1:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
      input(%input : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
      weights(%weights1 : memref<96x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
      parent_input(%input : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
      parent_output(%outBuff1 : !ConvOut1)
      parent_output_sparsity_map(%outSM1 : !ConvSMOut1)
      outputs(%outBuff1 : !ConvOut1)
      output_sparsity_map(%outSM1 : !ConvSMOut1)
  ->  !ConvOut1 ,  !ConvSMOut1 variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [13, 13, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 13, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 0 : i64, inEnd = [13, 13, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 13, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 0 : i64, inEnd = [13, 13, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 13, 95], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
  } PPE : {
  }

  %distributedCast1 = VPUIP.DistributedCast inputs(%conv1#0 : !ConvOut1) -> !DistribCast1

  %subview1 = VPUIP.SubView %concatBuff [0, 48, 0, 0] [1, 96, 14, 14] : !ConcatOut to !ConcatIn1
  %concatIn1 = VPUIP.Copy
      inputs(%distributedCast1 : !DistribCast1)
      outputs(%subview1 : !ConcatIn1)  ->  !ConcatIn1

  %distributedCastSM1 = VPUIP.DistributedCast inputs(%conv1#1 : !ConvSMOut1) -> !DistribCastSM1

  %subviewSM1 = VPUIP.SubView %concatSMBuff [0, 48, 0, 0] [1, 96, 14, 14] : !ConcatSMOut to !ConcatSMIn1
  %concatSMIn1 = VPUIP.Copy
      inputs(%distributedCastSM1 : !DistribCastSM1)
      outputs(%subviewSM1 : !ConcatSMIn1)  ->  !ConcatSMIn1
  %concat = VPUIP.ConcatView
      inputs(%concatIn0, %concatIn1 : !ConcatIn0, !ConcatIn1)
      outputs(%concatBuff : !ConcatOut) -> !ConcatOut
  %concatSM = VPUIP.ConcatView
      inputs(%concatSMIn0, %concatSMIn1 : !ConcatSMIn0, !ConcatSMIn1)
      outputs(%concatSMBuff : !ConcatSMOut) -> !ConcatSMOut

  return %concat, %concatSM : !ConcatOut, !ConcatSMOut

  // CHECK:       [[ALLOC_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x14x14xi1, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x14x14xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 48, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST0:%.+]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW0]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[SUBVIEW0_SM:%.+]] = VPUIP.SubView [[ALLOC_SM]] [0, 0, 0, 0] [1, 48, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x48x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST0_SM:%.+]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW0_SM]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x48x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[CONV0:%.+]]:2 = VPUIP.NCEClusterTask <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
  // CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}>
  // CHECK-SAME:     input([[INPUT]] : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:     weights([[WEIGHTS0]] : memref<48x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:     parent_input([[INPUT]] : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:     parent_output([[DCAST0]] : !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN
  // CHECK-SAME{LITERAL}:  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     parent_output_sparsity_map([[DCAST0_SM]] : !VPUIP.DistributedBuffer<1x48x14x14xi1
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     outputs([[DCAST0]] : !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN
  // CHECK-SAME{LITERAL}:  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     output_sparsity_map([[DCAST0_SM]] : !VPUIP.DistributedBuffer<1x48x14x14xi1
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)


  // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 48, 0, 0] [1, 96, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x96x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST1:%.+]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW1]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x96x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[SUBVIEW1_SM:%.+]] = VPUIP.SubView [[ALLOC_SM]] [0, 48, 0, 0] [1, 96, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x96x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST1_SM:%.+]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW1_SM]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x96x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[CONV1:%.+]]:2 = VPUIP.NCEClusterTask <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
  // CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}>
  // CHECK-SAME:     input([[INPUT]] : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:     weights([[WEIGHTS1]] : memref<96x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:     parent_input([[INPUT]] : memref<1x16x14x14xf16, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:     parent_output([[DCAST1]] : !VPUIP.DistributedBuffer<1x96x14x14xf16,
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]], memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     parent_output_sparsity_map([[DCAST1_SM]] : !VPUIP.DistributedBuffer<1x96x14x14xi1,
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]], memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     outputs([[DCAST1]] : !VPUIP.DistributedBuffer<1x96x14x14xf16,
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]], memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     output_sparsity_map([[DCAST1_SM]] : !VPUIP.DistributedBuffer<1x96x14x14xi1,
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]], memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)



  // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CONV0]]#0, [[CONV1]]#0
  // CHECK-SAME:    outputs([[ALLOC]] : !VPUIP.DistributedBuffer<1x144x14x14xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)

  // CHECK:       [[CONCAT_SM:%.+]] = VPUIP.ConcatView inputs([[CONV0]]#1, [[CONV1]]#1
  // CHECK-SAME:    outputs([[ALLOC_SM]] : !VPUIP.DistributedBuffer<1x144x14x14xi1, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)

  // CHECK: return [[CONCAT]], [[CONCAT_SM]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMX2CMXCopyWithSubview
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.+]]: memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[OUTPUT1:%.+]]: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
// CHECK-SAME:  [[OUTPUT2:%.+]]: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
func.func @CMX2CMXCopyWithSubview(
    %inData : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>,
    %inSparsityMap : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>,
    %inWeights : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>,
    %outDataConcatPart : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
    %outSparsityMapConcatPart : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    -> (memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>, memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>)
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>
    // Convolution
    %2:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%inData : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>)
        weights(%inWeights : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%inData : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>)
        parent_input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        parent_output_sparsity_map(%1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        output_sparsity_map(%1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        -> memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    // SubView for data
    %3 = memref.alloc() : memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 128, 14, 14] : memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>
        to memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // SubView for sparsity map
    %5 = memref.alloc() : memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 128, 14, 14] : memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>
        to memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CMX->CMX copy for data
    %7 = VPUIP.Copy
        inputs(%2#0: memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%4: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
        -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CMX->CMX copy for sparsity map
    %8 = VPUIP.Copy
        inputs(%2#1: memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%6: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
        -> memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // Concat for data
    %9 = VPUIP.ConcatView
        inputs(%7, %outDataConcatPart :
            memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
            memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
        )
        outputs(%3 : memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>

    // Concat for sparsity map
    %10 = VPUIP.ConcatView
        inputs(%8, %outSparsityMapConcatPart :
            memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
            memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
        )
        outputs(%5 : memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>)
        -> memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>

    return %9, %10 : memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>, memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK: [[SUBVIEW_DATA:%.+]] = VPUIP.SubView [[ALLOC0]]

    // CHECK: [[SUBVIEW_SM:%.+]] = VPUIP.SubView [[ALLOC1]]

    // CHECK: [[TASK_RES:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[INPUT1]]
    // CHECK-SAME:  input_sparsity_map([[INPUT2]]
    // CHECK-SAME:  weights([[INPUT3]]
    // CHECK-SAME:  outputs([[SUBVIEW_DATA]]
    // CHECK-SAME:  output_sparsity_map([[SUBVIEW_SM]]

    // CHECK: [[CONCAT_DATA:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#0, [[OUTPUT1]]
    // CHECK-SAME:  outputs([[ALLOC0]]
    // CHECK-SAME:  -> memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[CONCAT_SM:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#1, [[OUTPUT2]]
    // CHECK-SAME:  outputs([[ALLOC1]]
    // CHECK-SAME:  -> memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>

    // CHECK: return [[CONCAT_DATA]], [[CONCAT_SM]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutDataBufferType = !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN,
    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!OutSMBufferType = !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN,
    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

!PartialDataBufferType = !VPUIP.DistributedBuffer<1x128x14x14xf16,
    {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN,
    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!PartialSMBufferType = !VPUIP.DistributedBuffer<1x128x14x14xi1,
    {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN,
    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// CopyOp is distributed
// CHECK-LABEL: @CMX2CMXTilingCopyWithSubview
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.+]]: memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[OUTPUT1:%.+]]: !VPUIP.DistributedBuffer<1x128x14x14xf16
// CHECK-SAME:  [[OUTPUT2:%.+]]: !VPUIP.DistributedBuffer<1x128x14x14xi1
func.func @CMX2CMXTilingCopyWithSubview(
    %inData : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>,
    %inSparsityMap : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>,
    %inWeights : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>,
    %outDataConcatPart : !PartialDataBufferType,
    %outSparsityMapConcatPart : !PartialSMBufferType)
    -> (!OutDataBufferType, !OutSMBufferType)
{
    // alloc for Conv data out
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // alloc for Conv sparsity map out
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // Convolution
    %2:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%inData : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>)
        input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>)
        weights(%inWeights : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%inData : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>)
        parent_input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>)         parent_output(%0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output_sparsity_map(%1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        output_sparsity_map(%1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    ->  !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
        !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    // SubView for data
    %3 = VPURT.AllocDistributed -> !OutDataBufferType
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 128, 14, 14] : !OutDataBufferType
        to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // SubView for sparsity map
    %5 = VPURT.AllocDistributed -> !OutSMBufferType
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 128, 14, 14] : !OutSMBufferType
        to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CMX->CMX tiling copy for data
    %7 = VPUIP.Copy
        inputs(%2#0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%4 : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CMX->CMX tiling copy for sparsity map
    %8 = VPUIP.Copy
        inputs(%2#1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%6 : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // Concat for data
    %9 = VPUIP.ConcatView
        inputs(%7, %outDataConcatPart : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !PartialDataBufferType)
        outputs(%3 : !OutDataBufferType) -> !OutDataBufferType

    // Concat for sparsity map
    %10 = VPUIP.ConcatView
        inputs(%8, %outSparsityMapConcatPart : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !PartialSMBufferType)
        outputs(%5 : !OutSMBufferType) -> !OutSMBufferType

    return %9, %10 : !OutDataBufferType, !OutSMBufferType

    // CHECK: [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN,
    // CHECK: [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN,
    // CHECK: [[SUBVIEW_DATA:%.+]] = VPUIP.SubView [[ALLOC0]]

    // CHECK: [[SUBVIEW_SM:%.+]] = VPUIP.SubView [[ALLOC1]]

    // CHECK: [[TASK_RES:%.+]]:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2],
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME:     input([[INPUT1]] : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     input_sparsity_map([[INPUT2]] : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     weights([[INPUT3]] : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_input([[INPUT1]] : memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_input_sparsity_map([[INPUT2]] : memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_output([[SUBVIEW_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     parent_output_sparsity_map([[SUBVIEW_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     output_sparsity_map([[SUBVIEW_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK: [[CONCAT_DATA:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#0, [[OUTPUT1]]
    // CHECK-SAME:  outputs([[ALLOC0]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN,

    // CHECK: [[CONCAT_SM:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#1, [[OUTPUT2]]
    // CHECK-SAME:  outputs([[ALLOC1]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN,

    // CHECK: return [[CONCAT_DATA]], [[CONCAT_SM]]
}
