//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-rewriters="rewriter=optimize-copies-set" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @CopiesWithSubViewOps(%act : memref<1x80x28x28xf16, #NHWC, @DDR>)
                              -> memref<1x80x28x27xf16, #NHWC, @DDR>
{
    %buf0 = memref.alloc() : memref<1x70x28x27xf16, #NHWC, @DDR>
    %buf1 = memref.alloc() : memref<1x80x28x27xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %act [0, 0, 0, 1] [1, 70, 28, 27] : memref<1x80x28x28xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>
    %1 = VPUIP.Copy inputs(%0 : memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>) outputs(%buf0 : memref<1x70x28x27xf16, #NHWC, @DDR>) -> memref<1x70x28x27xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %buf1 [0, 0, 0, 0] [1, 70, 28, 27] : memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %3 = VPUIP.Copy inputs(%1 : memref<1x70x28x27xf16, #NHWC, @DDR>) outputs(%2 : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) -> memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %4 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 10, 28, 27] : memref<1x70x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>
    %5 = VPUIP.SubView %buf1 [0, 70, 0, 0] [1, 10, 28, 27] : memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>) outputs(%5 : memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) -> memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %7 = VPUIP.ConcatView inputs(%3, %6 : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) outputs(%buf1 : memref<1x80x28x27xf16, #NHWC, @DDR>) -> memref<1x80x28x27xf16, #NHWC, @DDR>
    return %7 : memref<1x80x28x27xf16, #NHWC, @DDR>

    // Copy %3 with parent Copy %1 will be tried to be optimized
    // do not optimize since Copy %1 is also used by SubView %4
    // and will not be removed.
    // implement a solution to optimize both E#35612

    // currently no changes after pass, desired outcome in E#35612

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x70x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x80x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 1] [1, 70, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x28xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>
    // CHECK:       [[VAR3:%.*]] = VPUIP.Copy inputs([[VAR2]] : memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>) outputs([[VAR0]] : memref<1x70x28x27xf16, #NHWC, @DDR>) ->
    // CHECK-SAME:      memref<1x70x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 70, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x70x28x27xf16, #NHWC, @DDR>) outputs([[VAR4]] : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) ->
    // CHECK-SAME:      memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR6:%.*]] = VPUIP.SubView [[VAR3]] [0, 0, 0, 0] [1, 10, 28, 27] :
    // CHECK-SAME:      memref<1x70x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>
    // CHECK:       [[VAR7:%.*]] = VPUIP.SubView [[VAR1]] [0, 70, 0, 0] [1, 10, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR8:%.*]] = VPUIP.Copy inputs([[VAR6]] : memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>) outputs([[VAR7]] : memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) ->
    // CHECK-SAME:      memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR9:%.*]] = VPUIP.ConcatView inputs([[VAR5]], [[VAR8]] : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x80x28x27xf16, #NHWC, @DDR>) -> memref<1x80x28x27xf16, #NHWC, @DDR>
    // CHECK:       return [[VAR9]] : memref<1x80x28x27xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @CopyWithSubViewOp(%in : memref<1x16x113x113xf16, #NHWC, @DDR>,
                        %in_sm : memref<1x16x113x113xi1, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
                        -> (memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>,
                            memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>) {

    %buf0 = memref.alloc() : memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    %buf0sm = memref.alloc() : memref<1x16x113x113xi1, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf1sm = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @CMX_NN>
    %buf2sm = memref.alloc() : memref<1x32x56x56xi1, #NHWC, @CMX_NN>

    // activation copy-in
    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x113x113xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x113x113xf16, #NHWC, @CMX_NN>

    %sm0 = VPUIP.Copy
            inputs(%in_sm : memref<1x16x113x113xi1, #NHWC, @DDR>)
            outputs(%buf0sm : memref<1x16x113x113xi1, #NHWC, @CMX_NN>)
             -> memref<1x16x113x113xi1, #NHWC, @CMX_NN>

    %1:2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%sm0 : memref<1x16x113x113xi1, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%buf1sm : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%buf1sm : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
        -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>, memref<1x16x56x56xi1, #NHWC, @CMX_NN>
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
        memref<1x32x56x56xf16, #NHWC, @CMX_NN> to
        memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    %sm2 = VPUIP.SubView %buf2sm [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56xi1, #NHWC, @CMX_NN> to
        memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // copy of the output NCE from NNCMX->NNCMX
    %3 = VPUIP.Copy
        inputs(%1#0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>)
        -> memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    %sm3 = VPUIP.Copy
        inputs(%1#1 : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
        outputs(%sm2 : memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>)
        -> memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    return %2, %sm2 :
        memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>,
        memref<1x16x56x56xi1, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // verify that the SubView operation is not removed along with the copy operation

    // CHECK:       [[VAL1SM:%.+]] = memref.alloc() : memref<1x32x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[VAL1:%.+]] = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAL0:%.+]] = memref.alloc() : memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAL0SM:%.+]] = memref.alloc() : memref<1x16x113x113xi1, #NHWC, @CMX_NN>

    // CHECK:       [[VAL2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs({{%.+}} : memref<1x16x113x113xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[VAL2SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs({{%.+}} : memref<1x16x113x113xi1, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[VAL0SM]] : memref<1x16x113x113xi1, #NHWC, @CMX_NN>)

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

func.func @DDR2DDRCopyOutput(%in : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                        -> memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
    // Input Tile 1
    %0 = memref.alloc() : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %1 = memref.alloc() : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %2 = VPUIP.Copy
        inputs(%in : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
        outputs(%0 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)  ->  memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 32, 64, 128] : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
            outputs(%3 : memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>)
                -> memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>

    // Input Tile 2
    %5 = memref.alloc() : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %6 = VPUIP.Copy
        inputs(%in : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
        outputs(%5 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)  ->  memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 32, 64, 128] : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>
    %8 = VPUIP.Copy inputs(%6 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
            outputs(%7 : memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>)
                    -> memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>

    // Concat
    %9 = VPUIP.ConcatView
        inputs(%4, %8 : memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>, memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>)
        outputs(%1 : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    return %9 : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

    // CHECK:       [[BUFF:%.*]] = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF]] [0, 0, 0, 0] [1, 32, 64, 128] :
    // CHECK-SAME:      memref<1x32x128x128xf16, #NHWC, @DDR> to memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>)  ->  memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>

    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF]] [0, 0, 64, 0] [1, 32, 64, 128] :
    // CHECK-SAME:      memref<1x32x128x128xf16, #NHWC, @DDR> to memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[SUBVIEW_2]] : memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>)  ->  memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_1]], [[COPY_2]] : memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>, memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFF]] : memref<1x32x128x128xf16, #NHWC, @DDR>) -> memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x32x128x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func.func @DDR2DDRCopyInput(%in : memref<1x144x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>,
                       %weights: memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                       %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
                        -> !OutputDistributed {
    %0 = VPUIP.SubView %in [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
                -> memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
        outputs(%3 : !OutputDistributed)  ->  !OutputDistributed
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%4 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
    ->  !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }
    return %6 : !OutputDistributed

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[NCE:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:     input([[COPY]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights(%arg1 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weight_table(%arg2 : memref<144x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:     parent_input([[COPY]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[BUFFER_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFFER_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       return [[NCE]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x48x192x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func.func @RemoveDDR2DDRCopyInputWithShapeCastOp(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
                        -> !OutputDistributed {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x144x64x128xf16, #NHWC, @DDR>
    %3 = VPUIP.ShapeCast {shape = [1, 48, 192, 128]} inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x48x192x128xf16, #NHWC, @DDR>
    %4 = VPURT.AllocDistributed -> !OutputDistributed
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x48x192x128xf16, #NHWC, @DDR>)
        outputs(%4 : !OutputDistributed)  ->  !OutputDistributed

    return %5 : !OutputDistributed

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 48, 192, 128]}
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:                        -> memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>
    // CHECK:       [[BUFFER:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST]] : memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER]] : !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[COPY]] : !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func.func @NotRemoveDDR2DDRCopyInputWithIllegalShapeCastOp(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
                        -> !OutputDistributed {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 128, 64]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x128x64xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x128x64xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x128x64xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<1x144x128x64xf16, #NHWC, @DDR>
    %3 = VPUIP.ShapeCast {shape = [1, 144, 64, 128]} inputs(%2 : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<1x144x64x128xf16, #NHWC, @DDR>
    %4 = VPURT.AllocDistributed -> !OutputDistributed
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%4 : !OutputDistributed)  ->  !OutputDistributed

    return %5 : !OutputDistributed

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 128, 64]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x128x64xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[DDRBUFFER:%.+]] = memref.alloc() : memref<1x144x128x64xf16, #NHWC, @DDR>
    // CHECK:       [[COPYTODDR:%.+]] =  VPUIP.Copy inputs([[SUBVIEW]] : memref<1x144x128x64xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK:                                       outputs([[DDRBUFFER]] : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<1x144x128x64xf16, #NHWC, @DDR>
    // CHECK:       [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 144, 64, 128]}
    // CHECK-SAME:      inputs([[COPYTODDR]] : memref<1x144x128x64xf16, #NHWC, @DDR>
    // CHECK-SAME:                          -> memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFFER:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST]] : memref<1x144x64x128xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[COPY]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    64x144x16x8xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x288x128x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func.func @NotRemoveDDR2DDRCopyInputWithOneIllegalShapeCastOp(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
                        -> (!OutputDistributed0, !OutputDistributed1) {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 128, 64]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x128x64xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x128x64xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x128x64xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<1x144x128x64xf16, #NHWC, @DDR>

    // Illegal ShapeCast cannot infer output strides with Subview output type
    %3 = VPUIP.ShapeCast {shape = [64, 144, 16, 8]} inputs(%2 : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<64x144x16x8xf16, #NHWC, @DDR>
    %4 = VPURT.AllocDistributed -> !OutputDistributed0
    %5 = VPUIP.Copy
        inputs(%3 : memref<64x144x16x8xf16, #NHWC, @DDR>)
        outputs(%4 : !OutputDistributed0)  ->  !OutputDistributed0

    // Legal ShapeCast can infer output strides with Subview output type
    %6 = VPUIP.ShapeCast {shape = [1, 288, 128, 32]} inputs(%2 : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<1x288x128x32xf16, #NHWC, @DDR>
    %7 = VPURT.AllocDistributed -> !OutputDistributed1
    %8 = VPUIP.Copy
        inputs(%6 : memref<1x288x128x32xf16, #NHWC, @DDR>)
        outputs(%7 : !OutputDistributed1)  ->  !OutputDistributed1

    return %5, %8 : !OutputDistributed0, !OutputDistributed1

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 128, 64]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x128x64xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[DDRBUFFER:%.+]] = memref.alloc() : memref<1x144x128x64xf16, #NHWC, @DDR>
    // CHECK:       [[COPYTODDR:%.+]] =  VPUIP.Copy inputs([[SUBVIEW]] : memref<1x144x128x64xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK:                                       outputs([[DDRBUFFER]] : memref<1x144x128x64xf16, #NHWC, @DDR>) -> memref<1x144x128x64xf16, #NHWC, @DDR>

    // CHECK:       [[SHAPECAST0:%.+]] = VPUIP.ShapeCast {shape = [64, 144, 16, 8]}
    // CHECK-SAME:      inputs([[COPYTODDR]] : memref<1x144x128x64xf16, #NHWC, @DDR>
    // CHECK-SAME:                          -> memref<64x144x16x8xf16, #NHWC, @DDR>
    // CHECK:       [[BUFFER0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<64x144x16x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST0]] : memref<64x144x16x8xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER0]] : !VPUIP.DistributedBuffer<64x144x16x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<64x144x16x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       [[SHAPECAST1:%.+]] = VPUIP.ShapeCast {shape = [1, 288, 128, 32]}
    // CHECK-SAME:      inputs([[COPYTODDR]] : memref<1x144x128x64xf16, #NHWC, @DDR>
    // CHECK-SAME:                          -> memref<1x288x128x32xf16, #NHWC, @DDR>
    // CHECK:       [[BUFFER1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x288x128x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST1]] : memref<1x288x128x32xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER1]] : !VPUIP.DistributedBuffer<1x288x128x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x288x128x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[COPY0]], [[COPY1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x48x192x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func.func @RemoveDDR2DDRCopyInputWithDiffUsers(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
                        -> (!OutputDistributed0, !OutputDistributed1) {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = VPUIP.ShapeCast {shape = [1, 48, 192, 128]} inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x48x192x128xf16, #NHWC, @DDR>
    %4 = VPURT.AllocDistributed -> !OutputDistributed0
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x48x192x128xf16, #NHWC, @DDR>)
        outputs(%4 : !OutputDistributed0)  ->  !OutputDistributed0

    %6 = VPURT.AllocDistributed -> !OutputDistributed1
    %7 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%6 : !OutputDistributed1)  ->  !OutputDistributed1

    return %5, %7 : !OutputDistributed0, !OutputDistributed1

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 48, 192, 128]}
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:                        -> memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>
    // CHECK:       [[BUFFER0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST]] : memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER0]] : !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x48x192x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[COPY0]], [[COPY1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
!qElemType1 = !quant.uniform<u8:f16, 7.1335556927849266:124>

func.func @ParallelDDR2DDRCopyOutput(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x64x48x176x!qElemType1, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        weights(%in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        parent_input(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    ->  memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>)  ->  memref<1x64x48x88x!qElemType1, #NHWC, @DDR>

    %4 = memref.alloc() : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 64, 48, 88] [1, 1, 1, 2] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
            to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %6 = VPUIP.Copy inputs(%3 : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>)
                    outputs(%5 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
                        -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %7 = VPUIP.SubView %4 [0, 0, 0, 1] [1, 64, 48, 88] [1, 1, 1, 2] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
            to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %8 = VPUIP.Copy inputs(%3 : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>)
                    outputs(%7 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
                        -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %9 = VPUIP.ConcatView
        inputs(%6, %8 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>, memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
        outputs(%4 : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>) -> memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
    return %9 : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    // CHECK:    [[ADD_0:%.*]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUFF_0]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK:     ->  memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN> variants : {
        // CHECK:     DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<MATRIX>, outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
        // CHECK:     PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }

    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 64, 48, 88] [1, 1, 1, 2] :
    // CHECK-SAME:      memref<1x64x48x176x!qElemType1, #NHWC, @DDR> to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)  ->  memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 1] [1, 64, 48, 88] [1, 1, 1, 2] :
    // CHECK-SAME:      memref<1x64x48x176x!qElemType1, #NHWC, @DDR> to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[SUBVIEW_2]] : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)  ->  memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_1]], [[COPY_2]] : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>, memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>) -> memref<1x64x48x176x!qElemType1, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ParallelDDR2DDRCopyOutputNoChangeToFixAccuracy(%in0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                                        %in1 : !VPUIP.DistributedBuffer<80x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                                        %in2 : !VPUIP.DistributedBuffer<80x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                                        %in3 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                                            -> memref<1x80x66x64xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 16540 : i64, task_type = #VPUIP.nce_task_type<DWCONV>}
        input(%in0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights(%in1 : !VPUIP.DistributedBuffer<80x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        weight_table(%in2 : !VPUIP.DistributedBuffer<80x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        parent_input(%in0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    ->  !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 31, 79], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 63, 63], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 63, 79], outStart = [0, 32, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = memref.alloc() : memref<1x80x64x64xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x80x64x64xf16, #NHWC, @DDR>)  ->  memref<1x80x64x64xf16, #NHWC, @DDR>
    %4 = VPUIP.SubView %3 [0, 0, 1, 0] [1, 80, 1, 64] : memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>
    %5 = VPUIP.SubView %3 [0, 0, 62, 0] [1, 80, 1, 64] : memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>
    %6 = memref.alloc() : memref<1x80x66x64xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 80, 1, 64] : memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %8 = VPUIP.Copy inputs(%4 : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>) outputs(%7 : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 0, 1, 0] [1, 80, 64, 64] : memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %10 = VPUIP.Copy inputs(%3 : memref<1x80x64x64xf16, #NHWC, @DDR>) outputs(%9 : memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) -> memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %11 = VPUIP.SubView %6 [0, 0, 65, 0] [1, 80, 1, 64] : memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %12 = VPUIP.Copy inputs(%5 : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>) outputs(%11 : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %13 = VPUIP.ConcatView
        inputs(%8, %10, %12 : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>, memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>, memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
        outputs(%6 : memref<1x80x66x64xf16, #NHWC, @DDR>) -> memref<1x80x66x64xf16, #NHWC, @DDR>

    return %13 : memref<1x80x66x64xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCETASK_0:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 16540 : i64, task_type = #VPUIP.nce_task_type<DWCONV>}
    // CHECK-SAME:          input(%arg0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:          weights(%arg1 : !VPUIP.DistributedBuffer<80x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weight_table(%arg2 : !VPUIP.DistributedBuffer<80x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          parent_input(%arg0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:          parent_output([[BUFF_0]] : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 31, 79], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                   DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 63, 63], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                   DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 63, 79], outStart = [0, 32, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE : {
    // CHECK:                   PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:           }

    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x80x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[NCETASK_0]] : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x80x64x64xf16, #NHWC, @DDR>)  ->  memref<1x80x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[COPY_1]] [0, 0, 1, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[COPY_1]] [0, 0, 62, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>

    // CHECK:       [[BUFF_2:%.*]] = memref.alloc() : memref<1x80x66x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_3:%.*]] = VPUIP.SubView [[BUFF_2]] [0, 0, 0, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_1]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_3]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:          -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>

    // CHECK:       [[SUBVIEW_4:%.*]] = VPUIP.SubView [[BUFF_2]] [0, 0, 1, 0] [1, 80, 64, 64] :
    // CHECK-SAME:      memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[COPY_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[COPY_1]] : memref<1x80x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_4]] : memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:          -> memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>

    // CHECK:       [[SUBVIEW_5:%.*]] = VPUIP.SubView [[BUFF_2]] [0, 0, 65, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[COPY_4:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_2]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_5]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:          -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_2]], [[COPY_3]], [[COPY_4]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>, memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>, memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFF_2]] : memref<1x80x66x64xf16, #NHWC, @DDR>) -> memref<1x80x66x64xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x80x66x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

func.func @DDR2DDRCopyOutputNOSubview(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x64x48x88x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        weights(%in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        parent_input(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    ->  memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %2 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)  ->  memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %4 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %5 = VPUIP.Copy inputs(%3 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
            outputs(%4 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %6 = VPUIP.ConcatView inputs(%5 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
            outputs(%4 : memref<1x64x48x88x!qElemType,#NHWC, @DDR>) -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    return %6 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[ADD_0:%.*]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[BUFF_0]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUFF_0]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK:     ->  memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants : {
    // CHECK:       DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<MATRIX>, outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:     } PPE : {
    // CHECK:       PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:     }
    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)  ->  memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
}

// -----

//  Optimize the left DDR2DDR copy in below case:
//                      ConcatView                                                     ConcatView
//                     /          \                                                  /          \
//          Copy(DDR2DDR)      SubView                                              |        SubView
//                  \             |                                                 |            |
//                   \        Copy(DDR2DDR)      =>                                 |       Copy(DDR2DDR)
//                    \        /                                                     \        /
//                        |                                                               |
//                        |                                                               |
//                    ConcatView                                                       ConcatView
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

func.func @DDR2DDROfConcatInput(%arg0: memref<1x512x18x32x!qElemType, #NHWC, @DDR>) -> memref<1x512x19x33x!qElemType, #NHWC, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    %1 = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 512, 18, 32] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>) outputs(%2 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %4 = VPUIP.SubView %1 [0, 0, 0, 32] [1, 512, 18, 1] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %5 = VPUIP.Copy inputs(%0 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>) outputs(%4 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %6 = VPUIP.ConcatView inputs(%3, %5 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%1 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    %7 = VPUIP.SubView %6 [0, 0, 17, 0] [1, 512, 1, 33] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %8 = memref.alloc() : memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 512, 18, 33] : memref<1x512x19x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %10 = VPUIP.Copy inputs(%6 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>) outputs(%9 : memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %11 = VPUIP.SubView %8 [0, 0, 18, 0] [1, 512, 1, 33] : memref<1x512x19x33x!qElemType, #NHWC, @DDR> to memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %12 = VPUIP.Copy inputs(%7 : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%11 : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %13 = VPUIP.ConcatView inputs(%10, %12 : memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) outputs(%8 : memref<1x512x19x33x!qElemType, #NHWC, @DDR>) -> memref<1x512x19x33x!qElemType, #NHWC, @DDR>
    return %13 : memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_FOR_COPY_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 33] :
    // CHECK-SAME:      memref<1x512x19x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 512, 18, 32] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_CONCAT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 32] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_SUBVIEW:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_FOR_COPY_0]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_FROM_CONCAT_0]], [[COPY_FROM_SUBVIEW]] :
    // CHECK-SAME:          memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] : memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_CONCAT_PARENT_0:%.*]] = VPUIP.SubView [[CONCAT_PARENT]] [0, 0, 17, 0] [1, 512, 1, 33] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_BUFF_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 18, 0] [1, 512, 1, 33] :
    // CHECK-SAME:      memref<1x512x19x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:       memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_CONCAT_PARENT_0]] : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_BUFF_0]] : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_CHILD:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CONCAT_PARENT]], [[COPY_RESULT]] :
    // CHECK-SAME:          memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x512x19x33x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_CHILD]] : memref<1x512x19x33x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDR2DDROfConcatWithConstInput(%arg0: memref<1x512x18x32xi8, #NHWC, @DDR>) -> memref<1x512x19x33xi8, #NHWC, @DDR> {
    %cst = const.Declare memref<1x512x1x33xi8, #NHWC> = dense<0> : tensor<1x512x1x33xi8, {order = #NHWC}>

    %0 = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] : memref<1x512x18x32xi8, #NHWC, @DDR> to memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    %1 = memref.alloc() : memref<1x512x18x33xi8, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 512, 18, 32] : memref<1x512x18x33xi8, #NHWC, @DDR> to memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x512x18x32xi8, #NHWC, @DDR>) outputs(%2 : memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %4 = VPUIP.SubView %1 [0, 0, 0, 32] [1, 512, 18, 1] : memref<1x512x18x33xi8, #NHWC, @DDR> to memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %5 = VPUIP.Copy inputs(%0 : memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>) outputs(%4 : memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %6 = VPUIP.ConcatView inputs(%3, %5 : memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%1 : memref<1x512x18x33xi8, #NHWC, @DDR>) -> memref<1x512x18x33xi8, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x512x19x33xi8, #NHWC, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 512, 18, 33] : memref<1x512x19x33xi8, #NHWC, @DDR> to memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %9 = VPUIP.Copy inputs(%6 : memref<1x512x18x33xi8, #NHWC, @DDR>) outputs(%8 : memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %10 = VPUIP.SubView %7 [0, 0, 18, 0] [1, 512, 1, 33] : memref<1x512x19x33xi8, #NHWC, @DDR> to memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %11 = VPUIP.Copy inputs(%cst : memref<1x512x1x33xi8, #NHWC>) outputs(%10 : memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) outputs(%7 : memref<1x512x19x33xi8, #NHWC, @DDR>) -> memref<1x512x19x33xi8, #NHWC, @DDR>
    return %12 : memref<1x512x19x33xi8, #NHWC, @DDR>

    // CHECK-DAG:       [[CONST:%.*]] = const.Declare memref<1x512x1x33xi8, #NHWC> = dense<0> : tensor<1x512x1x33xi8, {order = #NHWC}>

    // CHECK:       [[SUBVIEW_FOR_COPY_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x32xi8, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x19x33xi8, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 33] :
    // CHECK-SAME:      memref<1x512x19x33xi8, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 512, 18, 32] :
    // CHECK-SAME:      memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_CONCAT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x18x32xi8, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] : memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 32] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_SUBVIEW:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_FOR_COPY_0]] : memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_FROM_CONCAT_0]], [[COPY_FROM_SUBVIEW]] :
    // CHECK-SAME:          memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] : memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_BUFF_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 18, 0] [1, 512, 1, 33] :
    // CHECK-SAME:      memref<1x512x19x33xi8, #NHWC, @DDR> to
    // CHECK-SAME:       memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x512x1x33xi8, #NHWC>)
    // CHECK-SAME:      outputs([[SUBVIEW_BUFF_0]] : memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_CHILD:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CONCAT_PARENT]], [[COPY_RESULT]] :
    // CHECK-SAME:          memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x512x19x33xi8, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x19x33xi8, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_CHILD]] : memref<1x512x19x33xi8, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDR2DDROfConcatWithConstAndTilingCopyInput(%arg0: memref<1x1x128x200xf16, #NHWC, @DDR>) -> memref<1x1x130x202xf16, #NHWC, @DDR> {
    %cst0 = const.Declare memref<1x1x128x1xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x128x1xf16, {order = #NHWC}>
    %cst1 = const.Declare memref<1x1x128x1xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x128x1xf16, {order = #NHWC}>

    %0 = memref.alloc() : memref<1x1x128x202xf16, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 1] [1, 1, 128, 200] : memref<1x1x128x202xf16, #NHWC, @DDR>
            to memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%arg0 : memref<1x1x128x200xf16, #NHWC, @DDR>)
        outputs(%1 : memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)  ->  memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 128, 1] : memref<1x1x128x202xf16, #NHWC, @DDR>
            to memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%cst0 : memref<1x1x128x1xf16, #NHWC, @DDR>)
            outputs(%3 : memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>

    %5 = VPUIP.SubView %0 [0, 0, 0, 201] [1, 1, 128, 1] : memref<1x1x128x202xf16, #NHWC, @DDR>
            to memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%cst1 : memref<1x1x128x1xf16, #NHWC, @DDR>)
            outputs(%5 : memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %7 = VPUIP.ConcatView
        inputs(%2, %4, %6 : memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>, memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>, memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
        outputs(%0 : memref<1x1x128x202xf16, #NHWC, @DDR>) -> memref<1x1x128x202xf16, #NHWC, @DDR>

    %cst2 = const.Declare memref<1x1x1x202xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x1x202xf16, {order = #NHWC}>
    %cst3 = const.Declare memref<1x1x1x202xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x1x202xf16, {order = #NHWC}>

    %8 = memref.alloc() : memref<1x1x130x202xf16, #NHWC, @DDR>

    %9 = VPUIP.SubView %8 [0, 0, 1, 0] [1, 1, 128, 202] : memref<1x1x130x202xf16, #NHWC, @DDR>
            to memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %10 = VPUIP.Copy inputs(%7 : memref<1x1x128x202xf16, #NHWC, @DDR>)
            outputs(%9 : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    %11 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 1, 1, 202] : memref<1x1x130x202xf16, #NHWC, @DDR>
            to memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %12 = VPUIP.Copy inputs(%cst2  : memref<1x1x1x202xf16, #NHWC, @DDR>)
            outputs(%11 : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
                -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    %13 = VPUIP.SubView %8 [0, 0, 129, 0] [1, 1, 1, 202] : memref<1x1x130x202xf16, #NHWC, @DDR>
            to memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %14 = VPUIP.Copy inputs(%cst3  : memref<1x1x1x202xf16, #NHWC, @DDR>)
            outputs(%13 : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
                -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %15 = VPUIP.ConcatView
        inputs(%10, %12, %14 : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>, memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>, memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
        outputs(%8 : memref<1x1x130x202xf16, #NHWC, @DDR>) -> memref<1x1x130x202xf16, #NHWC, @DDR>

    return %15 : memref<1x1x130x202xf16, #NHWC, @DDR>

    // CHECK:       [[CONST:%.*]] = const.Declare memref<1x1x1x202xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x1x202xf16, {order = #NHWC}>

    // CHECK:       [[CONST_0:%.*]] = const.Declare memref<1x1x128x1xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x128x1xf16, {order = #NHWC}>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x1x130x202xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 1, 0] [1, 1, 128, 202] :
    // CHECK-SAME:      memref<1x1x130x202xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 1] [1, 1, 128, 200] :
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> to
    // CHECK-SAME:      memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x1x128x200xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_1]] : memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)  ->  memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 1, 128, 1] :
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> to
    // CHECK-SAME:      memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST_0]] : memref<1x1x128x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_0_3:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 201] [1, 1, 128, 1] :
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> to
    // CHECK-SAME:      memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST_0]] : memref<1x1x128x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_3]] : memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    // CHECK:    [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_1]], [[COPY_RESULT_0]], [[COPY_RESULT_1]] : memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>, memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>, memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>) -> memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_1_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 1, 1, 202] :
    // CHECK-SAME:      memref<1x1x130x202xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x1x1x202xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_1_2]] : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_1_3:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 129, 0] [1, 1, 1, 202] :
    // CHECK-SAME:      memref<1x1x130x202xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x1x1x202xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_1_3]] : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    // CHECK:    [[CONCAT_CHILD:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[CONCAT_PARENT]], [[COPY_RESULT_2]], [[COPY_RESULT_3]] : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>, memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>, memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFF_0]] : memref<1x1x130x202xf16, #NHWC, @DDR>) -> memref<1x1x130x202xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_CHILD]] : memref<1x1x130x202xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDR2DDROfConcatWithMemAllocOpInBlock(%arg0: memref<1x32x3x3xf16, #NHWC, @DDR>, %arg1 : memref<1x20x512x512xf16, #NHWC, @DDR>) -> (memref<1x32x512x512xf16, #NHWC, @DDR>, memref<9x32x3x3xf16, #NHWC, @DDR>) {
    %cst = const.Declare memref<6x32x3x3xf16, #NHWC> = dense<0.000000e+00> : tensor<3145728xf16>, [#const.SubView<[0], [1728]>, #const.Reshape<[6, 32, 3, 3]>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<1x12x512x512xf16, #NHWC> = dense<0.000000e+00> : tensor<3145728xf16>, [#const.Reshape<[1, 12, 512, 512]>, #const.Reorder<#NHWC>]

    %alloc_0 = memref.alloc() : memref<3x32x3x3xf16, #NHWC, @DDR>

    %0 = VPUIP.SubView %alloc_0 [0, 0, 0, 0] [1, 32, 3, 3] : memref<3x32x3x3xf16, #NHWC, @DDR> to memref<1x32x3x3xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x3x3xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x32x3x3xf16, #NHWC, @DDR>)  ->  memref<1x32x3x3xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %alloc_0 [1, 0, 0, 0] [1, 32, 3, 3] : memref<3x32x3x3xf16, #NHWC, @DDR> to memref<1x32x3x3xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x3x3xf16, #NHWC, @DDR>)
        outputs(%2 : memref<1x32x3x3xf16, #NHWC, @DDR>)  ->  memref<1x32x3x3xf16, #NHWC, @DDR>
    %4 = VPUIP.SubView %alloc_0 [2, 0, 0, 0] [1, 32, 3, 3] : memref<3x32x3x3xf16, #NHWC, @DDR> to memref<1x32x3x3xf16, #NHWC, @DDR>
    %5 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x3x3xf16, #NHWC, @DDR>)
        outputs(%4 : memref<1x32x3x3xf16, #NHWC, @DDR>)  ->  memref<1x32x3x3xf16, #NHWC, @DDR>
    %6 = VPUIP.ConcatView
        inputs(%1, %3, %5 : memref<1x32x3x3xf16, #NHWC, @DDR>, memref<1x32x3x3xf16, #NHWC, @DDR>, memref<1x32x3x3xf16, #NHWC, @DDR>)
        outputs(%alloc_0 : memref<3x32x3x3xf16, #NHWC, @DDR>) -> memref<3x32x3x3xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x32x512x512xf16, #NHWC, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 20, 512, 512] : memref<1x32x512x512xf16, #NHWC, @DDR> to memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>
    %9 = VPUIP.Copy inputs(%arg1 : memref<1x20x512x512xf16, #NHWC, @DDR>) outputs(%8 : memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>) -> memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>
    %10 = VPUIP.SubView %7 [0, 20, 0, 0] [1, 12, 512, 512] : memref<1x32x512x512xf16, #NHWC, @DDR> to memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>
    %11 = VPUIP.Copy inputs(%cst_0 : memref<1x12x512x512xf16, #NHWC>) outputs(%10 : memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>) -> memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>
    %12 = VPUIP.ConcatView
        inputs(%9, %11 : memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>, memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>)
        outputs(%7 : memref<1x32x512x512xf16, #NHWC, @DDR>) -> memref<1x32x512x512xf16, #NHWC, @DDR>

    %13 = memref.alloc() : memref<9x32x3x3xf16, #NHWC, @DDR>

    %14 = VPUIP.SubView %13 [0, 0, 0, 0] [3, 32, 3, 3] : memref<9x32x3x3xf16, #NHWC, @DDR> to memref<3x32x3x3xf16, #NHWC, @DDR>
    %15 = VPUIP.Copy inputs(%6 : memref<3x32x3x3xf16, #NHWC, @DDR>) outputs(%14 : memref<3x32x3x3xf16, #NHWC, @DDR>) -> memref<3x32x3x3xf16, #NHWC, @DDR>
    %16 = VPUIP.SubView %13 [10, 0, 0, 0] [6, 32, 3, 3] : memref<9x32x3x3xf16, #NHWC, @DDR> to memref<6x32x3x3xf16, #NHWC, @DDR>
    %17 = VPUIP.Copy inputs(%cst : memref<6x32x3x3xf16, #NHWC>) outputs(%16 : memref<6x32x3x3xf16, #NHWC, @DDR>) -> memref<6x32x3x3xf16, #NHWC, @DDR>
    %18 = VPUIP.ConcatView
        inputs(%15, %17 : memref<3x32x3x3xf16, #NHWC, @DDR>, memref<6x32x3x3xf16, #NHWC, @DDR>)
        outputs(%13 : memref<9x32x3x3xf16, #NHWC, @DDR>) -> memref<9x32x3x3xf16, #NHWC, @DDR>

    return %12, %18 : memref<1x32x512x512xf16, #NHWC, @DDR>, memref<9x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[CST:%.*]] = const.Declare memref<6x32x3x3xf16, #NHWC> = dense<0.000000e+00> :
    // CHECK-SAME:          tensor<3145728xf16>, [#const.SubView<[0], [1728]>, #const.Reshape<[6, 32, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_0:%.*]] = const.Declare memref<1x12x512x512xf16, #NHWC> = dense<0.000000e+00> :
    // CHECK-SAME:          tensor<3145728xf16>, [#const.Reshape<[1, 12, 512, 512]>, #const.Reorder<#NHWC>]

    // CHECK:       [[ALLOC_0:%.*]] = memref.alloc() : memref<9x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[VIEW_0:%.*]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 0] [3, 32, 3, 3] :
    // CHECK-SAME:          memref<9x32x3x3xf16, #NHWC, @DDR> to memref<3x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[VIEW_1:%.*]] = VPUIP.SubView [[VIEW_0]] [0, 0, 0, 0] [1, 32, 3, 3] : memref<3x32x3x3xf16, #NHWC, @DDR> to memref<1x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[TILE_COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[VIEW_1]] : memref<1x32x3x3xf16, #NHWC, @DDR>) -> memref<1x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[VIEW_2:%.*]] = VPUIP.SubView [[VIEW_0]] [1, 0, 0, 0] [1, 32, 3, 3] : memref<3x32x3x3xf16, #NHWC, @DDR> to memref<1x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[TILE_COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[VIEW_2]] : memref<1x32x3x3xf16, #NHWC, @DDR>)  -> memref<1x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[VIEW_3:%.*]] = VPUIP.SubView [[VIEW_0]] [2, 0, 0, 0] [1, 32, 3, 3] : memref<3x32x3x3xf16, #NHWC, @DDR> to memref<1x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[TILE_COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[VIEW_3]] : memref<1x32x3x3xf16, #NHWC, @DDR>)  -> memref<1x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILE_COPY_0]], [[TILE_COPY_1]], [[TILE_COPY_2]] : memref<1x32x3x3xf16, #NHWC, @DDR>, memref<1x32x3x3xf16, #NHWC, @DDR>, memref<1x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[VIEW_0]] : memref<3x32x3x3xf16, #NHWC, @DDR>) -> memref<3x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_1:%.*]] = memref.alloc() : memref<1x32x512x512xf16, #NHWC, @DDR>

    // CHECK:       [[VIEW_4:%.*]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 0, 0] [1, 20, 512, 512] :
    // CHECK-SAME:          memref<1x32x512x512xf16, #NHWC, @DDR> to memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>
    // CHECK:       [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg1 : memref<1x20x512x512xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[VIEW_4]] : memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>) -> memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>

    // CHECK:       [[VIEW_5:%.*]] = VPUIP.SubView [[ALLOC_1]] [0, 20, 0, 0] [1, 12, 512, 512] :
    // CHECK-SAME:          memref<1x32x512x512xf16, #NHWC, @DDR> to memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CST_0]] : memref<1x12x512x512xf16, #NHWC>)
    // CHECK-SAME:          outputs([[VIEW_5]] : memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>) -> memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>

    // CHECK:    [[CONCAT_0:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : memref<1x20x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>, memref<1x12x512x512xf16, {order = #NHWC, strides = [8388608, 1, 16384, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[ALLOC_1]] : memref<1x32x512x512xf16, #NHWC, @DDR>) -> memref<1x32x512x512xf16, #NHWC, @DDR>

    // CHECK:       [[VIEW_6:%.*]] = VPUIP.SubView [[ALLOC_0]] [10, 0, 0, 0] [6, 32, 3, 3] : memref<9x32x3x3xf16, #NHWC, @DDR> to memref<6x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CST]] : memref<6x32x3x3xf16, #NHWC>)
    // CHECK-SAME:          outputs([[VIEW_6]] : memref<6x32x3x3xf16, #NHWC, @DDR>) -> memref<6x32x3x3xf16, #NHWC, @DDR>
    // CHECK:    [[CONCAT_1:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[CONCAT]], [[COPY_3]] : memref<3x32x3x3xf16, #NHWC, @DDR>, memref<6x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[ALLOC_0]] : memref<9x32x3x3xf16, #NHWC, @DDR>) -> memref<9x32x3x3xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_0]], [[CONCAT_1]] : memref<1x32x512x512xf16, #NHWC, @DDR>, memref<9x32x3x3xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

func.func @DDR2DDROfConcatInputStrideCopy(%arg0: memref<1x512x9x32x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x33x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 512, 9, 32] [1, 1, 2, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>) outputs(%1 : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>) -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 512, 9, 32] [1, 1, 2, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>
    %4 = VPUIP.Copy inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>) outputs(%3 : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>) -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>) outputs(%0 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    %6 = VPUIP.SubView %5 [0, 0, 0, 31] [1, 512, 18, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    %7 = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 512, 18, 32] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %9 = VPUIP.Copy inputs(%5 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>) outputs(%8 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %10 = VPUIP.SubView %7 [0, 0, 0, 32] [1, 512, 18, 1] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %11 = VPUIP.Copy inputs(%6 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>) outputs(%10 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%7 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x33x!qElemType, #NHWC, @DDR>
    return %12 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[STRID_BUFF_1:%.*]] = memref.alloc() : memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_SB_10:%.*]] = VPUIP.SubView [[STRID_BUFF_1:%.*]] [0, 0, 0, 0] [1, 512, 9, 32] [1, 1, 2, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[COPY_SB_10:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_SB_10]] : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_SB_11:%.*]] = VPUIP.SubView [[STRID_BUFF_1:%.*]] [0, 0, 1, 0] [1, 512, 9, 32] [1, 1, 2, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[COPY_SB_11:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_SB_11]] : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[CONCAT_SB_1:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_SB_10]], [[COPY_SB_11]] :
    // CHECK-SAME:          memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[STRID_BUFF_1:%.*]] : memref<1x512x18x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_FOR_COPY_0:%.*]] = VPUIP.SubView [[CONCAT_SB_1]] [0, 0, 0, 31] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 32] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_CONCAT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONCAT_SB_1]] : memref<1x512x18x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 32] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_SUBVIEW:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_FOR_COPY_0]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_FROM_CONCAT_0]], [[COPY_FROM_SUBVIEW]] :
    // CHECK-SAME:          memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x512x18x33x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_PARENT]] : memref<1x512x18x33x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

func.func @ParallelDDR2DDRCopyOutputWithSlice(%in0 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x512x10x17x!qElemType, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @DDR>
    %2 = VPUIP.Copy inputs(%0 : !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
                           outputs(%1 : memref<1x512x9x17x!qElemType, #NHWC, @DDR>)
                               -> memref<1x512x9x17x!qElemType, #NHWC, @DDR>

    %3 = VPUIP.SubView %2 [0, 0, 8, 0] [1, 512, 1, 17] : memref<1x512x9x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>

    %4 = memref.alloc() : memref<1x512x1x17x!qElemType, #NHWC, @DDR>

    %5 = VPUIP.Copy inputs(%3 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>) outputs(%4 : memref<1x512x1x17x!qElemType, #NHWC, @DDR>)
    -> memref<1x512x1x17x!qElemType, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %8 = VPUIP.Copy inputs(%2 : memref<1x512x9x17x!qElemType, #NHWC, @DDR>) outputs(%7 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
    -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %9 = VPUIP.SubView %6 [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %10 = VPUIP.Copy inputs(%5 : memref<1x512x1x17x!qElemType, #NHWC, @DDR>) outputs(%9 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
    -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %11 = VPUIP.ConcatView inputs(%8, %10 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs(%6 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>

        return %11 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

        // CHECK:   [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        // CHECK:   [[BUFF_1:%.+]] = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
        // CHECK:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
        // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME: outputs([[SUBVIEW_0]] : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
        // CHECK:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
        // CHECK:   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 8, 0] [1, 512, 1, 17] : !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        // CHECK:   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : !VPUIP.DistributedBuffer<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME: outputs([[SUBVIEW_1]] : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
        // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs([[BUFF_1]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>
        // CHECK:   return [[CONCAT]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

func.func @ParallelDDR2DDRCopyOutputWithSubview(%in0 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x512x10x17x!qElemType, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @DDR>
    %2 = VPUIP.Copy inputs(%0 : !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
                           outputs(%1 : memref<1x512x9x17x!qElemType, #NHWC, @DDR>)
                               -> memref<1x512x9x17x!qElemType, #NHWC, @DDR>

    %3 = VPUIP.SubView %2 [0, 0, 8, 0] [1, 512, 1, 17] : memref<1x512x9x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>

    %4 = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %6 = VPUIP.Copy inputs(%2 : memref<1x512x9x17x!qElemType, #NHWC, @DDR>) outputs(%5 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
    -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %7 = VPUIP.SubView %4 [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %8 = VPUIP.Copy inputs(%3 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>) outputs(%7 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
    -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

    %9 = VPUIP.ConcatView inputs(%6, %8 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs(%4 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    return %9 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    // CHECK:   [[BUFF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   [[BUFF_1:%.*]] = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.Copy inputs([[BUFF_0]]
    // CHECK-SAME: outputs([[SUBVIEW_0]]
    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
    // CHECK:   [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 8, 0] [1, 512, 1, 17] : !VPUIP.DistributedBuffer<1x512x9x17x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs([[SUBVIEW_2]]
    // CHECK-SAME: outputs([[SUBVIEW_1]]
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs([[BUFF_1]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[CONCAT]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequence() -> !DuplicatedType {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.Copy
        inputs(%0 : !DuplicatedType)
        outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = VPURT.AllocDistributed -> !DuplicatedType

    // read to NN_CMX
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%3 : !DuplicatedType)  ->  !DuplicatedType

    return %4 : !DuplicatedType

    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.Copy
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       return [[BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!CompatibleType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceWithCastedInput() -> memref<1x144x64x128xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x144x64x128xf16, #NHWC>
    %1 = VPURT.AllocDistributed -> !DuplicatedType
    %2 = VPUIP.Copy
        inputs(%0 : memref<1x144x64x128xf16, #NHWC>)
        outputs(%1 : !DuplicatedType)  ->  !DuplicatedType
    %3 = VPUIP.DistributedCast inputs(%2 : !DuplicatedType) -> !CompatibleType
    %4 = memref.alloc() : memref<1x144x64x128xf16, #NHWC>
    %5 = VPUIP.Copy
        inputs(%3 : !CompatibleType)
        outputs(%4 : memref<1x144x64x128xf16, #NHWC>)  ->  memref<1x144x64x128xf16, #NHWC>

    return %5 : memref<1x144x64x128xf16, #NHWC>

    // CHECK:       [[BUFFER:%.*]] = memref.alloc()
    // CHECK-NOT:   VPUIP.Copy
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       return [[BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!CompatibleType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceWithCast() -> !CompatibleType {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.Copy
        inputs(%0 : !DuplicatedType)
        outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = VPURT.AllocDistributed -> !CompatibleType

    // read to NN_CMX
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%3 : !CompatibleType)  ->  !CompatibleType

    return %4 : !CompatibleType

    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.Copy
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[CAST:%.*]] = VPUIP.DistributedCast inputs([[BUFFER]]
    // CHECK:       return [[CAST]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!DuplicatedDDRType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @DDR, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceNoChange() -> !DuplicatedDDRType {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.Copy
        inputs(%0 : !DuplicatedType)
        outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = VPURT.AllocDistributed -> !DuplicatedDDRType

    // DDR to DDR
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%3 : !DuplicatedDDRType)  ->  !DuplicatedDDRType

    return %4 : !DuplicatedDDRType

    // CHECK:       VPURT.AllocDistributed
    // CHECK:       memref.alloc()
    // CHECK:       VPUIP.Copy
    // CHECK:       VPURT.AllocDistributed
    // CHECK:       [[COPY1:%.*]] = VPUIP.Copy
    // CHECK:       return [[COPY1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootBufferType = !VPUIP.DistributedBuffer<
    2x2560x8x8xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[2, 2560, 8, 8], [2, 2560, 8, 8], [2, 2560, 8, 8], [2, 2560, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[2, 2560, 8, 8], [2, 2560, 8, 8], [2, 2560, 8, 8], [2, 2560, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!SubviewBufferType = !VPUIP.DistributedBuffer<
    1x2560x8x8xf16, {order = #NHWC, strides = [163840, 1, 20480, 2560]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InputType = !VPUIP.DistributedBuffer<
    1x2560x8x8xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8], [1, 2560, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!WeightType = !VPUIP.DistributedBuffer<
    128x2560x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 2560, 3, 3], [32, 2560, 3, 3], [32, 2560, 3, 3], [32, 2560, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 2560, 3, 3], [32, 2560, 3, 3], [32, 2560, 3, 3], [32, 2560, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

!WeightTableType = !VPUIP.DistributedBuffer<
    128x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

!OutputType = !VPUIP.DistributedBuffer<
    1x128x8x8xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 8, 8], [1, 32, 8, 8], [1, 32, 8, 8], [1, 32, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 128, 8, 8], [1, 128, 8, 8], [1, 128, 8, 8], [1, 128, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

func.func @NCEClusterCopyOpSequenceWithSubviewInput() -> !OutputType {
    %0 = VPURT.AllocDistributed -> !RootBufferType
    %1 = VPUIP.SubView %0 [1, 0, 0, 0] [1, 2560, 8, 8] : !RootBufferType to !SubviewBufferType
    %2 = memref.alloc() : memref<1x2560x8x8xf16, #NHWC, @DDR>

    // spill to DDR
    %3 = VPUIP.Copy
        inputs(%1 : !SubviewBufferType)
        outputs(%2 : memref<1x2560x8x8xf16, #NHWC, @DDR>) -> memref<1x2560x8x8xf16, #NHWC, @DDR>

    %4 = VPURT.AllocDistributed -> !InputType
    // read to NN_CMX
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x2560x8x8xf16, #NHWC, @DDR>)
        outputs(%4 : !InputType) -> !InputType

    %6 = VPURT.AllocDistributed -> !WeightType
    %7 = VPURT.AllocDistributed -> !WeightTableType
    %8 = VPURT.AllocDistributed -> !OutputType
    %9 = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
        input(%5 : !InputType)
        weights(%6 : !WeightType)
        weight_table(%7 : !WeightTableType)
        parent_input(%5 : !InputType)
        parent_output(%8 : !OutputType)
        outputs(%8 : !OutputType)
    -> !OutputType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [7, 7, 2559], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 7, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [7, 7, 2559], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 7, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [7, 7, 2559], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 7, 95], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
        DPUTask {cluster_id = 3 : i64, inEnd = [7, 7, 2559], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [7, 7, 127], outStart = [0, 0, 96], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
    }

    return %9 : !OutputType


    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[CMX_TO_DDR:%.+]] = VPUIP.Copy
    // CHECK:       VPURT.AllocDistributed
    // CHECK:       [[DDR_TO_CMX:%.+]] = VPUIP.Copy inputs([[CMX_TO_DDR]] : memref<1x2560x8x8xf16, #NHWC, @DDR>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @FuseDDRToDDRCopyToTheFrontOfTillingCopy() -> memref<1x144x64x128xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.Copy
        inputs(%0 : !DuplicatedType)
        outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // DDR to DDR
    %4 = VPUIP.Copy
                inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
                outputs(%3 : memref<1x144x64x128xf16, #NHWC, @DDR>)
                    -> memref<1x144x64x128xf16, #NHWC, @DDR>

    return %4 : memref<1x144x64x128xf16, #NHWC, @DDR>

    // CHECK:       [[BUF0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF1:%.*]] = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[BUF0]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUF1]] : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY0]] : memref<1x144x64x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!NCEOutputDuplicatedType = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!SubviewOutputDuplicatedType = !VPUIP.DistributedBuffer<
    1x3x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @FuseDDRToCMXCopyToTheFrontOfTillingCopy() -> memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.AllocDistributed -> !NCEOutputDuplicatedType
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 32, 32] : !NCEOutputDuplicatedType to !SubviewOutputDuplicatedType

    %2 = memref.alloc() : memref<1x3x32x32xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !SubviewOutputDuplicatedType)
        outputs(%2 : memref<1x3x32x32xf16, #NHWC, @DDR>)  ->  memref<1x3x32x32xf16, #NHWC, @DDR>

    %4 = memref.alloc() : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>

    %5 = VPUIP.Copy
                inputs(%3 : memref<1x3x32x32xf16, #NHWC, @DDR>)
                outputs(%4 : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>)
                    -> memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>

    return %5 : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 3, 32, 32] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:                           to !VPUIP.DistributedBuffer<1x3x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[OUTPUT:%.+]] = memref.alloc() : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x3x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[OUTPUT]] : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>)  ->  memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       return [[COPY]] : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!NCEOutputDuplicatedType = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!SubviewOutputDuplicatedType = !VPUIP.DistributedBuffer<
    1x3x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 3, 16, 32], [1, 3, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 3, 32, 32], [1, 3, 32, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @FuseDDRToCMXCopyToTheFrontOfTillingCopyExplicitDistribution() -> memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.AllocDistributed -> !NCEOutputDuplicatedType
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 32, 32] : !NCEOutputDuplicatedType to !SubviewOutputDuplicatedType

    %2 = memref.alloc() : memref<1x3x32x32xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
        inputs(%1 : !SubviewOutputDuplicatedType)
        outputs(%2 : memref<1x3x32x32xf16, #NHWC, @DDR>)  ->  memref<1x3x32x32xf16, #NHWC, @DDR>

    %4 = memref.alloc() : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>

    %5 = VPUIP.Copy
                inputs(%3 : memref<1x3x32x32xf16, #NHWC, @DDR>)
                outputs(%4 : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>)
                    -> memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>

    return %5 : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 3, 32, 32] :
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x3x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 16, 32], [1, 3, 16, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 32, 32], [1, 3, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[OUTPUT:%.+]] = memref.alloc() : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x3x32x32xf16,
    // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 3, 16, 32], [1, 3, 16, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]], memory_shapes = [[1, 3, 32, 32], [1, 3, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:      outputs([[OUTPUT]] : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       return [[COPY]] : memref<1x3x32x32xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x12x12xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!InputStub_CMX = memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x3x12x12xf16, #NHWC, @DDR>

func.func @FuseCMXCopyToTheFrontOfTillingCopy() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.Copy
      inputs(%0 : !InputDistributedType)
      outputs(%1 : !SpilledOutput_DDR)  ->  !SpilledOutput_DDR

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX
  // CHECK:  [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x12x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
  // CHECK:  [[BUF_1:%.*]] = memref.alloc() : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy
  // CHECK-SAME:     inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x3x12x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
  // CHECK-SAME:     outputs([[BUF_1]] : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>)  ->  memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:  return [[COPY_0]] : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> memref<1x16x128x128xf16, #NHWC, @CMX_NN> {
  %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  %7 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%6 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    return %7 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>)
    // CHECK:                                outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>,
    // CHECK:       memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>)
    // CHECK:       outputs([[OUT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>)

    // CHECK:   return [[CONCAT]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithCopylastCopiesWithSubview(%arg0: memref<1x4x128x128xf16, #NHWC, @DDR>)
                                    -> memref<1x16x128x128xf16, #NHWC, @CMX_NN> {
  %cst = const.Declare memref<1x4x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<65536xf16>, [#const.Reshape<[1, 4, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x8x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 4, 128, 128] : memref<1x8x128x128xf16, #NHWC, @DDR>
        to memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x4x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>) -> memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 4, 0, 0] [1, 4, 128, 128] : memref<1x8x128x128xf16, #NHWC, @DDR>
        to memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x4x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>) -> memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>, memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>)
                        outputs(%0 : memref<1x8x128x128xf16, #NHWC, @DDR>) -> memref<1x8x128x128xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  %11 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1]: memref<1x16x128x128xf16, #NHWC, @CMX_NN>
        to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %7 = VPUIP.Copy inputs(%5 : memref<1x8x128x128xf16, #NHWC, @DDR>)
                  outputs(%11 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %12 = VPUIP.SubView %6 [0, 8, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1] : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
        to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %8 = VPUIP.Copy inputs(%5 : memref<1x8x128x128xf16, #NHWC, @DDR>)
                  outputs(%12 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %9 = VPUIP.ConcatView inputs(%7, %8 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>, memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>)
                        outputs(%6 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  return %9 :memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x4x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<65536xf16>, [#const.Reshape<[1, 4, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[CONCAT_BUFF:%.*]] = memref.alloc() : memref<1x8x128x128xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 0] [1, 4, 128, 128]
    // CHECK:       memref<1x8x128x128xf16, #NHWC, @DDR> to memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<1x4x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_0]] : memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>) -> memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 4, 0, 0] [1, 4, 128, 128]
    // CHECK:       memref<1x8x128x128xf16, #NHWC, @DDR> to memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs([[CST]] : memref<1x4x128x128xf16, #NHWC>)
    // CHECK:                                outputs([[SUBVIEW_1]] : memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>) -> memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>

    // CHECK:   [[CONCAT1:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>,
    // CHECK:       memref<1x4x128x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @DDR>)
    // CHECK:       outputs([[CONCAT_BUFF]] :  memref<1x8x128x128xf16, #NHWC, @DDR>)

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   [[SUBVIEW_3:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_3:%.*]] =  VPUIP.Copy inputs([[CONCAT1]] : memref<1x8x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_3]] : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_4:%.*]] = VPUIP.SubView [[OUT_BUFF]]  [0, 8, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_4:%.*]] =  VPUIP.Copy inputs([[CONCAT1]] : memref<1x8x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_4]] : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

    // CHECK:   [[CONCAT2:%.*]] = VPUIP.ConcatView inputs([[COPY_3]], [[COPY_4]]
    // CHECK:       memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>,
    // CHECK:       memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>)
    // CHECK:       outputs([[OUT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>)

    // CHECK:   return [[CONCAT2]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithMultiCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> (memref<1x16x128x128xf16, #NHWC, @CMX_NN>,memref<1x16x128x128xf16, #NHWC, @CMX_NN>) {
  %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  %7 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%6 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  %8 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  %9 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%8 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  return %7, %9 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>)
    // CHECK:                                outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>,
    // CHECK:       memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>)
    // CHECK:       outputs([[OUT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>)

    // CHECK:   return [[CONCAT]], [[CONCAT]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @SkipDDRToCMXCopyWithConcatViewInCaseNotDuplicatedCopyOutput
// CHECK-SAME:   ([[INPUT:%.+]]: memref<1x3x128x128xf16, #NHWC, @DDR>)
func.func @SkipDDRToCMXCopyWithConcatViewInCaseNotDuplicatedCopyOutput(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                                                       -> (memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>) {
    %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
    %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
                       to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                    outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                    -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
                       to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                    outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                    -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                          outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                          -> memref<1x16x128x128xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @CMX_NN>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 128, 128] : memref<1x32x128x128xf16, #NHWC, @CMX_NN>
                       to memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                    outputs(%7 : memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>)
                    -> memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>

    %9 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
    %10 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                     outputs(%9 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>)
                     -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    return %8, %10 : memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
    // CHECK:   [[CONCAT_BUFF:%.+]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK:                               outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    // CHECK:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>)
    // CHECK:                               outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>,
    // CHECK:       memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
    // CHECK:       outputs([[CONCAT_BUFF]] : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

    // CHECK:   [[OUT_BUFF_0:%.+]] = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @CMX_NN>
    // CHECK:   [[OUT0_SUBVIEW:%.+]] = VPUIP.SubView [[OUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 128, 128]
    // CHECK:       memref<1x32x128x128xf16, #NHWC, @CMX_NN> to memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>
    // CHECK:   [[OUT0_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x16x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                  outputs([[OUT0_SUBVIEW]] : memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>) -> memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>

    // CHECK:   [[OUT_BUFF_1:%.+]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
    // CHECK:   [[OUT1_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x16x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                  outputs([[OUT_BUFF_1]] : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   return [[OUT0_COPY]], [[OUT1_COPY]] : memref<1x16x128x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @DDRToCMXCopyWithConcatViewWithClusterCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
  %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %5 = VPUIP.ConcatView
      inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
      outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.Copy
      inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
      outputs(%6 : !OutputDistributed)  ->  !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed
    // CHECK:       -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  -> !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:       outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   return [[CONCAT]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    80x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @DDRToCMXCopyWithConcatViewNoChange(%arg0: memref<76x64x1x1xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
  %cst = const.Declare memref<4x64x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<22800xf16>, [#const.SubView<[0], [256]>, #const.Reshape<[4, 64, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<80x64x1x1xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [76, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR>
        to memref<76x64x1x1xf16, #NHWC, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<76x64x1x1xf16, #NHWC, @DDR>)
                  outputs(%1 : memref<76x64x1x1xf16, #NHWC, @DDR>) -> memref<76x64x1x1xf16, #NHWC, @DDR>

  %3 = VPUIP.SubView %0 [76, 0, 0, 0] [4, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR>
        to memref<4x64x1x1xf16, #NHWC, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<4x64x1x1xf16, #NHWC>)
                outputs(%3 : memref<4x64x1x1xf16, #NHWC, @DDR>) -> memref<4x64x1x1xf16, #NHWC, @DDR>
  %5 = VPUIP.ConcatView
      inputs(%2, %4 : memref<76x64x1x1xf16, #NHWC, @DDR>, memref<4x64x1x1xf16, #NHWC, @DDR>)
      outputs(%0 : memref<80x64x1x1xf16, #NHWC, @DDR>) -> memref<80x64x1x1xf16, #NHWC, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.Copy
      inputs(%5 : memref<80x64x1x1xf16, #NHWC, @DDR>)
      outputs(%6 : !OutputDistributed)  ->  !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<4x64x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<22800xf16>, [#const.SubView<[0], [256]>, #const.Reshape<[4, 64, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<80x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [76, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR> to memref<76x64x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<76x64x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_0]] : memref<76x64x1x1xf16, #NHWC, @DDR>) -> memref<76x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [76, 0, 0, 0] [4, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR> to memref<4x64x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs(%cst : memref<4x64x1x1xf16, #NHWC>) outputs([[SUBVIEW_1]] : memref<4x64x1x1xf16, #NHWC, @DDR>) -> memref<4x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<76x64x1x1xf16, #NHWC, @DDR>,
    // CHECK:       memref<4x64x1x1xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[OUT_BUFF]] : memref<80x64x1x1xf16, #NHWC, @DDR>)
    // CHECK:   [[CLUSTER_COPY:%.*]] = VPUIP.Copy
    // CHECK:   return [[CLUSTER_COPY]] : !VPUIP.DistributedBuffer<80x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x129x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @DDRToCMXCopyWithConcatViewWithNotBalancedClusterCopy(%arg0: memref<1x3x129x128xf16, #NCHW, @DDR>)
                                    -> !OutputDistributed {
  %cst = const.Declare memref<1x13x129x128xf16, #NCHW> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 129, 128]>]
  %0 = memref.alloc() : memref<1x16x129x128xf16, #NCHW, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 129, 128] : memref<1x16x129x128xf16, #NCHW, @DDR>
        to memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x129x128xf16, #NCHW, @DDR>)
                outputs(%1 : memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>) -> memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 129, 128] : memref<1x16x129x128xf16, #NCHW, @DDR>
        to memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x129x128xf16, #NCHW>)
                outputs(%3 : memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>) -> memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
  %5 = VPUIP.ConcatView
      inputs(%2, %4 : memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>, memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>)
      outputs(%0 : memref<1x16x129x128xf16, #NCHW, @DDR>) -> memref<1x16x129x128xf16, #NCHW, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.Copy
      inputs(%5 : memref<1x16x129x128xf16, #NCHW, @DDR>)
      outputs(%6 : !OutputDistributed)  ->  !OutputDistributed

  return %7 : !OutputDistributed

    // CHECK:    [[CST:%.*]] = const.Declare memref<1x13x129x128xf16> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 129, 128]>]
    // CHECK:    [[BUF_0:%.*]] = memref.alloc() : memref<1x16x129x128xf16, @DDR>
    // CHECK:    [[SUBIVEW_0:%.*]] = VPUIP.SubView [[BUF_0]] [0, 0, 0, 0] [1, 3, 129, 128] : memref<1x16x129x128xf16, @DDR> to memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x3x129x128xf16, @DDR>) outputs([[SUBVIEW_0]] : memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>) -> memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUF_0]] [0, 3, 0, 0] [1, 13, 129, 128] : memref<1x16x129x128xf16, @DDR> to memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy inputs([[CST]] : memref<1x13x129x128xf16>) outputs([[SUBVIEW_1]] : memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>) -> memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : memref<1x3x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>, memref<1x13x129x128xf16, {order = #NCHW, strides = [264192, 16512, 128, 1]}, @DDR>)
    // CHECK-SAME:     outputs(%alloc : memref<1x16x129x128xf16, @DDR>) -> memref<1x16x129x128xf16, @DDR>
    // CHECK:    [[BUF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x129x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCAT]] : memref<1x16x129x128xf16, @DDR>)
    // CHECK-SAME:     outputs([[BUF_1]] : !VPUIP.DistributedBuffer<1x16x129x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x16x129x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    return [[COPY_2]] : !VPUIP.DistributedBuffer<1x16x129x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithCopyStaticStrides(%arg0: memref<1x64x3x10xf16, #NHWC, @DDR>)
                                    -> memref<1x64x6x10xf16, #NHWC, @CMX_NN> {
  %0 = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1]: memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1]: memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %4 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%3 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>, memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>)
                        outputs(%0 : memref<1x64x6x10xf16, #NHWC, @DDR>) -> memref<1x64x6x10xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @CMX_NN>
  %7 = VPUIP.Copy inputs(%5 : memref<1x64x6x10xf16, #NHWC, @DDR>)
                  outputs(%6 : memref<1x64x6x10xf16, #NHWC, @CMX_NN>) -> memref<1x64x6x10xf16, #NHWC, @CMX_NN>

    return %7 : memref<1x64x6x10xf16, #NHWC, @CMX_NN>

    // CHECK: [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @CMX_NN>

    // CHECK: [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       memref<1x64x6x10xf16, #NHWC, @CMX_NN> to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>
    // CHECK: [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_0]] : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>

    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       memref<1x64x6x10xf16, #NHWC, @CMX_NN> to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>
    // CHECK: [[COPY_1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_1]] : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>

    // CHECK: [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>,
    // CHECK:       memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) outputs([[OUT_BUFF]] : memref<1x64x6x10xf16, #NHWC, @CMX_NN>) -> memref<1x64x6x10xf16, #NHWC, @CMX_NN>
    // CHECK: return [[CONCAT]] : memref<1x64x6x10xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x6x10xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @DDRToCMXCopyWithConcatViewWithClusterCopyStaticStrides(%arg0: memref<1x64x3x10xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
  %0 = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1] : memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1] : memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %4 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%3 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %5 = VPUIP.ConcatView
      inputs(%2, %4 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>, memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>)
      outputs(%0 : memref<1x64x6x10xf16, #NHWC, @DDR>) -> memref<1x64x6x10xf16, #NHWC, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.Copy
      inputs(%5 : memref<1x64x6x10xf16, #NHWC, @DDR>)
      outputs(%6 : !OutputDistributed)  ->  !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK: [[OUT_BUFF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: return [[CONCAT]] : !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    64x1504x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>


func.func @DDRToCMXCopyWithConcatViewPermuteCastWithClusterCopy(%arg0: memref<64x1500x1x1xf16, @DDR>)
                                    -> !OutputDistributed {
    %cst = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]
    %0 = memref.alloc() : memref<64x1504x1x1xf16, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [64, 1500, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %2 = VPUIP.Copy inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
                    outputs(%1 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 1500, 0, 0] [64, 4, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %4 = VPUIP.Copy inputs(%cst : memref<64x4x1x1xf16>)
                    outputs(%3 : memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>, memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>)
        outputs(%0 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1504x1x1xf16, @DDR>

    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%5 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1504x1x1xf16, #NHWC, @DDR>

    %7 = VPURT.AllocDistributed -> !OutputDistributed
    %8 = VPUIP.Copy
        inputs(%6 : memref<64x1504x1x1xf16, #NHWC, @DDR>)
        outputs(%7 : !OutputDistributed)  ->  !OutputDistributed

    return %8 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]

    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed
    // CHECK:       -> !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [64, 1500, 1, 1]
    // CHECK:       !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  -> !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 1500, 0, 0] [64, 4, 1, 1]
    // CHECK:       !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<64x4x1x1xf16>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  -> !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
    // CHECK:       outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT]] : !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   return [[PERMUTE_CAST]] : !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewPermuteCastWithCopy(%arg0: memref<64x1500x1x1xf16, @DDR>)
                                    -> memref<64x1504x1x1xf16, #NHWC, @CMX_NN> {
    %cst = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]
    %0 = memref.alloc() : memref<64x1504x1x1xf16, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [64, 1500, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %2 = VPUIP.Copy inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
                    outputs(%1 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 1500, 0, 0] [64, 4, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %4 = VPUIP.Copy inputs(%cst : memref<64x4x1x1xf16>)
                    outputs(%3 : memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>,
                                          memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>)
                          outputs(%0 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1504x1x1xf16, @DDR>

    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%5 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1504x1x1xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<64x1504x1x1xf16, #NHWC, @CMX_NN>

    %8 = VPUIP.Copy inputs(%6 : memref<64x1504x1x1xf16, #NHWC, @DDR>)
                    outputs(%7 : memref<64x1504x1x1xf16, #NHWC, @CMX_NN>) -> memref<64x1504x1x1xf16, #NHWC, @CMX_NN>

    return %8 : memref<64x1504x1x1xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<64x1504x1x1xf16, @CMX_NN>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [64, 1500, 1, 1]
    // CHECK:       memref<64x1504x1x1xf16, @CMX_NN>
    // CHECK:       memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
    // CHECK:       outputs(%0 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>)
    // CHECK:       -> memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 1500, 0, 0] [64, 4, 1, 1]
    // CHECK:       memref<64x1504x1x1xf16, @CMX_NN>
    // CHECK:       memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs(%cst : memref<64x4x1x1xf16>)
    // CHECK:       outputs(%2 : memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>)
    // CHECK:       -> memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:       memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:       outputs([[OUT_BUFF]] : memref<64x1504x1x1xf16, @CMX_NN>) -> memref<64x1504x1x1xf16, @CMX_NN>

    // CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT]] : memref<64x1504x1x1xf16, @CMX_NN>) -> memref<64x1504x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:   return [[PERMUTE_CAST]] : memref<64x1504x1x1xf16, #NHWC, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @NotConvertDDRToCMXCopyWithConcatViewPermuteCastWithCopyNoRootBuffer(%arg0: memref<72x1x1x1xf16, #NHWC, @DDR>)
                                    -> memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR> {
    %cst = const.Declare memref<8x1x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<8xf16>, [#const.Reshape<[8, 1, 1, 1]>, #const.Reorder<#NHWC>]
    %0 = memref.alloc() : memref<80x1x1x1xf16, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [72, 1, 1, 1] : memref<80x1x1x1xf16, #NHWC, @DDR> to memref<72x1x1x1xf16, #NHWC, @DDR>

    %2 = VPUIP.Copy inputs(%arg0 : memref<72x1x1x1xf16, #NHWC, @DDR>)
                    outputs(%1 : memref<72x1x1x1xf16, #NHWC, @DDR>) -> memref<72x1x1x1xf16, #NHWC, @DDR>

    %3 = VPUIP.SubView %0 [72, 0, 0, 0] [8, 1, 1, 1] : memref<80x1x1x1xf16, #NHWC, @DDR> to memref<8x1x1x1xf16, #NHWC, @DDR>

    %4 = VPUIP.Copy inputs(%cst : memref<8x1x1x1xf16, #NHWC>)
                    outputs(%3 : memref<8x1x1x1xf16, #NHWC, @DDR>) -> memref<8x1x1x1xf16, #NHWC, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<72x1x1x1xf16, #NHWC, @DDR>,
                                          memref<8x1x1x1xf16, #NHWC, @DDR>)
                          outputs(%0 : memref<80x1x1x1xf16, #NHWC, @DDR>) -> memref<80x1x1x1xf16, #NHWC, @DDR>

    %6 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%5 : memref<80x1x1x1xf16, #NHWC, @DDR>) -> memref<80x1x1x1xf16, @DDR>

    %7 = memref.alloc() : memref<80x16x1x1xf16, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [80, 1, 1, 1] : memref<80x16x1x1xf16, @DDR> to memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %9 = VPUIP.Copy inputs(%6 : memref<80x1x1x1xf16, @DDR>)
                    outputs(%8 : memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    return %9 : memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<8x1x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<8xf16>, [#const.Reshape<[8, 1, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK:   [[BUFF_0:%.*]] = memref.alloc() : memref<80x1x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [72, 1, 1, 1]
    // CHECK:       memref<80x1x1x1xf16, #NHWC, @DDR>
    // CHECK:       memref<72x1x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<72x1x1x1xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_0]] : memref<72x1x1x1xf16, #NHWC, @DDR>)
    // CHECK:       -> memref<72x1x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_0]] [72, 0, 0, 0] [8, 1, 1, 1]
    // CHECK:       memref<80x1x1x1xf16, #NHWC, @DDR>
    // CHECK:       memref<8x1x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs(%cst : memref<8x1x1x1xf16, #NHWC>)
    // CHECK:       outputs([[SUBVIEW_1]] : memref<8x1x1x1xf16, #NHWC, @DDR>)
    // CHECK:       -> memref<8x1x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<72x1x1x1xf16, #NHWC, @DDR>,
    // CHECK:       memref<8x1x1x1xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[BUFF_0]] : memref<80x1x1x1xf16, #NHWC, @DDR>) -> memref<80x1x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[CONCAT]] : memref<80x1x1x1xf16, #NHWC, @DDR>) -> memref<80x1x1x1xf16, @DDR>

    // CHECK:   [[BUFF_1:%.*]] = memref.alloc() : memref<80x16x1x1xf16, @DDR>
    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView %alloc_0 [0, 0, 0, 0] [80, 1, 1, 1]
    // CHECK:       memref<80x16x1x1xf16, @DDR>
    // CHECK:       memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    // CHECK:   [[COPY_OUT:%.*]] =  VPUIP.Copy inputs([[PERMUTE_CAST]] : memref<80x1x1x1xf16, @DDR>)
    // CHECK:       outputs(%6 : memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>)
    // CHECK:       -> memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    // CHECK:   return [[COPY_OUT]] : memref<80x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    64x1x1504x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>


func.func @DDRToCMXCopyWithConcatViewPermuteCastWithClusterCopy_ShapeChanged(%arg0: memref<64x1500x1x1xf16, @DDR>)
                                    -> !OutputDistributed {
    %cst = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]
    %0 = memref.alloc() : memref<64x1504x1x1xf16, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [64, 1500, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %2 = VPUIP.Copy inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
                    outputs(%1 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 1500, 0, 0] [64, 4, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %4 = VPUIP.Copy inputs(%cst : memref<64x4x1x1xf16>)
                    outputs(%3 : memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>, memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>)
        outputs(%0 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1504x1x1xf16, @DDR>

    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%5 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1x1504x1xf16, #NHWC, @DDR>

    %7 = VPURT.AllocDistributed -> !OutputDistributed
    %8 = VPUIP.Copy
        inputs(%6 : memref<64x1x1504x1xf16, #NHWC, @DDR>)
        outputs(%7 : !OutputDistributed)  ->  !OutputDistributed

    return %8 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]

    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed
    // CHECK:       -> !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [64, 1500, 1, 1]
    // CHECK:       !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  -> !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 1500, 0, 0] [64, 4, 1, 1]
    // CHECK:       !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<64x4x1x1xf16>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  -> !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       !VPUIP.DistributedBuffer<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
    // CHECK:       outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT]] : !VPUIP.DistributedBuffer<64x1504x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<64x1x1504x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:   return [[PERMUTE_CAST]] : !VPUIP.DistributedBuffer<64x1x1504x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewPermuteCastWithCopy_ShapeChanged(%arg0: memref<64x1500x1x1xf16, @DDR>)
                                    -> memref<64x1x1504x1xf16, #NHWC, @CMX_NN> {
    %cst = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]
    %0 = memref.alloc() : memref<64x1504x1x1xf16, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [64, 1500, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %2 = VPUIP.Copy inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
                    outputs(%1 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 1500, 0, 0] [64, 4, 1, 1] : memref<64x1504x1x1xf16, @DDR> to memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %4 = VPUIP.Copy inputs(%cst : memref<64x4x1x1xf16>)
                    outputs(%3 : memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>) -> memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>,
                                          memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @DDR>)
                          outputs(%0 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1504x1x1xf16, @DDR>

    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%5 : memref<64x1504x1x1xf16, @DDR>) -> memref<64x1x1504x1xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<64x1x1504x1xf16, #NHWC, @CMX_NN>

    %8 = VPUIP.Copy inputs(%6 : memref<64x1x1504x1xf16, #NHWC, @DDR>)
                    outputs(%7 : memref<64x1x1504x1xf16, #NHWC, @CMX_NN>) -> memref<64x1x1504x1xf16, #NHWC, @CMX_NN>

    return %8 : memref<64x1x1504x1xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<64x4x1x1xf16> = dense<0.000000e+00> : tensor<64x4x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<64x1504x1x1xf16, @CMX_NN>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [64, 1500, 1, 1]
    // CHECK:       memref<64x1504x1x1xf16, @CMX_NN>
    // CHECK:       memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<64x1500x1x1xf16, @DDR>)
    // CHECK:       outputs(%0 : memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>)
    // CHECK:       -> memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 1500, 0, 0] [64, 4, 1, 1]
    // CHECK:       memref<64x1504x1x1xf16, @CMX_NN>
    // CHECK:       memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs(%cst : memref<64x4x1x1xf16>)
    // CHECK:       outputs(%2 : memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>)
    // CHECK:       -> memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:       memref<64x4x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN>
    // CHECK:       outputs([[OUT_BUFF]] : memref<64x1504x1x1xf16, @CMX_NN>) -> memref<64x1504x1x1xf16, @CMX_NN>

    // CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT]] : memref<64x1504x1x1xf16, @CMX_NN>) -> memref<64x1x1504x1xf16, #NHWC, @CMX_NN>

    // CHECK:   return [[PERMUTE_CAST]] : memref<64x1x1504x1xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingTilingCopyOptimization(
        %in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>,
        %in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>,
        %in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 : memref<128x1x1x4xsi32, @CMX_NN>)
        -> (!VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, #NHWC, @DDR>, memref<1x128x36x36xi1, #NHWC, @DDR>) {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %9 = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @DDR>
    %sm9 = memref.alloc() : memref<1x128x36x36xi1, #NHWC, @DDR>
    %2:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>)
        weights(%in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%in2 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
        parent_input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>)
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
    %3:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>)
        weights(%in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%in2 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
        parent_input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>)
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
        outputs(%9 : memref<1x128x36x36xf16, #NHWC, @DDR>)  ->  memref<1x128x36x36xf16, #NHWC, @DDR>
    %sm11 = VPUIP.Copy
        inputs(%2#1 : !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm9 : memref<1x128x36x36xi1, #NHWC, @DDR>)  ->  memref<1x128x36x36xi1, #NHWC, @DDR>

    return %10, %sm10, %11, %sm11: !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, #NHWC, @DDR>, memref<1x128x36x36xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_1_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[BUFF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.*]] = memref.alloc() : memref<1x128x36x36xi1, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[SUBVIEW_0_SM:%.*]] = VPUIP.SubView [[BUFF_1_SM]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[NCETASK_0:%.*]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          outputs([[SUBVIEW_0]]
    // CHECK-SAME:          output_sparsity_map([[SUBVIEW_0_SM]]

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[SUBVIEW_1_SM:%.*]] = VPUIP.SubView [[BUFF_1_SM]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK:       [[NCETASK_1:%.*]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          outputs([[SUBVIEW_1]]
    // CHECK-SAME:          output_sparsity_map([[SUBVIEW_1_SM]]

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCETASK_0:%.*]]#0, [[NCETASK_1:%.*]]#0
    // CHECK-SAME:      outputs([[BUFF_1]]

    // CHECK:       [[CONCAT_SM:%.*]] = VPUIP.ConcatView inputs([[NCETASK_0:%.*]]#1, [[NCETASK_1:%.*]]#1
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[TILING:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCETASK_0]]#0
    // CHECK-SAME:      outputs([[BUFF_0]]

    // CHECK:       [[TILING_SM:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[NCETASK_0]]#1
    // CHECK-SAME:      outputs([[BUFF_0_SM]]

    // CHECK:       return [[CONCAT]], [[CONCAT_SM]], [[TILING]], [[TILING_SM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingTilingCopyOptimizationSameParent(
    %in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>,
    %in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>,
    %in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
    %in2 : memref<128x1x1x4xsi32, @CMX_NN>) ->
    (!VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {

    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>)
        weights(%in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%in2 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
        parent_input_sparsity_map(%in_sm0 : memref<1x256x36x36xi1, #NHWC, @CMX_NN>)
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

    // CHECK: [[CONCAT_OUT_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[CONCAT_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONCAT_OUT]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[SUBVIEW_0_SM:%.*]] = VPUIP.SubView [[CONCAT_OUT_SM]] [0, 0, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[CONV:%.*]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[SUBVIEW_0]]
    // CHECK-SAME:      output_sparsity_map([[SUBVIEW_0_SM]]

    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONCAT_OUT]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[SUBVIEW_1_SM:%.*]] = VPUIP.SubView [[CONCAT_OUT_SM]] [0, 128, 0, 0] [1, 128, 36, 36]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x36x36xi1, {order = #NHWC, strides = [331776, 1, 9216, 256]}

    // CHECK: [[SUBVIEW_1_COPY:%.*]] = VPUIP.Copy inputs([[CONV]]#0
    // CHECK-SAME:  outputs([[SUBVIEW_1]]

    // CHECK: [[SUBVIEW_1_COPY_SM:%.*]] = VPUIP.Copy inputs([[CONV]]#1
    // CHECK-SAME:  outputs([[SUBVIEW_1_SM]]

    // CHECK: [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[CONV]]#0, [[SUBVIEW_1_COPY]]
    // CHECK-SAME:  outputs([[CONCAT_OUT]]
    // CHECK: [[CONCAT_SM:%.*]] = VPUIP.ConcatView inputs([[CONV]]#1, [[SUBVIEW_1_COPY_SM]]
    // CHECK-SAME:  outputs([[CONCAT_OUT_SM]]

    // CHECK:  return [[CONCAT]], [[CONCAT_SM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedTypeAligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!DuplicatedTypeUnaligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceDUPAlign() -> !DuplicatedTypeUnaligned {
    %0 = VPURT.AllocDistributed -> !DuplicatedTypeAligned
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.Copy
        inputs(%0 : !DuplicatedTypeAligned)
        outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = VPURT.AllocDistributed -> !DuplicatedTypeUnaligned

    // read to NN_CMX
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%3 : !DuplicatedTypeUnaligned) -> !DuplicatedTypeUnaligned

    return %4 : !DuplicatedTypeUnaligned

    // Don't need to check the alignment when the tensor is not split
    // The Copy ops are optimized to DistributedCast
    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.Copy
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[CAST:%.*]] = VPUIP.DistributedCast
    // return [[CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!SegmentedTypeAligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!SegmentedTypeUnaligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceSEGAlign() -> !SegmentedTypeUnaligned {
    %0 = VPURT.AllocDistributed -> !SegmentedTypeAligned
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.Copy
        inputs(%0 : !SegmentedTypeAligned)
        outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>)  ->  memref<1x144x64x128xf16, #NHWC, @DDR>

    %3 = VPURT.AllocDistributed -> !SegmentedTypeUnaligned

    // read to NN_CMX
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
        outputs(%3 : !SegmentedTypeUnaligned) -> !SegmentedTypeUnaligned

    return %4 : !SegmentedTypeUnaligned

    // The alignments have to be compatible for split tensors
    // The Copies are not optimized because of incompatibility
    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[ALLOC:%.*]] = memref.alloc()
    // CHECK:       [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[BUFFER]]
    // CHECK-SAME:     outputs([[ALLOC]]
    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK:       [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[COPY]]
    // CHECK-SAME:     outputs([[BUFFER_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!WeightsType = memref<64x64x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<32x64x1x1xf16, #NHWC, @DDR>
!OutputStub_CMX = memref<32x64x1x1xf16, #NHWC, @CMX_NN>

func.func @MoveTilingCopyBeforeSubviewForSegmentedOnN(%arg0: !WeightsType) -> (!Output_DDR, !Output_DDR) {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>
    %weights1 = VPUIP.SubView %arg0 [32, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>

    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights0_copy = VPUIP.Copy
        inputs(%weights0 : memref<32x64x1x1xf16, #NHWC, @DDR>)
        outputs(%weights0_cmx : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    %weights1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights1_copy = VPUIP.Copy
        inputs(%weights1 : memref<32x64x1x1xf16, #NHWC, @DDR>)
        outputs(%weights1_cmx : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // simulate nce task 0
    %output0_buf  = memref.alloc() : !Output_DDR
    %output0 = VPUIP.Copy
        inputs(%weights0_copy : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        outputs(%output0_buf : !Output_DDR)  ->  !Output_DDR

    // simulate nce task 1
    %output1_buf  = memref.alloc() : !Output_DDR
    %output1 = VPUIP.Copy
        inputs(%weights1_copy : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        outputs(%output1_buf : !Output_DDR)  ->  !Output_DDR

    return %output0, %output1: !Output_DDR, !Output_DDR


    // CHECK:       [[WEIGHTS_BUF_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[WEIGHTS_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<64x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[WEIGHTS_BUF_CMX]] : !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[WEIGHTS_COPY]] [0, 0, 0, 0] [32, 64, 1, 1] : !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [8, 1, 1, 1]}>
    // CHECK:       [[CAST0:%.*]] = VPUIP.DistributedCast inputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [8, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView %1 [32, 0, 0, 0] [32, 64, 1, 1] : !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [8, 1, 1, 1]}>
    // CHECK:       [[CAST1:%.*]] = VPUIP.DistributedCast inputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [8, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[OUTBUF0:%.*]] = memref.alloc() : memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[COPY0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CAST0]] : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
    // CHECK-SAME:     outputs([[OUTBUF0]] : memref<32x64x1x1xf16, #NHWC, @DDR>)  ->  memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[OUTBUF1:%.*]] = memref.alloc() : memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[COPY1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CAST1]] : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
    // CHECK-SAME:     outputs([[OUTBUF1]] : memref<32x64x1x1xf16, #NHWC, @DDR>)  ->  memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY0]], [[COPY1]] : memref<32x64x1x1xf16, #NHWC, @DDR>, memref<32x64x1x1xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!WeightsType = memref<1568x32x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<784x32x1x1xf16, #NHWC, @DDR>
!OutputStub_CMX = memref<784x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsDistributedtype = !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1],
        num_clusters = 4 : i64,
        alignment = [16, 1, 1, 1],
        uniform_distributed_segments,
        compute_shapes = [[208, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [400, 0, 0, 0], [592, 0, 0, 0]],
        memory_shapes = [[208, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [400, 0, 0, 0], [592, 0, 0, 0]]}>

// CHECK-LABEL: @MoveTilingCopyBeforeSubviewWithExplicitShapesAndOffsetsOnN
// CHECK-SAME: ([[ARG0:%.+]]: memref<1568x32x1x1xf16, #NHWC, @DDR>)
func.func @MoveTilingCopyBeforeSubviewWithExplicitShapesAndOffsetsOnN(%arg0: !WeightsType) -> (!Output_DDR, !Output_DDR) {

    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [784, 32, 1, 1] : memref<1568x32x1x1xf16, #NHWC, @DDR> to memref<784x32x1x1xf16, #NHWC, @DDR>
    %weights1 = VPUIP.SubView %arg0 [784, 0, 0, 0] [784, 32, 1, 1] : memref<1568x32x1x1xf16, #NHWC, @DDR> to memref<784x32x1x1xf16, #NHWC, @DDR>

    %alloc0 = memref.alloc() : memref<784x32x1x1xf16, #NHWC, @DDR>
    %weights0_DDR = VPUIP.Copy inputs(%weights0 : memref<784x32x1x1xf16, #NHWC, @DDR>) outputs(%alloc0 : memref<784x32x1x1xf16, #NHWC, @DDR>) -> memref<784x32x1x1xf16, #NHWC, @DDR>

    %alloc1 = memref.alloc() : memref<784x32x1x1xf16, #NHWC, @DDR>
    %weights1_DDR = VPUIP.Copy inputs(%weights1 : memref<784x32x1x1xf16, #NHWC, @DDR>) outputs(%alloc1 : memref<784x32x1x1xf16, #NHWC, @DDR>) -> memref<784x32x1x1xf16, #NHWC, @DDR>

    %weights0_alloc_CMX = VPURT.AllocDistributed -> !WeightsDistributedtype
    %weights0_CMX = VPUIP.Copy
        inputs(%weights0_DDR : memref<784x32x1x1xf16, #NHWC, @DDR>)
        outputs(%weights0_alloc_CMX : !WeightsDistributedtype)  ->  !WeightsDistributedtype

    %weights1_alloc_CMX = VPURT.AllocDistributed -> !WeightsDistributedtype
    %weights1_CMX = VPUIP.Copy
        inputs(%weights1_DDR : memref<784x32x1x1xf16, #NHWC, @DDR>)
        outputs(%weights1_alloc_CMX : !WeightsDistributedtype)  ->  !WeightsDistributedtype

    // simulate nce task 0
    %output0_buf  = memref.alloc() : !Output_DDR
    %output0 = VPUIP.Copy
        inputs(%weights0_CMX : !WeightsDistributedtype)
        outputs(%output0_buf : !Output_DDR)  ->  !Output_DDR

    // simulate nce task 1
    %output1_buf  = memref.alloc() : !Output_DDR
    %output1 = VPUIP.Copy
        inputs(%weights1_CMX : !WeightsDistributedtype)
        outputs(%output1_buf : !Output_DDR)  ->  !Output_DDR

    return %output0, %output1: !Output_DDR, !Output_DDR

    // CHECK:       [[WEIGHTS_BUF_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1568x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[WEIGHTS_COPY:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1568x32x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[WEIGHTS_BUF_CMX]] : !VPUIP.DistributedBuffer<1568x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>)
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1568x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[WEIGHTS_COPY]] [0, 0, 0, 0] [784, 32, 1, 1] : !VPUIP.DistributedBuffer<1568x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [8, 1, 1, 1]}>
    // CHECK:       [[CAST0:%.*]] = VPUIP.DistributedCast inputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [8, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = {{\[\[}}208, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [208, 0, 0, 0], [400, 0, 0, 0], [592, 0, 0, 0]], memory_shapes = {{\[\[}}784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[WEIGHTS_COPY]] [784, 0, 0, 0] [784, 32, 1, 1] : !VPUIP.DistributedBuffer<1568x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [8, 1, 1, 1]}>
    // CHECK:       [[CAST1:%.*]] = VPUIP.DistributedCast inputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [8, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = {{\[\[}}208, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [208, 0, 0, 0], [400, 0, 0, 0], [592, 0, 0, 0]], memory_shapes = {{\[\[}}784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:       [[OUTBUF0:%.*]] = memref.alloc() : memref<784x32x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[COPY0:%.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[CAST0]] : !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "DUPLICATED|SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[208, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1]], compute_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [400, 0, 0, 0], [592, 0, 0, 0]], memory_shapes = [[784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:  outputs([[OUTBUF0]] : memref<784x32x1x1xf16, #NHWC, @DDR>) -> memref<784x32x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[OUTBUF1:%.*]] = memref.alloc() : memref<784x32x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[COPY1:%.*]] = VPUIP.Copy inputs([[CAST1]] : !VPUIP.DistributedBuffer<784x32x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "DUPLICATED|SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[208, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1], [192, 32, 1, 1]], compute_offsets = [[0, 0, 0, 0], [208, 0, 0, 0], [400, 0, 0, 0], [592, 0, 0, 0]], memory_shapes = [[784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1], [784, 32, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   outputs([[OUTBUF1]] : memref<784x32x1x1xf16, #NHWC, @DDR>) -> memref<784x32x1x1xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY0]], [[COPY1]] : memref<784x32x1x1xf16, #NHWC, @DDR>, memref<784x32x1x1xf16, #NHWC, @DDR>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
!WeightsType = memref<512x3584x1x1x!qElemType, #NHWC, @DDR>
!OutputDistributedType = !VPUIP.DistributedBuffer<1x176x43x4xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
        compute_shapes = [[1, 64, 43, 4], [1, 64, 43, 4], [1, 48, 43, 4]],
        compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0]],
        memory_shapes = [[1, 176, 43, 4], [1, 176, 43, 4], [1, 176, 43, 4]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!WeightsDistributedType = !VPUIP.DistributedBuffer<176x3584x1x1x!quant.uniform<i4:f16, 1.000000e+00>, #NHWC, @CMX_NN, {
        mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
        compute_shapes = [[64, 3584, 1, 1], [64, 3584, 1, 1], [48, 3584, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]],
        memory_shapes = [[64, 3584, 1, 1], [64, 3584, 1, 1], [48, 3584, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]]}>
!WTDistributedType = !VPUIP.DistributedBuffer<176x1x1x4xsi32, #NHWC, @CMX_NN, {
        mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
        compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [48, 1, 1, 4]],
        compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]],
        memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [48, 1, 1, 4]],
        memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]]}>
!InDistributedType = !VPUIP.DistributedBuffer<1x3584x43x4xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
        compute_shapes = [[1, 3584, 43, 4], [1, 3584, 43, 4], [1, 3584, 43, 4]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[1, 3584, 43, 4], [1, 3584, 43, 4], [1, 3584, 43, 4]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>


// CHECK-LABEL: @NotMoveTilingCopyBeforeSubviewIfExceedCMXSize
// CHECK-SAME: ([[ARG0:%.+]]: memref<512x3584x1x1x!qElemType, #NHWC, @DDR>)
func.func @NotMoveTilingCopyBeforeSubviewIfExceedCMXSize(%arg0: !WeightsType) -> (!OutputDistributedType, !OutputDistributedType) {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [176, 3584, 1, 1] : memref<512x3584x1x1x!qElemType, #NHWC, @DDR> to memref<176x3584x1x1x!qElemType, #NHWC, @DDR>

    %weights0_alloc_cmx = VPURT.AllocDistributed -> !WeightsDistributedType
    %weights0_copy_cmx = VPUIP.Copy inputs(%weights0 : memref<176x3584x1x1x!qElemType, #NHWC, @DDR>) outputs(%weights0_alloc_cmx : !WeightsDistributedType) -> !WeightsDistributedType
    %input0_alloc_cmx = VPURT.AllocDistributed -> !InDistributedType
    %wt0_alloc_cmx = VPURT.AllocDistributed -> !WTDistributedType
    %conv0_out_alloc_cmx = VPURT.AllocDistributed -> !OutputDistributedType

    %conv0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 51103 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
    input(%input0_alloc_cmx : !InDistributedType)
    weights(%weights0_copy_cmx : !WeightsDistributedType)
    weight_table(%wt0_alloc_cmx : !WTDistributedType)
    parent_input(%input0_alloc_cmx : !InDistributedType)
    parent_output(%conv0_out_alloc_cmx : !OutputDistributedType)
    outputs(%conv0_out_alloc_cmx : !OutputDistributedType) -> !OutputDistributedType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 42, 3583], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 42, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [3, 42, 3583], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 42, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [3, 42, 3583], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 42, 175], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    %weights1 = VPUIP.SubView %arg0 [176, 0, 0, 0] [176, 3584, 1, 1] : memref<512x3584x1x1x!qElemType, #NHWC, @DDR> to memref<176x3584x1x1x!qElemType, #NHWC, @DDR>

    %weights1_alloc_cmx = VPURT.AllocDistributed -> !WeightsDistributedType
    %weights1_copy_cmx = VPUIP.Copy inputs(%weights1 : memref<176x3584x1x1x!qElemType, #NHWC, @DDR>) outputs(%weights1_alloc_cmx : !WeightsDistributedType) -> !WeightsDistributedType
    %input1_alloc_cmx = VPURT.AllocDistributed -> !InDistributedType
    %wt1_alloc_cmx = VPURT.AllocDistributed -> !WTDistributedType
    %conv1_out_alloc_cmx = VPURT.AllocDistributed -> !OutputDistributedType

    %conv1 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 51103 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
    input(%input1_alloc_cmx : !InDistributedType)
    weights(%weights1_copy_cmx : !WeightsDistributedType)
    weight_table(%wt1_alloc_cmx : !WTDistributedType)
    parent_input(%input1_alloc_cmx : !InDistributedType)
    parent_output(%conv1_out_alloc_cmx : !OutputDistributedType)
    outputs(%conv1_out_alloc_cmx : !OutputDistributedType) -> !OutputDistributedType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 42, 3583], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 42, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [3, 42, 3583], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 42, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [3, 42, 3583], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 42, 175], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    return %conv0, %conv1 : !OutputDistributedType, !OutputDistributedType


    // CHECK:       [[WEIGHTS0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [176, 3584, 1, 1]
    // CHECK:       [[WEIGHTS0_CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[WEIGHTS0_CMX_COPY:%.+]] = VPUIP.Copy
    // CHECK:       [[INPUT0_CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[WT0_CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV0:%.+]] = VPUIP.NCEClusterTask

    // CHECK:       [[WEIGHTS1:%.+]] = VPUIP.SubView [[ARG0]] [176, 0, 0, 0] [176, 3584, 1, 1]
    // CHECK:       [[WEIGHTS1_CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[WEIGHTS1_CMX_COPY:%.+]] = VPUIP.Copy
    // CHECK:       [[INPUT1_CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[WT1_CMX_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV1:%.+]] = VPUIP.NCEClusterTask

    // CHECK:       return [[CONV0]], [[CONV1]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!WeightsType = memref<64x64x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<32x64x1x1xf16, #NHWC, @DDR>
!OutputStub_CMX = memref<32x64x1x1xf16, #NHWC, @CMX_NN>

func.func @NotMoveTilingCopyBeforeSubviewForSingleUser(%arg0: !WeightsType) -> !Output_DDR {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>

    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights0_copy = VPUIP.Copy
        inputs(%weights0 : memref<32x64x1x1xf16, #NHWC, @DDR>)
        outputs(%weights0_cmx : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // simulate nce task 0
    %output0_buf  = memref.alloc() : !Output_DDR
    %output0 = VPUIP.Copy
        inputs(%weights0_copy : !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        outputs(%output0_buf : !Output_DDR)  ->  !Output_DDR

    return %output0: !Output_DDR

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK-NOT:   [[CAST:%.*]]  VPUIP.DistributedCast
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    64x1504x1x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!SubOutputDistributed = !VPUIP.DistributedBuffer<
    64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @NotMoveTilingCopyBeforeSubviewForStridedOutput(%arg0: memref<512x1500x1x1xf16, @DDR>) -> (!SubOutputDistributed, !SubOutputDistributed) {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [64, 1500, 1, 1] : memref<512x1500x1x1xf16, @DDR> to memref<64x1500x1x1xf16, @DDR>
    %1 = VPUIP.SubView %arg0 [64, 0, 0, 0] [64, 1500, 1, 1] : memref<512x1500x1x1xf16, @DDR> to memref<64x1500x1x1xf16, @DDR>

    %2 = VPURT.AllocDistributed -> !OutputDistributed
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [64, 1500, 1, 1] : !OutputDistributed to !SubOutputDistributed
    %4 = VPUIP.Copy
        inputs(%0 : memref<64x1500x1x1xf16, @DDR>)
        outputs(%3 : !SubOutputDistributed)  ->  !SubOutputDistributed

    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [64, 1500, 1, 1] : !OutputDistributed to !SubOutputDistributed
    %7 = VPUIP.Copy
        inputs(%1 : memref<64x1500x1x1xf16, @DDR>)
        outputs(%6 : !SubOutputDistributed)  ->  !SubOutputDistributed

    return %4, %7: !SubOutputDistributed, !SubOutputDistributed


    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [64, 1500, 1, 1] : memref<512x1500x1x1xf16, @DDR> to memref<64x1500x1x1xf16, @DDR>
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [64, 0, 0, 0] [64, 1500, 1, 1] : memref<512x1500x1x1xf16, @DDR> to memref<64x1500x1x1xf16, @DDR>

    // CHECK:       [[BUFFER0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:                                         -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1504x1x1xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]
    // CHECK-SAME:                                                                    }>
    // CHECK:       [[SUB_BUFFER0:%.*]] = VPUIP.SubView [[BUFFER0]] [0, 0, 0, 0] [64, 1500, 1, 1] :
    // CHECK-SAME:                                            !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1504x1x1xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}> to
    // CHECK-SAME:                                            !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW0]] : memref<64x1500x1x1xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUB_BUFFER0]] : !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:       [[BUFFER1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:                                         -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1504x1x1xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}>
    // CHECK:       [[SUB_BUFFER1:%.*]] = VPUIP.SubView [[BUFFER1]] [0, 0, 0, 0] [64, 1500, 1, 1] :
    // CHECK-SAME:                                            !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1504x1x1xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}> to
    // CHECK-SAME:                                            !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW1]] : memref<64x1500x1x1xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUB_BUFFER1]] : !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
    // CHECK-SAME:     !VPUIP.DistributedBuffer<64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:       return [[COPY0]], [[COPY1]] :
    // CHECK-SAME:                                            !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}>,
    // CHECK-SAME:                                            !VPUIP.DistributedBuffer<
    // CHECK-SAME:                                                                    64x1500x1x1xf16, {order = #NCHW, strides = [1504, 1, 1, 1]}, @CMX_NN, {
    // CHECK-SAME:                                                                    mode = "SEGMENTED",
    // CHECK-SAME:                                                                    num_tiles = [2, 1, 1, 1],
    // CHECK-SAME:                                                                    num_clusters = 2 : i64,
    // CHECK-SAME:                                                                    alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x256x1500x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>
!WeightsType = memref<8192x256x1x1xf16, #NHWC, @DDR>
!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x512x1500x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @NotMoveTilingCopyBeforeSubviewForNonSuitableCMXRequirements(%arg0: !WeightsType) -> (!OutputDistributedType, !OutputDistributedType) {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [512, 256, 1, 1] : !WeightsType to memref<512x256x1x1xf16, #NHWC, @DDR>
    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights0_copy = VPUIP.Copy
        inputs(%weights0 : memref<512x256x1x1xf16, #NHWC, @DDR>)
        outputs(%weights0_cmx : !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    %input0 = VPURT.AllocDistributed -> !InputDistributedType
    %weights_table0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %output0_cmx = VPURT.AllocDistributed -> !OutputDistributedType
    // user nce task 0
    %nce_0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 102752 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%input0 : !InputDistributedType)
        weights(%weights0_copy : !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        weight_table(%weights_table0 : !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        parent_input(%input0 : !InputDistributedType)
        parent_output(%output0_cmx : !OutputDistributedType)
        outputs(%output0_cmx : !OutputDistributedType)
    ->  !OutputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 1499, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 1499, 511], outStart = [0, 0, 256], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %weights1 = VPUIP.SubView %arg0 [512, 0, 0, 0] [512, 256, 1, 1] : !WeightsType to memref<512x256x1x1xf16, #NHWC, @DDR>
    %weights1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights1_copy = VPUIP.Copy
        inputs(%weights1 : memref<512x256x1x1xf16, #NHWC, @DDR>)
        outputs(%weights0_cmx : !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %input1 = VPURT.AllocDistributed -> !InputDistributedType
    %weights_table1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %output1_cmx = VPURT.AllocDistributed -> !OutputDistributedType
    // user nce task 0
    %nce_1 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 102752 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%input1 : !InputDistributedType)
        weights(%weights1_copy : !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        weight_table(%weights_table1 : !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        parent_input(%input1 : !InputDistributedType)
        parent_output(%output1_cmx : !OutputDistributedType)
        outputs(%output1_cmx : !OutputDistributedType)
    ->  !OutputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 1499, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 1499, 511], outStart = [0, 0, 256], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %nce_0, %nce_1: !OutputDistributedType, !OutputDistributedType

    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [512, 256, 1, 1] : memref<8192x256x1x1xf16, #NHWC, @DDR> to memref<512x256x1x1xf16, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[NCE_0:%.*]] = VPUIP.NCEClusterTask
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [512, 0, 0, 0] [512, 256, 1, 1] : memref<8192x256x1x1xf16, #NHWC, @DDR> to memref<512x256x1x1xf16, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[NCE_1:%.*]] = VPUIP.NCEClusterTask
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 16, 66, 128], [1, 16, 67, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 61, 0]]
}>

func.func @NoOptimizeDDRToCMXCopyWithOverlappedModeClusterCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
    %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
    %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

    %6 = VPURT.AllocDistributed -> !OutputDistributed
    %7 = VPUIP.Copy
        inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
        outputs(%6 : !OutputDistributed)  ->  !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
    // CHECK:   [[CONCAT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>) outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>) outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
    // CHECK-SAME:     outputs([[CONCAT_BUFF]] : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 66, 128], [1, 16, 67, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 61, 0]]
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCAT]] : memref<1x16x128x128xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 66, 128], [1, 16, 67, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 61, 0]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 66, 128], [1, 16, 67, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 61, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 2.0:123>


func.func @DDR2DDRCopyMultiInputsWithDifferentType(%in : memref<1x144x128x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>,
                       %arg1: memref<1x144x64x128x!qElemType1, #NHWC>,
                       %weights: memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                       %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
                        -> (!OutputDistributed,  memref<1x144x64x128x!qElemType1, #NHWC, @DDR>) {
    %0 = VPUIP.SubView %in [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x144x64x128x!qElemType, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
                -> memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %4 = VPUIP.QuantizeCast
        inputs(%2 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
        -> memref<1x144x64x128x!qElemType1, #NHWC, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %7 = VPUIP.Copy
        inputs(%2 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
        outputs(%3 : !OutputDistributed)  ->  !OutputDistributed
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%7 : !OutputDistributed)
        weights(%weights : memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
        weight_table(%weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
        parent_input(%7 : !OutputDistributed)
        parent_output(%5 : !OutputDistributed)
        outputs(%5 : !OutputDistributed)
    ->  !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    return %6 , %4 : !OutputDistributed, memref<1x144x64x128x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128x!qElemType, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[COPY_BUFF:%.*]] = memref.alloc() : memref<1x144x64x128x!qElemType, #NHWC, @DDR>
    // CHECK:       [[DDRCOPY:%.*]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x144x64x128x!qElemType, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[COPY_BUFF]] : memref<1x144x64x128x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x144x64x128x!qElemType, #NHWC, @DDR>

    // CHECK:       [[QC:%.*]] = VPUIP.QuantizeCast inputs(
    // CHECK-SAME:      [[DDRCOPY]] : memref<1x144x64x128x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x144x64x128x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[DDRCOPY]] : memref<1x144x64x128x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUFFER_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[NCE:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:     input([[COPY]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights(%arg2 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weight_table(%arg3 : memref<144x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:     parent_input([[COPY]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[BUFFER_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFFER_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       return [[NCE]], [[QC]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, memref<1x144x64x128x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeDDR2CMXCopies
func.func @OptimizeDDR2CMXCopies(%arg0: memref<1x32x56x56xf16, #NHWC, @DDR>)
    -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {
                mode = "DUPLICATED",
                num_clusters = 2 : i64,
                alignment = [1, 16, 1, 1]
            }> {
    %ALLOC_DDR = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @DDR>
    %DDR_TO_DDR = VPUIP.Copy
        inputs(%arg0: memref<1x32x56x56xf16, #NHWC, @DDR>)
        outputs(%ALLOC_DDR : memref<1x32x56x56xf16, #NHWC, @DDR>)
            -> memref<1x32x56x56xf16, #NHWC, @DDR>

    %ALLOC_CMX = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 2 : i64,
        alignment = [1, 16, 1, 1]
    }>
    %DDR_TO_CMX = VPUIP.Copy
        inputs(%DDR_TO_DDR : memref<1x32x56x56xf16, #NHWC, @DDR>)
        outputs(%ALLOC_CMX : !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    return %DDR_TO_CMX : !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 2 : i64,
        alignment = [1, 16, 1, 1]
    }>

    // CHECK:   ([[FUNC_ARG:%.*]]: memref<1x32x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN

    // CHECK:   [[DDR_TO_CMX:%.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[FUNC_ARG]] : memref<1x32x56x56xf16, #NHWC, @DDR>)

    // CHECK:   return [[DDR_TO_CMX]] : !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @removeDistributedOpCMXToCMXCopyForHighDimInputStrideCopy
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x256x26x26xf16, @CMX_NN>
func.func @removeDistributedOpCMXToCMXCopyForHighDimInputStrideCopy(%in0 : memref<1x256x26x26xf16, @CMX_NN>)
                                                                    -> (memref<1x255x26x26xf16, @DDR>) {
    %0 = VPUIP.SubView %in0 [0, 0, 0, 0] [1, 255, 26, 26] :
             memref<1x256x26x26xf16, #NCHW, @CMX_NN> to
             memref<1x255x26x26xf16, {order = #NCHW, strides = [173056, 676, 26, 1]}, @CMX_NN>

    // copy of the output from NNCMX->NNCMX
    %1 = memref.alloc() : memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>
    %2 = VPUIP.Copy
        inputs(%0 : memref<1x255x26x26xf16, {order = #NCHW, strides = [173056, 676, 26, 1]}, @CMX_NN>)
        outputs(%1 : memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>)  ->  memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>

    %3 = memref.alloc() : memref<1x255x26x26xf16, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>)
                    outputs(%3 : memref<1x255x26x26xf16, @DDR>)
                        -> memref<1x255x26x26xf16, @DDR>

    return %4: memref<1x255x26x26xf16, @DDR>

    //CHECK:      [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]
    //CHECK:      [[COPY_BUFF:%.+]] = memref.alloc() : memref<1x255x26x26xf16, @DDR>
    //CHECK:      [[COPY:%.+]] = VPUIP.Copy
    //CHECK:      return [[COPY]] : memref<1x255x26x26xf16, @DDR>
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
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x14x14xf16, #NHWC, @CMX_NN>
// CHECK-SAME:  [[WEIGHTS0:%.+]]: memref<48x16x1x1xf16, #NHWC, @CMX_NN>
// CHECK-SAME:  [[WEIGHTSTABLE0:%.+]]: memref<48x1x1x4xsi32, @CMX_NN>
// CHECK-SAME:  [[WEIGHTS1:%.+]]: memref<96x16x1x1xf16, #NHWC, @CMX_NN>
// CHECK-SAME:  [[WEIGHTSTABLE1:%.+]]: memref<96x1x1x4xsi32, @CMX_NN>
func.func @CMX2CMXCopyOptimizationWithDuplicatedExplicitDistributedAttr(
  %input: memref<1x16x14x14xf16, #NHWC, @CMX_NN>, %weights0: memref<48x16x1x1xf16, #NHWC, @CMX_NN>, %weightsTable0: memref<48x1x1x4xsi32, @CMX_NN>,
  %weights1: memref<96x16x1x1xf16, #NHWC, @CMX_NN>, %weightsTable1: memref<96x1x1x4xsi32, @CMX_NN>)
      -> (!ConcatOut, !ConcatSMOut) {

  %concatBuff = VPURT.AllocDistributed -> !ConcatOut
  %concatSMBuff = VPURT.AllocDistributed -> !ConcatSMOut

  %outBuff0 = VPURT.AllocDistributed -> !ConvOut0
  %outSM0 = VPURT.AllocDistributed -> !ConvSMOut0
  %conv0:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
      input(%input : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
      weights(%weights0 : memref<48x16x1x1xf16, #NHWC, @CMX_NN>)
      weight_table(%weightsTable0 : memref<48x1x1x4xsi32, @CMX_NN>)
      parent_input(%input : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
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
  %conv1:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
      input(%input : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
      weights(%weights1 : memref<96x16x1x1xf16, #NHWC, @CMX_NN>)
      weight_table(%weightsTable1 : memref<96x1x1x4xsi32, @CMX_NN>)
      parent_input(%input : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
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

  // CHECK:       [[ALLOC_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x14x14xi1, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[ALLOC:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x14x14xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 144, 14, 14], [1, 144, 14, 14], [1, 144, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 48, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST0:%.*]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW0]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[SUBVIEW0_SM:%.*]] = VPUIP.SubView [[ALLOC_SM]] [0, 0, 0, 0] [1, 48, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x48x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST0_SM:%.*]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW0_SM]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x48x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[CONV0:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
  // CHECK-SAME:     input([[INPUT]] : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME:     weights([[WEIGHTS0]] : memref<48x16x1x1xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME:     weight_table([[WEIGHTSTABLE0:%.+]] : memref<48x1x1x4xsi32, @CMX_NN>)
  // CHECK-SAME:     parent_input([[INPUT]] : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME:     parent_output([[DCAST0]] : !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN
  // CHECK-SAME{LITERAL}:  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     parent_output_sparsity_map([[DCAST0_SM]] : !VPUIP.DistributedBuffer<1x48x14x14xi1
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     outputs([[DCAST0]] : !VPUIP.DistributedBuffer<1x48x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN
  // CHECK-SAME{LITERAL}:  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:     output_sparsity_map([[DCAST0_SM]] : !VPUIP.DistributedBuffer<1x48x14x14xi1
  // CHECK-SAME{LITERAL}:  {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 16, 14, 14], [1, 16, 14, 14], [1, 16, 14, 14]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]], memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)


  // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[ALLOC]] [0, 48, 0, 0] [1, 96, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x96x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST1:%.*]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW1]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x96x14x14xf16, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[SUBVIEW1_SM:%.*]] = VPUIP.SubView [[ALLOC_SM]] [0, 48, 0, 0] [1, 96, 14, 14]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x96x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[DCAST1_SM:%.*]] = VPUIP.DistributedCast
  // CHECK-SAME:        inputs([[SUBVIEW1_SM]]
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x96x14x14xi1, {order = #NHWC, strides = [28224, 1, 2016, 144]}, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
  // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 14, 14], [1, 96, 14, 14], [1, 96, 14, 14]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[CONV1:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
  // CHECK-SAME:     input([[INPUT]] : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME:     weights([[WEIGHTS1]] : memref<96x16x1x1xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME:     weight_table([[WEIGHTSTABLE1]] : memref<96x1x1x4xsi32, @CMX_NN>)
  // CHECK-SAME:     parent_input([[INPUT]] : memref<1x16x14x14xf16, #NHWC, @CMX_NN>)
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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!inputDistributed = !VPUIP.DistributedBuffer<
    1x256x26x26xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!outputDistributed = !VPUIP.DistributedBuffer<
    1x255x26x26xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [173056, 676, 26, 1]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

module @VPU.SW {
    func.func private @builtin_RegionYolo(%input : memref<*xf16, @CMX_NN>, %output : memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "single_shave_region_yolo.cpp", VPU.kernel_entry = "single_shave_region_yolo"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @notRemoveDistributedOpCMXToCMXCopyForIncompatibleType
// CHECK-SAME:    [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x256x26x26xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
func.func @notRemoveDistributedOpCMXToCMXCopyForIncompatibleType(%in0 : !inputDistributed)
                                                                 -> (memref<1x255x26x26xf16, [@CMX_NN, 0]>) {
    %0 = VPUIP.SubView %in0 [0, 0, 0, 0] [1, 255, 26, 26] : !inputDistributed to !outputDistributed

    // copy of the output from NNCMX->NNCMX
    %1 = memref.alloc() : memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>
    %2 = VPUIP.Copy
        inputs(%0 : !outputDistributed)
        outputs(%1 : memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>)  ->  memref<1x255x26x26xf16, #NCHW, [@CMX_NN, 0]>

    %3 = memref.alloc() : memref<1x255x26x26xf16, [@CMX_NN, 0]>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RegionYolo
                          inputs(%2 as %arg3: memref<1x255x26x26xf16, [@CMX_NN, 0]>)
                          outputs(%3 as %arg4: memref<1x255x26x26xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x255x26x26xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, 80, 6, false, 3, [0, 1, 2, 0, 0, 0, 0, 0, 0], 1, 3]}(%arg3, %arg4) : memref<1x255x26x26xf16, [@CMX_NN, 0]>, memref<1x255x26x26xf16, [@CMX_NN, 0]>
    }

    return %4: memref<1x255x26x26xf16, [@CMX_NN, 0]>

    //CHECK:      [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]
    //CHECK:      [[COPY_BUFF1:%.+]] = memref.alloc() : memref<1x255x26x26xf16, [@CMX_NN, 0]>
    //CHECK:      [[TILINGCOPY:%.+]] = VPUIP.Copy
    //CHECK:      [[COPY_BUFF2:%.+]] = memref.alloc() : memref<1x255x26x26xf16, [@CMX_NN, 0]>
    //CHECK:      [[SWKERNEL:%.+]] = VPUIP.SW.Kernel
    //CHECK:      return [[SWKERNEL]] : memref<1x255x26x26xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyOpSequenceWithInPlaceEltwiseUser
// CHECK-SAME:    [[INPUT_0:%.+]]: memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>, [[INPUT_1:%.+]]: memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>
func.func @CopyOpSequenceWithInPlaceEltwiseUser(%arg0: memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>,
                                                %arg1: memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>)
                                                -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]> {
    // First CopyOp sequence
    %0 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.ConcatView
              inputs(%arg0, %arg0 : memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>, memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>)
              outputs(%0 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    %2 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, @DDR>
    %3 = VPUIP.Copy
              inputs(%1 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%2 : memref<1x48x64x96xf16, #NHWC, @DDR>)
              -> memref<1x48x64x96xf16, #NHWC, @DDR>

    %4 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPUIP.Copy
              inputs(%3 : memref<1x48x64x96xf16, #NHWC, @DDR>)
              outputs(%4 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    // Second CopyOp sequence
    %6 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPUIP.ConcatView
              inputs(%arg1, %arg1 : memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>, memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>)
              outputs(%6 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    %8 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, @DDR>
    %9 = VPUIP.Copy
              inputs(%7 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%8 : memref<1x48x64x96xf16, #NHWC, @DDR>)
              -> memref<1x48x64x96xf16, #NHWC, @DDR>

    %10 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    %11 = VPUIP.Copy
              inputs(%9 : memref<1x48x64x96xf16, #NHWC, @DDR>)
              outputs(%10 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    // Eltwise AddOp with two inputs of CopyOp sequence
    %12 = VPUIP.NCEClusterTask
              {
                  is_inplace = true,
                  task_type = #VPUIP.nce_task_type<ELTWISE>
              }
              input(%5 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              weights(%11 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              parent_input(%5 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              parent_output(%10 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%10 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
              variants :
              {
                  DPUTask
                      {
                          inEnd = [95, 63, 47], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                          outEnd = [95, 63, 47], outStart = [0, 0, 0],
                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                      }
              }
              PPE :
              {
                  PPETask
                      {
                          ppe = #VPU.PPEStub<>
                      }
              }

    return %12 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[COPY_BUFF1:%.+]] = memref.alloc()
    // CHECK:       [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[INPUT_0]], [[INPUT_0]]
    // CHECK-SAME:          memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>,
    // CHECK-SAME:          memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[COPY_BUFF1]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[COPY_BUFF2:%.+]] = memref.alloc()
    // CHECK:       [[CONCAT2:%.+]] = VPUIP.ConcatView inputs([[INPUT_1]], [[INPUT_1]]
    // CHECK-SAME:          memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>,
    // CHECK-SAME:          memref<1x48x32x96xf16, {order = #NHWC, strides = [294912, 1, 4608, 48]}, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[COPY_BUFF2]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[ADD:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[CONCAT1]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[CONCAT2]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[CONCAT1]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[COPY_BUFF2]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[COPY_BUFF2]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]> variants : {
    // CHECK:               DPUTask {inEnd = [95, 63, 47], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [95, 63, 47], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:       }
    // CHECK:       return [[ADD]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!CompatibleTypeAct = !VPUIP.DistributedBuffer<
    1x64x128x88xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CompatibleTypeWeights = !VPUIP.DistributedBuffer<
    32x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!CompatibleTypeTable = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!CompatibleTypeConv = !VPUIP.DistributedBuffer<
    1x32x128x88xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CompatibleTypeOutput = !VPUIP.DistributedBuffer<
    1x32x64x176xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @NCEClusterCopyOpSequenceInPlaceEltwiseUserWithTypeIncompatible
func.func @NCEClusterCopyOpSequenceInPlaceEltwiseUserWithTypeIncompatible() -> !CompatibleTypeOutput {
    // Eltwise Add Input 1
    %0 = VPURT.AllocDistributed -> !CompatibleTypeAct
    %1 = VPURT.AllocDistributed -> !CompatibleTypeWeights
    %2 = VPURT.AllocDistributed -> !CompatibleTypeTable
    %3 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    %4 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
        input(%0 : !CompatibleTypeAct)
        weights(%1 : !CompatibleTypeWeights)
        weight_table(%2 : !CompatibleTypeTable)
        parent_input(%0 : !CompatibleTypeAct)
        parent_output(%3 : !CompatibleTypeOutput)
        outputs(%3 : !CompatibleTypeOutput)
    ->  !CompatibleTypeOutput variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [87, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [87, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %5 = VPUIP.ShapeCast {shape = [1, 32, 64, 176]} inputs(%4 : !CompatibleTypeOutput) -> !CompatibleTypeOutput

    %6 = memref.alloc() : memref<1x32x64x176xf16, #NHWC, @DDR>
    // spill to DDR
    %7 = VPUIP.Copy
        inputs(%5 : !CompatibleTypeOutput)
        outputs(%6 : memref<1x32x64x176xf16, #NHWC, @DDR>)  ->  memref<1x32x64x176xf16, #NHWC, @DDR>

    %8 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    // read to NN_CMX
    %9 = VPUIP.Copy
        inputs(%7 : memref<1x32x64x176xf16, #NHWC, @DDR>)
        outputs(%8 : !CompatibleTypeOutput)  ->  !CompatibleTypeOutput

    // Eltwise Add Input 2
    %10 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    %11 = memref.alloc() : memref<1x32x64x176xf16, #NHWC, @DDR>
    // spill to DDR
    %12 = VPUIP.Copy
        inputs(%10 : !CompatibleTypeOutput)
        outputs(%11 : memref<1x32x64x176xf16, #NHWC, @DDR>)  ->  memref<1x32x64x176xf16, #NHWC, @DDR>

    %13 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    // read to NN_CMX
    %14 = VPUIP.Copy
        inputs(%12 : memref<1x32x64x176xf16, #NHWC, @DDR>)
        outputs(%13 : !CompatibleTypeOutput)  ->  !CompatibleTypeOutput

    // Eltwise with is_inplace = true
    %15 = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%9 : !CompatibleTypeOutput)
        weights(%14 : !CompatibleTypeOutput)
        parent_input(%9 : !CompatibleTypeOutput)
        parent_output(%8 : !CompatibleTypeOutput)
        outputs(%8 : !CompatibleTypeOutput)
    ->  !CompatibleTypeOutput variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %15 : !CompatibleTypeOutput

    // CHECK:       [[CONV_ACT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_WEIGHTS:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_TABLE:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_OUTPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK:       [[COPY_BUFF1:%.+]] = memref.alloc()
    // CHECK:       [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK:       [[COPY_BUFF2:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK:       [[ELTWISE_INPUT2:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[ELTWISE:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:         input([[TILING_COPY2]]
    // CHECK-SAME:         weights([[ELTWISE_INPUT2]]
    // CHECK-SAME:         parent_input([[TILING_COPY2]]
    // CHECK-SAME:         parent_output([[COPY_BUFF2]]
    // CHECK-SAME:         outputs([[COPY_BUFF2]]
    // CHECK:             DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:             DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:         } PPE : {
    // CHECK:             PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:         }
    // CHECK:       return [[ELTWISE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!CompatibleTypeAct = !VPUIP.DistributedBuffer<
    1x64x64x176xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CompatibleTypeWeights = !VPUIP.DistributedBuffer<
    32x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!CompatibleTypeTable = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!CompatibleTypeOutput = !VPUIP.DistributedBuffer<
    1x32x64x176xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @NCEClusterCopyOpSequenceInPlaceEltwiseUserWithSameType
func.func @NCEClusterCopyOpSequenceInPlaceEltwiseUserWithSameType() -> !CompatibleTypeOutput {
    %0 = VPURT.AllocDistributed -> !CompatibleTypeAct
    %1 = VPURT.AllocDistributed -> !CompatibleTypeWeights
    %2 = VPURT.AllocDistributed -> !CompatibleTypeTable
    %3 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    %4 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
        input(%0 : !CompatibleTypeAct)
        weights(%1 : !CompatibleTypeWeights)
        weight_table(%2 : !CompatibleTypeTable)
        parent_input(%0 : !CompatibleTypeAct)
        parent_output(%3 : !CompatibleTypeOutput)
        outputs(%3 : !CompatibleTypeOutput)
    ->  !CompatibleTypeOutput variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %5 = memref.alloc() : memref<1x32x64x176xf16, #NHWC, @DDR>
    // spill to DDR
    %6 = VPUIP.Copy
        inputs(%4 : !CompatibleTypeOutput)
        outputs(%5 : memref<1x32x64x176xf16, #NHWC, @DDR>)  ->  memref<1x32x64x176xf16, #NHWC, @DDR>

    %7 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    // read to NN_CMX
    %8 = VPUIP.Copy
        inputs(%6 : memref<1x32x64x176xf16, #NHWC, @DDR>)
        outputs(%7 : !CompatibleTypeOutput)  ->  !CompatibleTypeOutput

    %9 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    // Eltwise with is_inplace = true
    %10 = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%8 : !CompatibleTypeOutput)
        weights(%9 : !CompatibleTypeOutput)
        parent_input(%8 : !CompatibleTypeOutput)
        parent_output(%7 : !CompatibleTypeOutput)
        outputs(%7 : !CompatibleTypeOutput)
    ->  !CompatibleTypeOutput variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %10 : !CompatibleTypeOutput

    // CHECK:       [[CONV_ACT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_WEIGHTS:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_TABLE:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_OUTPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[ELTWISE_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[ELTWISE:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:         input([[CONV]]
    // CHECK-SAME:         weights([[ELTWISE_INPUT]]
    // CHECK-SAME:         parent_input([[CONV]]
    // CHECK-SAME:         parent_output([[CONV_OUTPUT]]
    // CHECK-SAME:         outputs([[CONV_OUTPUT]]
    // CHECK:             DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:             DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:         } PPE : {
    // CHECK:             PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:         }
    // CHECK:       return [[ELTWISE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!CompatibleTypeAct = !VPUIP.DistributedBuffer<
    1x64x64x176xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CompatibleTypeWeights = !VPUIP.DistributedBuffer<
    32x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!CompatibleTypeTable = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!CompatibleTypeConvOutput = !VPUIP.DistributedBuffer<
    1x32x64x176xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CompatibleTypeOutput = !VPUIP.DistributedBuffer<
    1x32x64x176xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
}>

// CHECK-LABEL: @NCEClusterCopyOpSequenceInPlaceEltwiseUserWithTypeCompatible
func.func @NCEClusterCopyOpSequenceInPlaceEltwiseUserWithTypeCompatible() -> !CompatibleTypeOutput {
    %0 = VPURT.AllocDistributed -> !CompatibleTypeAct
    %1 = VPURT.AllocDistributed -> !CompatibleTypeWeights
    %2 = VPURT.AllocDistributed -> !CompatibleTypeTable
    %3 = VPURT.AllocDistributed -> !CompatibleTypeConvOutput
    %4 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
        input(%0 : !CompatibleTypeAct)
        weights(%1 : !CompatibleTypeWeights)
        weight_table(%2 : !CompatibleTypeTable)
        parent_input(%0 : !CompatibleTypeAct)
        parent_output(%3 : !CompatibleTypeConvOutput)
        outputs(%3 : !CompatibleTypeConvOutput)
    ->  !CompatibleTypeConvOutput variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %5 = memref.alloc() : memref<1x32x64x176xf16, #NHWC, @DDR>
    // spill to DDR
    %6 = VPUIP.Copy
        inputs(%4 : !CompatibleTypeConvOutput)
        outputs(%5 : memref<1x32x64x176xf16, #NHWC, @DDR>)  ->  memref<1x32x64x176xf16, #NHWC, @DDR>

    %7 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    // read to NN_CMX
    %8 = VPUIP.Copy
        inputs(%6 : memref<1x32x64x176xf16, #NHWC, @DDR>)
        outputs(%7 : !CompatibleTypeOutput)  ->  !CompatibleTypeOutput

    %9 = VPURT.AllocDistributed -> !CompatibleTypeOutput
    // Eltwise with is_inplace = true
    %10 = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%8 : !CompatibleTypeOutput)
        weights(%9 : !CompatibleTypeOutput)
        parent_input(%8 : !CompatibleTypeOutput)
        parent_output(%7 : !CompatibleTypeOutput)
        outputs(%7 : !CompatibleTypeOutput)
    ->  !CompatibleTypeOutput variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %10 : !CompatibleTypeOutput

    // CHECK:       [[CONV_ACT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_WEIGHTS:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_TABLE:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV_OUTPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast
    // CHECK:       [[ELTWISE_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[ELTWISE:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input([[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>)
    // CHECK-SAME:     weights([[ELTWISE_INPUT]] : !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>)
    // CHECK-SAME:     parent_input([[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>)
    // CHECK-SAME:     parent_output([[CONV_OUTPUT]] : !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[CONV_OUTPUT]] : !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:  {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:             DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:             DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [175, 63, 31], outStart = [0, 32, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:         } PPE : {
    // CHECK:             PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:         }
    // CHECK:   [[ELTWISE_CAST:%.+]] = VPUIP.DistributedCast inputs(%7 : !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN
    // CHECK-SAME{LITERAL}: {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x32x64x176xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 32, 32, 176], [1, 32, 32, 176]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:       return [[ELTWISE_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NotOptimizeConcatWithMultipleUsers(%arg0: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>,
                                              %arg1: memref<1x32x56x56xui8, #NHWC>,
                                              %arg2: memref<1x1x224x448x!qElemType, #NHWC>)
                                              -> (memref<1x32x56x56xui8, #NHWC>, memref<1x1x224x448x!qElemType, #NHWC>) {

    %ddr_buf = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %0 = VPUIP.SubView %ddr_buf [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56x!qElemType, #NHWC, @DDR> to
        memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %2 = VPUIP.SubView %ddr_buf [0, 16, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56x!qElemType, #NHWC, @DDR> to
        memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %4 = VPUIP.ConcatView
        inputs(%1, %3 :
            memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>,
            memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
        )
        outputs(%ddr_buf : memref<1x32x56x56x!qElemType, #NHWC, @DDR>)
        -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %5 = VPUIP.QuantizeCast
        inputs(%4 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>)
        -> memref<1x32x56x56xui8, #NHWC, @DDR>
    %6 = VPUIP.Copy
        inputs(%5 : memref<1x32x56x56xui8, #NHWC, @DDR>)
        outputs(%arg1 : memref<1x32x56x56xui8, #NHWC>)
        -> memref<1x32x56x56xui8, #NHWC>
    %7 = VPUIP.ShapeCast
        {shape = [1, 1, 224, 448]} inputs(%4 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>)
        -> memref<1x1x224x448x!qElemType, #NHWC, @DDR>
    %8 = VPUIP.Copy
        inputs(%7 : memref<1x1x224x448x!qElemType, #NHWC, @DDR>)
        outputs(%arg2 : memref<1x1x224x448x!qElemType, #NHWC>)
        -> memref<1x1x224x448x!qElemType, #NHWC>
    return %6, %8 : memref<1x32x56x56xui8, #NHWC>, memref<1x1x224x448x!qElemType, #NHWC>

    // CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    // CHECK:    VPUIP.SubView
    // CHECK:    VPUIP.Copy inputs(%arg0
    // CHECK-SAME:          outputs(
    // CHECK:    VPUIP.SubView
    // CHECK:    VPUIP.Copy inputs(%arg0
    // CHECK-SAME:          outputs(
    // CHECK:    VPUIP.ConcatView
    // CHECK:    VPUIP.QuantizeCast
    // CHECK:    VPUIP.Copy inputs(
    // CHECK-SAME:          outputs(%arg1
    // CHECK:    VPUIP.ShapeCast
    // CHECK:    VPUIP.Copy inputs(
    // CHECK-SAME:          outputs(%arg2
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x12x12xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!InputStub_CMX = memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x3x12x12xf16, #NHWC, @DDR>

func.func @FuseDuplicatedCopy() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.Copy
      inputs(%0 : !InputDistributedType)
      outputs(%1 : !SpilledOutput_DDR)  ->  !SpilledOutput_DDR

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX
  // CHECK:  [[BUF_0:%.*]] = VPURT.AllocDistributed
  // CHECK-NOT:  VPUIP.Copy
  // CHECK:  [[CAST:%.*]] = VPUIP.NonDistributedCastOp inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x3x12x12xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:  return [[CAST]] : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x1x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!InputStub_CMX = memref<1x1x3x3xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x1x3x3xf16, #NHWC, @DDR>

func.func @DoNotFuseDuplicatedCopyWhenNoTiling() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.Copy
      inputs(%0 : !InputDistributedType)
      outputs(%1 : !SpilledOutput_DDR)  ->  !SpilledOutput_DDR

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX
  // CHECK-NOT:  VPUIP.NonDistributedCastOp
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#Strides = [154560, 1, 1344, 6]
func.func @TileAfterConcatWithNHWCConcat_C_Tiling_H(%arg0: memref<1x3x115x224xf16, #NHWC, @DDR>) -> !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %alloc = memref.alloc() : memref<1x6x115x224xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 3, 115, 224] : memref<1x6x115x224xf16, #NHWC, @DDR> to memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NHWC, @DDR>) outputs(%0 : memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>) -> memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 3, 0, 0] [1, 3, 115, 224] : memref<1x6x115x224xf16, #NHWC, @DDR> to memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NHWC, @DDR>) outputs(%2 : memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>) -> memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>, memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>)
        outputs(%alloc : memref<1x6x115x224xf16, #NHWC, @DDR>) -> memref<1x6x115x224xf16, #NHWC, @DDR>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %6 = VPUIP.Copy
        inputs(%4 : memref<1x6x115x224xf16, #NHWC, @DDR>)
        outputs(%5 : !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    return %6: !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[FUNC_ARG:%.+]] : memref<1x3x115x224xf16, #NHWC, @DDR>)
    // CHECK-SAME:  outputs(%1 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer

    // CHECK:       [[VAR4:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NHWC, @DDR>)
    // CHECK-SAME:   outputs(%3 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[VAR5:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs(%2, %4 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs(%0 : !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: return [[VAR5:%.*]] : !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#Strides = [154560, 1, 1344, 6]
func.func @TileAfterConcatWithNHWCConcat_C_Tiling_W(%arg0: memref<1x3x115x224xf16, #NHWC, @DDR>) -> !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}> {
    %alloc = memref.alloc() : memref<1x6x115x224xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 3, 115, 224] : memref<1x6x115x224xf16, #NHWC, @DDR> to memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NHWC, @DDR>) outputs(%0 : memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>) -> memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 3, 0, 0] [1, 3, 115, 224] : memref<1x6x115x224xf16, #NHWC, @DDR> to memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NHWC, @DDR>) outputs(%2 : memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>) -> memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>, memref<1x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>)
        outputs(%alloc : memref<1x6x115x224xf16, #NHWC, @DDR>) -> memref<1x6x115x224xf16, #NHWC, @DDR>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>
    %6 = VPUIP.Copy
        inputs(%4 : memref<1x6x115x224xf16, #NHWC, @DDR>)
        outputs(%5 : !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>)  ->  !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>
    return %6: !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>

    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[FUNC_ARG:%.*]] : memref<1x3x115x224xf16, #NHWC, @DDR>)
    // CHECK-SAME:  outputs(%1 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>


    // CHECK:       [[VAR4:%.*]] = VPUIP.Copy inputs([[FUNC_ARG]] : memref<1x3x115x224xf16, #NHWC, @DDR>)
    // CHECK-SAME:   outputs(%3 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>

    // CHECK:    [[VAR5:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs(%2, %4 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>, !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>)
    // CHECK-SAME:     outputs(%0 : !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>) -> !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>

    // CHECK: return [[VAR5:%.*]] : !VPUIP.DistributedBuffer<1x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#Strides = [154560, 51520, 224, 1]
func.func @TileAfterConcatWithNCHWConcat_H_Tiling_C(%arg0: memref<1x3x115x224xf16, #NCHW, @DDR>) -> !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    %alloc = memref.alloc() : memref<1x3x230x224xf16, #NCHW, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 3, 115, 224] : memref<1x3x230x224xf16, #NCHW, @DDR> to memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NCHW, @DDR>) outputs(%0 : memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>) -> memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 115, 0] [1, 3, 115, 224] : memref<1x3x230x224xf16, #NCHW, @DDR> to memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x3x115x224xf16, #NCHW, @DDR>) outputs(%2 : memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>) -> memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>, memref<1x3x115x224xf16, {order = #NCHW, strides = #Strides}, @DDR>)
        outputs(%alloc : memref<1x3x230x224xf16, #NCHW, @DDR>) -> memref<1x3x230x224xf16, #NCHW, @DDR>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %6 = VPUIP.Copy
        inputs(%4 : memref<1x3x230x224xf16, #NCHW, @DDR>)
        outputs(%5 : !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    return %6: !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[FUNC_ARG:%.*]] : memref<1x3x115x224xf16, @DDR>)
    // CHECK-SAME:  outputs(%1 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NCHW, strides = [154560, 51520, 224, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NCHW, strides = [154560, 51520, 224, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>


    // CHECK:       [[VAR4:%.*]] = VPUIP.Copy inputs([[FUNC_ARG]] : memref<1x3x115x224xf16, @DDR>)
    // CHECK-SAME:   outputs(%3 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NCHW, strides = [154560, 51520, 224, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NCHW, strides = [154560, 51520, 224, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[VAR5:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs(%2, %4 : !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NCHW, strides = [154560, 51520, 224, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x115x224xf16, {order = #NCHW, strides = [154560, 51520, 224, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs(%0 : !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK: return [[VAR5:%.*]] : !VPUIP.DistributedBuffer<1x3x230x224xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#Strides = [154560, 1, 1344, 6]
func.func @TileAfterConcatWithBatchOf2NHWCConcat_C_Tiling_H(%arg0: memref<2x3x115x224xf16, #NHWC, @DDR>) -> !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %alloc = memref.alloc() : memref<2x6x115x224xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [2, 3, 115, 224] : memref<2x6x115x224xf16, #NHWC, @DDR> to memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<2x3x115x224xf16, #NHWC, @DDR>) outputs(%0 : memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>) -> memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 3, 0, 0] [2, 3, 115, 224] : memref<2x6x115x224xf16, #NHWC, @DDR> to memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<2x3x115x224xf16, #NHWC, @DDR>) outputs(%2 : memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>) -> memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>, memref<2x3x115x224xf16, {order = #NHWC, strides = #Strides}, @DDR>)
        outputs(%alloc : memref<2x6x115x224xf16, #NHWC, @DDR>) -> memref<2x6x115x224xf16, #NHWC, @DDR>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %6 = VPUIP.Copy
        inputs(%4 : memref<2x6x115x224xf16, #NHWC, @DDR>)
        outputs(%5 : !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    return %6: !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[FUNC_ARG:%.*]] : memref<2x3x115x224xf16, #NHWC, @DDR>)
    // CHECK-SAME:  outputs(%1 : !VPUIP.DistributedBuffer<2x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<2x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>


    // CHECK:       [[VAR4:%.*]] = VPUIP.Copy inputs([[FUNC_ARG]] : memref<2x3x115x224xf16, #NHWC, @DDR>)
    // CHECK-SAME:   outputs(%3 : !VPUIP.DistributedBuffer<2x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<2x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[VAR5:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs(%2, %4 : !VPUIP.DistributedBuffer<2x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<2x3x115x224xf16, {order = #NHWC, strides = [154560, 1, 1344, 6]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs(%0 : !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: return [[VAR5:%.*]] : !VPUIP.DistributedBuffer<2x6x115x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMX2CMXCopyWithSubview
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x256x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.*]]: memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT4:%.*]]: memref<128x1x1x4xsi32, @CMX_NN>,
// CHECK-SAME:  [[OUTPUT1:%.*]]: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
// CHECK-SAME:  [[OUTPUT2:%.*]]: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
func.func @CMX2CMXCopyWithSubview(
    %inData : memref<1x256x28x28xf16, #NHWC, @CMX_NN>,
    %inSparsityMap : memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
    %inWeights : memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
    %inWeightsTable : memref<128x1x1x4xsi32, @CMX_NN>,
    %outDataConcatPart : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
    %outSparsityMapConcatPart : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    -> (memref<1x256x14x14xf16, #NHWC, @CMX_NN>, memref<1x256x14x14xi1, #NHWC, @CMX_NN>)
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>
    // Convolution
    %2:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%inData : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
        weights(%inWeights : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%inWeightsTable : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%inData : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
        parent_input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    // SubView for data
    %3 = memref.alloc() : memref<1x256x14x14xf16, #NHWC, @CMX_NN>
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 128, 14, 14] : memref<1x256x14x14xf16, #NHWC, @CMX_NN>
        to memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // SubView for sparsity map
    %5 = memref.alloc() : memref<1x256x14x14xi1, #NHWC, @CMX_NN>
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 128, 14, 14] : memref<1x256x14x14xi1, #NHWC, @CMX_NN>
        to memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CMX->CMX copy for data
    %7 = VPUIP.Copy
        inputs(%2#0: memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        outputs(%4: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
        -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CMX->CMX copy for sparsity map
    %8 = VPUIP.Copy
        inputs(%2#1: memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%6: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
        -> memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // Concat for data
    %9 = VPUIP.ConcatView
        inputs(%7, %outDataConcatPart :
            memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
            memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
        )
        outputs(%3 : memref<1x256x14x14xf16, #NHWC, @CMX_NN>)
        -> memref<1x256x14x14xf16, #NHWC, @CMX_NN>

    // Concat for sparsity map
    %10 = VPUIP.ConcatView
        inputs(%8, %outSparsityMapConcatPart :
            memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
            memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
        )
        outputs(%5 : memref<1x256x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x256x14x14xi1, #NHWC, @CMX_NN>

    return %9, %10 : memref<1x256x14x14xf16, #NHWC, @CMX_NN>, memref<1x256x14x14xi1, #NHWC, @CMX_NN>

    // CHECK: [[ALLOC1:%.*]] = memref.alloc() : memref<1x256x14x14xi1, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC0:%.*]] = memref.alloc() : memref<1x256x14x14xf16, #NHWC, @CMX_NN>
    // CHECK: [[SUBVIEW_DATA:%.*]] = VPUIP.SubView [[ALLOC0]]

    // CHECK: [[SUBVIEW_SM:%.*]] = VPUIP.SubView [[ALLOC1]]

    // CHECK: [[TASK_RES:%.*]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[INPUT1]]
    // CHECK-SAME:  input_sparsity_map([[INPUT2]]
    // CHECK-SAME:  weights([[INPUT3]]
    // CHECK-SAME:  weight_table([[INPUT4]]
    // CHECK-SAME:  outputs([[SUBVIEW_DATA]]
    // CHECK-SAME:  output_sparsity_map([[SUBVIEW_SM]]

    // CHECK: [[CONCAT_DATA:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#0, [[OUTPUT1]]
    // CHECK-SAME:  outputs([[ALLOC0]]
    // CHECK-SAME:  -> memref<1x256x14x14xf16, #NHWC, @CMX_NN>

    // CHECK: [[CONCAT_SM:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#1, [[OUTPUT2]]
    // CHECK-SAME:  outputs([[ALLOC1]]
    // CHECK-SAME:  -> memref<1x256x14x14xi1, #NHWC, @CMX_NN>

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
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x256x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.*]]: memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT4:%.*]]: memref<128x1x1x4xsi32, @CMX_NN>,
// CHECK-SAME:  [[OUTPUT1:%.*]]: !VPUIP.DistributedBuffer<1x128x14x14xf16
// CHECK-SAME:  [[OUTPUT2:%.*]]: !VPUIP.DistributedBuffer<1x128x14x14xi1
func.func @CMX2CMXTilingCopyWithSubview(
    %inData : memref<1x256x28x28xf16, #NHWC, @CMX_NN>,
    %inSparsityMap : memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
    %inWeights : memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
    %inWeightsTable : memref<128x1x1x4xsi32, @CMX_NN>,
    %outDataConcatPart : !PartialDataBufferType,
    %outSparsityMapConcatPart : !PartialSMBufferType)
    -> (!OutDataBufferType, !OutSMBufferType)
{
    // alloc for Conv data out
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // alloc for Conv sparsity map out
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // Convolution
    %2:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%inData : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
        input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
        weights(%inWeights : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%inWeightsTable : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%inData : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
        parent_input_sparsity_map(%inSparsityMap : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)         parent_output(%0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
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

    // CHECK: [[ALLOC1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN,
    // CHECK: [[ALLOC0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN,
    // CHECK: [[SUBVIEW_DATA:%.*]] = VPUIP.SubView [[ALLOC0]]

    // CHECK: [[SUBVIEW_SM:%.*]] = VPUIP.SubView [[ALLOC1]]

    // CHECK: [[TASK_RES:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:     input([[INPUT1]] : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     input_sparsity_map([[INPUT2]] : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights([[INPUT3]] : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weight_table([[INPUT4]] : memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:     parent_input([[INPUT1]] : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input_sparsity_map([[INPUT2]] : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[SUBVIEW_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     parent_output_sparsity_map([[SUBVIEW_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     output_sparsity_map([[SUBVIEW_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK: [[CONCAT_DATA:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#0, [[OUTPUT1]]
    // CHECK-SAME:  outputs([[ALLOC0]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN,

    // CHECK: [[CONCAT_SM:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[TASK_RES]]#1, [[OUTPUT2]]
    // CHECK-SAME:  outputs([[ALLOC1]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN,

    // CHECK: return [[CONCAT_DATA]], [[CONCAT_SM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputCMX = !VPUIP.DistributedBuffer<
    1x16x128x256xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!SpillDDR = memref<1x16x128x256xf16, #NHWC, @DDR>
!InputCMX = memref<1x16x128x256xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @NotFuseCopyWithTilingCopyIntoTilingCopyCannotFitCMX
// CHECK-SAME:  [[ARG0:%.*]]: memref<1x16x128x256xf16, #NHWC, @CMX_NN>
func.func @NotFuseCopyWithTilingCopyIntoTilingCopyCannotFitCMX(%arg0 : !InputCMX) -> !OutputCMX {
    %alloc_0 = memref.alloc() : !SpillDDR
    %0 = VPUIP.Copy inputs(%arg0 : !InputCMX) outputs(%alloc_0 : !SpillDDR) -> !SpillDDR
    %1 = VPURT.AllocDistributed -> !OutputCMX
    %2 = VPUIP.Copy
        inputs(%0 : !SpillDDR)
        outputs(%1 : !OutputCMX)  ->  !OutputCMX

    return %2 : !OutputCMX

    // CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<1x16x128x256xf16, #NHWC, @DDR>
    // CHECK:      [[COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x16x128x256xf16, #NHWC, @CMX_NN>) outputs([[ALLOC]] : memref<1x16x128x256xf16, #NHWC, @DDR>) -> memref<1x16x128x256xf16, #NHWC, @DDR>
    // CHECK:      [[BUFFER:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x128x256xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[CLUSTER_COPY:%.+]] = VPUIP.Copy inputs([[COPY]] : memref<1x16x128x256xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUFFER]] : !VPUIP.DistributedBuffer<1x16x128x256xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x128x256xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      return [[CLUSTER_COPY]] : !VPUIP.DistributedBuffer<1x16x128x256xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputCMX = !VPUIP.DistributedBuffer<
    1x1x128x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!SpillDDR = memref<1x1x128x256xf16, #NHWC, @DDR>
!InputCMX = memref<1x1x128x256xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @FuseCopyWithTilingCopyIntoTilingCopy
// CHECK-SAME:  [[ARG0:%.*]]: memref<1x1x128x256xf16, #NHWC, @CMX_NN>
func.func @FuseCopyWithTilingCopyIntoTilingCopy(%arg0 : !InputCMX) -> !OutputCMX {
    %alloc_0 = memref.alloc() : !SpillDDR
    %0 = VPUIP.Copy inputs(%arg0 : !InputCMX) outputs(%alloc_0 : !SpillDDR) -> !SpillDDR
    %1 = VPURT.AllocDistributed -> !OutputCMX
    %2 = VPUIP.Copy
        inputs(%0 : !SpillDDR)
        outputs(%1 : !OutputCMX)  ->  !OutputCMX

    return %2 : !OutputCMX

    // CHECK:      [[BUFFER:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:      [[CLUSTER_COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1x128x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER]] : !VPUIP.DistributedBuffer<1x1x128x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x1x128x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:      return [[CLUSTER_COPY]] : !VPUIP.DistributedBuffer<1x1x128x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)>

!OutputBufferType = !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN,
                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
                     compute_shapes = [[1, 32, 14, 28], [1, 32, 14, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]],
                     memory_shapes = [[1, 32, 16, 28], [1, 32, 16, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]}>

// CHECK-LABEL: @FuseCopiesThroughReshape
// CHECK-SAME:  [[INPUT:%.*]]: memref<1x32x50x28x28xf16, #map, @DDR>
func.func @FuseCopiesThroughReshape(%input : memref<1x32x50x28x28xf16, #map, @DDR>) -> !OutputBufferType {
    %subview = VPUIP.SubView %input [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    %alloc = memref.alloc() : memref<1x32x1x28x28xf16, #map, @DDR>
    %copy = VPUIP.Copy inputs(%subview : memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
                       outputs(%alloc : memref<1x32x1x28x28xf16, #map, @DDR>) -> memref<1x32x1x28x28xf16, #map, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%copy : memref<1x32x1x28x28xf16, #map, @DDR>) -> memref<1x32x28x28xf16, #NHWC, @DDR>
    %alloc_dist = VPURT.AllocDistributed -> !OutputBufferType
    %cluster_copy = VPUIP.Copy
        inputs(%reshape : memref<1x32x28x28xf16, #NHWC, @DDR>)
        outputs(%alloc_dist : !OutputBufferType)  ->  !OutputBufferType

    return %cluster_copy : !OutputBufferType

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    // CHECK: [[ALLOC:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>
    // CHECK: [[CLUSTER_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
    // CHECK-SAME:    outputs([[ALLOC]] : !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CLUSTER_COPY]] :
    // CHECK-SAME{LITERAL}: !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>)
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 14, 28], [1, 32, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 16, 28], [1, 32, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]}>
    // CHECK: return [[RESHAPE]] : !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 14, 28], [1, 32, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 16, 28], [1, 32, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]}>
}

// -----

!qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

!OutputBufferType = !VPUIP.DistributedBuffer<1x2048x1x128x!qElemType, #NCHW, @CMX_NN,
                    {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                     compute_shapes = [[1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128]],
                     compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
                     memory_shapes = [[1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128]],
                     memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]}>


// CHECK-LABEL: @FuseCopiesThroughReshapeQuantType
// CHECK-SAME:  [[INPUT:%.+]]: memref<2048x1x128x!qElemType, {order = #CHW, strides = [2048, 128, 1]}, @DDR>
func.func @FuseCopiesThroughReshapeQuantType(%input : memref<2048x1x128x!qElemType, {order = #CHW, strides = [2048, 128, 1]}, @DDR>) -> !OutputBufferType {
    %0 = memref.alloc() : memref<2048x1x128x!qElemType, @DDR>
    %1 = VPUIP.Copy inputs(%input : memref<2048x1x128x!qElemType, {order = #CHW, strides = [2048, 128, 1]}, @DDR>) outputs(%0 : memref<2048x1x128x!qElemType, @DDR>) -> memref<2048x1x128x!qElemType, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<2048x1x128x!qElemType, @DDR>) -> memref<1x2048x1x128x!qElemType, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputBufferType
    %4 = VPUIP.Copy inputs(%2 : memref<1x2048x1x128x!qElemType, @DDR>) outputs(%3 : !OutputBufferType) -> !OutputBufferType
    return %4 : !OutputBufferType

    // CHECK: [[DDR_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<2048x1x128x!qElemType, #CHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[512, 1, 128], [512, 1, 128], [512, 1, 128], [512, 1, 128]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0], [512, 0, 0], [1024, 0, 0], [1536, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[512, 1, 128], [512, 1, 128], [512, 1, 128], [512, 1, 128]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0], [512, 0, 0], [1024, 0, 0], [1536, 0, 0]]}>

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<2048x1x128x!qElemType, {order = #CHW, strides = [2048, 128, 1]}, @DDR>) outputs([[DDR_BUF]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<2048x1x128x!qElemType, #CHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[512, 1, 128], [512, 1, 128], [512, 1, 128], [512, 1, 128]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0], [512, 0, 0], [1024, 0, 0], [1536, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[512, 1, 128], [512, 1, 128], [512, 1, 128], [512, 1, 128]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0], [512, 0, 0], [1024, 0, 0], [1536, 0, 0]]}>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]]
    // CHECK-SAME:  !VPUIP.DistributedBuffer<2048x1x128x!qElemType, #CHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[512, 1, 128], [512, 1, 128], [512, 1, 128], [512, 1, 128]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0], [512, 0, 0], [1024, 0, 0], [1536, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[512, 1, 128], [512, 1, 128], [512, 1, 128], [512, 1, 128]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0], [512, 0, 0], [1024, 0, 0], [1536, 0, 0]]}>
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x2048x1x128x!qElemType, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]}>

    // CHECK: return [[RESHAPE]] : !VPUIP.DistributedBuffer<1x2048x1x128x!qElemType, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128], [1, 512, 1, 128]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)>

!OutputBufferType = !VPUIP.DistributedBuffer<1x32x56x28xf16, #NHWC, @CMX_NN,
                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
                     compute_shapes = [[1, 32, 28, 28], [1, 32, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
                     memory_shapes = [[1, 32, 30, 28], [1, 32, 30, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]}>

// CHECK-LABEL: @NotFuseCopiesTileAxisShapeChanges
// CHECK-SAME:  [[INPUT:%.*]]: memref<1x32x50x28x28xf16, #map, @DDR>
func.func @NotFuseCopiesTileAxisShapeChanges(%input : memref<1x32x50x28x28xf16, #map, @DDR>) -> !OutputBufferType {
    %subview = VPUIP.SubView %input [0, 0, 0, 0, 0] [1, 32, 2, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x2x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    %alloc = memref.alloc() : memref<1x32x2x28x28xf16, #map, @DDR>
    %copy = VPUIP.Copy inputs(%subview : memref<1x32x2x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
                       outputs(%alloc : memref<1x32x2x28x28xf16, #map, @DDR>) -> memref<1x32x2x28x28xf16, #map, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%copy : memref<1x32x2x28x28xf16, #map, @DDR>) -> memref<1x32x56x28xf16, #NHWC, @DDR>
    %alloc_dist = VPURT.AllocDistributed -> !OutputBufferType
    %cluster_copy = VPUIP.Copy
        inputs(%reshape : memref<1x32x56x28xf16, #NHWC, @DDR>)
        outputs(%alloc_dist : !OutputBufferType)  ->  !OutputBufferType

    return %cluster_copy : !OutputBufferType

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 2, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x2x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x2x28x28xf16, #map, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x32x2x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>) outputs([[ALLOC]] : memref<1x32x2x28x28xf16, #map, @DDR>) -> memref<1x32x2x28x28xf16, #map, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x32x2x28x28xf16, #map, @DDR>) -> memref<1x32x56x28xf16, #NHWC, @DDR>
    // CHECK: [[DISTRIBUTED_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x32x56x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 28, 28], [1, 32, 28, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 30, 28], [1, 32, 30, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]}>
    // CHECK: [[CLUSTER_COPY:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x32x56x28xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[DISTRIBUTED_BUF]] : !VPUIP.DistributedBuffer<1x32x56x28xf16, #NHWC, @CMX_NN
    // CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x32x56x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 28, 28], [1, 32, 28, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 30, 28], [1, 32, 30, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]}>
    // CHECK: return [[CLUSTER_COPY]] : !VPUIP.DistributedBuffer<1x32x56x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 28, 28], [1, 32, 28, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 28, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 30, 28], [1, 32, 30, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)>

!DistributedBufferType = !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN,
                    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1], num_clusters = 2 : i64,
                     compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]], compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
                     memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]], memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>

// CHECK-LABEL: @NotFuseDistributedCopiesThroughReshape
// CHECK-SAME:  [[INPUT:%.*]]: memref<1x32x50x28x28xf16, #map, @DDR>
func.func @NotFuseDistributedCopiesThroughReshape(%input : memref<1x32x50x28x28xf16, #map, @DDR>) -> memref<1x32x28x28xf16, #NHWC,@DDR> {
    %subview = VPUIP.SubView %input [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    %alloc_dist = VPURT.AllocDistributed -> !DistributedBufferType
    %cluster_copy = VPUIP.Copy
        inputs(%subview : memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
        outputs(%alloc_dist : !DistributedBufferType)  ->  !DistributedBufferType
    %reshape = VPUIP.GenericReshape inputs(%cluster_copy : !DistributedBufferType) -> memref<1x32x28x28xf16, #NHWC, @CMX_NN>
    %alloc = memref.alloc() : memref<1x32x28x28xf16, #NHWC, @DDR>
    %cluster_copy1 = VPUIP.Copy
        inputs(%reshape : memref<1x32x28x28xf16, #NHWC, @CMX_NN>)
        outputs(%alloc : memref<1x32x28x28xf16, #NHWC, @DDR>)  ->  memref<1x32x28x28xf16, #NHWC, @DDR>

    return %cluster_copy1 : memref<1x32x28x28xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    // CHECK: [[ALLOC_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>
    // CHECK: [[CLUSTER_TILING:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
    // CHECK-SAME:   outputs([[ALLOC_DIST]] : !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CLUSTER_TILING]] : !VPUIP.DistributedBuffer<1x32x1x28x28xf16, #map, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2, 1],
    // CHECK-SAME{LITERAL}: num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 1, 14, 28], [1, 32, 1, 14, 28]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 14, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 1, 16, 28], [1, 32, 1, 16, 28]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 12, 0]]}>)
    // CHECK-SAME{LITERAL}: -> memref<1x32x28x28xf16, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x28x28xf16, #NHWC, @DDR>
    // CHECK: [[CLUSTER_TILING1:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x32x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[ALLOC]] : memref<1x32x28x28xf16, #NHWC, @DDR>) -> memref<1x32x28x28xf16, #NHWC, @DDR>
    // CHECK: return [[CLUSTER_TILING1]] : memref<1x32x28x28xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)>

// CHECK-LABEL: @NotFuseCopiesThroughReshapeOutBufIsSubView
// CHECK-SAME:  [[INPUT:%.*]]: memref<1x32x50x28x28xf16, #map, @DDR>
func.func @NotFuseCopiesThroughReshapeOutBufIsSubView(%input : memref<1x32x50x28x28xf16, #map, @DDR>) -> memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN> {
    %subview = VPUIP.SubView %input [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    %alloc_ddr = memref.alloc() : memref<1x32x1x28x28xf16, #map, @DDR>
    %copy = VPUIP.Copy inputs(%subview : memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
                       outputs(%alloc_ddr : memref<1x32x1x28x28xf16, #map, @DDR>) -> memref<1x32x1x28x28xf16, #map, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%copy : memref<1x32x1x28x28xf16, #map, @DDR>) -> memref<1x32x28x28xf16, #NHWC, @DDR>
    %alloc_cmx = memref.alloc() : memref<1x64x28x28xf16, #NHWC, @CMX_NN>
    %out_subview = VPUIP.SubView %alloc_cmx [0, 0, 0, 0] [1, 32, 28, 28] : memref<1x64x28x28xf16, #NHWC, @CMX_NN> to  memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>
    %cluster_copy = VPUIP.Copy
        inputs(%reshape : memref<1x32x28x28xf16, #NHWC, @DDR>)
        outputs(%out_subview : memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>)  ->  memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>

    return %cluster_copy : memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>

    // CHECK: [[SUBVIEW_IN:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : memref<1x32x50x28x28xf16, #map, @DDR> to memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>
    // CHECK: [[ALLOC_DDR:%.+]] = memref.alloc() : memref<1x32x1x28x28xf16, #map, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_IN]] : memref<1x32x1x28x28xf16, {order = #map, strides = [1254400, 50, 1, 44800, 1600]}, @DDR>)
    // CHECK-SAME:    outputs([[ALLOC_DDR]] : memref<1x32x1x28x28xf16, #map, @DDR>) -> memref<1x32x1x28x28xf16, #map, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x32x1x28x28xf16, #map, @DDR>) -> memref<1x32x28x28xf16, #NHWC, @DDR>
    // CHECK: [[ALLOC_CMX:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, @CMX_NN>
    // CHECK: [[SUBVIEW_OUT:%.+]] = VPUIP.SubView [[ALLOC_CMX]] [0, 0, 0, 0] [1, 32, 28, 28] : memref<1x64x28x28xf16, #NHWC, @CMX_NN> to memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>
    // CHECK: [[CLUSTER_TILING:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x32x28x28xf16, #NHWC, @DDR>)
    // CHECK-SAME:    outputs([[SUBVIEW_OUT]] : memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>)
    // CHECK-SAME:    -> memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>
    // CHECK: return [[CLUSTER_TILING]] : memref<1x32x28x28xf16, {order = #NHWC, strides = [50176, 1, 1792, 64]}, @CMX_NN>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x128x32x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    memory_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
}>

// CHECK-LABEL: @DDRToCMXCopyWithConcatViewPermuteCastWithClusterCopyWithSegmentedLikeDistribution
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x32x127x128xf16, @DDR>
func.func @DDRToCMXCopyWithConcatViewPermuteCastWithClusterCopyWithSegmentedLikeDistribution(%arg0: memref<1x32x127x128xf16, @DDR>)
                                    -> !OutputDistributed {
    %cst = const.Declare memref<1x32x1x128xf16> = dense<0.000000e+00> : tensor<1x32x1x128xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]
    %0 = memref.alloc() : memref<1x32x128x128xf16, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 32, 127, 128] : memref<1x32x128x128xf16, @DDR> to memref<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>

    %2 = VPUIP.Copy inputs(%arg0 : memref<1x32x127x128xf16, @DDR>)
                    outputs(%1 : memref<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>) -> memref<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 127, 0] [1, 32, 1, 128] : memref<1x32x128x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>

    %4 = VPUIP.Copy inputs(%cst : memref<1x32x1x128xf16>)
                    outputs(%3 : memref<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @DDR>)
        outputs(%0 : memref<1x32x128x128xf16, @DDR>) -> memref<1x32x128x128xf16, @DDR>

    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%5 : memref<1x32x128x128xf16, @DDR>) -> memref<1x128x32x128xf16, #NHWC, @DDR>

    %7 = VPURT.AllocDistributed -> !OutputDistributed
    %8 = VPUIP.Copy
        inputs(%6 : memref<1x128x32x128xf16, #NHWC, @DDR>)
        outputs(%7 : !OutputDistributed)  ->  !OutputDistributed

    return %8 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<1x32x1x128xf16> = dense<0.000000e+00> : tensor<1x32x1x128xf16, {order = #NHWC}>, [#const.Reorder<#NCHW>]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x128x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 32, 127, 128] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x128x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}> to
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 127, 128], [1, 6, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 127, 128], [1, 6, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INPUT]] : memref<1x32x127x128xf16, @DDR>)
    // CHECK-SAME:              outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 127, 128], [1, 6, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 127, 128], [1, 6, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 127, 0] [1, 32, 1, 128] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x128x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}> to
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 1, 128], [1, 6, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 1, 128], [1, 6, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:       [[COPY_1:%.+]] =  VPUIP.Copy
    // CHECK-SAME:              inputs([[CST]] : memref<1x32x1x128xf16>)
    // CHECK-SAME:              outputs([[SUBVIEW_1]] :  !VPUIP.DistributedBuffer<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 1, 128], [1, 6, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 1, 128], [1, 6, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:              inputs([[COPY_0]], [[COPY_1]]
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x127x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 127, 128], [1, 6, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 127, 128], [1, 6, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128], [1, 5, 127, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x1x128xf16, {order = #NCHW, strides = [524288, 16384, 128, 1]}, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 1, 128], [1, 6, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 1, 128], [1, 6, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128], [1, 5, 1, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>)

    // CHECK-SAME:              outputs([[OUT_BUFF]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x128x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>) ->
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x128x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:              inputs([[CONCAT]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x128x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 128, 128], [1, 6, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128], [1, 5, 128, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>) ->
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x128x32x128xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:       [[DISTRIBUTED_CAST:%.+]] = VPUIP.DistributedCast
    // CHECK-SAME:              inputs([[PERMUTE_CAST]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x128x32x128xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>) ->
    // CHECK-SAME{LITERAL}: !VPUIP.DistributedBuffer<1x128x32x128xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:       return [[DISTRIBUTED_CAST]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x128x32x128xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 128, 6, 128], [1, 128, 6, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128], [1, 128, 5, 128]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x32x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

// CHECK-LABEL: @RemoveDDR2DDRCopyInputWithSubViewOp
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x144x128x128xf16, #NHWC, @DDR>
func.func @RemoveDDR2DDRCopyInputWithSubViewOp(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
                        -> (!OutputDistributed, !OutputDistributed) {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x144x64x128xf16, #NHWC, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 144, 32, 128]
            : memref<1x144x64x128xf16, #NHWC, @DDR>
            to memref<1x144x32x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [1179648, 1, 18432, 144]}, @DDR>
    %4 = VPURT.AllocDistributed -> !OutputDistributed
    %5 = VPUIP.Copy
        inputs(%3 : memref<1x144x32x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [1179648, 1, 18432, 144]}, @DDR>)
        outputs(%4 : !OutputDistributed)  ->  !OutputDistributed
    %6 = VPUIP.SubView %2 [0, 0, 32, 0] [1, 144, 32, 128]
            : memref<1x144x64x128xf16, #NHWC, @DDR>
            to memref<1x144x32x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [1179648, 1, 18432, 144]}, @DDR>
    %7 = VPURT.AllocDistributed -> !OutputDistributed
    %8 = VPUIP.Copy
        inputs(%6 : memref<1x144x32x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [1179648, 1, 18432, 144]}, @DDR>)
        outputs(%7 : !OutputDistributed)  ->  !OutputDistributed

    return %5, %8 : !OutputDistributed, !OutputDistributed

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 144, 32, 128]
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR> to
    // CHECK-SAME:      memref<1x144x32x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x32x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : memref<1x144x32x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_1]] : !VPUIP.DistributedBuffer<1x144x32x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x144x32x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 32, 0] [1, 144, 32, 128]
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR> to
    // CHECK-SAME:      memref<1x144x32x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_2:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x32x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_2]] : memref<1x144x32x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_2]] : !VPUIP.DistributedBuffer<1x144x32x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x144x32x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[COPY_1]], [[COPY_2]]
}
// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SubViewWithTrivialCopy
func.func @SubViewWithTrivialCopy(
    %INPUT: memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
) -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR> {
    // CHECK:   [[INPUT:%.+]]: memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
    %SUBVIEW = VPUIP.SubView %INPUT [0, 0, 0, 0] [1, 1, 357604, 1] :
        memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
        to memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 1, 357604, 1]

    %ALLOC = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK-NOT:   memref.alloc

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR>)
        outputs(%ALLOC : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>)
            -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:   [[VIEW_OP:%.+]] = VPUIP.ViewOp [[SUBVIEW]] : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR> to memref<1x1x357604x1x!qElemType, #NHWC, @DDR>

    return %COPY : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[VIEW_OP]] : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipSubViewWithNonTrivialCopy
func.func @SkipSubViewWithNonTrivialCopy(
    %INPUT: memref<1x3x224x224x!qElemType, #NHWC, @DDR>
) -> memref<1x1x224x224x!qElemType, #NHWC, @DDR> {
    // CHECK:   [[INPUT:%.+]]: memref<1x3x224x224x!qElemType, #NHWC, @DDR>
    %SUBVIEW = VPUIP.SubView %INPUT [0, 0, 0, 0] [1, 1, 224, 224] :
        memref<1x3x224x224x!qElemType, #NHWC, @DDR>
        to memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [150528, 1, 672, 3]}, @DDR>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]

    %ALLOC = memref.alloc() : memref<1x1x224x224x!qElemType, #NHWC, @DDR>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x224x224x!qElemType, #NHWC, @DDR>

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [150528, 1, 672, 3]}, @DDR>)
        outputs(%ALLOC : memref<1x1x224x224x!qElemType, #NHWC, @DDR>)
            -> memref<1x1x224x224x!qElemType, #NHWC, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy

    return %COPY : memref<1x1x224x224x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[COPY]] : memref<1x1x224x224x!qElemType, #NHWC, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @SubViewWithOffset([[INPUT:%.+]]: memref<1x1x363584x1x!qElemType, #NHWC, @DDR>)
// CHECK-SAME: -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
func.func @SubViewWithOffset(
    %INPUT: memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
) -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR> {
    %SUBVIEW = VPUIP.SubView %INPUT [0, 0, 1, 0] [1, 1, 357604, 1] :
        memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
        to memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]

    %ALLOC = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK-NOT:   memref.alloc()

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR>)
        outputs(%ALLOC : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>)
            -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.Copy

    // CHECK: [[VIEWOP:%.+]] = VPUIP.ViewOp [[SUBVIEW]] : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR> to memref<1x1x357604x1x!qElemType, #NHWC, @DDR>

    return %COPY : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[VIEWOP]] : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipSubViewWithCMX2DDR
func.func @SkipSubViewWithCMX2DDR(
    %INPUT: memref<1x1x363584x1x!qElemType, #NHWC, [@CMX_NN, 0]>
) -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR> {
    // CHECK:   [[INPUT:%.+]]: memref<1x1x363584x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %SUBVIEW = VPUIP.SubView %INPUT [0, 0, 0, 0] [1, 1, 357604, 1] :
        memref<1x1x363584x1x!qElemType, #NHWC, [@CMX_NN, 0]>
        to memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]

    %ALLOC = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, [@CMX_NN, 0]>)
        outputs(%ALLOC : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>)
            -> memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy

    return %COPY : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[COPY]] : memref<1x1x357604x1x!qElemType, #NHWC, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipSubViewWithDDR2CMX
func.func @SkipSubViewWithDDR2CMX(
    %INPUT: memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
) -> memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]> {
    // CHECK:   [[INPUT:%.+]]: memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
    %SUBVIEW = VPUIP.SubView %INPUT [0, 0, 0, 0] [1, 1, 357604, 1] :
        memref<1x1x363584x1x!qElemType, #NHWC, @DDR>
        to memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]

    %ALLOC = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]>

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, @DDR>)
        outputs(%ALLOC : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy

    return %COPY : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:   return [[COPY]] : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipSubViewWithCMX2CMX
func.func @SkipSubViewWithCMX2CMX(
    %INPUT: memref<1x1x363584x1x!qElemType, #NHWC, [@CMX_NN, 0]>
) -> memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]> {
    // CHECK:   [[INPUT:%.+]]: memref<1x1x363584x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %SUBVIEW = VPUIP.SubView %INPUT [0, 0, 0, 0] [1, 1, 357604, 1] :
        memref<1x1x363584x1x!qElemType, #NHWC, [@CMX_NN, 0]>
        to memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]

    %ALLOC = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]>

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<1x1x357604x1x!qElemType, {order = #NHWC, strides = [363584, 1, 1, 1]}, [@CMX_NN, 0]>)
        outputs(%ALLOC : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]>)
            -> memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy

    return %COPY : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:   return [[COPY]] : memref<1x1x357604x1x!qElemType, #NHWC, [@CMX_NN, 1]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: SkipSubViewWithStridedCopy
func.func @SkipSubViewWithStridedCopy(
    %INPUT: memref<387072x3xf16, @DDR>
) -> memref<387072x1xf16, @DDR> {
    // CHECK:   [[INPUT:%.+]]: memref<387072x3xf16, @DDR>

    %SUBVIEW = VPUIP.SubView %INPUT [0, 0] [387072, 1] :
        memref<387072x3xf16, @DDR>
        to memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>
    // CHECK:   [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]]

    %ALLOC = memref.alloc() : memref<387072x1xf16, @DDR>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<387072x1xf16, @DDR>

    %COPY = VPUIP.Copy
        inputs(%SUBVIEW : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>)
        outputs(%ALLOC : memref<387072x1xf16, @DDR>) -> memref<387072x1xf16, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy

    return %COPY : memref<387072x1xf16, @DDR>
    // CHECK:   return [[COPY]] : memref<387072x1xf16, @DDR>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.012237719928517061:128>
!qElemType1 = !quant.uniform<u8:f16, 0.011216485266591988:128>
!qElemType2 = !quant.uniform<u8:f16, 0.068572803572112442:128>
!qElemType3 = !quant.uniform<u8:f16, 0.0068572806377036897:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @CopyOpSequenceWithInPlaceEltwiseUserAndViewOpOutBuffer
// CHECK-SAME:   [[INPUT0:%.+]]: memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>
// CHECK-SAME:   [[INPUT1:%.+]]: memref<1x32x32x128x!qElemType, #NHWC, [@CMX_NN, 0]>
func.func @CopyOpSequenceWithInPlaceEltwiseUserAndViewOpOutBuffer(%arg0: memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>,
                                                                  %arg1: memref<1x32x32x128x!qElemType, #NHWC, [@CMX_NN, 0]>)
                                                                  -> memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>

    %1 = VPUIP.NCEClusterTask
             {
                eltwise_type = #VPU.eltwise_type<ADD>,
                task_type = #VPUIP.nce_task_type<ELTWISE>
             }
             input(%arg0 : memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>)
             weights(%arg0 : memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>)
             parent_input(%arg0 : memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>)
             parent_output(%0 : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>)
             outputs(%0 : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>)
             -> memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>
             variants :
             {
                 DPUTask
                     {
                         inEnd = [127, 31, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                         outEnd = [127, 31, 31], outStart = [0, 0, 0],
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                     }
             }
             PPE :
             {
                 PPETask {ppe = #VPU.PPEStub<>}
             }

    %2 = VPUIP.QuantizeCast
              inputs(%1 : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>

    %3 = memref.alloc() : memref<1x32x32x128x!qElemType3, #NHWC, @DDR>

    // CMX2DDR
    %4 = VPUIP.Copy
              inputs(%2 : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>)
              outputs(%3 : memref<1x32x32x128x!qElemType3, #NHWC, @DDR>)
              -> memref<1x32x32x128x!qElemType3, #NHWC, @DDR>

    %5 = memref.alloc() : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>

    // ViewOp
    %6 = VPUIP.ViewOp %5 : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>

    // DDR2CMX
    %7 = VPUIP.Copy
              inputs(%4 : memref<1x32x32x128x!qElemType3, #NHWC, @DDR>)
              outputs(%5 : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>

    // EltwiseOp with is_inplace = true
    %8 = VPUIP.NCEClusterTask
            {
                eltwise_type = #VPU.eltwise_type<ADD>,
                is_inplace = true,
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }
            input(%7 : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x32x32x128x!qElemType, #NHWC, [@CMX_NN, 0]>)
            parent_input(%7 : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>)
            parent_output(%6 : memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>)
            outputs(%6 : memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask
                    {
                        inEnd = [127, 31, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                        outEnd = [127, 31, 31], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
            }
            PPE :
            {
                PPETask {ppe = #VPU.PPEStub<>}
            }

    return %8 : memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[COPY_BUFF:%.+]] = memref.alloc()

    // CHECK:       [[ADD:%.+]] = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[INPUT0]] : memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[INPUT0]] : memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[INPUT0]] : memref<1x32x32x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[COPY_BUFF]] : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[COPY_BUFF]] : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]> variants : {
    // CHECK:               DPUTask {inEnd = [127, 31, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:       }

    // CHECK:       [[QUANTIZE_CAST:%.+]] = VPUIP.QuantizeCast inputs([[ADD]] : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[VIEWOP:%.+]] = VPUIP.ViewOp [[COPY_BUFF]] : memref<1x32x32x128x!qElemType2, #NHWC, [@CMX_NN, 0]> to memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[ADD_INPLACE:%.+]] = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[QUANTIZE_CAST]] : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[INPUT1]] : memref<1x32x32x128x!qElemType, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[QUANTIZE_CAST]] : memref<1x32x32x128x!qElemType3, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[VIEWOP]] : memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[VIEWOP]] : memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]> variants : {
    // CHECK:               DPUTask {inEnd = [127, 31, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:       }

    // CHECK:       return [[ADD_INPLACE]] : memref<1x32x32x128x!qElemType1, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyOpSequenceWithInPlaceEltwiseUserAndShapeCastParent
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>
func.func @CopyOpSequenceWithInPlaceEltwiseUserAndShapeCastParent(%arg0: memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>,
                                                                  %arg1: memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>)
                                                                  -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]> {
    // CopyOp sequence
    %0 = memref.alloc() : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>

    %1 = VPUIP.NCEClusterTask
             {
                eltwise_type = #VPU.eltwise_type<ADD>,
                task_type = #VPUIP.nce_task_type<ELTWISE>
             }
             input(%arg1 : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>)
             weights(%arg1 : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>)
             parent_input(%arg1 : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>)
             parent_output(%0 : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>)
             outputs(%0 : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>)
             -> memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>
             variants :
             {
                 DPUTask
                     {
                         inEnd = [576, 31, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                         outEnd = [576, 31, 15], outStart = [0, 0, 0],
                         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                     }
             }
             PPE :
             {
                 PPETask {ppe = #VPU.PPEStub<>}
             }

    %2 = VPUIP.ShapeCast {shape = [1, 48, 64, 96]} inputs(%1 : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    %3 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, @DDR>
    %4 = VPUIP.Copy
              inputs(%2 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%3 : memref<1x48x64x96xf16, #NHWC, @DDR>)
              -> memref<1x48x64x96xf16, #NHWC, @DDR>

    %5 = memref.alloc() : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.Copy
              inputs(%4 : memref<1x48x64x96xf16, #NHWC, @DDR>)
              outputs(%5 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    // Eltwise AddOp with inplace
    %7 = VPUIP.NCEClusterTask
              {
                  is_inplace = true,
                  task_type = #VPUIP.nce_task_type<ELTWISE>
              }
              input(%arg0 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              weights(%6 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              parent_input(%arg0 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              parent_output(%5 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%5 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
              -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
              variants :
              {
                  DPUTask
                      {
                          inEnd = [95, 63, 47], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                          outEnd = [95, 63, 47], outStart = [0, 0, 0],
                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                      }
              }
              PPE :
              {
                  PPETask
                      {
                          ppe = #VPU.PPEStub<>
                      }
              }

    return %7 : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[COPY_BUFF:%.+]] = memref.alloc()
    // CHECK:       [[ADD1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:       [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 48, 64, 96]} inputs([[ADD1]] : memref<1x16x32x576xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[ADD2_INPLACE:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[INPUT0]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[SHAPECAST]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[INPUT0]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[SHAPECAST]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[SHAPECAST]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]> variants : {
    // CHECK:               DPUTask {inEnd = [95, 63, 47], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [95, 63, 47], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:       }

    // CHECK:       return [[ADD2_INPLACE]] : memref<1x48x64x96xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutDistBufferType = !VPUIP.DistributedBuffer<
  2x1x1x160xf16, #NCHW, @CMX_NN,
  {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
   compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
   memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}
>

!SubViewOutDistBufferType = !VPUIP.DistributedBuffer<
  2x1x1x160xf16, {order = #NCHW, strides = [160, 160, 160, 1]}, @CMX_NN,
  {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
   compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
   memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}
>

// CHECK: func.func @SkipSubViewWithTilingCopyDueToSubViewUserWithExplicitOutputShape
// CHECK-SAME: ([[INPUT:%.+]]: memref<8x1x1x160xf16, @DDR>)
func.func @SkipSubViewWithTilingCopyDueToSubViewUserWithExplicitOutputShape(
    %input: memref<8x1x1x160xf16, @DDR>
) -> (!SubViewOutDistBufferType, !SubViewOutDistBufferType) {

    %0 = VPUIP.SubView %input [2, 0, 0, 0] [2, 1, 1, 160] : memref<8x1x1x160xf16, @DDR> to memref<2x1x1x160xf16, @DDR>
    %1 = VPUIP.SubView %input [0, 0, 0, 0] [2, 1, 1, 160] : memref<8x1x1x160xf16, @DDR> to memref<2x1x1x160xf16, @DDR>

    %2 = VPURT.AllocDistributed -> !OutDistBufferType
    %3 = VPUIP.Copy
        inputs(%1 : memref<2x1x1x160xf16, @DDR>)
        outputs(%2 : !OutDistBufferType)  ->  !OutDistBufferType
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [2, 1, 1, 160] {explicit_output_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]]} : !OutDistBufferType to !SubViewOutDistBufferType

    %5 = VPURT.AllocDistributed -> !OutDistBufferType
    %6 = VPUIP.Copy
        inputs(%0 : memref<2x1x1x160xf16, @DDR>)
        outputs(%5 : !OutDistBufferType)  ->  !OutDistBufferType

    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [2, 1, 1, 160] {explicit_output_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]]} : !OutDistBufferType to !SubViewOutDistBufferType

    return %4, %7 : !SubViewOutDistBufferType, !SubViewOutDistBufferType

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [2, 0, 0, 0] [2, 1, 1, 160] : memref<8x1x1x160xf16, @DDR> to memref<2x1x1x160xf16, @DDR>
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [2, 1, 1, 160] : memref<8x1x1x160xf16, @DDR> to memref<2x1x1x160xf16, @DDR>

    // CHECK: [[ALLOC_0:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<2x1x1x160xf16, #NCHW, @CMX_NN,
    // CHECK:      mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[SUBVIEW_1]]
    // CHECK:  outputs([[ALLOC_0]]
    // CHECK:  -> !VPUIP.DistributedBuffer<2x1x1x160xf16, #NCHW, @CMX_NN
    // CHECK:      mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

    // CHECK: [[SUBVIEW_USER_0:%.+]] = VPUIP.SubView [[COPY_0]] [0, 0, 0, 0] [2, 1, 1, 160]
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]]
    // CHECK: !VPUIP.DistributedBuffer
    // CHECK:   to !VPUIP.DistributedBuffer<2x1x1x160xf16, {order = #NCHW, strides = [160, 160, 160, 1]}, @CMX_NN
    // CHECK:      mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]

    // CHECK: [[ALLOC_1:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<2x1x1x160xf16, #NCHW, @CMX_NN,
    // CHECK:      mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[SUBVIEW_0]] : memref<2x1x1x160xf16, @DDR>
    // CHECK:  outputs([[ALLOC_1]] : !VPUIP.DistributedBuffer<2x1x1x160xf16, #NCHW, @CMX_NN
    // CHECK:  -> !VPUIP.DistributedBuffer<2x1x1x160xf16, #NCHW, @CMX_NN
    // CHECK:      mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

    // CHECK: [[SUBVIEW_USER_1:%.+]] = VPUIP.SubView [[COPY_1]] [0, 0, 0, 0] [2, 1, 1, 160]
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]]
    // CHECK: !VPUIP.DistributedBuffer
    // CHECK:   to !VPUIP.DistributedBuffer<2x1x1x160xf16, {order = #NCHW, strides = [160, 160, 160, 1]}, @CMX_NN
    // CHECK:      mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]

    // CHECK: return [[SUBVIEW_USER_0]], [[SUBVIEW_USER_1]]
    // CHECK: !VPUIP.DistributedBuffer<2x1x1x160xf16, {order = #NCHW, strides = [160, 160, 160, 1]}, @CMX_NN
    // CHECK:   mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>
    // CHECK: !VPUIP.DistributedBuffer<2x1x1x160xf16, {order = #NCHW, strides = [160, 160, 160, 1]}, @CMX_NN
    // CHECK:   mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 160], [1, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputBufferType = !VPUIP.DistributedBuffer<1x8x256x336xf16, #NCHW, @CMX_NN,
                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
                     compute_shapes = [[1, 8, 128, 336], [1, 8, 128, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]],
                     memory_shapes = [[1, 8, 130, 336], [1, 8, 130, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 126, 0]]}>

// CHECK-LABEL: @NotFuseCopiesThroughReshapeWith3DInputAndOverlappedUser
// CHECK-SAME:  [[INPUT:%.+]]: memref<8x512x336xf16, @DDR>
func.func @NotFuseCopiesThroughReshapeWith3DInputAndOverlappedUser(%input : memref<8x512x336xf16, @DDR>) -> !OutputBufferType {
    %subview = VPUIP.SubView %input [0, 0, 0] [8, 256, 336] : memref<8x512x336xf16, @DDR> to memref<8x256x336xf16, {order = #CHW, strides = [172032, 336, 1]}, @DDR>
    %alloc = memref.alloc() : memref<8x256x336xf16, @DDR>
    %copy = VPUIP.Copy inputs(%subview : memref<8x256x336xf16, {order = #CHW, strides = [172032, 336, 1]}, @DDR>)
                       outputs(%alloc : memref<8x256x336xf16, @DDR>)
                       -> memref<8x256x336xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%copy : memref<8x256x336xf16, @DDR>) -> memref<1x8x256x336xf16, @DDR>
    %alloc_dist = VPURT.AllocDistributed -> !OutputBufferType
    %cluster_copy = VPUIP.Copy
        inputs(%reshape : memref<1x8x256x336xf16, @DDR>)
        outputs(%alloc_dist : !OutputBufferType)  ->  !OutputBufferType

    return %cluster_copy : !OutputBufferType

    // CHECK: [[SUBVIEW_IN:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0] [8, 256, 336] : memref<8x512x336xf16, @DDR> to memref<8x256x336xf16, {order = #CHW, strides = [172032, 336, 1]}, @DDR>
    // CHECK: [[ALLOC_DDR:%.+]] = memref.alloc() : memref<8x256x336xf16, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_IN]] : memref<8x256x336xf16, {order = #CHW, strides = [172032, 336, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[ALLOC_DDR]] : memref<8x256x336xf16, @DDR>) -> memref<8x256x336xf16, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY]] : memref<8x256x336xf16, @DDR>) -> memref<1x8x256x336xf16, @DDR>
    // CHECK: [[ALLOC_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x256x336xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 8, 128, 336], [1, 8, 128, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]]
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 8, 130, 336], [1, 8, 130, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 126, 0]]}>
    // CHECK: [[CLUSTER_TILING:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : memref<1x8x256x336xf16, @DDR>)
    // CHECK-SAME:     outputs([[ALLOC_CMX]] : !VPUIP.DistributedBuffer<1x8x256x336xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x8x256x336xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 8, 128, 336], [1, 8, 128, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]]
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 8, 130, 336], [1, 8, 130, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 126, 0]]}>

    // CHECK: return [[CLUSTER_TILING]] : !VPUIP.DistributedBuffer<1x8x256x336xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 8, 128, 336], [1, 8, 128, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]]
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 8, 130, 336], [1, 8, 130, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 126, 0]]}>
}
