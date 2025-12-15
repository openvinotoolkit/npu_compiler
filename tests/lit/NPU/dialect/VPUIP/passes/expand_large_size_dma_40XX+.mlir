//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-expand-dma %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 0.0040670955882352944:128>

// CHECK-LABEL: @UnrollExpandDMAWithLargeSizeAndDiffWithExpandAxis
func.func @UnrollExpandDMAWithLargeSizeAndDiffWithExpandAxis() -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <0> -> memref<1x400x4096x4096x!qElemType, #NWCH, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <6710886400> -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
                inputs(%input : memref<1x400x4096x4096x!qElemType, #NWCH, @DDR>)
                outputs(%output : memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>)
                -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>
    }

    return %output: memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>

    //CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x400x4096x2621x!qElemType, #NWCH, @DDR>
    //CHECK:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <4294246400> -> memref<1x400x4096x1475x!qElemType, #NWCH, @DDR>
    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <6710886400> -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>
    //CHECK:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <6710886400> -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>
    //CHECK:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <11133960192> -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {
    //CHECK-SAME:   pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x400x4096x2621x!qElemType, #NWCH, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>) -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {
    //CHECK-SAME:   pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x400x4096x1475x!qElemType, #NWCH, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>) -> memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>

    //CHECK:    return [[OUTPUT]] : memref<1x412x4096x4096x!qElemType, #NWCH, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0040670955882352944:128>

// CHECK-LABEL: @UnrollExpandDMAWithLargeSizeAndSameWithExpandAxis
func.func @UnrollExpandDMAWithLargeSizeAndSameWithExpandAxis() -> memref<1x412x4096x4096x!qElemType, #NCHW, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <0> -> memref<1x400x4096x4096x!qElemType, #NCHW, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <6710886400> -> memref<1x412x4096x4096x!qElemType, #NCHW, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
                inputs(%input : memref<1x400x4096x4096x!qElemType, #NCHW, @DDR>)
                outputs(%output : memref<1x412x4096x4096x!qElemType, #NCHW, @DDR>)
                -> memref<1x412x4096x4096x!qElemType, #NCHW, @DDR>
    }

    return %output: memref<1x412x4096x4096x!qElemType, #NCHW, @DDR>

    //CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x255x4096x4096x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <4278190080> -> memref<1x145x4096x4096x!qElemType, @DDR>
    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <6710886400> -> memref<1x412x4096x4096x!qElemType, @DDR>
    //CHECK:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <DDR> <6710886400> -> memref<1x412x4096x4096x!qElemType, @DDR>
    //CHECK:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <DDR> <10989076480> -> memref<1x412x4096x4096x!qElemType, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {
    //CHECK-SAME:     pads_begin = [0, 0, 0, 0], pads_end = [0, 157, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x255x4096x4096x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x412x4096x4096x!qElemType, @DDR>) -> memref<1x412x4096x4096x!qElemType, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {
    //CHECK-SAME:     pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x145x4096x4096x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x412x4096x4096x!qElemType, @DDR>) -> memref<1x412x4096x4096x!qElemType, @DDR>

    //CHECK:    return [[OUTPUT]] : memref<1x412x4096x4096x!qElemType, @DDR>
}
