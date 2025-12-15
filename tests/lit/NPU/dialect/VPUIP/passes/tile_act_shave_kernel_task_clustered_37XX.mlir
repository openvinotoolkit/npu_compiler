//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW  allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithCAlignment(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>) on tile 0 -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>,!VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
        }

    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                   -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                  outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NoTileSegmentedShaveWithCAlignment(%arg0: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x64x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>) on tile 0 -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
        }

    %5 = memref.alloc() : memref<1x64x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x64x64x32xf16>)  ->  memref<1x64x64x32xf16>

    return %6: memref<1x64x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x64x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[MVN:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:              inputs([[COPY0]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:              outputs([[OUTPUT_CMX]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK-NOT:                      VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x64x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[MVN]] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[OUTPUT_DDR]] : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x64x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithCAlignment(%arg0: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
        }

    %5 = memref.alloc() : memref<1x32x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x32x64x32xf16>)  ->  memref<1x32x64x32xf16>

    return %6: memref<1x32x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:               inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:                      [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:              outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:                      [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>,
    // CHECK-SAME:           !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x32x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                          outputs([[OUTPUT_DDR]] : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x32x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NoTileDuplicatedShaveWithCAlignment(%arg0: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) on tile 0 -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
        }

    %5 = memref.alloc() : memref<1x16x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x16x64x32xf16>)  ->  memref<1x16x64x32xf16>

    return %6: memref<1x16x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x16x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[MVN:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                 inputs([[COPY0]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16,
    // CHECK-SAME:                 outputs([[OUTPUT_CMX]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16
    // CHECK-SAME:                 -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK-NOT:                      VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x16x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[MVN]] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[OUTPUT_DDR]] : memref<1x16x64x32xf16>)  ->  memref<1x16x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x16x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterStridedMVN(%arg0: memref<1x128x64x32xf16, #NWHC>)
        -> memref<1x128x64x32xf16, #NWHC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16, #NWHC>)  ->  memref<1x128x64x32xf16, #NWHC>
    return %6: memref<1x128x64x32xf16, #NWHC>


    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN
    // CHECK-SAME:                   -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                               outputs([[BUFF_OUT_DDR]] : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:    return [[COPY_OUTPUT_TO_DDR]] : memref<1x128x64x32xf16, #NWHC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileNoSymmetricClusterStridedMVNWithNCHW(%arg0: memref<1x33x16x1xf16>)
        -> memref<1x33x16x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x33x16x1xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>{
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x33x16x1xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x33x16x1xf16>) -> memref<1x33x16x1xf16>
    return %6: memref<1x33x16x1xf16>


    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x33x16x1xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x33x16x1xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                          outputs([[OUTPUT_DDR]] : memref<1x33x16x1xf16>) -> memref<1x33x16x1xf16>
    // CHECK:    return [[COPY1]] : memref<1x33x16x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileNoSymmetricClusterStridedMVNWithNHWC(%arg0: memref<1x33x16x1xf16, #NHWC>)
        -> memref<1x33x16x1xf16, #NHWC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x33x16x1xf16, #NHWC>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>{
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x33x16x1xf16, #NHWC>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x33x16x1xf16, #NHWC>) -> memref<1x33x16x1xf16, #NHWC>
    return %6: memref<1x33x16x1xf16, #NHWC>


    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x33x16x1xf16, #NHWC>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x33x16x1xf16, #NHWC>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                          outputs([[OUTPUT_DDR]] : memref<1x33x16x1xf16, #NHWC>) -> memref<1x33x16x1xf16, #NHWC>
    // CHECK:    return [[COPY1]] : memref<1x33x16x1xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileClusterMVNForUnalignedShape(%arg0: memref<1x32x1x10240xf16>)
        -> memref<1x32x1x10240xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x1x10240xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%1 as %arg3 : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg4 : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>{
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
        }
    %5 = memref.alloc() : memref<1x32x1x10240xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x32x1x10240xf16>)  ->  memref<1x32x1x10240xf16>
    return %6: memref<1x32x1x10240xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x1x10240xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[MVN:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:     inputs([[COPY0]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTPUT_CMX]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) on tile 0 -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>{
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK-SAME:    }
    // CHECK:    [[BUFF_OUT_DDR:%.+]] = memref.alloc() : memref<1x32x1x10240xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[MVN]] : !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[BUFF_OUT_DDR]] : memref<1x32x1x10240xf16>)  ->  memref<1x32x1x10240xf16>
    // CHECK:    return [[COPY_OUTPUT_TO_DDR]] : memref<1x32x1x10240xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x1x96x160xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>) -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate
            inputs(%1 as %arg2: !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
                                        compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
                                        memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>)
            outputs(%2 as %arg3: !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
            -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>, !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }
    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %5 : memref<1x1x192x320xf16>

    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}:     VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW0]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:                          outputs([[INPUT_BUF0]]
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW1]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:     outputs([[INPUT_BUF1]] : !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CLUSTER_INTERPOLATE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate
 // CHECK-SAME:     inputs([[COPY1]]        as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16,
 // CHECK-SAME:             [[COPY0]]       as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16,
 // CHECK-SAME:     outputs([[OUTPUT_BUF0]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16,
 // CHECK-SAME:             [[OUTPUT_BUF1]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16,
// CHECK-SAME:     -> (!VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
// CHECK:               VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}([[INN_0]], [[OUT_0]])
// CHECK:               VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}([[INN_1]], [[OUT_1]])
// CHECK-SAME:     }
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy inputs([[CLUSTER_INTERPOLATE]]#0
    // CHECK-SAME:                   outputs([[SUBVIEW2]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY3:%.+]] = VPUIP.Copy inputs([[CLUSTER_INTERPOLATE]]#1
    // CHECK-SAME:                   outputs([[SUBVIEW3]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY2]], [[COPY3]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) outputs([[OUTPUT_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[CONCAT]] : memref<1x1x192x320xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x96x160xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x192x320xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 96, 320], [1, 1, 96, 320]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 1, 96, 320], [1, 1, 96, 320]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]
}>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterHalfPixelInterpolateWithExplicitDistribution
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x1x96x160xf16>)
func.func @TileClusterHalfPixelInterpolateWithExplicitDistribution(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = VPURT.AllocDistributed -> !InputDistributedBuffer
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x1x96x160xf16>)
        outputs(%0 : !InputDistributedBuffer) -> !InputDistributedBuffer
    %2 = VPURT.AllocDistributed -> !OutputDistributedBuffer
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Interpolate
        inputs(%1 as %arg4 : !InputDistributedBuffer)
        outputs(%2 as %arg5 : !OutputDistributedBuffer) on tile 0 -> !OutputDistributedBuffer{
            VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]} (%arg4, %arg5) : !InputDistributedBuffer, !OutputDistributedBuffer
        }

    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !OutputDistributedBuffer)
        outputs(%4 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %5 : memref<1x1x192x320xf16>

    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 47, 0] [1, 1, 49, 160] :
    // CHECK-SAME:  memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>

    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[SUBVIEW0]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:              outputs([[INPUT_BUF0]] : !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>)

    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1, 49, 160] :
    // CHECK-SAME:  memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>

    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[SUBVIEW1]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:           outputs([[INPUT_BUF1]] : !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>)

    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]]}>

    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]]}>

    // CHECK:    [[CLUSTER_INTERPOLATE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate
    // CHECK-SAME:    inputs([[COPY1]] as        [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16,
    // CHECK-SAME:           [[COPY0]] as        [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16,
    // CHECK-SAME:    outputs([[OUTPUT_BUF0]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16,
    // CHECK-SAME:            [[OUTPUT_BUF1]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16,
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK-SAME:      {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}
    // CHECK-SAME:      ([[INN_0]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK-SAME:      {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}
    // CHECK-SAME:      ([[INN_1]], [[OUT_1]])


    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 96, 320] :
    // CHECK-SAME:  memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>

    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[CLUSTER_INTERPOLATE]]#0
    // CHECK-SAME:  outputs([[SUBVIEW2]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>)
    // CHECK-SAME:    -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>

    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 96, 0] [1, 1, 96, 320] :
    // CHECK-SAME:    memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>

    // CHECK:    [[COPY3:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[CLUSTER_INTERPOLATE]]#1
    // CHECK-SAME:  outputs([[SUBVIEW3]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>)
    // CHECK-SAME:    -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY2]], [[COPY3]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>,
    // CHECK-SAME:                                memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs([[OUTPUT_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>

    // CHECK:    return [[CONCAT]] : memref<1x1x192x320xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterAlignCornersInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x1x96x160xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>)  ->  !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Interpolate
        inputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
         -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
            VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]} (%arg4, %arg5) :  !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>, !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x1x192x320xf16>)  ->  memref<1x1x192x320xf16>
    return %5 : memref<1x1x192x320xf16>


    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}:     VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW0]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUF0]]
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[COPY4:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW1]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:     outputs([[INPUT_BUF1]] : !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                          compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>)  ->  !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CLUSTER_INTERPOLATE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate
    // CHECK-SAME:     inputs([[COPY4]]        as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16,
    // CHECK-SAME:            [[COPY2]]        as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16,
    // CHECK-SAME:     outputs([[OUTPUT_BUF0]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16,
    // CHECK-SAME:           [[OUTPUT_BUF1]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16,
    // CHECK-SAME:     -> (!VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}([[INN_0]], [[OUT_0]])
    // CHECK: }
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY5:%.+]] = VPUIP.Copy inputs([[CLUSTER_INTERPOLATE]]#0
    // CHECK-SAME:               outputs([[SUBVIEW2]]
    // CHECK-SAME:               -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY6:%.+]] = VPUIP.Copy inputs([[CLUSTER_INTERPOLATE]]#1
    // CHECK-SAME:              outputs([[SUBVIEW3]]
    // CHECK-SAME:              -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY5]], [[COPY6]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) outputs([[OUTPUT_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[CONCAT]] : memref<1x1x192x320xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterPytorchHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x1x96x160xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>) -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Interpolate
        inputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
            VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]} (%arg4, %arg5) :
            !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}>, !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %5 : memref<1x1x192x320xf16>

    // CHECK:    [[INPUT_SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUFF1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:            !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[INPUT_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_SUBVIEW1]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:     outputs([[INPUT_BUFF1]] : !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                           compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>)  ->  !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[INPUT_SUBVIEW0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUFF0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME{Literal}:             !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[INPUT_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_SUBVIEW0]] : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:     outputs([[INPUT_BUFF0]] : !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                           compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>)  ->  !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[OUTPUT_BUFF1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUFF0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INTERP:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate
    // CHECK-SAME:     inputs([[INPUT_COPY0]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>,
    // CHECK-SAME:          [[INPUT_COPY1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>)
    // CHECK-SAME:     outputs([[OUTPUT_BUFF0]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:           [[OUTPUT_BUFF1]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> (!VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:            VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}([[INN_0]], [[OUT_0]])
    // CHECK:            VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}([[INN_1]], [[OUT_1]])
    // CHECK-SAME:     }
    // CHECK:    [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[OUTPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[OUTPUT_COPY0:%.+]] = VPUIP.Copy inputs([[INTERP]]#0
    // CHECK-SAME:                          outputs([[OUTPUT_SUBVIEW0]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[OUTPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[OUTPUT_COPY1:%.+]] = VPUIP.Copy inputs([[INTERP]]#1
    // CHECK-SAME:                          outputs([[OUTPUT_SUBVIEW1]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUTPUT_COPY0]], [[OUTPUT_COPY1]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) outputs([[OUTPUT_BUFF]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[CONCAT:%.+]] : memref<1x1x192x320xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterGelu(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[IN_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]   = VPUIP.SubView %1 [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]   = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[GELU:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu
    // CHECK-SAME:                   inputs([[IN_SV1]]  as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                          [[IN_SV0]]  as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                  outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                          [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                       -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                             outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x128x64x32xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
}>

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterGeluWithExplicitDistributedAttr
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x128x64x32xf16>)
func.func @TileClusterGeluWithExplicitDistributedAttr(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuffer
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !DistributedBuffer) -> !DistributedBuffer
    %3 = VPURT.AllocDistributed -> !DistributedBuffer
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%1 as %arg6 : !DistributedBuffer)
        outputs(%3 as %arg7 : !DistributedBuffer) on tile 0 -> !DistributedBuffer{
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !DistributedBuffer, !DistributedBuffer
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !DistributedBuffer)
        outputs(%5 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:                 [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[IN_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:                 [[IN_SV0:%.+]]   = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:                 [[IN_SV1:%.+]]   = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:                 [[OUT_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:                 [[OUT_SV0:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:                 [[OUT_SV1:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK: [[GELU:%.+]]:2   = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu
    // CHECK-SAME:                  inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:                         [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:                outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:                        [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>){
    // CHECK:                                VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                                VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:                 [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:          [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:          [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                              outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:          return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_HardSigmoid(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_hardsigmoid.cpp", VPU.kernel_entry = "activation_hardsigmoid"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterHardSigmoid(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_HardSigmoid
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK: [[IN_BUF:%.+]]   = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[IN_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[IN_SV0:%.+]]   = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_SV1:%.+]]   = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[OUT_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_SV0:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_SV1:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[HARD_SIGMOID:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_HardSigmoid
    // CHECK-SAME:   inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:          [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME: outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:         [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_0]], [[OUT_0]])
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_1]], [[OUT_1]])

    // CHECK: [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[HARD_SIGMOID]]#0, [[HARD_SIGMOID]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK: [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                          outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:  return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterGeluWithCMXInput(%arg0: !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%arg0 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%0 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %2 = memref.alloc() : memref<1x128x64x32xf16>
    %3 = VPUIP.Copy
        inputs(%1 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %3: memref<1x128x64x32xf16>

    // CHECK:    [[IN_SV0:%.+]] = VPUIP.SubView %arg0 [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[GELU:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu
    // CHECK-SAME:                    inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                           [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                  outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                          [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%2 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                             outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK: return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x128x64x32xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
}>

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterGeluWithCMXInputAndExplicitDistribution
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN
func.func @TileClusterGeluWithCMXInputAndExplicitDistribution(%arg0: !DistributedBuffer)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuffer
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Gelu

        inputs(%arg0 as %arg6 : !DistributedBuffer)
        outputs(%0 as %arg7 : !DistributedBuffer)  on tile 0 ->  !DistributedBuffer {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !DistributedBuffer, !DistributedBuffer
        }
    %2 = memref.alloc() : memref<1x128x64x32xf16>
    %3 = VPUIP.Copy
        inputs(%1 : !DistributedBuffer)
        outputs(%2 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %3: memref<1x128x64x32xf16>

    // CHECK:             [[IN_SV0:%.+]]   = VPUIP.SubView %arg0 [0, 64, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:             [[IN_SV1:%.+]]   = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:             [[OUT_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:             [[OUT_SV0:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32]
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:             [[OUT_SV1:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:             [[GELU:%.+]]:2   = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu
    // CHECK-SAME:                                    inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                                           [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                                  outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                                          [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME{LITERAL}:  -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) outputs(%2 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                             outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterGeluWithdifferentTiles(%arg0: memref<1x128x6x32xf16>)
        -> memref<1x128x6x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x6x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x6x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x6x32xf16>)  ->  memref<1x128x6x32xf16>
    return %6: memref<1x128x6x32xf16>

    // CHECK:     [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x6x32xf16>)
    // CHECK-SAME:     outputs([[IN_BUF]] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[GELU:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu
    // CHECK-SAME:                    inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]},
    // CHECK-SAME:                           [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]},
    // CHECK-SAME:                   outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]},
    // CHECK-SAME:                           [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]},
    // CHECK-SAME:                   -> (!VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:     [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x6x32xf16>
    // CHECK:     [[OUT_COPY:%.+]]  = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                          outputs([[OUT_DDR]] : memref<1x128x6x32xf16>) -> memref<1x128x6x32xf16>

    // CHECK:      return [[OUT_COPY]] : memref<1x128x6x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterSoftmax(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[SOFTMAX:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                      -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}([[INN_0]], [[OUT_0]])
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}([[INN_1]], [[OUT_1]])
    // CHECK:}

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                                      outputs([[BUFF_OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY_OUTPUT_TO_DDR]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSoftmaxWhenAxisIsHighestDim(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [2]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 32, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 32, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SOFTMAX:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer
    // CHECK-SAME:                    -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [2]}([[INN_0]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [2]}([[INN_1]], [[OUT_1]])
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUTPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                                  outputs([[BUFF_OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY_OUTPUT_TO_DDR]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileClusterSoftmaxForUnsupportedAxis(%arg0: memref<1x128x2x1xf16>)
        -> memref<1x128x2x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x2x1xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [2]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x2x1xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x2x1xf16>)  ->  memref<1x128x2x1xf16>
    return %6: memref<1x128x2x1xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x2x1xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SOFTMAX:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:     inputs([[COPY0]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTPUT_CMX]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
    // CHECK:                       VPUIP.SW.Kernel.run {attrs = [2]}([[INN_0]], [[OUT_0]])
    // CHECK:    [[BUFF_OUT_DDR:%.+]] = memref.alloc() : memref<1x128x2x1xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SOFTMAX]] : !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_OUT_DDR]] : memref<1x128x2x1xf16>)  ->  memref<1x128x2x1xf16>
    // CHECK:    return [[COPY_OUTPUT_TO_DDR]] : memref<1x128x2x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileClusterHalfPixelInterpolateForCMXSizeRequirement(%arg0: memref<1x16x154x160xf16>) -> memref<1x16x308x320xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x154x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]], memory_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x154x160xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x16x154x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]], memory_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]]}>)  ->  !VPUIP.DistributedBuffer<1x16x154x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]], memory_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]]}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Interpolate
        inputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x16x154x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]], memory_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]]}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 154, 16, 1], [320, 308, 16, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]} (%arg4, %arg5) :
            !VPUIP.DistributedBuffer<1x16x154x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]], memory_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]]}>, !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %4 = memref.alloc() : memref<1x16x308x320xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x16x308x320xf16>)  ->  memref<1x16x308x320xf16>
    return %5 : memref<1x16x308x320xf16>

    // CHECK:    [[INPUT_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs
    // CHECK-SAME:                        outputs([[INPUT_BUF]]
    // CHECK:    [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK:    [[INTERP:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate
    // CHECK-SAME:                             inputs([[COPY0]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x154x160xf16,
    // CHECK-SAME:                       outputs([[OUTPUT_BUF]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x308x320xf16,
    // CHECK-SAME:                       -> !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
    // CHECK:                             VPUIP.SW.Kernel.run
    // CHECK-NOT:                         VPUIP.SW.Kernel.run
    // CHECK:                          }
    // CHECK:    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterMultiply() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Multiply

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Multiply First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Multiply Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTIPLY:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[MULTI_OUT_TILE0]]  as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[MULTIPLY]]#0, [[MULTIPLY]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterMultiplyAtBroadcastAxis() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Multiply

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Multiply First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Multiply Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>


    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTIPLY:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}
    // CHECK:                            [[INPUT1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[MULTIPLY]]#0, [[MULTIPLY]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterMultiplyWithSameInputs() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Multiply

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %0 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%1 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %3 = memref.alloc() : memref<1x64x88x88xf16>
    %4 = VPUIP.Copy
        inputs(%2 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %4: memref<1x64x88x88xf16>

    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTIPLY:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                     inputs([[INPUT1_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[MULTIPLY]]#0, [[MULTIPLY]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterMinimum() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Minimum

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Minimum Input & Output
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MIN_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MIN_OUT_TILE0:%.+]] = VPUIP.SubView [[MIN_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MIN_OUT_TILE1:%.+]] = VPUIP.SubView [[MIN_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]

    // CHECK:    [[MINIMUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Minimum
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INPUT0_TILE1_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INPUT1_TILE1_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INPUT0_TILE0_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INPUT1_TILE0_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     outputs([[MIN_OUT_TILE1]] as [[MIN_OUT_TILE1_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                             [[MIN_OUT_TILE0]] as [[MIN_OUT_TILE0_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE1_ARG]], [[INPUT1_TILE1_ARG]], [[MIN_OUT_TILE1_ARG]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE0_ARG]], [[INPUT1_TILE0_ARG]], [[MIN_OUT_TILE0_ARG]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[MINIMUM]]#0, [[MINIMUM]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MIN_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipMVNTiling
module @SkipMVNTiling attributes {} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private
            @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64)
            attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%NET_INPUT: memref<1x32x5120x1xf16>,
                %NET_OUTPUT: memref<1x32x5120x1xf16>)
        -> memref<1x32x5120x1xf16> {
    %ALLOC_INPUT_DIST = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<
        1x32x5120x1xf16,
        #NCHW,
        @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [1, 2, 1, 1],
            num_clusters = 2 : i64,
            alignment = [1, 16, 1, 1]
        }>
    %INPUT_TILING_COPY = VPUIP.Copy
        inputs(%NET_INPUT : memref<1x32x5120x1xf16>)
        outputs(%ALLOC_INPUT_DIST : !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %MVN_OUTPUT = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<
        1x32x5120x1xf16,
        #NCHW,
        @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [1, 2, 1, 1],
            num_clusters = 2 : i64
        }>
    %MVN_TILING = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_MVN
        inputs(%INPUT_TILING_COPY as %arg4 : !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%MVN_OUTPUT as %arg5 : !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 9.9999997473787516E-6]} (%arg4, %arg5) :!VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,  !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }
    %OUTPUT_TILING_COPY = VPUIP.Copy
        inputs(%MVN_TILING : !VPUIP.DistributedBuffer<1x32x5120x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%NET_OUTPUT : memref<1x32x5120x1xf16>)  ->  memref<1x32x5120x1xf16>

    return %OUTPUT_TILING_COPY : memref<1x32x5120x1xf16>

    // CHECK: ([[NET_INPUT:%.+]]: memref<1x32x5120x1xf16>, [[NET_OUTPUT:%.+]]: memref<1x32x5120x1xf16>)

    // CHECK:   [[ALLOC_INPUT_DIST:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16
    // CHECK-SAME:      alignment = [1, 16, 1, 1]
    // CHECK:    [[INPUT_TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[NET_INPUT]]
    // CHECK-SAME:     outputs([[ALLOC_INPUT_DIST]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16,
    // CHECK-SAME:      alignment = [1, 16, 1, 1]

    // CHECK:   [[MVN_OUTPUT:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16,

    // CHECK:   [[MVN_TILING:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:  inputs([[INPUT_TILING_COPY]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x5120x1xf16,
    // CHECK-SAME:  outputs([[MVN_OUTPUT]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x5120x1xf16,
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16,
    // CHECK:       VPUIP.SW.Kernel.run {{.*}}
    // CHECK-NOT:   VPUIP.SW.Kernel.run {{.*}}
    // CHECK:    [[OUTPUT_TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[MVN_TILING]]
    // CHECK-SAME:     outputs([[NET_OUTPUT]] : memref<1x32x5120x1xf16>)  -> memref<1x32x5120x1xf16>

    // CHECK:   return [[OUTPUT_TILING_COPY]] : memref<1x32x5120x1xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileConvertOpTest(%arg0: memref<1x64x16x16xf32>)
        -> memref<1x64x16x16xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x64x16x16xf32>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Convert
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x64x16x16xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x64x16x16xf16>)  ->  memref<1x64x16x16xf16>
    return %6: memref<1x64x16x16xf16>
    // CHECK:   [[SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 8, 0] [1, 64, 8, 16] : memref<1x64x16x16xf32> to memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>
    // CHECK:   [[ALLOC_DSTR1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW1]] : memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>)
    // CHECK-SAME:     outputs([[ALLOC_DSTR1]] : !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[SUBVIEW2:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 8, 16] : memref<1x64x16x16xf32> to memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>
    // CHECK:   [[ALLOC_DSTR2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>)
    // CHECK-SAME:     outputs([[ALLOC_DSTR2]] : !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[ALLOC_DSTR3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[ALLOC_DSTR4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Convert
    // CHECK-SAME:     inputs([[TILING_COPY2]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[TILING_COPY1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[ALLOC_DSTR4]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[ALLOC_DSTR3]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> (!VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [0]}([[INN_0]], [[OUT_0]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileConvertOpTest_AlignedShape(%arg0: memref<1x64x16x16xf32>)
        -> memref<1x64x16x16xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x64x16x16xf32>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Convert
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
            VPUIP.SW.Kernel.run {attrs = [0]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x64x16x16xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x64x16x16xf16>)  ->  memref<1x64x16x16xf16>
    return %6: memref<1x64x16x16xf16>
    // CHECK:   [[SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 8, 0] [1, 64, 8, 16] : memref<1x64x16x16xf32> to memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>
    // CHECK:   [[ALLOC_DSTR1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW1]] : memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>)
    // CHECK-SAME:     outputs([[ALLOC_DSTR1]] : !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:   [[SUBVIEW2:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 8, 16] : memref<1x64x16x16xf32> to memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>
    // CHECK:   [[ALLOC_DSTR2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>)
    // CHECK-SAME:     outputs([[ALLOC_DSTR2]] : !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:   [[ALLOC_DSTR3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[ALLOC_DSTR4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Convert
    // CHECK-SAME:     inputs([[TILING_COPY2]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, [[TILING_COPY1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[ALLOC_DSTR4]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[ALLOC_DSTR3]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> (!VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [0]}([[INN_0]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [0]}([[INN_1]], [[OUT_1]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterTanh(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Tanh
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[IN_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[TANH:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Tanh
    // CHECK-SAME:                        inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                               [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                      outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                              [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                       -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[TANH]]#0, [[TANH]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                             outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterTopKWithCopyUser
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x8x128x128xf16, #NHWC>
// CHECK-SAME:  [[ARG1:%.+]]: memref<1x1x128x128xsi32, #NHWC>
func.func @TileClusterTopKWithCopyUser(%arg0: memref<1x8x128x128xf16, #NHWC>, %arg1: memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC> {
    %cst = const.Declare memref<1x1x1x128xui8> = dense<0> : tensor<1x1x1x128xui8>
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x128x128xf16, #NHWC>) outputs(%0 : !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
             -> !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %11 = VPUIP.Copy inputs(%cst : memref<1x1x1x128xui8>) outputs(%10 : !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                    -> !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK
            inputs(%1 as %arg2: !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
            %11 as %arg8: !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
            outputs(%2 as %arg3: !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
            %3 as %arg4: !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
            -> (!VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg2, %arg8, %arg3, %arg4) :
        !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
        !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
        !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
        !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }
    %5 = memref.alloc() : memref<1x1x128x128xsi32, #NHWC>
    %6 = VPUIP.Copy inputs(%4#1 : !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    outputs(%5 : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    %7 = VPUIP.Copy inputs(%6 : memref<1x1x128x128xsi32, #NHWC>) outputs(%arg1 : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    return %7 : memref<1x1x128x128xsi32, #NHWC>

    // CHECK:    [[CONST:%.+]] = const.Declare memref<1x1x1x128xui8> = dense<0> : tensor<1x1x1x128xui8>
    // CHECK:    [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x8x128x128xf16, #NHWC>) outputs([[ALLOC0]] : !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 64, 0] [1, 8, 64, 128] : !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 8, 64, 128] : !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONST]] : memref<1x1x1x128xui8>) outputs([[ALLOC1]] : !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 64] [1, 1, 1, 64] : !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x1x64xui8, {order = #NHWC, strides = [128, 1, 128, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 1, 1, 64] : !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x1x64xui8, {order = #NHWC, strides = [128, 1, 128, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW4:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 64, 0] [1, 1, 64, 128] : !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x64x128xf16, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW5:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0, 0] [1, 1, 64, 128] : !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x64x128xf16, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC3]] [0, 0, 64, 0] [1, 1, 64, 128] : !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW7:%.+]] = VPUIP.SubView [[ALLOC3]] [0, 0, 0, 0] [1, 1, 64, 128] : !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[RESULTS:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK inputs([[SUBVIEW1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[SUBVIEW3]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x64xui8, {order = #NHWC, strides = [128, 1, 128, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, [[SUBVIEW0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[SUBVIEW2]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x64xui8, {order = #NHWC, strides = [128, 1, 128, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs([[SUBVIEW5]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x64x128xf16, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[SUBVIEW7]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[SUBVIEW4]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x64x128xf16, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[SUBVIEW6]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.DistributedBuffer<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x64xui8, {order = #NHWC, strides = [128, 1, 128, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x64x128xf16, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.DistributedBuffer<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x64xui8, {order = #NHWC, strides = [128, 1, 128, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x64x128xf16, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#1, [[RESULTS]]#3 : !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[ALLOC3]] : !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC_MEMREF:%.+]] = memref.alloc() : memref<1x1x128x128xsi32, #NHWC>
    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[ALLOC_MEMREF]] : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    // CHECK:    [[COPY3:%.+]] = VPUIP.Copy inputs([[COPY2]] : memref<1x1x128x128xsi32, #NHWC>) outputs([[ARG1]] : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    // CHECK:    return [[COPY3]] : memref<1x1x128x128xsi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterTopKWithoutCopyUser(%arg0: memref<1x5x256x512xf16>, %arg1: memref<1x1x256x512xf16>, %arg2: memref<1x1x256x512xsi32>) -> (memref<1x1x256x512xf16>, memref<1x1x256x512xsi32>) {
    %cst = const.Declare memref<1x1x1x80xui8> = dense<0> : tensor<1x1x1x80xui8>
    %0 = memref.alloc() : memref<1x5x256x512xf16>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x5x256x512xf16>) outputs(%0 : memref<1x5x256x512xf16>) -> memref<1x5x256x512xf16>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.Copy inputs(%1 : memref<1x5x256x512xf16>)
            outputs(%2 : !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %15 = VPUIP.Copy inputs(%cst : memref<1x1x1x80xui8>)
            outputs(%14 : !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
            -> !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %6:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK
                inputs(%3 as %arg3: !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                %15 as %arg9: !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                outputs(%4 as %arg4: !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                %5 as %arg5: !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
                -> (!VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
        VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg3, %arg9, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }
    %7 = memref.alloc() : memref<1x1x256x512xf16>
    %8 = VPUIP.Copy inputs(%6#0 : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            outputs(%7 : memref<1x1x256x512xf16>)
             -> memref<1x1x256x512xf16>
    %9 = memref.alloc() : memref<1x1x256x512xsi32>
    %10 = VPUIP.Copy inputs(%6#1 : !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            outputs(%9 : memref<1x1x256x512xsi32>)
             -> memref<1x1x256x512xsi32>
    %11 = VPUIP.Copy inputs(%8 : memref<1x1x256x512xf16>) outputs(%arg1 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    %12 = VPUIP.Copy inputs(%10 : memref<1x1x256x512xsi32>) outputs(%arg2 : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
    return %11, %12 : memref<1x1x256x512xf16>, memref<1x1x256x512xsi32>

    // CHECK:           [[CST:%.+]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 40]>]
    // CHECK:           [[CST0:%.+]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 40], [1, 1, 1, 40]>]
    // CHECK:           [[ALLOC0:%.+]] = memref.alloc() : memref<1x5x256x512xf16>
    // CHECK:           [[COPY0:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x5x256x512xf16>) outputs([[ALLOC0]] : memref<1x5x256x512xf16>) -> memref<1x5x256x512xf16>
    // CHECK:           [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 128, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:           [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_INPUT0:%.+]] = VPUIP.Copy inputs([[SUBVIEW0]]
    // CHECK-SAME:                                   outputs([[ALLOC1]]
    // CHECK-SAME:                                   -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:           [[ALLOC2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_INPUT1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]]
    // CHECK-SAME:                                   outputs([[ALLOC2]]
    // CHECK-SAME:                                   -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[ALLOC13:%.+]] = VPURT.AllocDistributed ->
    // CHECK:           !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:           [[TILING_AUX_INPUT_0:%.+]] = VPUIP.Copy inputs([[CST0]]
    // CHECK-SAME:                                   outputs([[ALLOC13]]
    // CHECK{LITERAL}:  -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:           [[ALLOC14:%.+]] = VPURT.AllocDistributed
    // CHECK:           -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:           [[TILING_AUX_INPUT_1:%.+]] = VPUIP.Copy inputs([[CST]]
    // CHECK-SAME:                                   outputs([[ALLOC14]]
    // CHECK{LITERAL}:  -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:           [[ALLOC3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC5:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC6:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CLUSTER_TOPK:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK
    // CHECK:           inputs([[TILING_INPUT1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x5x128x512xf16,
    // CHECK:           [[TILING_AUX_INPUT_1]]   as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x40xui8,
    // CHECK:           [[TILING_INPUT0]]        as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x5x128x512xf16,
    // CHECK:           [[TILING_AUX_INPUT_0]]   as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x40xui8,
    // CHECK:           outputs([[ALLOC4]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xf16,
    // CHECK:           [[ALLOC6]]         as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xsi32,
    // CHECK:           [[ALLOC3]]         as [[OUT_2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xf16,
    // CHECK:           [[ALLOC5]]         as [[OUT_3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xsi32,
    // CHECK:           -> (!VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}([[INN_0]], [[INN_1]], [[OUT_0]], [[OUT_1]])
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}([[INN_2]], [[INN_3]], [[OUT_2]], [[OUT_3]])
    // CHECK:           }
    // CHECK:           [[ALLOC7:%.+]] = memref.alloc() : memref<1x1x256x512xf16, @DDR>
    // CHECK:           [[ALLOC8:%.+]] = memref.alloc() : memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC7]] [0, 0, 0, 0] [1, 1, 128, 512] : memref<1x1x256x512xf16, @DDR> to memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT0:%.+]] = VPUIP.Copy inputs([[CLUSTER_TOPK]]#0
    // CHECK-SAME:                                   outputs([[SUBVIEW2]] : memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>)
    // CHECK-SAME:                                   -> memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC8]] [0, 0, 0, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT1:%.+]] = VPUIP.Copy inputs([[CLUSTER_TOPK]]#1
    // CHECK-SAME:                                   outputs([[SUBVIEW3]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>)
    // CHECK-SAME:                                   -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[SUBVIEW4:%.+]] = VPUIP.SubView [[ALLOC7]] [0, 0, 128, 0] [1, 1, 128, 512] : memref<1x1x256x512xf16, @DDR> to memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT2:%.+]] = VPUIP.Copy inputs([[CLUSTER_TOPK]]#2
    // CHECK-SAME:                                   outputs([[SUBVIEW4]] : memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>)
    // CHECK-SAME:                                   -> memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[SUBVIEW5:%.+]] = VPUIP.SubView [[ALLOC8]] [0, 0, 128, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT3:%.+]] = VPUIP.Copy inputs([[CLUSTER_TOPK]]#3
    // CHECK-SAME:                                   utputs([[SUBVIEW5]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[TILING_OUTPUT0]], [[TILING_OUTPUT2]] : memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>, memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) outputs([[ALLOC7]] : memref<1x1x256x512xf16, @DDR>) -> memref<1x1x256x512xf16, @DDR>
    // CHECK:           [[ALLOC9:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_OUTPUT4:%.+]] = VPUIP.Copy inputs([[CONCAT0]]
    // CHECK-SAME:                                   outputs([[ALLOC9]]
    // CHECK-SAME:                                   -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[TILING_OUTPUT1]], [[TILING_OUTPUT3]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>, memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) outputs([[ALLOC8]] : memref<1x1x256x512xsi32, @DDR>) -> memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[ALLOC10:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_OUTPUT5:%.+]] = VPUIP.Copy inputs([[CONCAT1]]
    // CHECK-SAME:                                   outputs([[ALLOC10]]
    // CHECK-SAME:                                   -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC11:%.+]] = memref.alloc() : memref<1x1x256x512xf16>
    // CHECK:           [[TILING_OUTPUT6:%.+]] = VPUIP.Copy inputs([[TILING_OUTPUT4]]
    // CHECK-SAME:                                   outputs([[ALLOC11]] : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    // CHECK:           [[ALLOC12:%.+]] = memref.alloc() : memref<1x1x256x512xsi32>
    // CHECK:           [[TILING_OUTPUT7:%.+]] = VPUIP.Copy inputs([[TILING_OUTPUT5]]
    // CHECK-SAME:                                   outputs([[ALLOC12]] : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
    // CHECK:           [[COPY_OUT0:%.+]] = VPUIP.Copy inputs([[TILING_OUTPUT6]] : memref<1x1x256x512xf16>) outputs(%arg1 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    // CHECK:           [[COPY_OUT1:%.+]] = VPUIP.Copy inputs([[TILING_OUTPUT7]] : memref<1x1x256x512xsi32>) outputs(%arg2 : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
    // CHECK:           return [[COPY_OUT0]], [[COPY_OUT1]] : memref<1x1x256x512xf16>, memref<1x1x256x512xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @builtin_Convert(memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterSingleOutputTopKWithoutCopyUser
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x5x256x512xf16>
func.func @TileClusterSingleOutputTopKWithoutCopyUser(%arg0: memref<1x5x256x512xf16>) -> memref<1x1x256x512xf16> {
    %cst = const.Declare memref<1x1x1x80xui8> = dense<0> : tensor<1x1x1x80xui8>
    %0 = memref.alloc() : memref<1x5x256x512xf16>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x5x256x512xf16>) outputs(%0 : memref<1x5x256x512xf16>) -> memref<1x5x256x512xf16>
    %2  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.Copy inputs(%1 : memref<1x5x256x512xf16>)
            outputs(%2 : !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
             -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %13 = VPUIP.Copy inputs(%cst : memref<1x1x1x80xui8>)
            outputs(%12 : !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
            -> !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %6:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK
                inputs(%3 as %arg1: !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                %13 as %arg7: !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                outputs(%4 as %arg2: !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                %5 as %arg3: !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
                -> (!VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
        VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg1, %arg7, %arg2, %arg3) :
        !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
        !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
        !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
                inputs(%6#1 as %arg3: !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
                outputs(%7 as %arg4: !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
     -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
        VPUIP.SW.Kernel.run(%arg3, %arg4) : !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    }
    %9 = memref.alloc() : memref<1x1x256x512xf16>
    %10 = VPUIP.Copy inputs(%8 : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            outputs(%9 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    return %10 : memref<1x1x256x512xf16>

    // CHECK:               [[CST:%.+]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 40]>]
    // CHECK:               [[CST0:%.+]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 40], [1, 1, 1, 40]>]
    // CHECK:               [[ALLOC0:%.+]] = memref.alloc() : memref<1x5x256x512xf16>
    // CHECK:               [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x5x256x512xf16>) outputs([[ALLOC0]] : memref<1x5x256x512xf16>) -> memref<1x5x256x512xf16>
    // CHECK:               [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 128, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:               [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:               [[TILING_INPUT0:%.+]] = VPUIP.Copy inputs([[SUBVIEW0]] : memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>)
    // CHECK-SAME:                                   outputs([[ALLOC1]]
    // CHECK-SAME:                                   -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:               [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:               [[ALLOC2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:               [[TILING_INPUT1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>)
    // CHECK-SAME:                                   outputs([[ALLOC2]]
    // CHECK-SAME:                                   > !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:               [[ALLOC11:%.+]] = VPURT.AllocDistributed
    // CHECK:                -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[TILING_AUX_INPUT_0:%.+]] = VPUIP.Copy inputs([[CST0]] : memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>)
    // CHECK-SAME:                                   outputs([[ALLOC11]]
    // CHECK{LITERAL}:      -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[ALLOC12:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[TILING_AUX_INPUT_1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>)
    // CHECK-SAME:                                   outputs([[ALLOC12]]
    // CHECK{LITERAL}:      -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:           [[ALLOC3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC5:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC6:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CLUSTER_TOPK:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK
    // CHECK:           inputs([[TILING_INPUT1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x5x128x512xf16,
    // CHECK:           [[TILING_AUX_INPUT_1]]   as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x40xui8,
    // CHECK:           [[TILING_INPUT0]]        as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x5x128x512xf16,
    // CHECK:           [[TILING_AUX_INPUT_0]]   as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x40xui8,
    // CHECK:           outputs([[ALLOC4]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xf16,
    // CHECK:           [[ALLOC6]]         as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xsi32,
    // CHECK:           [[ALLOC3]]         as [[OUT_2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xf16,
    // CHECK:           [[ALLOC5]]         as [[OUT_3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xsi32,
    // CHECK:           -> (!VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}([[INN_0]], [[INN_1]], [[OUT_0]], [[OUT_1]])
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}([[INN_2]], [[INN_3]], [[OUT_2]], [[OUT_3]])
    // CHECK:           }
    // CHECK:           [[ALLOC7:%.+]] = memref.alloc() : memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC7]] [0, 0, 0, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT0:%.+]] = VPUIP.Copy inputs([[CLUSTER_TOPK]]#1
    // CHECK-SAME:                                   outputs([[SUBVIEW2]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC7]] [0, 0, 128, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT1:%.+]] = VPUIP.Copy inputs([[CLUSTER_TOPK]]#3
    // CHECK-SAME:                                   outputs([[SUBVIEW3]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[TILING_OUTPUT0]], [[TILING_OUTPUT1]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>, memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) outputs([[ALLOC7]] : memref<1x1x256x512xsi32, @DDR>) -> memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[ALLOC8:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_OUTPUT2:%.+]] = VPUIP.Copy inputs([[CONCAT0]] : memref<1x1x256x512xsi32, @DDR>)
    // CHECK-SAME:                                   outputs([[ALLOC8]]
    // CHECK-SAME:                                   -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW4:%.+]] = VPUIP.SubView [[TILING_OUTPUT2]] [0, 0, 128, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW5:%.+]] = VPUIP.SubView [[TILING_OUTPUT2]] [0, 0, 0, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC9:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC9]] [0, 0, 128, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW7:%.+]] = VPUIP.SubView [[ALLOC9]] [0, 0, 0, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CLUSTER_CONVERT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Convert
    // CHECK-SAME:                               inputs([[SUBVIEW5]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]},
    // CHECK-SAME:                                      [[SUBVIEW4]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]},
    // CHECK-SAME:                              outputs([[SUBVIEW7]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]},
    // CHECK-SAME:                                      [[SUBVIEW6]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]},
    // CHECK-SAME:                              -> (!VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])
    // CHECK:           }
    // CHECK:           [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[CLUSTER_CONVERT]]#0, [[CLUSTER_CONVERT]]#1 : !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[ALLOC9]] : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC10:%.+]] = memref.alloc() : memref<1x1x256x512xf16>
    // CHECK:           [[TILING_OUTPUT3:%.+]] = VPUIP.Copy inputs([[CONCAT1]]
    // CHECK-SAME:                                      outputs([[ALLOC10]] : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    // CHECK:           return [[TILING_OUTPUT3]] : memref<1x1x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Sigmoid(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterSigmoid(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[SIGMOID:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:                       inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                              [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                             [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                       -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_0]], [[OUT_0]])
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]   = VPUIP.ConcatView inputs([[SIGMOID]]#0, [[SIGMOID]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]]  = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                        outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:     return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Log(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_log.cpp", VPU.kernel_entry = "activation_log"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterLog
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterLog(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Log
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:     [[CMX_IN_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[COPY_CMX:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>)
    // CHECK-SAME:                           outputs([[CMX_IN_BUF]]
    // CHECK-SAME:                   -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[PERMUTE_IN:%.+]] = VPUIP.PermuteCast {dst_order = #NHCW, mem_perm = #NHCW}
    // CHECK-SAME:          inputs([[COPY_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SUBVIEW_IN0:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_IN1:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CMX_OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CLUSTER_LOG:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Log
    // CHECK-SAME:            inputs([[SUBVIEW_IN1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:                   [[SUBVIEW_IN0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:          outputs([[SUBVIEW_OUT1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:                  [[SUBVIEW_OUT0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_0]], [[OUT_0]])
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_1]], [[OUT_1]])

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CLUSTER_LOG]]#0, [[CLUSTER_LOG]]#1
    // CHECK:     [[PERMUTE_OUT:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:          inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Sqrt(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_sqrt.cpp", VPU.kernel_entry = "activation_sqrt"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterSqrt
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterSqrt(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Sqrt
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]} (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:     [[CMX_IN_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[COPY_CMX:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>)
    // CHECK-SAME:                       outputs([[CMX_IN_BUF]]
    // CHECK-SAME:                   -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[PERMUTE_IN:%.+]] = VPUIP.PermuteCast {dst_order = #NHCW, mem_perm = #NHCW}
    // CHECK-SAME:          inputs([[COPY_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SUBVIEW_IN0:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_IN1:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CMX_OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CLUSTER_LOG:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sqrt
    // CHECK-SAME:            inputs([[SUBVIEW_IN1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:                   [[SUBVIEW_IN0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:          outputs([[SUBVIEW_OUT1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:                  [[SUBVIEW_OUT0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_0]], [[OUT_0]])
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_1]], [[OUT_1]])

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CLUSTER_LOG]]#0, [[CLUSTER_LOG]]#1
    // CHECK:     [[PERMUTE_OUT:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:          inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_DepthToSpace(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterSWDepthToSpace(%arg0: memref<1x128x12x270xf16, #NHWC>)
        -> memref<1x8x48x1080xf16, #NHWC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
        inputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg5 : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> {
            VPUIP.SW.Kernel.run {attrs = [4, 1]} (%arg4, %arg5) : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
        }
    %5 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)
        outputs(%5 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    return %6: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:                [[INPUT_BUF:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:      !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CLUSTER_COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>)
    // CHECK-SAME:     outputs([[INPUT_BUF]] : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[SUBVIEW0:%.+]] = VPUIP.SubView [[CLUSTER_COPY_INPUT]] [0, 0, 6, 0] [1, 128, 6, 270] : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[SUBVIEW1:%.+]] = VPUIP.SubView [[CLUSTER_COPY_INPUT]] [0, 0, 0, 0] [1, 128, 6, 270] : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:                [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    // CHECK:                [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 24, 0] [1, 8, 24, 1080] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> to !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    // CHECK:                [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 8, 24, 1080] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> to !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    // CHECK:                [[CLUSTER_D2S:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DepthToSpace
    // CHECK-SAME:                inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}
    // CHECK-SAME:                       [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}
    // CHECK-SAME:               outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}
    // CHECK-SAME:                       [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}
    // CHECK-SAME:               -> (!VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [4, 1]}([[INN_0]], [[OUT_0]])
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [4, 1]}([[INN_1]], [[OUT_1]])
    // CHECK:                }

    // CHECK:                [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:               inputs([[CLUSTER_D2S]]#0, [[CLUSTER_D2S]]#1 :
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
    // CHECK-SAME:               outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)
    // CHECK-SAME:                   -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    // CHECK:                [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:                [[CLUSTER_COPY_OUTPUT:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                                          outputs([[OUTPUT_DDR]] : memref<1x8x48x1080xf16, #NHWC>)
    // CHECK:                return [[CLUSTER_COPY_OUTPUT]] : memref<1x8x48x1080xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_DepthToSpace(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileClusterDMADepthToSpace(%arg0: memref<1x128x12x270xf16, #NHWC>)
        -> memref<1x8x48x1080xf16, #NHWC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
        inputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg5 : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> {
            VPUIP.SW.Kernel.run {attrs = [4, 0]} (%arg4, %arg5) : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
        }
    %5 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)
        outputs(%5 : memref<1x8x48x1080xf16, #NHWC>)  ->  memref<1x8x48x1080xf16, #NHWC>
    return %6: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:    [[INPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPYIN:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>)
    // CHECK-SAME:     outputs([[INPUT_BUF]] : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    // CHECK:    [[CLUSTER_D2S:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
    // CHECK-SAME:     inputs([[COPYIN]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 0]}([[INN_0]], [[OUT_0]])
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    [[COPYOUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CLUSTER_D2S]] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)
    // CHECK-SAME:     outputs([[OUTPUT_DDR]] : memref<1x8x48x1080xf16, #NHWC>)  ->  memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    return [[COPYOUT]] : memref<1x8x48x1080xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Clamp(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_clamp.cpp", VPU.kernel_entry = "activation_clamp"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterClamp(%arg0: memref<1x5x34x60xf16, #NHWC>)
        -> memref<1x5x34x60xf16, #NHWC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Clamp
        inputs(%1 as %arg5 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg6 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]} (%arg5, %arg6) : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x5x34x60xf16, #NHWC>)  ->  memref<1x5x34x60xf16, #NHWC>
    return %6: memref<1x5x34x60xf16, #NHWC>

    // CHECK:           [[INPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SHAPE_CAST_IN:%.+]] = VPUIP.ShapeCast {shape = [1, 1, 10200, 1]} inputs([[INPUT_COPY]] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[SHAPE_CAST_IN]] [0, 0, 5100, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[SHAPE_CAST_IN]] [0, 0, 0, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-DAG:       [[OUTPUT_BUF_SUBVIEW0:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 5100, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-DAG:       [[OUTPUT_BUF_SUBVIEW1:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[CLUSTER_CLAMP:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Clamp
    // CHECK-SAME:        inputs([[INPUT_SUBVIEW1]] as [[CLUSTER_CLAMP_ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]},
    // CHECK-SAME:               [[INPUT_SUBVIEW0]] as [[CLUSTER_CLAMP_ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]},
    // CHECK-SAME:        outputs([[OUTPUT_BUF_SUBVIEW1]] as [[CLUSTER_CLAMP_ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]},
    // CHECK-SAME:               [[OUTPUT_BUF_SUBVIEW0]] as [[CLUSTER_CLAMP_ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]},
    // CHECK-SAME:                -> (!VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}([[CLUSTER_CLAMP_ARG1]], [[CLUSTER_CLAMP_ARG3]])
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}([[CLUSTER_CLAMP_ARG2]], [[CLUSTER_CLAMP_ARG4]])
    // CHECK:               }

    // CHECK:           [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:                inputs([[CLUSTER_CLAMP]]#0, [[CLUSTER_CLAMP]]#1 :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SHAPE_CAST_OUT:%.+]] = VPUIP.ShapeCast {shape = [1, 5, 34, 60]} inputs([[CONCAT_VIEW]] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:           [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK:           return [[OUTPUT_COPY]] : memref<1x5x34x60xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Divide(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_div.cpp", VPU.kernel_entry = "eltwise_div"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterDivide() -> memref<1x64x1x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Divide

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x1x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x1x88xf16>)  ->  memref<1x64x1x88xf16>

    return %5: memref<1x64x1x88xf16>

    // For Divide First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // For Divide Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[DIVIDE_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[DIVIDE_OUT_TILE0:%.+]] = VPUIP.SubView [[DIVIDE_OUT]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[DIVIDE_OUT_TILE1:%.+]] = VPUIP.SubView [[DIVIDE_OUT]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[DIVIDE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Divide
    // CHECK:                          inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                                 [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                                 [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                                 [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                     outputs([[DIVIDE_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[DIVIDE_OUT_TILE0]]  as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[DIVIDE]]#0, [[DIVIDE]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[DIVIDE_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x1x88xf16>)  -> memref<1x64x1x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x1x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Power(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_power.cpp", VPU.kernel_entry = "eltwise_power"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterPower() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Power

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Power First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Power Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[POWER_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[POWER_OUT_TILE0:%.+]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER_OUT_TILE1:%.+]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Power
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                 outputs([[POWER_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                         [[POWER_OUT_TILE0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[POWER]]#0, [[POWER]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[POWER_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Power(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_power.cpp", VPU.kernel_entry = "eltwise_power"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterPowerAtBroadcastAxis() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Power

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Power First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Power Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[POWER_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[POWER_OUT_TILE0:%.+]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER_OUT_TILE1:%.+]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Power
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN
    // CHECK:                            [[INPUT1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN
    // CHECK:                            [[INPUT1]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                 outputs([[POWER_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN
    // CHECK:                         [[POWER_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[POWER]]#0, [[POWER]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[POWER_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Greater(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_greater.cpp", VPU.kernel_entry = "eltwise_greater"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterGreater
func.func @TileClusterGreater() -> memref<1x64x1x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Greater

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x1x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x1x88xf16>)  ->  memref<1x64x1x88xf16>

    return %5: memref<1x64x1x88xf16>

    // For Greater First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // For Greater Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[GREATER_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[GREATER_OUT_TILE0:%.+]] = VPUIP.SubView [[GREATER_OUT]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[GREATER_OUT_TILE1:%.+]] = VPUIP.SubView [[GREATER_OUT]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[GREATER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Greater
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:               outputs([[GREATER_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                      [[GREATER_OUT_TILE0]]  as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[GREATER]]#0, [[GREATER]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[GREATER_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x1x88xf16>)  -> memref<1x64x1x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x1x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Less(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_less.cpp", VPU.kernel_entry = "eltwise_less"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterLess
func.func @TileClusterLess() -> memref<1x64x1x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Less

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x1x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x1x88xf16>)  ->  memref<1x64x1x88xf16>

    return %5: memref<1x64x1x88xf16>

    // For Less First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // For Less Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[LESS_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[LESS_OUT_TILE0:%.+]] = VPUIP.SubView [[LESS_OUT]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[LESS_OUT_TILE1:%.+]] = VPUIP.SubView [[LESS_OUT]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[LESS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Less
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                  outputs([[LESS_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                          [[LESS_OUT_TILE0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[LESS]]#0, [[LESS]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[LESS_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x1x88xf16>)  -> memref<1x64x1x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x1x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Equal(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_equal.cpp", VPU.kernel_entry = "eltwise_equal"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterEqual
func.func @TileClusterEqual() -> memref<1x64x1x88xi8> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Equal

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x1x88xi8>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x1x88xi8>)  ->  memref<1x64x1x88xi8>

    return %5: memref<1x64x1x88xi8>

    // For Equal First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // For Equal Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[EQUAL_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[EQUAL_OUT_TILE0:%.+]] = VPUIP.SubView [[EQUAL_OUT]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[EQUAL_OUT_TILE1:%.+]] = VPUIP.SubView [[EQUAL_OUT]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[EQUAL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Equal
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INPUT0_TILE1_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INPUT1_TILE1_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INPUT0_TILE0_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INPUT1_TILE0_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                 outputs([[EQUAL_OUT_TILE1]] as [[EQUAL_OUT_TILE1_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                         [[EQUAL_OUT_TILE0]] as [[EQUAL_OUT_TILE0_ARG:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE1_ARG]], [[INPUT1_TILE1_ARG]], [[EQUAL_OUT_TILE1_ARG]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE0_ARG]], [[INPUT1_TILE0_ARG]], [[EQUAL_OUT_TILE0_ARG]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[EQUAL]]#0, [[EQUAL]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[EQUAL_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xi8>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x1x88xi8>)  -> memref<1x64x1x88xi8>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x1x88xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Subtract(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_sub.cpp", VPU.kernel_entry = "eltwise_sub"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterSubtract() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Subtract

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Subtract First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Subtract Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[SUBTRACT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Subtract
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                 outputs([[MULTI_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                         [[MULTI_OUT_TILE0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[SUBTRACT]]#0, [[SUBTRACT]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Subtract(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_sub.cpp", VPU.kernel_entry = "eltwise_sub"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterSubtractAtBroadcastAxis() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Subtract

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Subtract First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Subtract Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[SUBTRACT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Subtract
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}
    // CHECK:                            [[INPUT1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}
    // CHECK:                            [[INPUT1]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                 outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}
    // CHECK:                         [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[SUBTRACT]]#0, [[SUBTRACT]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Subtract(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_sub.cpp", VPU.kernel_entry = "eltwise_sub"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterSubtractWithSameInputs() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Subtract

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %0 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%1 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %3 = memref.alloc() : memref<1x64x88x88xf16>
    %4 = VPUIP.Copy
        inputs(%2 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %4: memref<1x64x88x88xf16>

    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[SUBTRACT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Subtract
    // CHECK:                     inputs([[INPUT1_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                 outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                         [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[SUBTRACT]]#0, [[SUBTRACT]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterAdd() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Add

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Add First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Add Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[ADD:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE1]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[INN_2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[INN_3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                 outputs([[MULTI_OUT_TILE1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                         [[MULTI_OUT_TILE0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[INN_1]], [[OUT_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INN_2]], [[INN_3]], [[OUT_1]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD]]#0, [[ADD]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterAddAtBroadcastAxis() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Add

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %5: memref<1x64x88x88xf16>

    // For Add First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Add Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[ADD:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x88xf16,
    // CHECK:                 outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                         [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD]]#0, [[ADD]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterAddWithSameInputs() -> memref<1x64x88x88xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Add
        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %0 as %arg4 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%1 as %arg5 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %3 = memref.alloc() : memref<1x64x88x88xf16>
    %4 = VPUIP.Copy
        inputs(%2 : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 : memref<1x64x88x88xf16>)  ->  memref<1x64x88x88xf16>

    return %4: memref<1x64x88x88xf16>

    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.+]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[ADD:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK:                     inputs([[INPUT1_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                 outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                         [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD]]#0, [[ADD]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCATVIEW]]
    // CHECK-SAME:     outputs([[OUTPUT_BUF]] : memref<1x64x88x88xf16>)  -> memref<1x64x88x88xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x64x88x88xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Abs(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_abs.cpp", VPU.kernel_entry = "activation_abs"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterAbs(%arg0: memref<1x5x34x60xf16, #NHWC>)
        -> memref<1x5x34x60xf16, #NHWC> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Abs
        inputs(%1 as %arg5 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg6 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg5, %arg6) : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x5x34x60xf16, #NHWC>)  ->  memref<1x5x34x60xf16, #NHWC>
    return %6: memref<1x5x34x60xf16, #NHWC>

    // CHECK:           [[INPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>)
    // CHECK-SAME:     outputs([[INPUT_BUF]] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 18, 0] [1, 5, 16, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 5, 18, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[OUTPUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_BUF_SUBVIEW0:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 18, 0] [1, 5, 16, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_BUF_SUBVIEW1:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 5, 18, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[CLUSTER_ABS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Abs
    // CHECK-SAME:                inputs([[INPUT_SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]},
    // CHECK-SAME:                       [[INPUT_SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]},
    // CHECK-SAME:          outputs([[OUTPUT_BUF_SUBVIEW1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]},
    // CHECK-SAME:                  [[OUTPUT_BUF_SUBVIEW0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]},
    // CHECK-SAME:                -> (!VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])
    // CHECK:               }

    // CHECK:           [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:                inputs([[CLUSTER_ABS]]#0, [[CLUSTER_ABS]]#1 :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:           [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:                inputs([[CONCAT_VIEW]]
    // CHECK-SAME:                outputs([[OUTPUT_DDR]] : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    // CHECK:           return [[OUTPUT_COPY]] : memref<1x5x34x60xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_PRelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "prelu_fp16.cpp", VPU.kernel_entry = "prelu_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterPRelu() -> memref<1x18x160x288xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_PRelu

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, %1 as %arg4 : !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        outputs(%2 as %arg5 : !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg3, %arg4, %arg5) : !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x18x160x288xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x18x160x288xf16>)  ->  memref<1x18x160x288xf16>

    return %5: memref<1x18x160x288xf16>

    // For PRelu First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 80, 0] [1, 18, 80, 288]
    // CHECK:         !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 18, 80, 288]
    // CHECK:         !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For PRelu Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[PRELU_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[PRELU_OUT_TILE0:%.+]] = VPUIP.SubView [[PRELU_OUT]] [0, 0, 80, 0] [1, 18, 80, 288]
    // CHECK:    [[PRELU_OUT_TILE1:%.+]] = VPUIP.SubView [[PRELU_OUT]] [0, 0, 0, 0] [1, 18, 80, 288]
    // CHECK:    [[PRELU:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PRelu
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]},
    // CHECK:                            [[INPUT1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x18x1x1xf16,
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]},
    // CHECK:                            [[INPUT1]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x18x1x1xf16,
    // CHECK:                 outputs([[PRELU_OUT_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]},
    // CHECK:                         [[PRELU_OUT_TILE0]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]},
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]])

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[PRELU]]#0, [[PRELU]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     outputs([[PRELU_OUT]] : !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x18x160x288xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCATVIEW]]
    // CHECK:                                                 outputs([[OUTPUT_BUF]] : memref<1x18x160x288xf16>) -> memref<1x18x160x288xf16>

    // CHECK:    return [[OUTPUT_COPY]] : memref<1x18x160x288xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Floor(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_floor.cpp", VPU.kernel_entry = "activation_floor"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterFloor() -> memref<1x16x16x512xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Floor

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x16x16x512xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x16x16x512xf16>)  ->  memref<1x16x16x512xf16>

    return %5 : memref<1x16x16x512xf16>

    // CHECK:    [[IN_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV0:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 8, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 8, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[FLOOR:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Floor
    // CHECK-SAME:           inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]},
    // CHECK-SAME:                  [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]},
    // CHECK-SAME:         outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]},
    // CHECK-SAME:                 [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]},
    // CHECK-SAME:         -> (!VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]   = VPUIP.ConcatView inputs([[FLOOR]]#0, [[FLOOR]]#1 : !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]]  = memref.alloc() : memref<1x16x16x512xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                      outputs([[OUT_DDR]] : memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16>
    // CHECK:    return [[OUT_COPY]] : memref<1x16x16x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Round(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "round_fp16.cpp", VPU.kernel_entry = "round_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterRound
func.func @TileClusterRound() -> memref<1x16x16x512xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Round

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x16x16x512xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x16x16x512xf16>)  ->  memref<1x16x16x512xf16>

    return %5 : memref<1x16x16x512xf16>

    // CHECK:     [[CMX_IN_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[PERMUTE_IN:%.+]] = VPUIP.PermuteCast {dst_order = #NHCW, mem_perm = #NHCW}
    // CHECK-SAME:          inputs([[CMX_IN_BUF]] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SUBVIEW_IN0:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 8, 0] [1, 16, 8, 512]
    // CHECK:     [[SUBVIEW_IN1:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 0, 0] [1, 16, 8, 512]

    // CHECK:     [[CMX_OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 8, 0] [1, 16, 8, 512]
    // CHECK:     [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 0, 0] [1, 16, 8, 512]

    // CHECK:     [[CLUSTER_LOG:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Round
    // CHECK-SAME:          inputs([[SUBVIEW_IN1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]},
    // CHECK-SAME:                 [[SUBVIEW_IN0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]},
    // CHECK-SAME:        outputs([[SUBVIEW_OUT1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]},
    // CHECK-SAME:                [[SUBVIEW_OUT0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]},
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_0]], [[OUT_0]])
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[INN_1]], [[OUT_1]])

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CLUSTER_LOG]]#0, [[CLUSTER_LOG]]#1
    // CHECK:     [[PERMUTE_OUT:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:          inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Sin(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_sin.cpp", VPU.kernel_entry = "activation_sin"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterSin
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterSin(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Sin
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                  -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[SIN:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sin
    // CHECK-SAME:               inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                      [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:             outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:            -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[SIN]]#0, [[SIN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                   outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Cos(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_cos.cpp", VPU.kernel_entry = "activation_cos"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterCos
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterCos(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Cos
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                 -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[COS:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Cos
    // CHECK-SAME:               inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                      [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:             outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[COS]]#0, [[COS]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                   outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Exp(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_exp.cpp", VPU.kernel_entry = "activation_exp"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterExp
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterExp(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Exp
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                 -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[EXP:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Exp
    // CHECK-SAME:              inputs([[IN_SV1]]  as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     [[IN_SV0]]  as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:             outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[EXP]]#0, [[EXP]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                  outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_LogSoftmax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "log_softmax.cpp", VPU.kernel_entry = "log_softmax"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileLogSoftmax() -> memref<1x16x16x512xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_LogSoftmax

        inputs(%0 as %arg3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%1 as %arg4 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [2]} (%arg3, %arg4) : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }

    %4 = memref.alloc() : memref<1x16x16x512xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%4 : memref<1x16x16x512xf16>)  ->  memref<1x16x16x512xf16>

    return %5 : memref<1x16x16x512xf16>

    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1x16x16x512xf16, @DDR>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]]
    // CHECK-SAME:                  outputs([[ALLOC0]] : memref<1x16x16x512xf16, @DDR>) -> memref<1x16x16x512xf16, @DDR>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 8, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16, @DDR> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>
    // CHECK:    [[BUF0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0_TILE0]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>)
    // CHECK-SAME:                           outputs([[BUF0]]
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x16x512xf16, @DDR>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT0]]
    // CHECK-SAME:                outputs([[ALLOC1]] : memref<1x16x16x512xf16, @DDR>) -> memref<1x16x16x512xf16, @DDR>
    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16, @DDR> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>
    // CHECK:    [[BUF1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1_TILE1]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>)
    // CHECK-SAME:                       outputs([[BUF1]]
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[BUF9:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[BUF10:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[CLUSTER_LOG_SOFTMAX:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LogSoftmax
    // CHECK-SAME:                inputs([[INPUT1_COPY1]] as [[COPY_ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16,
    // CHECK-SAME:                       [[INPUT0_COPY0]] as [[COPY_ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16,
    // CHECK-SAME:                outputs([[BUF10]]       as [[COPY_ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16,
    // CHECK-SAME:                               [[BUF9]] as [[COPY_ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x16x8x512xf16,
    // CHECK-SAME:                -> (!VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                      VPUIP.SW.Kernel.run {attrs = [2]}([[COPY_ARG0]], [[COPY_ARG2]])
    // CHECK:                      VPUIP.SW.Kernel.run {attrs = [2]}([[COPY_ARG1]], [[COPY_ARG3]])
    // CHECK:    }

    // CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<1x16x16x512xf16>
    // CHECK:    [[SUBVIEW12:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    [[COPY13:%.+]] = VPUIP.Copy inputs([[CLUSTER_LOG_SOFTMAX]]#0
    // CHECK-SAME:                   outputs([[SUBVIEW12]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>)
    // CHECK-SAME:                -> memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>

    // CHECK:    [[SUBVIEW14:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 8, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    [[COPY15:%.+]] = VPUIP.Copy inputs([[CLUSTER_LOG_SOFTMAX]]#1
    // CHECK-SAME:                      outputs([[SUBVIEW14]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>)
    // CHECK-SAME:                -> memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView inputs([[COPY13]], [[COPY15]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>, memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>) outputs(%alloc_1 : memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16>

    // CHECK:    return [[CONCAT_VIEW]] : memref<1x16x16x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileImplicitDistributedSoftmax(%arg0: memref<1x4096x18x4xf16, #NHWC>) -> memref<1x4096x18x4xf16, #NHWC> {
    %in_alloc = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %in_copy = VPUIP.Copy
        inputs(%arg0 : memref<1x4096x18x4xf16, #NHWC>)
        outputs(%in_alloc : !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %out_alloc = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %sw = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        inputs(%in_copy as %arg4 : !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%out_alloc as %arg5 : !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run {attrs = [0, 0]} (%arg4, %arg5) : !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %res_alloc = memref.alloc() : memref<1x4096x18x4xf16, #NHWC>
    %out_copy = VPUIP.Copy
        inputs(%sw : !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%res_alloc : memref<1x4096x18x4xf16, #NHWC>) -> memref<1x4096x18x4xf16, #NHWC>
    return %out_copy : memref<1x4096x18x4xf16, #NHWC>

    // CHECK:   [[IN_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK:   [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME:      shape = [1, 4096, 72, 1]
    // CHECK-SAME:      inputs([[IN_COPY]]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x4096x72x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-DAG:   [[IN_SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 4096, 36, 1]
    // CHECK-DAG:   [[IN_SUBVIEW_2:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 36, 0] [1, 4096, 36, 1]
    // CHECK:   [[OUT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x72x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-DAG:   [[OUT_SUBVIEW_2:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 36, 0] [1, 4096, 36, 1]
    // CHECK-DAG:   [[OUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0, 0] [1, 4096, 36, 1]
    // CHECK:   [[SW_CLUSTERING:%.+]]:2 =  VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:          [[IN_SUBVIEW_1]] as [[SW_CLUSTERING_ARG1:[^:]+]]
    // CHECK-SAME:          [[IN_SUBVIEW_2]] as [[SW_CLUSTERING_ARG2:[^:]+]]
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}([[SW_CLUSTERING_ARG1]],
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}([[SW_CLUSTERING_ARG2]],
    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_CLUSTERING]]#0, [[SW_CLUSTERING]]#1
    // CHECK:   [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 4096, 18, 4]} inputs([[CONCAT]]
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:                  inputs([[SHAPE_CAST]] : !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2
    // CHECK-SAME:                  -> memref<1x4096x18x4xf16, #NHWC>
    // CHECK:   return [[OUT_COPY]] : memref<1x4096x18x4xf16, #NHWC>
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Mish(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_mish.cpp", VPU.kernel_entry = "activation_mish"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterMish
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterMish(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Mish
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                 -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MISH:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Mish
    // CHECK-SAME:                  inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                         [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                        [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                 -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                 !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MISH]]#0, [[MISH]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                 outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Negative(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_negative.cpp", VPU.kernel_entry = "activation_negative"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterNegative
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterNegative(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Negative
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                  -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[NEGATIVE:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Negative
    // CHECK-SAME:               inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                      [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:             outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:            -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[NEGATIVE]]#0, [[NEGATIVE]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                   outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_LogicalNot(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_logical_not.cpp", VPU.kernel_entry = "eltwise_logical_not"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterLogicalNot
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterLogicalNot(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_LogicalNot
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                  -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[PERMUTE_IN:%.+]] = VPUIP.PermuteCast {dst_order = #NHCW, mem_perm = #NHCW}
    // CHECK-SAME:          inputs([[IN_COPY]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 32, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 0, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 32, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[LOGICALNOT:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LogicalNot
    // CHECK-SAME:               inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:                      [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:             outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:                     [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]},
    // CHECK-SAME:            -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[LOGICALNOT]]#0, [[LOGICALNOT]]#1 : !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[PERMUTE_OUT:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:          inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTE_OUT]]
    // CHECK-SAME:                   outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_ReLU(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterReLU
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x128x64x32xf16>
func.func @TileClusterReLU(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_ReLU
        inputs(%1 as %arg6 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%3 as %arg7 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  on tile 0 ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPUIP.SW.Kernel.run (%arg6, %arg7) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x128x64x32xf16>) outputs([[IN_BUF]]
    // CHECK-SAME:                  -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[RELU:%.+]]:2  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReLU
    // CHECK-SAME:               inputs([[IN_SV1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                      [[IN_SV0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:             outputs([[OUT_SV1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:                     [[OUT_SV0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK-SAME:            -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_0]], [[OUT_0]])
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}([[INN_1]], [[OUT_1]])

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[RELU]]#0, [[RELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                   outputs([[OUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}
