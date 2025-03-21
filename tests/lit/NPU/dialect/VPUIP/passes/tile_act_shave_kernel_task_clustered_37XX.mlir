//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW  allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithCAlignment(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }

    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NoTileSegmentedShaveWithCAlignment(%arg0: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x64x32xf16>) outputs(%0 as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16>) outputs(%arg2 : memref<1x64x64x32xf16, @CMX_NN>) -> memref<1x64x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x64x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x64x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x64x64x32xf16, @CMX_NN>, memref<1x64x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x64x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    }

    return %6: memref<1x64x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16>) outputs(%arg2 : memref<1x64x64x32xf16, @CMX_NN>) -> memref<1x64x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[MVN:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY0]] as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs([[OUTPUT_CMX]] as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x64x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x64x64x32xf16, @CMX_NN>{
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x64x64x32xf16, @CMX_NN>, memref<1x64x64x32xf16, @CMX_NN>
    // CHECK-NOT:                      VPUIP.SW.Kernel.run
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x64x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[MVN]] as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x64x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithCAlignment(%arg0: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x64x32xf16>) outputs(%0 as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16>) outputs(%arg2 : memref<1x32x64x32xf16, @CMX_NN>) -> memref<1x32x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x32x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x32x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x32x64x32xf16, @CMX_NN>, memref<1x32x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x32x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    }

    return %6: memref<1x32x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16>) outputs(%arg2 : memref<1x32x64x32xf16, @CMX_NN>) -> memref<1x32x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) strides([[65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1]]) on tile 0 -> (memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x32x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x32x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NoTileDuplicatedShaveWithCAlignment(%arg0: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x32xf16>) outputs(%0 as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16>) outputs(%arg2 : memref<1x16x64x32xf16, @CMX_NN>) -> memref<1x16x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x16x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x16x64x32xf16, @CMX_NN>, memref<1x16x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x16x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    }

    return %6: memref<1x16x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16>) outputs(%arg2 : memref<1x16x64x32xf16, @CMX_NN>) -> memref<1x16x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[MVN:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY0]] as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs([[OUTPUT_CMX]] as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK{LITERAL}:      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x16x64x32xf16, @CMX_NN>{
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x16x64x32xf16,  @CMX_NN>, memref<1x16x64x32xf16,  @CMX_NN>
    // CHECK-NOT:                      VPUIP.SW.Kernel.run
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x16x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[MVN]] as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16, #NWHC>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x64x32xf16, #NWHC, @CMX_NN>, memref<1x128x64x32xf16, #NWHC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    }
    return %6: memref<1x128x64x32xf16, #NWHC>


    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>) strides([[262144, 1, 128, 8192], [262144, 1, 128, 8192]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs  = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
    // CHECK:      }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs([[BUFF_OUT_DDR]] as %arg2: memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x33x16x1xf16>) outputs(%0 as %arg2: memref<1x33x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16>) outputs(%arg2 : memref<1x33x16x1xf16, @CMX_NN>) -> memref<1x33x16x1xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x33x16x1xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x33x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x33x16x1xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x33x16x1xf16, @CMX_NN>) on tile 0 -> memref<1x33x16x1xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x33x16x1xf16, @CMX_NN>, memref<1x33x16x1xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x33x16x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x33x16x1xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x33x16x1xf16>) -> memref<1x33x16x1xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x33x16x1xf16>) -> memref<1x33x16x1xf16>
    }
    return %6: memref<1x33x16x1xf16>


    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x33x16x1xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x33x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16>) outputs(%arg2 : memref<1x33x16x1xf16, @CMX_NN>) -> memref<1x33x16x1xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>) strides([[272, 16, 1, 1], [256, 16, 1, 1]]) on tile 0 -> (memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, memref<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>, memref<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NCHW, strides = [528, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x33x16x1xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x33x16x1xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x33x16x1xf16>) -> memref<1x33x16x1xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x33x16x1xf16>) -> memref<1x33x16x1xf16>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x33x16x1xf16, #NHWC>) outputs(%0 as %arg2: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16, #NHWC>) outputs(%arg2 : memref<1x33x16x1xf16, #NHWC, @CMX_NN>) -> memref<1x33x16x1xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x33x16x1xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x33x16x1xf16, #NHWC, @CMX_NN>, memref<1x33x16x1xf16, #NHWC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x33x16x1xf16, #NHWC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg2: memref<1x33x16x1xf16, #NHWC>) -> memref<1x33x16x1xf16, #NHWC> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x33x16x1xf16, #NHWC>) -> memref<1x33x16x1xf16, #NHWC>
    }
    return %6: memref<1x33x16x1xf16, #NHWC>


    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x33x16x1xf16, #NHWC>) outputs([[INPUT_CMX]] as %arg2: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:             VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16, #NHWC>) outputs(%arg2 : memref<1x33x16x1xf16, #NHWC, @CMX_NN>) -> memref<1x33x16x1xf16, #NHWC, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 18, 0, 0] [1, 15, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 18, 16, 1] : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
    // CHECK:                     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, %arg2 as %arg6: memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, %arg4 as %arg8: memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>)
    // CHECK{LITERAL}:              strides([[272, 1, 17, 17], [256, 1, 16, 16]]) on tile 0 -> (memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, memref<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>, memref<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN>
    // CHECK:                     }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x18x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x15x16x1xf16, {order = #NHWC, strides = [528, 1, 33, 33]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x33x16x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x33x16x1xf16, #NHWC>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x33x16x1xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x33x16x1xf16, #NHWC>) -> memref<1x33x16x1xf16, #NHWC> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x33x16x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x33x16x1xf16, #NHWC>) -> memref<1x33x16x1xf16, #NHWC>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x1x10240xf16>) outputs(%0 as %arg2: memref<1x32x1x10240xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x32x1x10240xf16>) outputs(%arg2 : memref<1x32x1x10240xf16, @CMX_NN>) -> memref<1x32x1x10240xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x32x1x10240xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x32x1x10240xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x32x1x10240xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x32x1x10240xf16, @CMX_NN>) on tile 0 -> memref<1x32x1x10240xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x32x1x10240xf16, @CMX_NN>, memref<1x32x1x10240xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x32x1x10240xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x32x1x10240xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x32x1x10240xf16>) -> memref<1x32x1x10240xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x32x1x10240xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x1x10240xf16>) -> memref<1x32x1x10240xf16>
    }
    return %6: memref<1x32x1x10240xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x1x10240xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x32x1x10240xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x32x1x10240xf16>) outputs(%arg2 : memref<1x32x1x10240xf16, @CMX_NN>) -> memref<1x32x1x10240xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[MVN:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY0]] as %arg1: memref<1x32x1x10240xf16, @CMX_NN>) outputs([[OUTPUT_CMX]] as %arg2: memref<1x32x1x10240xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x1x10240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x32x1x10240xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x32x1x10240xf16, @CMX_NN>) on tile 0 -> memref<1x32x1x10240xf16, @CMX_NN>{
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x32x1x10240xf16, @CMX_NN>, memref<1x32x1x10240xf16, @CMX_NN>
    // CHECK:      }
    // CHECK:    }
    // CHECK:    [[BUFF_OUT_DDR:%.*]] = memref.alloc() : memref<1x32x1x10240xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NCEClusterTiling inputs([[MVN]] as %arg1: memref<1x32x1x10240xf16, @CMX_NN>) outputs([[BUFF_OUT_DDR]] as %arg2: memref<1x32x1x10240xf16>) -> memref<1x32x1x10240xf16> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x32x1x10240xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x1x10240xf16>) -> memref<1x32x1x10240xf16>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1x96x160xf16>) outputs(%0 as %arg3: memref<1x1x96x160xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x96x160xf16>) outputs(%arg3 : memref<1x1x96x160xf16, @CMX_NN>) -> memref<1x1x96x160xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x1x96x160xf16, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x192x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg2 as %arg4: memref<1x1x96x160xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x192x320xf16, @CMX_NN>) on tile 0 -> memref<1x1x192x320xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : memref<1x1x96x160xf16, @CMX_NN>, memref<1x1x192x320xf16, @CMX_NN>
      }
    }
    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1x192x320xf16, @CMX_NN>) outputs(%4 as %arg3: memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x192x320xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    }
    return %5 : memref<1x1x192x320xf16>

    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}:     VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUF0]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK:                       VPUIP.Copy inputs(%arg1 : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs(%arg2 : memref<1x1x49x160xf16, @CMX_NN>) -> memref<1x1x49x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUF1]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK:                       VPUIP.Copy inputs(%arg1 : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs(%arg2 : memref<1x1x49x160xf16, @CMX_NN>) -> memref<1x1x49x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUF1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUF0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CLUSTER_INTERPOLATE:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[COPY1]] as %arg1: memref<1x1x49x160xf16, @CMX_NN>, [[COPY0]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>) outputs([[OUTPUT_BUF0]] as %arg3: memref<1x1x96x320xf16, @CMX_NN>, [[OUTPUT_BUF1]] as %arg4: memref<1x1x96x320xf16, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                   %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg1 as %arg5: memref<1x1x49x160xf16, @CMX_NN>, %arg2 as %arg6: memref<1x1x49x160xf16, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x1x96x320xf16, @CMX_NN>, %arg4 as %arg8: memref<1x1x96x320xf16, @CMX_NN>) on tile 0 -> (memref<1x1x96x320xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>){
    // CHECK:                     VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg5, %arg7) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:                     VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg6, %arg8) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:                   }
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY2:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_INTERPOLATE]]#0 as %arg1: memref<1x1x96x320xf16, @CMX_NN>) outputs([[SUBVIEW2]] as %arg2: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x96x320xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    }
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY3:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_INTERPOLATE]]#1 as %arg1: memref<1x1x96x320xf16, @CMX_NN>) outputs([[SUBVIEW3]] as %arg2: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x96x320xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY2]], [[COPY3]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) outputs([[OUTPUT_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1x96x160xf16>) outputs(%0 as %arg3: memref<1x1x96x160xf16, @CMX_NN>)
        -> !InputDistributedBuffer {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x96x160xf16>) outputs(%arg3 : memref<1x1x96x160xf16, @CMX_NN>) -> memref<1x1x96x160xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !OutputDistributedBuffer
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x1x96x160xf16, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x192x320xf16, @CMX_NN>)
        -> !OutputDistributedBuffer {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate
        inputs(%arg2 as %arg4: memref<1x1x96x160xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x192x320xf16, @CMX_NN>) on tile 0
          -> memref<1x1x192x320xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : memref<1x1x96x160xf16, @CMX_NN>, memref<1x1x192x320xf16, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1x192x320xf16, @CMX_NN>) outputs(%4 as %arg3: memref<1x1x192x320xf16>)
        -> memref<1x1x192x320xf16> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x192x320xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    }
    return %5 : memref<1x1x192x320xf16>

    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[ARG0]] [0, 0, 47, 0] [1, 1, 49, 160] :
    // CHECK-SAME:  memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>

    // CHECK:    [[INPUT_BUF0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>

    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[SUBVIEW0]] as [[IN_ARG0:%.*]]: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:  outputs([[INPUT_BUF0]] as [[IN_ARG1:%.*]]: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK-NEXT:     VPUIP.Copy

    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1, 49, 160] :
    // CHECK-SAME:  memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>

    // CHECK:    [[INPUT_BUF1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>

    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[SUBVIEW1]] as [[IN_ARG2:%.*]]: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>)
    // CHECK-SAME:  outputs([[INPUT_BUF1]] as [[IN_ARG3:%.*]]: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK-NEXT:     VPUIP.Copy

    // CHECK:    [[OUTPUT_BUF1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]]}>

    // CHECK:    [[OUTPUT_BUF0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 48, 320], [1, 1, 48, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0]]}>

    // CHECK:    [[CLUSTER_INTERPOLATE:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[COPY1]] as [[IN_ARG4:%.*]]: memref<1x1x49x160xf16, @CMX_NN>,
    // CHECK-SAME:           [[COPY0]] as [[IN_ARG5:%.*]]: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK-SAME:    outputs([[OUTPUT_BUF0]] as [[IN_ARG6:%.*]]: memref<1x1x96x320xf16, @CMX_NN>,
    // CHECK-SAME:            [[OUTPUT_BUF1]] as [[IN_ARG7:%.*]]: memref<1x1x96x320xf16, @CMX_NN>)
    // CHECK:       %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate
    // CHECK-SAME:      inputs([[IN_ARG4]] as [[IN_ARG8:%.*]]: memref<1x1x49x160xf16, @CMX_NN>,
    // CHECK-SAME:             [[IN_ARG5]] as [[IN_ARG9:%.*]]: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK-SAME:      outputs([[IN_ARG6]] as [[IN_ARG10:%.*]]: memref<1x1x96x320xf16, @CMX_NN>,
    // CHECK-SAME:              [[IN_ARG7]] as [[IN_ARG11:%.*]]: memref<1x1x96x320xf16, @CMX_NN>) on tile 0
    // CHECK-SAME:    -> (memref<1x1x96x320xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>){
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK-SAME:      {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}
    // CHECK-SAME:      ([[IN_ARG8]], [[IN_ARG10]]) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK-SAME:      {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}
    // CHECK-SAME:      ([[IN_ARG9]], [[IN_ARG11]]) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>


    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 96, 320] :
    // CHECK-SAME:  memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>

    // CHECK:    [[COPY2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[CLUSTER_INTERPOLATE]]#0 as [[IN_ARG12:%.*]]: memref<1x1x96x320xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[SUBVIEW2]] as [[IN_ARG13:%.*]]: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>)
    // CHECK-SAME:    -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK-NEXT:           VPUIP.Copy

    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 96, 0] [1, 1, 96, 320] :
    // CHECK-SAME:    memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>

    // CHECK:    [[COPY3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[CLUSTER_INTERPOLATE]]#1 as [[IN_ARG14:%.*]]: memref<1x1x96x320xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[SUBVIEW3]] as [[IN_ARG15:%.*]]: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>)
    // CHECK-SAME:    -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK-NEXT:        VPUIP.Copy

    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1x96x160xf16>) outputs(%0 as %arg3: memref<1x1x96x160xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x96x160xf16>) outputs(%arg3 : memref<1x1x96x160xf16, @CMX_NN>) -> memref<1x1x96x160xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x1x96x160xf16, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x192x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg2 as %arg4: memref<1x1x96x160xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x192x320xf16, @CMX_NN>) on tile 0 -> memref<1x1x192x320xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : memref<1x1x96x160xf16, @CMX_NN>, memref<1x1x192x320xf16, @CMX_NN>
      }
    }
    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1x192x320xf16, @CMX_NN>) outputs(%4 as %arg3: memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x192x320xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    }
    return %5 : memref<1x1x192x320xf16>


    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME{LITERAL}:     VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[COPY2:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUF0]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK:                       VPUIP.Copy inputs(%arg1 : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs(%arg2 : memref<1x1x49x160xf16, @CMX_NN>) -> memref<1x1x49x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUF1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[COPY4:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUF1]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>)
    // CHECK:                       VPUIP.Copy inputs(%arg1 : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs(%arg2 : memref<1x1x49x160xf16, @CMX_NN>) -> memref<1x1x49x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUF1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUF0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CLUSTER_INTERPOLATE:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[COPY4]] as %arg1: memref<1x1x49x160xf16, @CMX_NN>, [[COPY2]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>) outputs([[OUTPUT_BUF0]] as %arg3: memref<1x1x96x320xf16, @CMX_NN>, [[OUTPUT_BUF1]] as %arg4: memref<1x1x96x320xf16, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                   %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg1 as %arg5: memref<1x1x49x160xf16, @CMX_NN>, %arg2 as %arg6: memref<1x1x49x160xf16, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x1x96x320xf16, @CMX_NN>, %arg4 as %arg8: memref<1x1x96x320xf16, @CMX_NN>) on tile 0 -> (memref<1x1x96x320xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>){
    // CHECK:                     VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg5, %arg7) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:                     VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg6, %arg8) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:                   }
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY5:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_INTERPOLATE]]#0 as %arg1: memref<1x1x96x320xf16, @CMX_NN>) outputs([[SUBVIEW2]] as %arg2: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x96x320xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    }
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[COPY6:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_INTERPOLATE]]#1 as %arg1: memref<1x1x96x320xf16, @CMX_NN>) outputs([[SUBVIEW3]] as %arg2: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x96x320xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY5]], [[COPY6]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) outputs([[OUTPUT_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1x96x160xf16>) outputs(%0 as %arg3: memref<1x1x96x160xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x96x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]], memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]}> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x96x160xf16>) outputs(%arg3 : memref<1x1x96x160xf16, @CMX_NN>) -> memref<1x1x96x160xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x1x96x160xf16, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x192x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg2 as %arg4: memref<1x1x96x160xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x192x320xf16, @CMX_NN>) on tile 0 -> memref<1x1x192x320xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : memref<1x1x96x160xf16, @CMX_NN>, memref<1x1x192x320xf16, @CMX_NN>
      }
    }
    %4 = memref.alloc() : memref<1x1x192x320xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1x192x320xf16, @CMX_NN>) outputs(%4 as %arg3: memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x192x320xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    }
    return %5 : memref<1x1x192x320xf16>

    // CHECK:    [[INPUT_SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUFF1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME{Literal}:            !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}>
    // CHECK:    [[INPUT_COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT_SUBVIEW1]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUFF1]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>) ->
    // CHECK-SAME{Literal}:            !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]], memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}> {
    // CHECK:                                   VPUIP.Copy inputs(%arg1 : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs(%arg2 : memref<1x1x49x160xf16, @CMX_NN>) -> memref<1x1x49x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[INPUT_SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>
    // CHECK:    [[INPUT_BUFF0:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME{Literal}:             !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}>
    // CHECK:    [[INPUT_COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs([[INPUT_BUFF0]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>) ->
    // CHECK-SAME{Literal}:             !VPUIP.DistributedBuffer<1x1x49x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]], memory_shapes = [[1, 1, 25, 160], [1, 1, 26, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0]]}> {
    // CHECK:                                   VPUIP.Copy inputs(%arg1 : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}>) outputs(%arg2 : memref<1x1x49x160xf16, @CMX_NN>) -> memref<1x1x49x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUFF1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_BUFF0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[INPUT_COPY0]] as %arg1: memref<1x1x49x160xf16, @CMX_NN>, [[INPUT_COPY1]] as %arg2: memref<1x1x49x160xf16, @CMX_NN>) outputs([[OUTPUT_BUFF0]] as %arg3: memref<1x1x96x320xf16, @CMX_NN>, [[OUTPUT_BUFF1]] as %arg4: memref<1x1x96x320xf16, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x96x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                                   %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg1 as %arg5: memref<1x1x49x160xf16, @CMX_NN>, %arg2 as %arg6: memref<1x1x49x160xf16, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x1x96x320xf16, @CMX_NN>, %arg4 as %arg8: memref<1x1x96x320xf16, @CMX_NN>) on tile 0 -> (memref<1x1x96x320xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>){
    // CHECK:                                         VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg5, %arg7) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:                                         VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg6, %arg8) : memref<1x1x49x160xf16, @CMX_NN>, memref<1x1x96x320xf16, @CMX_NN>
    // CHECK:                                   }
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUFF:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[OUTPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[OUTPUT_COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[INTERP]]#0 as %arg1: memref<1x1x96x320xf16, @CMX_NN>) outputs([[OUTPUT_SUBVIEW0]] as %arg2: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK:                                    VPUIP.Copy inputs(%arg1 : memref<1x1x96x320xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    }
    // CHECK:    [[OUTPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    [[OUTPUT_COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[INTERP]]#1 as %arg1: memref<1x1x96x320xf16, @CMX_NN>) outputs([[OUTPUT_SUBVIEW1]] as %arg2: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}> {
    // CHECK:                                     VPUIP.Copy inputs(%arg1 : memref<1x1x96x320xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) -> memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[OUTPUT_COPY0]], [[OUTPUT_COPY1]] : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}>) outputs([[OUTPUT_BUFF]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[CONCAT:%.*]] : memref<1x1x192x320xf16>
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.*]]   = VPUIP.SubView %1 [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.*]]   = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[GELU:%.*]]:2  = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK{LITERAL}:                %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>)
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling
      inputs(%arg0 as %arg1: memref<1x128x64x32xf16>)
      outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>)
        -> !DistributedBuffer {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>)
        -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !DistributedBuffer
    %4 = VPUIP.NCEClusterTiling
      inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>)
      outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>)
        -> !DistributedBuffer {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>)
        outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0
          -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling
      inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>)
      outputs(%5 as %arg2: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:                 [[IN_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:                 [[IN_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> {
    // CHECK:                               VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:                 [[IN_SV0:%.*]]   = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:                 [[IN_SV1:%.*]]   = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:                 [[OUT_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:                 [[OUT_SV0:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:                 [[OUT_SV1:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:       explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK: [[GELU:%.*]]:2   = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>)
    // CHECK-SAME:                outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) {
    // CHECK{LITERAL}:                       %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                                VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                                VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:                 [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:          [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:          [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                                VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_HardSigmoid inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK: [[IN_BUF:%.*]]   = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_COPY:%.*]]  = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                       VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK: [[IN_SV0:%.*]]   = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_SV1:%.*]]   = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[OUT_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_SV0:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_SV1:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[HARD_SIGMOID:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                             %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_HardSigmoid inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:                  outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK: [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[HARD_SIGMOID]]#0, [[HARD_SIGMOID]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK: [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                       VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %2 = memref.alloc() : memref<1x128x64x32xf16>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%2 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %4 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %3: memref<1x128x64x32xf16>

    // CHECK:    [[IN_SV0:%.*]] = VPUIP.SubView %arg0 [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[GELU:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                           %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:                outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%2 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                      VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling
      inputs(%arg0 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>)
        -> !DistributedBuffer {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0
          -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %2 = memref.alloc() : memref<1x128x64x32xf16>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%2 as %arg2: memref<1x128x64x32xf16>)
      -> memref<1x128x64x32xf16> {
      %4 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16>
    }
    return %3: memref<1x128x64x32xf16>

    // CHECK:             [[IN_SV0:%.*]]   = VPUIP.SubView %arg0 [0, 64, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:             [[IN_SV1:%.*]]   = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:             [[OUT_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:             [[OUT_SV0:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32]
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>
    // CHECK:             [[OUT_SV1:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] {
    // CHECK-SAME{LITERAL}:  explicit_output_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]]} : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 32, 32], [1, 128, 32, 32]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>

    // CHECK:             [[GELU:%.*]]:2   = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:  -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]], memory_shapes = [[1, 64, 32, 32], [1, 64, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}>) {
    // CHECK{LITERAL}:           %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
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

    // CHECK:    [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%6 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%alloc as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x6x32xf16>) outputs(%0 as %arg2: memref<1x128x6x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x6x32xf16>) outputs(%arg2 : memref<1x128x6x32xf16, @CMX_NN>) -> memref<1x128x6x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x6x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x6x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg6: memref<1x128x6x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x6x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x6x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x6x32xf16, @CMX_NN>, memref<1x128x6x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x6x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x6x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x6x32xf16>) -> memref<1x128x6x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x6x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x6x32xf16>) -> memref<1x128x6x32xf16>
    }
    return %6: memref<1x128x6x32xf16>

    // CHECK:     [[IN_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[IN_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x6x32xf16>) outputs([[IN_BUF]] as %arg2: memref<1x128x6x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x6x32xf16>) outputs(%arg2 : memref<1x128x6x32xf16, @CMX_NN>) -> memref<1x128x6x32xf16, @CMX_NN>

    // CHECK:     [[IN_SV0:%.*]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[IN_SV1:%.*]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[OUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUT_SV0:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUT_SV1:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 6, 32] : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[GELU:%.*]]:2  = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, %2 as %arg2: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>) outputs(%6 as %arg3: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, %5 as %arg4: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                          %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gelu inputs(%arg1 as %arg5: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>)
     // CHECK-SAME{LITERAL}:              outputs(%arg3 as %arg7: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>) strides([[12288, 96, 32, 1], [12288, 96, 32, 1]]) on tile 0 -> (memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>, memref<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN>

    // CHECK:     [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x6x32xf16, {order = #NCHW, strides = [24576, 192, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x6x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x6x32xf16>
    // CHECK:     [[OUT_COPY:%.*]]  = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x6x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x6x32xf16>) -> memref<1x128x6x32xf16> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x128x6x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x6x32xf16>) -> memref<1x128x6x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[SOFTMAX:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:   %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:  }
    // CHECK:}

    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[BUFF_OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [2]}(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:         VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 32, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 32, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 128, 32, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SOFTMAX:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg5: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [2]}(%arg5, %arg7) : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [2]}(%arg6, %arg8) : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:      }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUTPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[BUFF_OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[BUFF_OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:         VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x2x1xf16>) outputs(%0 as %arg2: memref<1x128x2x1xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x2x1xf16>) outputs(%arg2 : memref<1x128x2x1xf16, @CMX_NN>) -> memref<1x128x2x1xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x2x1xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x2x1xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg6: memref<1x128x2x1xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x2x1xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x2x1xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [2]}(%arg6, %arg7) : memref<1x128x2x1xf16, @CMX_NN>, memref<1x128x2x1xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x2x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x2x1xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x2x1xf16>) -> memref<1x128x2x1xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x2x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x2x1xf16>) -> memref<1x128x2x1xf16>
    }
    return %6: memref<1x128x2x1xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x2x1xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x2x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x2x1xf16>) outputs(%arg2 : memref<1x128x2x1xf16, @CMX_NN>) -> memref<1x128x2x1xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SOFTMAX:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY0]] as %arg1: memref<1x128x2x1xf16, @CMX_NN>) outputs([[OUTPUT_CMX]] as %arg2: memref<1x128x2x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x2x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg3: memref<1x128x2x1xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x2x1xf16, @CMX_NN>) on tile 0 -> memref<1x128x2x1xf16, @CMX_NN>{
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [2]}(%arg3, %arg4) : memref<1x128x2x1xf16, @CMX_NN>, memref<1x128x2x1xf16, @CMX_NN>
    // CHECK:      }
    // CHECK:    }
    // CHECK:    [[BUFF_OUT_DDR:%.*]] = memref.alloc() : memref<1x128x2x1xf16>
    // CHECK:    [[COPY_OUTPUT_TO_DDR:%.*]] = VPUIP.NCEClusterTiling inputs([[SOFTMAX]] as %arg1: memref<1x128x2x1xf16, @CMX_NN>) outputs([[BUFF_OUT_DDR]] as %arg2: memref<1x128x2x1xf16>) -> memref<1x128x2x1xf16> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x2x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x2x1xf16>) -> memref<1x128x2x1xf16>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x16x154x160xf16>) outputs(%0 as %arg3: memref<1x16x154x160xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x154x160xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]], memory_shapes = [[1, 16, 77, 160], [1, 16, 77, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 76, 0]]}> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x16x154x160xf16>) outputs(%arg3 : memref<1x16x154x160xf16, @CMX_NN>) -> memref<1x16x154x160xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x16x154x160xf16, @CMX_NN>) outputs(%2 as %arg3: memref<1x16x308x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg2 as %arg4: memref<1x16x154x160xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x16x308x320xf16, @CMX_NN>) on tile 0 -> memref<1x16x308x320xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 154, 16, 1], [320, 308, 16, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : memref<1x16x154x160xf16, @CMX_NN>, memref<1x16x308x320xf16, @CMX_NN>
      }
    }
    %4 = memref.alloc() : memref<1x16x308x320xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x16x308x320xf16, @CMX_NN>) outputs(%4 as %arg3: memref<1x16x308x320xf16>) -> memref<1x16x308x320xf16> {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x16x308x320xf16, @CMX_NN>) outputs(%arg3 : memref<1x16x308x320xf16>) -> memref<1x16x308x320xf16>
    }
    return %5 : memref<1x16x308x320xf16>

    // CHECK:    [[INPUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x154x160xf16>) outputs([[INPUT_BUF]] as %arg2: memref<1x16x154x160xf16, @CMX_NN>)
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x16x154x160xf16>) outputs(%arg2 : memref<1x16x154x160xf16, @CMX_NN>) -> memref<1x16x154x160xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK:    [[INTERP:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY0]] as %arg1: memref<1x16x154x160xf16, @CMX_NN>) outputs([[OUTPUT_BUF]] as %arg2: memref<1x16x308x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x308x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg1 as %arg3: memref<1x16x154x160xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x308x320xf16, @CMX_NN>) on tile 0 -> memref<1x16x308x320xf16, @CMX_NN>{
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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Multiply First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Multiply Second Input
    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTIPLY:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as %arg4: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as %arg5: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[MULTIPLY]]#0, [[MULTIPLY]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                       %1 as %arg1: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Multiply First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Multiply Second Input
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>


    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTIPLY:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[ARG1:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG3:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[MULTIPLY]]#0, [[MULTIPLY]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as {{%.+}}: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as {{%.+}}: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %0 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%1 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %3 = memref.alloc() : memref<1x64x88x88xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %4: memref<1x64x88x88xf16>

    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTIPLY:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT1_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE1]] as [[ARG1:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG3:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[MULTIPLY]]#0, [[MULTIPLY]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Minimum Input & Output
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MIN_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MIN_OUT_TILE0:%.*]] = VPUIP.SubView [[MIN_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MIN_OUT_TILE1:%.*]] = VPUIP.SubView [[MIN_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]

    // CHECK:    [[MINIMUM:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INPUT0_TILE1_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[INPUT1_TILE1_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[INPUT0_TILE0_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[INPUT1_TILE0_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MIN_OUT_TILE1]] as [[MIN_OUT_TILE1_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[MIN_OUT_TILE0]] as [[MIN_OUT_TILE0_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Minimum
    // CHECK:                     inputs([[INPUT0_TILE1_ARG]] as [[INPUT0_TILE1_SW_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1_ARG]] as [[INPUT1_TILE1_SW_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0_ARG]] as [[INPUT0_TILE0_SW_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0_ARG]] as [[INPUT1_TILE0_SW_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MIN_OUT_TILE1_ARG]] as [[MIN_OUT_TILE1_SW_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[MIN_OUT_TILE0_ARG]] as [[MIN_OUT_TILE0_SW_ARG:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE1_SW_ARG]], [[INPUT1_TILE1_SW_ARG]], [[MIN_OUT_TILE1_SW_ARG]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE0_SW_ARG]], [[INPUT1_TILE0_SW_ARG]], [[MIN_OUT_TILE0_SW_ARG]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[MINIMUM]]#0, [[MINIMUM]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MIN_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as [[INPUT0_TILE1_ARG]]: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as [[INPUT1_TILE1_ARG]]: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %INPUT_TILING_COPY = VPUIP.NCEClusterTiling
        inputs(%NET_INPUT as %arg2: memref<1x32x5120x1xf16>)
        outputs(%ALLOC_INPUT_DIST as %arg3: memref<1x32x5120x1xf16, @CMX_NN>)
            -> !VPUIP.DistributedBuffer<
                1x32x5120x1xf16,
                #NCHW,
                @CMX_NN, {
                    mode = "SEGMENTED",
                    num_tiles = [1, 2, 1, 1],
                    num_clusters = 2 : i64,
                    alignment = [1, 16, 1, 1]
                }> {
      %INPUT_COPY = VPUIP.Copy
          inputs(%arg2 : memref<1x32x5120x1xf16>)
          outputs(%arg3 : memref<1x32x5120x1xf16, @CMX_NN>)
            -> memref<1x32x5120x1xf16, @CMX_NN>
    }

    %MVN_OUTPUT = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<
        1x32x5120x1xf16,
        #NCHW,
        @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [1, 2, 1, 1],
            num_clusters = 2 : i64
        }>

    %MVN_TILING = VPUIP.NCEClusterTiling
        inputs(%INPUT_TILING_COPY as %arg2: memref<1x32x5120x1xf16, @CMX_NN>)
        outputs(%MVN_OUTPUT as %arg3: memref<1x32x5120x1xf16, @CMX_NN>)
            -> !VPUIP.DistributedBuffer<
                    1x32x5120x1xf16,
                    #NCHW,
                    @CMX_NN, {
                        mode = "SEGMENTED",
                        num_tiles = [1, 2, 1, 1],
                        num_clusters = 2 : i64
                    }> {
      %results = VPUIP.SW.Kernel {
          resultSegmentSizes = array<i32: 1, 0, 0>
      } @VPU.SW::@builtin_MVN
        inputs(%arg2 as %arg4: memref<1x32x5120x1xf16, @CMX_NN>)
        outputs(%arg3 as %arg5: memref<1x32x5120x1xf16, @CMX_NN>) on tile 0
            -> memref<1x32x5120x1xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {
            attrs = [false, true, 9.9999997473787516E-6]
        }(%arg4, %arg5) : memref<1x32x5120x1xf16, @CMX_NN>, memref<1x32x5120x1xf16, @CMX_NN>
      }
    }

    %OUTPUT_TILING_COPY = VPUIP.NCEClusterTiling
        inputs(%MVN_TILING as %arg2: memref<1x32x5120x1xf16, @CMX_NN>)
        outputs(%NET_OUTPUT as %arg3: memref<1x32x5120x1xf16>)
            -> memref<1x32x5120x1xf16> {
      %OUTPUT_COPY = VPUIP.Copy
          inputs(%arg2 : memref<1x32x5120x1xf16, @CMX_NN>)
          outputs(%arg3 : memref<1x32x5120x1xf16>)
            -> memref<1x32x5120x1xf16>
    }

    return %OUTPUT_TILING_COPY : memref<1x32x5120x1xf16>

    // CHECK: ([[NET_INPUT:%.*]]: memref<1x32x5120x1xf16>, [[NET_OUTPUT:%.*]]: memref<1x32x5120x1xf16>)

    // CHECK:   [[ALLOC_INPUT_DIST:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16
    // CHECK-SAME:      alignment = [1, 16, 1, 1]

    // CHECK:   [[INPUT_TILING_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[NET_INPUT]] as %arg2: memref<1x32x5120x1xf16>)
    // CHECK-SAME:  outputs([[ALLOC_INPUT_DIST]] as %arg3: memref<1x32x5120x1xf16, @CMX_NN>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16,
    // CHECK-SAME:      alignment = [1, 16, 1, 1]

    // CHECK:   [[MVN_OUTPUT:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16,

    // CHECK:   [[MVN_TILING:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[INPUT_TILING_COPY]] as %arg2: memref<1x32x5120x1xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[MVN_OUTPUT]] as %arg3: memref<1x32x5120x1xf16, @CMX_NN>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x5120x1xf16,
    // CHECK:       VPUIP.SW.Kernel.run {{.*}} : memref<1x32x5120x1xf16
    // CHECK-NOT:   VPUIP.SW.Kernel.run {{.*}} : memref<1x16x5120x1xf16

    // CHECK:   [[OUTPUT_TILING_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[MVN_TILING]] as %arg2: memref<1x32x5120x1xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[NET_OUTPUT]] as %arg3: memref<1x32x5120x1xf16>)
    // CHECK-SAME:  -> memref<1x32x5120x1xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x16x16xf32>) outputs(%0 as %arg2: memref<1x64x16x16xf32, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x64x16x16xf32>) outputs(%arg2 : memref<1x64x16x16xf32, @CMX_NN>) -> memref<1x64x16x16xf32, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x64x16x16xf32, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x16x16xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg1 as %arg6: memref<1x64x16x16xf32, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x64x16x16xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x64x16x16xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg7) : memref<1x64x16x16xf32, @CMX_NN>, memref<1x64x16x16xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x64x16x16xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x64x16x16xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x64x16x16xf16>) -> memref<1x64x16x16xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x64x16x16xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x16x16xf16>) -> memref<1x64x16x16xf16>
    }
    return %6: memref<1x64x16x16xf16>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 8, 0] [1, 64, 8, 16] : memref<1x64x16x16xf32> to memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>
    // CHECK:   [[ALLOC_DSTR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[TILING_COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>) outputs(%1 as %arg2: memref<1x64x8x16xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:   VPUIP.Copy inputs(%arg1 : memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>) outputs(%arg2 : memref<1x64x8x16xf32, @CMX_NN>) -> memref<1x64x8x16xf32, @CMX_NN>
    // CHECK:   }
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 8, 16] : memref<1x64x16x16xf32> to memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>
    // CHECK:   [[ALLOC_DSTR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[TILING_COPY2:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW2]] as %arg1: memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>) outputs([[ALLOC_DSTR2]] as %arg2: memref<1x64x8x16xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x8x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:   VPUIP.Copy inputs(%arg1 : memref<1x64x8x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}>) outputs(%arg2 : memref<1x64x8x16xf32, @CMX_NN>) -> memref<1x64x8x16xf32, @CMX_NN>
    // CHECK:   }
    // CHECK:   [[ALLOC_DSTR3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[ALLOC_DSTR4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[TILING_KERNEL:%.*]] = VPUIP.NCEClusterTiling inputs([[TILING_COPY2]] as %arg1: memref<1x64x8x16xf32, @CMX_NN>, [[TILING_COPY1]] as %arg2: memref<1x64x8x16xf32, @CMX_NN>) outputs([[ALLOC_DSTR4]] as %arg3: memref<1x64x8x16xf16, @CMX_NN>, [[ALLOC_DSTR3]] as %arg4: memref<1x64x8x16xf16, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x8x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg1 as %arg5: memref<1x64x8x16xf32, @CMX_NN>, %arg2 as %arg6: memref<1x64x8x16xf32, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x8x16xf16, @CMX_NN>, %arg4 as %arg8: memref<1x64x8x16xf16, @CMX_NN>) on tile 0 -> (memref<1x64x8x16xf16, @CMX_NN>, memref<1x64x8x16xf16, @CMX_NN>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}(%arg5, %arg7) : memref<1x64x8x16xf32, @CMX_NN>, memref<1x64x8x16xf16, @CMX_NN>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg8) : memref<1x64x8x16xf32, @CMX_NN>, memref<1x64x8x16xf16, @CMX_NN>
    // CHECK:   }
    // CHECK:   }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileConvertOpTest(%arg0: memref<1x64x16x16xf32>)
        -> memref<1x64x16x16xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x16x16xf32>) outputs(%0 as %arg2: memref<1x64x16x16xf32, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x64x16x16xf32>) outputs(%arg2 : memref<1x64x16x16xf32, @CMX_NN>) -> memref<1x64x16x16xf32, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x64x16x16xf32, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x16x16xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg1 as %arg6: memref<1x64x16x16xf32, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x64x16x16xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x64x16x16xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg7) : memref<1x64x16x16xf32, @CMX_NN>, memref<1x64x16x16xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x64x16x16xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x64x16x16xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x64x16x16xf16>) -> memref<1x64x16x16xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x64x16x16xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x16x16xf16>) -> memref<1x64x16x16xf16>
    }
    return %6: memref<1x64x16x16xf16>
    // CHECK:   [[ALLOC_DSTR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>


    // CHECK:   [[TILING_COPY1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x16x16xf32>) outputs(%0 as %arg2: memref<1x64x16x16xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x16x16xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   }
    // CHECK:	[[ALLOC_DSTR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[TILING_KERNEL:%.*]] = VPUIP.NCEClusterTiling inputs([[TILING_COPY1]] as %arg1: memref<1x64x16x16xf32, @CMX_NN>) outputs([[ALLOC_DSTR2]] as %arg2: memref<1x64x16x16xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg1 as %arg3: memref<1x64x16x16xf32, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x64x16x16xf16, @CMX_NN>) on tile 0 -> memref<1x64x16x16xf16, @CMX_NN>{
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x64x16x16xf32, @CMX_NN>, memref<1x64x16x16xf16, @CMX_NN>
    // CHECK:   }
    // CHECK:   }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tanh inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.*]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.*]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[TANH:%.*]]:2  = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:                %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Tanh inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[TANH]]#0, [[TANH]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClusterTopKWithCopyUser(%arg0: memref<1x8x128x128xf16, #NHWC>, %arg1: memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC> {
    %cst = const.Declare memref<1x1x1x128xui8> = dense<0> : tensor<1x1x1x128xui8>
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x128x128xf16, #NHWC>) outputs(%0 as %arg3: memref<1x8x128x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x8x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %8 = VPUIP.Copy inputs(%arg2 : memref<1x8x128x128xf16, #NHWC>) outputs(%arg3 : memref<1x8x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x8x128x128xf16, #NHWC, @CMX_NN>
    }
    %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %11 = VPUIP.NCEClusterTiling inputs(%cst as %arg3: memref<1x1x1x128xui8, #NHWC>) outputs(%10 as %arg4: memref<1x1x1x128xui8, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x128xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %8 = VPUIP.Copy inputs(%arg3 : memref<1x1x1x128xui8, #NHWC>) outputs(%arg4 : memref<1x1x1x128xui8, #NHWC, @CMX_NN>) -> memref<1x1x1x128xui8, #NHWC, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4:2 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x8x128x128xf16, #NHWC, @CMX_NN>, %11 as %arg8: memref<1x1x1x128xui8, #NHWC, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x128x128xf16, #NHWC, @CMX_NN>, %3 as %arg4: memref<1x1x128x128xsi32, #NHWC, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%arg2 as %arg5: memref<1x8x128x128xf16, #NHWC, @CMX_NN>, %arg8 as %arg9: memref<1x1x1x128xui8, #NHWC, @CMX_NN>) outputs(%arg3 as %arg6: memref<1x1x128x128xf16, #NHWC, @CMX_NN>, %arg4 as %arg7: memref<1x1x128x128xsi32, #NHWC, @CMX_NN>) on tile 0 -> (memref<1x1x128x128xf16, #NHWC, @CMX_NN>, memref<1x1x128x128xsi32, #NHWC, @CMX_NN>){
        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg5, %arg9, %arg6, %arg7) : memref<1x8x128x128xf16, #NHWC, @CMX_NN>, memref<1x1x1x128xui8, #NHWC, @CMX_NN>, memref<1x1x128x128xf16, #NHWC, @CMX_NN>, memref<1x1x128x128xsi32, #NHWC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x1x128x128xsi32, #NHWC>
    %6 = VPUIP.NCEClusterTiling inputs(%4#1 as %arg2: memref<1x1x128x128xsi32, #NHWC, @CMX_NN>) outputs(%5 as %arg3: memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC> {
      %8 = VPUIP.Copy inputs(%arg2 : memref<1x1x128x128xsi32, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    }
    %7 = VPUIP.Copy inputs(%6 : memref<1x1x128x128xsi32, #NHWC>) outputs(%arg1 : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    return %7 : memref<1x1x128x128xsi32, #NHWC>



    // CHECK:                [[CST:%.*]] = const.Declare memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}> = dense<0> : tensor<1x1x1x128xui8>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 64]>]
    // CHECK:                [[CST0:%.*]] = const.Declare memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}> = dense<0> : tensor<1x1x1x128xui8>, [#const.SubView<[0, 0, 0, 64], [1, 1, 1, 64]>]
  
    // CHECK:                [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 64, 0] [1, 8, 64, 128] : memref<1x8x128x128xf16, #NHWC> to memref<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}>
    // CHECK:                [[BUFFER0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[CLUSTER0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[SUBVIEW0]] as %arg2: memref<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}>)
    // CHECK-SAME:               outputs([[BUFFER0]] as %arg3: memref<1x8x64x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x8x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:               inputs(%arg2 : memref<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}>)
    // CHECK-SAME:               outputs(%arg3 : memref<1x8x64x128xf16, #NHWC, @CMX_NN>) -> memref<1x8x64x128xf16, #NHWC, @CMX_NN>
    // CHECK:                    }
    
    // CHECK:                [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 8, 64, 128] : memref<1x8x128x128xf16, #NHWC> to memref<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}>
    // CHECK:                [[BUFFER1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[CLUSTER1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[SUBVIEW1]] as %arg2: memref<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}>)
    // CHECK-SAME:               outputs([[BUFFER1]] as %arg3: memref<1x8x64x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x8x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:               inputs(%arg2 : memref<1x8x64x128xf16, {order = #NHWC, strides = [131072, 1, 1024, 8]}>)
    // CHECK-SAME:               outputs(%arg3 : memref<1x8x64x128xf16, #NHWC, @CMX_NN>) -> memref<1x8x64x128xf16, #NHWC, @CMX_NN>
    // CHECK:                    }

    // CHECK:                [[BUFFER2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x64xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:                [[CLUSTER2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[CST0]] as %arg2: memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}>)
    // CHECK-SAME:               outputs([[BUFFER2]] as %arg3: memref<1x1x1x64xui8, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:                [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:               inputs(%arg2 : memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}>)
    // CHECK-SAME:               outputs(%arg3 : memref<1x1x1x64xui8, #NHWC, @CMX_NN>) -> memref<1x1x1x64xui8, #NHWC, @CMX_NN>
    // CHECK:                    }

    // CHECK:                [[BUFFER3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x64xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:                [[CLUSTER3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[CST]] as %arg2: memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}>)
    // CHECK-SAME:               outputs([[BUFFER3]] as %arg3: memref<1x1x1x64xui8, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:                [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:               inputs(%arg2 : memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}>)
    // CHECK-SAME:               outputs(%arg3 : memref<1x1x1x64xui8, #NHWC, @CMX_NN>) -> memref<1x1x1x64xui8, #NHWC, @CMX_NN>
    // CHECK:                    }

    // CHECK:                [[BUFFER4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[BUFFER5:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[BUFFER6:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x64x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[BUFFER7:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x64x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:                [[CLUSTER_TOPK:%.*]]:4 = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[CLUSTER1]] as %arg2: memref<1x8x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:               [[CLUSTER3]] as %arg3: memref<1x1x1x64xui8, #NHWC, @CMX_NN>,
    // CHECK-SAME:               [[CLUSTER0]] as %arg4: memref<1x8x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:               [[CLUSTER2]] as %arg5: memref<1x1x1x64xui8, #NHWC, @CMX_NN>)
    // CHECK-SAME:               outputs(%11 as %arg6: memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:               [[BUFFER7]] as %arg7: memref<1x1x64x128xsi32, #NHWC, @CMX_NN>,
    // CHECK-SAME:               [[BUFFER4]] as %arg8: memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:               [[BUFFER6]] as %arg9: memref<1x1x64x128xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:               -> (!VPUIP.DistributedBuffer<1x1x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x1x64x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x1x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x1x64x128xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                  results:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK
    // CHECK-SAME:                   inputs(%arg2 as %arg10: memref<1x8x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   %arg3 as %arg11: memref<1x1x1x64xui8, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   %arg4 as %arg12: memref<1x8x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   %arg5 as %arg13: memref<1x1x1x64xui8, #NHWC, @CMX_NN>)
    // CHECK-SAME:                   outputs(%arg6 as %arg14: memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   %arg7 as %arg15: memref<1x1x64x128xsi32, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   %arg8 as %arg16: memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME{LITERAL}:          %arg9 as %arg17: memref<1x1x64x128xsi32, #NHWC, @CMX_NN>) on tile 0
    // CHECK-SAME:                   -> (memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xsi32, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xsi32, #NHWC, @CMX_NN>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg10, %arg11, %arg14, %arg15) :
    // CHECK-SAME:                   memref<1x8x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x1x64xui8, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xsi32, #NHWC, @CMX_NN>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg12, %arg13, %arg16, %arg17) :
    // CHECK-SAME:                   memref<1x8x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x1x64xui8, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                   memref<1x1x64x128xsi32, #NHWC, @CMX_NN>
    // CHECK:                    }
    // CHECK:                }

    // CHECK:                [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x1x128x128xsi32, #NHWC>
    // CHECK:                [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 0, 0] [1, 1, 64, 128] : memref<1x1x128x128xsi32, #NHWC> to memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>
    // CHECK:                [[CLUSTER4:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[CLUSTER_TOPK]]#1 as %arg2: memref<1x1x64x128xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:               outputs([[SUBVIEW2]] as %arg3: memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>) -> memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}> {
    // CHECK:                [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:               inputs(%arg2 : memref<1x1x64x128xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:               outputs(%arg3 : memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>) -> memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>
    // CHECK:                    }

    // CHECK:                [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_DDR]] [0, 0, 64, 0] [1, 1, 64, 128] : memref<1x1x128x128xsi32, #NHWC> to memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>
    // CHECK:                [[CLUSTER5:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[CLUSTER_TOPK]]#3 as %arg2: memref<1x1x64x128xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:               outputs([[SUBVIEW3]] as %arg3: memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>) -> memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}> {
    // CHECK:                [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:               inputs(%arg2 : memref<1x1x64x128xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:               outputs(%arg3 : memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>) -> memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>
    // CHECK:                    }

    // CHECK:                [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:           inputs([[CLUSTER4]], [[CLUSTER5]] : memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>, memref<1x1x64x128xsi32, {order = #NHWC, strides = [16384, 1, 128, 1]}>)
    // CHECK-SAME:           outputs([[OUTPUT_DDR]] : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    // CHECK:                [[OUTPUT_COPY:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x128x128xsi32, #NHWC>) outputs(%arg1 : memref<1x1x128x128xsi32, #NHWC>) -> memref<1x1x128x128xsi32, #NHWC>
    // CHECK:                return [[OUTPUT_COPY]] : memref<1x1x128x128xsi32, #NHWC>
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
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg3: memref<1x5x256x512xf16>) outputs(%2 as %arg4: memref<1x5x256x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %13 = VPUIP.Copy inputs(%arg3 : memref<1x5x256x512xf16>) outputs(%arg4 : memref<1x5x256x512xf16, @CMX_NN>) -> memref<1x5x256x512xf16, @CMX_NN>
    }
    %14 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %15 = VPUIP.NCEClusterTiling inputs(%cst as %arg3: memref<1x1x1x80xui8>) outputs(%14 as %arg4: memref<1x1x1x80xui8, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x80xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
      %13 = VPUIP.Copy inputs(%arg3 : memref<1x1x1x80xui8>) outputs(%arg4 : memref<1x1x1x80xui8, @CMX_NN>) -> memref<1x1x1x80xui8, @CMX_NN>
  }

    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %6:2 = VPUIP.NCEClusterTiling inputs(%3 as %arg3: memref<1x5x256x512xf16, @CMX_NN>, %15 as %arg9: memref<1x1x1x80xui8, @CMX_NN>) outputs(%4 as %arg4: memref<1x1x256x512xf16, @CMX_NN>, %5 as %arg5: memref<1x1x256x512xsi32, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
      %results_0:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%arg3 as %arg6: memref<1x5x256x512xf16, @CMX_NN>, %arg9 as %arg10: memref<1x1x1x80xui8, @CMX_NN>) outputs(%arg4 as %arg7: memref<1x1x256x512xf16, @CMX_NN>, %arg5 as %arg8: memref<1x1x256x512xsi32, @CMX_NN>) on tile 0 -> (memref<1x1x256x512xf16, @CMX_NN>, memref<1x1x256x512xsi32, @CMX_NN>){
        VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg6, %arg10, %arg7, %arg8) : memref<1x5x256x512xf16, @CMX_NN>, memref<1x1x1x80xui8, @CMX_NN>, memref<1x1x256x512xf16, @CMX_NN>, memref<1x1x256x512xsi32, @CMX_NN>
      }
    }
    %7 = memref.alloc() : memref<1x1x256x512xf16>
    %8 = VPUIP.NCEClusterTiling inputs(%6#0 as %arg3: memref<1x1x256x512xf16, @CMX_NN>) outputs(%7 as %arg4: memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16> {
      %13 = VPUIP.Copy inputs(%arg3 : memref<1x1x256x512xf16, @CMX_NN>) outputs(%arg4 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    }
    %9 = memref.alloc() : memref<1x1x256x512xsi32>
    %10 = VPUIP.NCEClusterTiling inputs(%6#1 as %arg3: memref<1x1x256x512xsi32, @CMX_NN>) outputs(%9 as %arg4: memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32> {
      %13 = VPUIP.Copy inputs(%arg3 : memref<1x1x256x512xsi32, @CMX_NN>) outputs(%arg4 : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
    }
    %11 = VPUIP.Copy inputs(%8 : memref<1x1x256x512xf16>) outputs(%arg1 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    %12 = VPUIP.Copy inputs(%10 : memref<1x1x256x512xsi32>) outputs(%arg2 : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
    return %11, %12 : memref<1x1x256x512xf16>, memref<1x1x256x512xsi32>

    // CHECK:           [[CST:%.*]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 40]>]
    // CHECK:           [[CST0:%.*]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 40], [1, 1, 1, 40]>]
    // CHECK:           [[ALLOC0:%.*]] = memref.alloc() : memref<1x5x256x512xf16>
    // CHECK:           [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x5x256x512xf16>) outputs([[ALLOC0]] : memref<1x5x256x512xf16>) -> memref<1x5x256x512xf16>
    // CHECK:           [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 128, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:           [[ALLOC1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_INPUT0:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg3: memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs([[ALLOC1]] as %arg4: memref<1x5x128x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs(%arg4 : memref<1x5x128x512xf16, @CMX_NN>) -> memref<1x5x128x512xf16, @CMX_NN>
    // CHECK:           }
    // CHECK:           [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:           [[ALLOC2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_INPUT1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg3: memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs([[ALLOC2]] as %arg4: memref<1x5x128x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs(%arg4 : memref<1x5x128x512xf16, @CMX_NN>) -> memref<1x5x128x512xf16, @CMX_NN>
    // CHECK:           }

    // CHECK:           [[ALLOC13:%.*]] = VPURT.AllocDistributed -> 
    // CHECK:           !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    
    // CHECK:           [[TILING_AUX_INPUT_0:%.*]] = VPUIP.NCEClusterTiling inputs([[CST0]] as %arg3: memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs([[ALLOC13]] as %arg4: memref<1x1x1x40xui8, @CMX_NN>)
    // CHECK{LITERAL}:  -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, 
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs(%arg4 : memref<1x1x1x40xui8, @CMX_NN>) -> memref<1x1x1x40xui8, @CMX_NN>
    // CHECK:           }

    // CHECK:           [[ALLOC14:%.*]] = VPURT.AllocDistributed 
    // CHECK:           -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    
    // CHECK:           [[TILING_AUX_INPUT_1:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]] as %arg3: memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs([[ALLOC14]] as %arg4: memref<1x1x1x40xui8, @CMX_NN>) ->
    // CHECK{LITERAL}:  !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, 
    // CHECK{LITERAL}:  compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
    // CHECK{LITERAL}:  memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs(%arg4 : memref<1x1x1x40xui8, @CMX_NN>) -> memref<1x1x1x40xui8, @CMX_NN>
    // CHECK:           }

    // CHECK:           [[ALLOC3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC5:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC6:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CLUSTER_TOPK:%.*]]:4 = VPUIP.NCEClusterTiling 
    // CHECK:           inputs([[TILING_INPUT1]] as %arg3: memref<1x5x128x512xf16, @CMX_NN>,
    // CHECK:           [[TILING_AUX_INPUT_1]] as %arg4: memref<1x1x1x40xui8, @CMX_NN>,
    // CHECK:           [[TILING_INPUT0]] as %arg5: memref<1x5x128x512xf16, @CMX_NN>,
    // CHECK:           [[TILING_AUX_INPUT_0]] as %arg6: memref<1x1x1x40xui8, @CMX_NN>) 
    // CHECK:           outputs([[ALLOC4]] as %arg7: memref<1x1x128x512xf16, @CMX_NN>,
    // CHECK:           [[ALLOC6]] as %arg8: memref<1x1x128x512xsi32, @CMX_NN>,
    // CHECK:           [[ALLOC3]] as %arg9: memref<1x1x128x512xf16, @CMX_NN>, 
    // CHECK:           [[ALLOC5]] as %arg10: memref<1x1x128x512xsi32, @CMX_NN>) 
    // CHECK:           -> (!VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:           %results:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK 
    // CHECK:             inputs(%arg3 as %arg11: memref<1x5x128x512xf16, @CMX_NN>, 
    // CHECK:             %arg4 as %arg12: memref<1x1x1x40xui8, @CMX_NN>, 
    // CHECK:             %arg5 as %arg13: memref<1x5x128x512xf16, @CMX_NN>, 
    // CHECK:             %arg6 as %arg14: memref<1x1x1x40xui8, @CMX_NN>) 
    // CHECK:             outputs(%arg7 as %arg15: memref<1x1x128x512xf16, @CMX_NN>, 
    // CHECK:             %arg8 as %arg16: memref<1x1x128x512xsi32, @CMX_NN>, 
    // CHECK:             %arg9 as %arg17: memref<1x1x128x512xf16, @CMX_NN>, 
    // CHECK{LITERAL}:    %arg10 as %arg18: memref<1x1x128x512xsi32, @CMX_NN>) on tile 0 
    // CHECK:             -> (memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>, memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg11, %arg12, %arg15, %arg16) : 
    // CHECK:              memref<1x5x128x512xf16, @CMX_NN>, memref<1x1x1x40xui8, @CMX_NN>, memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg13, %arg14, %arg17, %arg18) : 
    // CHECK:              memref<1x5x128x512xf16, @CMX_NN>, memref<1x1x1x40xui8, @CMX_NN>, memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>
    // CHECK:               }
    // CHECK:           }
    // CHECK:           [[ALLOC7:%.*]] = memref.alloc() : memref<1x1x256x512xf16, @DDR>
    // CHECK:           [[ALLOC8:%.*]] = memref.alloc() : memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC7]] [0, 0, 0, 0] [1, 1, 128, 512] : memref<1x1x256x512xf16, @DDR> to memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT0:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_TOPK]]#0 as %arg3: memref<1x1x128x512xf16, @CMX_NN>) outputs([[SUBVIEW2]] as %arg4: memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x128x512xf16, @CMX_NN>) outputs(%arg4 : memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           }
    // CHECK:           [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC8]] [0, 0, 0, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT1:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_TOPK]]#1 as %arg3: memref<1x1x128x512xsi32, @CMX_NN>) outputs([[SUBVIEW3]] as %arg4: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x128x512xsi32, @CMX_NN>) outputs(%arg4 : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           }
    // CHECK:           [[SUBVIEW4:%.*]] = VPUIP.SubView [[ALLOC7]] [0, 0, 128, 0] [1, 1, 128, 512] : memref<1x1x256x512xf16, @DDR> to memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT2:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_TOPK]]#2 as %arg3: memref<1x1x128x512xf16, @CMX_NN>) outputs([[SUBVIEW4]] as %arg4: memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x128x512xf16, @CMX_NN>) outputs(%arg4 : memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           }
    // CHECK:           [[SUBVIEW5:%.*]] = VPUIP.SubView [[ALLOC8]] [0, 0, 128, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT3:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_TOPK]]#3 as %arg3: memref<1x1x128x512xsi32, @CMX_NN>) outputs([[SUBVIEW5]] as %arg4: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x128x512xsi32, @CMX_NN>) outputs(%arg4 : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           }
    // CHECK:           [[CONCAT0:%.*]] = VPUIP.ConcatView inputs([[TILING_OUTPUT0]], [[TILING_OUTPUT2]] : memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>, memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) outputs([[ALLOC7]] : memref<1x1x256x512xf16, @DDR>) -> memref<1x1x256x512xf16, @DDR>
    // CHECK:           [[ALLOC9:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_OUTPUT4:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT0]] as %arg3: memref<1x1x256x512xf16, @DDR>) outputs([[ALLOC9]] as %arg4: memref<1x1x256x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x256x512xf16, @DDR>) outputs(%arg4 : memref<1x1x256x512xf16, @CMX_NN>) -> memref<1x1x256x512xf16, @CMX_NN>
    // CHECK:           }
    // CHECK:           [[CONCAT1:%.*]] = VPUIP.ConcatView inputs([[TILING_OUTPUT1]], [[TILING_OUTPUT3]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>, memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) outputs([[ALLOC8]] : memref<1x1x256x512xsi32, @DDR>) -> memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[ALLOC10:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_OUTPUT5:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT1]] as %arg3: memref<1x1x256x512xsi32, @DDR>) outputs([[ALLOC10]] as %arg4: memref<1x1x256x512xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x256x512xsi32, @DDR>) outputs(%arg4 : memref<1x1x256x512xsi32, @CMX_NN>) -> memref<1x1x256x512xsi32, @CMX_NN>
    // CHECK:           }
    // CHECK:           [[ALLOC11:%.*]] = memref.alloc() : memref<1x1x256x512xf16>
    // CHECK:           [[TILING_OUTPUT6:%.*]] = VPUIP.NCEClusterTiling inputs([[TILING_OUTPUT4]] as %arg3: memref<1x1x256x512xf16, @CMX_NN>) outputs([[ALLOC11]] as %arg4: memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x256x512xf16, @CMX_NN>) outputs(%arg4 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    // CHECK:           }
    // CHECK:           [[ALLOC12:%.*]] = memref.alloc() : memref<1x1x256x512xsi32>
    // CHECK:           [[TILING_OUTPUT7:%.*]] = VPUIP.NCEClusterTiling inputs([[TILING_OUTPUT5]] as %arg3: memref<1x1x256x512xsi32, @CMX_NN>) outputs([[ALLOC12]] as %arg4: memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32> {
    // CHECK:               VPUIP.Copy inputs(%arg3 : memref<1x1x256x512xsi32, @CMX_NN>) outputs(%arg4 : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
    // CHECK:           }
    // CHECK:           [[COPY_OUT0:%.*]] = VPUIP.Copy inputs([[TILING_OUTPUT6]] : memref<1x1x256x512xf16>) outputs(%arg1 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    // CHECK:           [[COPY_OUT1:%.*]] = VPUIP.Copy inputs([[TILING_OUTPUT7]] : memref<1x1x256x512xsi32>) outputs(%arg2 : memref<1x1x256x512xsi32>) -> memref<1x1x256x512xsi32>
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
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x5x256x512xf16>) outputs(%2 as %arg2: memref<1x5x256x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %11 = VPUIP.Copy inputs(%arg1 : memref<1x5x256x512xf16>) outputs(%arg2 : memref<1x5x256x512xf16, @CMX_NN>) -> memref<1x5x256x512xf16, @CMX_NN>
    }

    %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %13 = VPUIP.NCEClusterTiling inputs(%cst as %arg3: memref<1x1x1x80xui8>) outputs(%2 as %arg4: memref<1x1x1x80xui8, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x80xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80], [1, 1, 1, 80]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
      %11 = VPUIP.Copy inputs(%arg3 : memref<1x1x1x80xui8>) outputs(%arg4 : memref<1x1x1x80xui8, @CMX_NN>) -> memref<1x1x1x80xui8, @CMX_NN>
    }

    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %6:2 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x5x256x512xf16, @CMX_NN>, %13 as %arg7: memref<1x1x1x80xui8, @CMX_NN>) outputs(%4 as %arg2: memref<1x1x256x512xf16, @CMX_NN>, %5 as %arg3: memref<1x1x256x512xsi32, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
      %results_0:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%arg1 as %arg4: memref<1x5x256x512xf16, @CMX_NN>, %arg7 as %arg8: memref<1x1x1x80xui8, @CMX_NN>) outputs(%arg2 as %arg5: memref<1x1x256x512xf16, @CMX_NN>, %arg3 as %arg6: memref<1x1x256x512xsi32, @CMX_NN>) on tile 0 -> (memref<1x1x256x512xf16, @CMX_NN>, memref<1x1x256x512xsi32, @CMX_NN>){
        VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg4, %arg8, %arg5, %arg6) : memref<1x5x256x512xf16, @CMX_NN>, memref<1x1x1x80xui8, @CMX_NN>, memref<1x1x256x512xf16, @CMX_NN>, memref<1x1x256x512xsi32, @CMX_NN>
      }
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTiling inputs(%6#1 as %arg1: memref<1x1x256x512xsi32, @CMX_NN>) outputs(%7 as %arg2: memref<1x1x256x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results_0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg1 as %arg3: memref<1x1x256x512xsi32, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x1x256x512xf16, @CMX_NN>) on tile 0 -> memref<1x1x256x512xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x256x512xsi32, @CMX_NN>, memref<1x1x256x512xf16, @CMX_NN>
      }
    }
    %9 = memref.alloc() : memref<1x1x256x512xf16>
    %10 = VPUIP.NCEClusterTiling inputs(%8 as %arg1: memref<1x1x256x512xf16, @CMX_NN>) outputs(%9 as %arg2: memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16> {
      %11 = VPUIP.Copy inputs(%arg1 : memref<1x1x256x512xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    }
    return %10 : memref<1x1x256x512xf16>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 40]>]
    // CHECK:               [[CST0:%.*]] = const.Declare memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}> = dense<0> : tensor<1x1x1x80xui8>, [#const.SubView<[0, 0, 0, 40], [1, 1, 1, 40]>]
    // CHECK:               [[ALLOC0:%.*]] = memref.alloc() : memref<1x5x256x512xf16>
    // CHECK:               [[COPY0:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x5x256x512xf16>) outputs([[ALLOC0]] : memref<1x5x256x512xf16>) -> memref<1x5x256x512xf16>
    // CHECK:               [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 128, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:               [[ALLOC1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:               [[TILING_INPUT0:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg1: memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs([[ALLOC1]] as %arg2: memref<1x5x128x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                 VPUIP.Copy inputs(%arg1 : memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs(%arg2 : memref<1x5x128x512xf16, @CMX_NN>) -> memref<1x5x128x512xf16, @CMX_NN>
    // CHECK:               }
    // CHECK:               [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 5, 128, 512] : memref<1x5x256x512xf16> to memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>
    // CHECK:               [[ALLOC2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:               [[TILING_INPUT1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs([[ALLOC2]] as %arg2: memref<1x5x128x512xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                 VPUIP.Copy inputs(%arg1 : memref<1x5x128x512xf16, {order = #NCHW, strides = [655360, 131072, 512, 1]}>) outputs(%arg2 : memref<1x5x128x512xf16, @CMX_NN>) -> memref<1x5x128x512xf16, @CMX_NN>
    // CHECK:               }

    // CHECK:               [[ALLOC11:%.*]] = VPURT.AllocDistributed 
    // CHECK:                -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    
    // CHECK:               [[TILING_AUX_INPUT_0:%.*]] = VPUIP.NCEClusterTiling inputs([[CST0]] as %arg1: memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs([[ALLOC11]] as %arg2: memref<1x1x1x40xui8, @CMX_NN>) ->
    // CHECK{LITERAL}:      !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, 
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:                 VPUIP.Copy inputs(%arg1 : memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs(%arg2 : memref<1x1x1x40xui8, @CMX_NN>) -> memref<1x1x1x40xui8, @CMX_NN>
    // CHECK:               }

    // CHECK:               [[ALLOC12:%.*]] = VPURT.AllocDistributed 
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    
    // CHECK:               [[TILING_AUX_INPUT_1:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]] as %arg1: memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs([[ALLOC12]] as %arg2: memref<1x1x1x40xui8, @CMX_NN>) ->
    // CHECK{LITERAL}:      !VPUIP.DistributedBuffer<1x1x1x40xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, 
    // CHECK{LITERAL}:      compute_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
    // CHECK{LITERAL}:      memory_shapes = [[1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40], [1, 1, 1, 40]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:                 VPUIP.Copy inputs(%arg1 : memref<1x1x1x40xui8, {order = #NCHW, strides = [80, 80, 80, 1]}>) outputs(%arg2 : memref<1x1x1x40xui8, @CMX_NN>) -> memref<1x1x1x40xui8, @CMX_NN>
    // CHECK:               }

    // CHECK:           [[ALLOC3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC5:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC6:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CLUSTER_TOPK:%.*]]:4 = VPUIP.NCEClusterTiling 
    // CHECK:           inputs([[TILING_INPUT1]] as %arg1: memref<1x5x128x512xf16, @CMX_NN>,
    // CHECK:           [[TILING_AUX_INPUT_1]] as %arg2: memref<1x1x1x40xui8, @CMX_NN>,
    // CHECK:           [[TILING_INPUT0]] as %arg3: memref<1x5x128x512xf16, @CMX_NN>,
    // CHECK:           [[TILING_AUX_INPUT_0]] as %arg4: memref<1x1x1x40xui8, @CMX_NN>) 
    // CHECK:           outputs([[ALLOC4]] as %arg5: memref<1x1x128x512xf16, @CMX_NN>,
    // CHECK:           [[ALLOC6]] as %arg6: memref<1x1x128x512xsi32, @CMX_NN>,
    // CHECK:           [[ALLOC3]] as %arg7: memref<1x1x128x512xf16, @CMX_NN>, 
    // CHECK:           [[ALLOC5]] as %arg8: memref<1x1x128x512xsi32, @CMX_NN>) 
    // CHECK:           -> (!VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:           %results:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK 
    // CHECK:             inputs(%arg1 as %arg9: memref<1x5x128x512xf16, @CMX_NN>, 
    // CHECK:             %arg2 as %arg10: memref<1x1x1x40xui8, @CMX_NN>, 
    // CHECK:             %arg3 as %arg11: memref<1x5x128x512xf16, @CMX_NN>, 
    // CHECK:             %arg4 as %arg12: memref<1x1x1x40xui8, @CMX_NN>) 
    // CHECK:             outputs(%arg5 as %arg13: memref<1x1x128x512xf16, @CMX_NN>, 
    // CHECK:             %arg6 as %arg14: memref<1x1x128x512xsi32, @CMX_NN>, 
    // CHECK:             %arg7 as %arg15: memref<1x1x128x512xf16, @CMX_NN>, 
    // CHECK{LITERAL}:    %arg8 as %arg16: memref<1x1x128x512xsi32, @CMX_NN>) on tile 0 
    // CHECK:             -> (memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>, memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg9, %arg10, %arg13, %arg14) : 
    // CHECK:              memref<1x5x128x512xf16, @CMX_NN>, memref<1x1x1x40xui8, @CMX_NN>, memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = [2, 0, 2, 1]}(%arg11, %arg12, %arg15, %arg16) : 
    // CHECK:              memref<1x5x128x512xf16, @CMX_NN>, memref<1x1x1x40xui8, @CMX_NN>, memref<1x1x128x512xf16, @CMX_NN>, memref<1x1x128x512xsi32, @CMX_NN>
    // CHECK:               }
    // CHECK:           }
    // CHECK:           [[ALLOC7:%.*]] = memref.alloc() : memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC7]] [0, 0, 0, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT0:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_TOPK]]#1 as %arg1: memref<1x1x128x512xsi32, @CMX_NN>) outputs([[SUBVIEW2]] as %arg2: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x128x512xsi32, @CMX_NN>) outputs(%arg2 : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           }
    // CHECK:           [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC7]] [0, 0, 128, 0] [1, 1, 128, 512] : memref<1x1x256x512xsi32, @DDR> to memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           [[TILING_OUTPUT1:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_TOPK]]#3 as %arg1: memref<1x1x128x512xsi32, @CMX_NN>) outputs([[SUBVIEW3]] as %arg2: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x128x512xsi32, @CMX_NN>) outputs(%arg2 : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) -> memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>
    // CHECK:           }
    // CHECK:           [[CONCAT0:%.*]] = VPUIP.ConcatView inputs([[TILING_OUTPUT0]], [[TILING_OUTPUT1]] : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>, memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @DDR>) outputs([[ALLOC7]] : memref<1x1x256x512xsi32, @DDR>) -> memref<1x1x256x512xsi32, @DDR>
    // CHECK:           [[ALLOC8:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[TILING_OUTPUT2:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT0]] as %arg1: memref<1x1x256x512xsi32, @DDR>) outputs([[ALLOC8]] as %arg2: memref<1x1x256x512xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x256x512xsi32, @DDR>) outputs(%arg2 : memref<1x1x256x512xsi32, @CMX_NN>) -> memref<1x1x256x512xsi32, @CMX_NN>
    // CHECK:           }
    // CHECK:           [[SUBVIEW4:%.*]] = VPUIP.SubView [[TILING_OUTPUT2]] [0, 0, 128, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW5:%.*]] = VPUIP.SubView [[TILING_OUTPUT2]] [0, 0, 0, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC9:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW6:%.*]] = VPUIP.SubView [[ALLOC9]] [0, 0, 128, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SUBVIEW7:%.*]] = VPUIP.SubView [[ALLOC9]] [0, 0, 0, 0] [1, 1, 128, 512] : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[CLUSTER_CONVERT:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW5]] as %arg1: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, [[SUBVIEW4]] as %arg2: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>) outputs([[SUBVIEW7]] as %arg3: memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, [[SUBVIEW6]] as %arg4: memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg1 as %arg5: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>) strides([[65536, 65536, 512, 1], [65536, 65536, 512, 1]]) on tile 0 -> (memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>){
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>
    // CHECK:                   VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x1x128x512xsi32, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>, memref<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:           }
    // CHECK:           [[CONCAT1:%.*]] = VPUIP.ConcatView inputs([[CLUSTER_CONVERT]]#0, [[CLUSTER_CONVERT]]#1 : !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x128x512xf16, {order = #NCHW, strides = [131072, 131072, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[ALLOC9]] : !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[ALLOC10:%.*]] = memref.alloc() : memref<1x1x256x512xf16>
    // CHECK:           [[TILING_OUTPUT3:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT1]] as %arg1: memref<1x1x256x512xf16, @CMX_NN>) outputs([[ALLOC10]] as %arg2: memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16> {
    // CHECK:               VPUIP.Copy inputs(%arg1 : memref<1x1x256x512xf16, @CMX_NN>) outputs(%arg2 : memref<1x1x256x512xf16>) -> memref<1x1x256x512xf16>
    // CHECK:           }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.*]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.*]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.*]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.*]]  = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[SIGMOID:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                           %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>)
    // CHEKC-SAME{LITERAL}:                outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[131072, 1024, 32, 1], [131072, 1024, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]]   = VPUIP.ConcatView inputs([[SIGMOID]]#0, [[SIGMOID]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.*]]  = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Log inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:     [[CMX_IN_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[COPY_CMX:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as {{[^:]+}}: memref<1x128x64x32xf16>) outputs([[CMX_IN_BUF]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:          VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16>) outputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:     [[PERMUTE_IN:%.+]] = VPUIP.PermuteCast {dst_order = #NHCW, mem_perm = #NHCW}
    // CHECK-SAME:          inputs([[COPY_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SUBVIEW_IN0:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_IN1:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CMX_OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CLUSTER_LOG:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN1]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>,
    // CHECK-SAME:                 [[SUBVIEW_IN0]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[SUBVIEW_OUT1]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>,
    // CHECK-SAME:                  [[SUBVIEW_OUT0]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:           %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Log
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sqrt inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:     [[CMX_IN_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[COPY_CMX:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as {{[^:]+}}: memref<1x128x64x32xf16>) outputs([[CMX_IN_BUF]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:          VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16>) outputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:     [[PERMUTE_IN:%.+]] = VPUIP.PermuteCast {dst_order = #NHCW, mem_perm = #NHCW}
    // CHECK-SAME:          inputs([[COPY_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SUBVIEW_IN0:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_IN1:%.+]] = VPUIP.SubView [[PERMUTE_IN]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CMX_OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 32, 0] [1, 128, 32, 32]
    // CHECK:     [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[CMX_OUT_BUF]] [0, 0, 0, 0] [1, 128, 32, 32]

    // CHECK:     [[CLUSTER_LOG:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN1]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>,
    // CHECK-SAME:                 [[SUBVIEW_IN0]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[SUBVIEW_OUT1]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>,
    // CHECK-SAME:                  [[SUBVIEW_OUT0]] as {{[^:]+}}: memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:           %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sqrt
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>, memref<1x128x32x32xf16, {order = #NHCW, strides = [262144, 32, 4096, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x12x270xf16, #NHWC>) outputs(%0 as %arg2: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x12x270xf16, #NHWC>) outputs(%arg2 : memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> memref<1x128x12x270xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg3: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%arg2 as %arg4: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg4, %arg5) : memref<1x128x12x270xf16, #NHWC, @CMX_NN>, memref<1x8x48x1080xf16, #NHWC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg2: memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    }
    return %6: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:                [[INPUT_BUF:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME{LITERAL}:      !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[CLUSTER_COPY_INPUT:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x12x270xf16, #NHWC>) outputs([[INPUT_BUF]] as %arg2: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                    VPUIP.Copy inputs(%arg1 : memref<1x128x12x270xf16, #NHWC>) outputs(%arg2 : memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> memref<1x128x12x270xf16, #NHWC, @CMX_NN>
    // CHECK:                }
    // CHECK:                [[SUBVIEW0:%.*]] = VPUIP.SubView [[CLUSTER_COPY_INPUT]] [0, 0, 6, 0] [1, 128, 6, 270] : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                [[SUBVIEW1:%.*]] = VPUIP.SubView [[CLUSTER_COPY_INPUT]] [0, 0, 0, 0] [1, 128, 6, 270] : !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:                [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    // CHECK:                [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 24, 0] [1, 8, 24, 1080] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> to !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    // CHECK:                [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 8, 24, 1080] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> to !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

    // CHECK:                [[CLUSTER_D2S:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:               inputs([[SUBVIEW1]] as %arg1: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN>,
    // CHECK-SAME:                      [[SUBVIEW0]] as %arg2: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN>)
    // CHECK-SAME:               outputs([[SUBVIEW3]] as %arg3: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>,
    // CHECK-SAME:                       [[SUBVIEW2]] as %arg4: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>)
    // CHECK-SAME:               -> (!VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>) {
    // CHECK:                    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DepthToSpace
    // CHECK-SAME:                   inputs(%arg1 as %arg5: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN>,
    // CHECK-SAME:                          %arg2 as %arg6: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN>)
    // CHECK-SAME:                   outputs(%arg3 as %arg7: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>,
    // CHECK-SAME{LITERAL}:                  %arg4 as %arg8: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>) strides([[207360, 1, 8640, 8], [207360, 1, 8640, 8]]) on tile 0
    // CHECK-SAME:                   -> (memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>,
    // CHECK-SAME:                       memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg5, %arg7) : memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg6, %arg8) : memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, @CMX_NN>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN>
    // CHECK:                    }
    // CHECK:                }

    // CHECK:                [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:               inputs([[CLUSTER_D2S]]#0, [[CLUSTER_D2S]]#1 :
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
    // CHECK-SAME:               outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>)
    // CHECK-SAME:                   -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    // CHECK:                [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:                [[CLUSTER_COPY_OUTPUT:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC> {
    // CHECK:                    VPUIP.Copy inputs(%arg1 : memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x12x270xf16, #NHWC>) outputs(%0 as %arg2: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x12x270xf16, #NHWC>) outputs(%arg2 : memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> memref<1x128x12x270xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg3: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%arg2 as %arg4: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [4, 0]}(%arg4, %arg5) : memref<1x128x12x270xf16, #NHWC, @CMX_NN>, memref<1x8x48x1080xf16, #NHWC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg2: memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    }
    return %6: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:    [[INPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x12x270xf16, #NHWC>) outputs([[INPUT_BUF]] as %arg2: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x12x270xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             VPUIP.Copy inputs(%arg1 : memref<1x128x12x270xf16, #NHWC>) outputs(%arg2 : memref<1x128x12x270xf16, #NHWC, @CMX_NN>) -> memref<1x128x12x270xf16, #NHWC, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}>
    // CHECK:    [[CLUSTER_D2S:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYIN]] as %arg1: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_BUF]] as %arg2: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x8x48x1080xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 4, 1]}> {
    // CHECK:           %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%arg1 as %arg3: memref<1x128x12x270xf16, #NHWC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, @CMX_NN>{
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [4, 0]}(%arg3, %arg4) : memref<1x128x12x270xf16, #NHWC, @CMX_NN>, memref<1x8x48x1080xf16, #NHWC, @CMX_NN>
    // CHECK:           }
    // CHECK:    }
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    [[COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_D2S]] as %arg1: memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC> {
    // CHECK:             VPUIP.Copy inputs(%arg1 : memref<1x8x48x1080xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    }
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x5x34x60xf16, #NHWC>) outputs(%0 as %arg2: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x5x34x60xf16, #NHWC>) outputs(%arg2 : memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> memref<1x5x34x60xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg3: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg4: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Clamp inputs(%arg3 as %arg5: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%arg4 as %arg6: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x5x34x60xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg5, %arg6) : memref<1x5x34x60xf16, #NHWC, @CMX_NN>, memref<1x5x34x60xf16, #NHWC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg7: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg8: memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC> {
      %7 = VPUIP.Copy inputs(%arg7 : memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%arg8 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    }
    return %6: memref<1x5x34x60xf16, #NHWC>

    // CHECK:           [[INPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                     VPUIP.Copy
    // CHECK:               }
    // CHECK:           [[SHAPE_CAST_IN:%.+]] = VPUIP.ShapeCast {shape = [1, 1, 10200, 1]} inputs([[INPUT_COPY]] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[SHAPE_CAST_IN]] [0, 0, 5100, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[SHAPE_CAST_IN]] [0, 0, 0, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-DAG:       [[OUTPUT_BUF_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 5100, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-DAG:       [[OUTPUT_BUF_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 1, 5100, 1] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[CLUSTER_CLAMP:%.*]]:2 =  VPUIP.NCEClusterTiling
    // CHECK-SAME-LITERAL:        inputs([[INPUT_SUBVIEW1]] as [[CLUSTER_CLAMP_ARG1:%.+]]: memref<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME-LITERAL:               [[INPUT_SUBVIEW0]] as [[CLUSTER_CLAMP_ARG2:%.+]]: memref<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME-LITERAL:        outputs([[OUTPUT_BUF_SUBVIEW1]] as [[CLUSTER_CLAMP_ARG3:%.+]]: memref<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN>,
    // CHECK-SAME-LITERAL:               [[OUTPUT_BUF_SUBVIEW0]] as [[CLUSTER_CLAMP_ARG4:%.+]]: memref<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME-LITERAL:                -> !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME-LITERAL:                   !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Clamp
    // CHECK-SAME:                      -> (memref<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN>, memref<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}
    // CHECK-SAME-LITERAL:                      memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>, memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}
    // CHECK-SAME-LITERAL:                      memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>, memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>
    // CHECK:                     }
    // CHECK:               }

    // CHECK:           [[CONCAT_VIEW:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:                inputs([[CLUSTER_CLAMP]]#0, [[CLUSTER_CLAMP]]#1 :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x1x5100x1xf16, {order = #NHWC, strides = [10200, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SHAPE_CAST_OUT:%.+]] = VPUIP.ShapeCast {shape = [1, 5, 34, 60]} inputs([[CONCAT_VIEW]] : !VPUIP.DistributedBuffer<1x1x10200x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:           [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK:                         VPUIP.Copy
    // CHECK:              }
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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Divide
                    inputs(%arg0 as %arg3: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x1x88xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x64x1x88xf16, #NCHW, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x1x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x1x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x1x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16>
    }

    return %5: memref<1x64x1x88xf16>

    // For Divide First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // For Divide Second Input
    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[DIVIDE_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[DIVIDE_OUT_TILE0:%.*]] = VPUIP.SubView [[DIVIDE_OUT]] [0, 32, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[DIVIDE_OUT_TILE1:%.*]] = VPUIP.SubView [[DIVIDE_OUT]] [0, 0, 0, 0] [1, 32, 1, 88]
    // CHECK:    [[DIVIDE:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs([[DIVIDE_OUT_TILE1]] as %arg4: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[DIVIDE_OUT_TILE0]] as %arg5: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Divide
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[DIVIDE]]#0, [[DIVIDE]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[DIVIDE_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x1x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x1x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Power
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Power First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Power Second Input
    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[POWER_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[POWER_OUT_TILE0:%.*]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER_OUT_TILE1:%.*]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[POWER_OUT_TILE1]] as %arg4: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[POWER_OUT_TILE0]] as %arg5: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Power
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[POWER]]#0, [[POWER]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[POWER_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                       %1 as %arg1: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Power
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Power First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Power Second Input
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[POWER_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[POWER_OUT_TILE0:%.*]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER_OUT_TILE1:%.*]] = VPUIP.SubView [[POWER_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[POWER:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[ARG1:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG3:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[POWER_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[POWER_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Power
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[POWER]]#0, [[POWER]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[POWER_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as {{%.+}}: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as {{%.+}}: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Greater
                    inputs(%arg0 as %arg3: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x1x88xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x64x1x88xf16, #NCHW, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x1x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x1x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x1x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16>
    }

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
    // CHECK:    [[GREATER:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs([[GREATER_OUT_TILE1]] as %arg4: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[GREATER_OUT_TILE0]] as %arg5: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Greater
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[GREATER]]#0, [[GREATER]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[GREATER_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x1x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Less
                    inputs(%arg0 as %arg3: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x1x88xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x64x1x88xf16, #NCHW, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x1x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x1x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x1x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16>
    }

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
    // CHECK:    [[LESS:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs([[LESS_OUT_TILE1]] as %arg4: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[LESS_OUT_TILE0]] as %arg5: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Less
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[LESS]]#0, [[LESS]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[LESS_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x1x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x1x88xf16>) -> memref<1x64x1x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x1x88xi8, #NCHW, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal
                    inputs(%arg0 as %arg3: memref<1x64x1x88xf16, #NCHW, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x1x88xf16, #NCHW, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x1x88xi8, #NCHW, @CMX_NN>) on tile 0 -> memref<1x64x1x88xi8, #NCHW, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xf16, #NCHW, @CMX_NN>, memref<1x64x1x88xi8, #NCHW, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x1x88xi8>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x1x88xi8, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x1x88xi8>) -> memref<1x64x1x88xi8> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x1x88xi8, @CMX_NN>) outputs(%arg2 : memref<1x64x1x88xi8>) -> memref<1x64x1x88xi8>
    }

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
    // CHECK:    [[EQUAL:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[INPUT0_TILE1_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[INPUT1_TILE1_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[INPUT0_TILE0_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[INPUT1_TILE0_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs([[EQUAL_OUT_TILE1]] as [[EQUAL_OUT_TILE1_ARG:[^:]+]]: memref<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                             [[EQUAL_OUT_TILE0]] as [[EQUAL_OUT_TILE0_ARG:[^:]+]]: memref<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Equal
    // CHECK:                     inputs([[INPUT0_TILE1_ARG]] as [[INPUT0_TILE1_SW_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1_ARG]] as [[INPUT1_TILE1_SW_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0_ARG]] as [[INPUT0_TILE0_SW_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0_ARG]] as [[INPUT1_TILE0_SW_ARG:[^:]+]]: memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                     outputs([[EQUAL_OUT_TILE1_ARG]] as [[EQUAL_OUT_TILE1_SW_ARG:[^:]+]]: memref<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:                             [[EQUAL_OUT_TILE0_ARG]] as [[EQUAL_OUT_TILE0_SW_ARG:[^:]+]]: memref<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE1_SW_ARG]], [[INPUT1_TILE1_SW_ARG]], [[EQUAL_OUT_TILE1_SW_ARG]]) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[INPUT0_TILE0_SW_ARG]], [[INPUT1_TILE0_SW_ARG]], [[EQUAL_OUT_TILE0_SW_ARG]]) : memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xf16, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>, memref<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[EQUAL]]#0, [[EQUAL]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x32x1x88xi8, {order = #NCHW, strides = [5632, 88, 88, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[EQUAL_OUT]] : !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x1x88xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x64x1x88xi8>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as [[INPUT0_TILE1_ARG]]: memref<1x64x1x88xi8, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as [[INPUT1_TILE1_ARG]]: memref<1x64x1x88xi8>) -> memref<1x64x1x88xi8>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Subtract First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Subtract Second Input
    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[SUBTRACT:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as %arg4: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as %arg5: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Subtract
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[SUBTRACT]]#0, [[SUBTRACT]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                       %1 as %arg1: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Subtract First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Subtract Second Input
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[SUBTRACT:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[ARG1:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG3:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Subtract
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[SUBTRACT]]#0, [[SUBTRACT]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as {{%.+}}: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as {{%.+}}: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %0 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%1 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %3 = memref.alloc() : memref<1x64x88x88xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %4: memref<1x64x88x88xf16>

    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[SUBTRACT:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT1_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE1]] as [[ARG1:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG3:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Subtract
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[SUBTRACT]]#0, [[SUBTRACT]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %1 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Add First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Add Second Input
    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[ADD:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as %arg0: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as %arg1: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as %arg2: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as %arg3: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as %arg4: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as %arg5: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK:                     inputs(%arg0 as %arg6: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg1 as %arg7: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg2 as %arg8: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            %arg3 as %arg9: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs(%arg4 as %arg10: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             %arg5 as %arg11: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg7, %arg10) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg11) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[ADD]]#0, [[ADD]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                       %1 as %arg1: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x1x1x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x64x88x88xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %5: memref<1x64x88x88xf16>

    // For Add First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For Add Second Input
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 1, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x1x88xf16, {order = #NHWC, strides = [88, 1, 88, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[ADD:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[ARG1:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG3:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x1x1x88xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x1x1x88xf16, #NHWC, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[ADD]]#0, [[ADD]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as {{%.+}}: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as {{%.+}}: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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

    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                                          %0 as %arg1: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                  outputs(%1 as %arg2: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
                    inputs(%arg0 as %arg3: memref<1x64x88x88xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x64x88x88xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x64x88x88xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x64x88x88xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>, memref<1x64x88x88xf16, #NHWC, @CMX_NN>
      }
    }

    %3 = memref.alloc() : memref<1x64x88x88xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg1: memref<1x64x88x88xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x88x88xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>
    }

    return %4: memref<1x64x88x88xf16>

    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:         !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MULTI_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[MULTI_OUT_TILE0:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 44, 0] [1, 64, 44, 88]
    // CHECK:    [[MULTI_OUT_TILE1:%.*]] = VPUIP.SubView [[MULTI_OUT]] [0, 0, 0, 0] [1, 64, 44, 88]
    // CHECK:    [[ADD:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT1_TILE1]] as [[ARG0:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE1]] as [[ARG1:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG2:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG3:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[MULTI_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[MULTI_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>, memref<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[ADD]]#0, [[ADD]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK:                     !VPUIP.DistributedBuffer<1x64x44x88xf16, {order = #NHWC, strides = [495616, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[MULTI_OUT]] : !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x64x88x88xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x64x88x88xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as %arg0: memref<1x64x88x88xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as %arg1: memref<1x64x88x88xf16>) -> memref<1x64x88x88xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x5x34x60xf16, #NHWC>) outputs(%0 as %arg2: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x5x34x60xf16, #NHWC>) outputs(%arg2 : memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> memref<1x5x34x60xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg3: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg4: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Abs inputs(%arg3 as %arg5: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%arg4 as %arg6: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x5x34x60xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run (%arg5, %arg6) : memref<1x5x34x60xf16, #NHWC, @CMX_NN>, memref<1x5x34x60xf16, #NHWC, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg7: memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg8: memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC> {
      %7 = VPUIP.Copy inputs(%arg7 : memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%arg8 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    }
    return %6: memref<1x5x34x60xf16, #NHWC>

    // CHECK:           [[INPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs(%arg0 as %arg1: memref<1x5x34x60xf16, #NHWC>)
    // CHECK-SAME:                outputs([[INPUT_BUF]] as %arg2: memref<1x5x34x60xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                     VPUIP.Copy inputs(%arg1 : memref<1x5x34x60xf16, #NHWC>) outputs(%arg2 : memref<1x5x34x60xf16, #NHWC, @CMX_NN>) -> memref<1x5x34x60xf16, #NHWC, @CMX_NN>
    // CHECK:               }
    // CHECK:           [[INPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 18, 0] [1, 5, 16, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[INPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 5, 18, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_BUF_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 18, 0] [1, 5, 16, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_BUF_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 5, 18, 60] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:           [[CLUSTER_ABS:%.*]]:2 =  VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs([[INPUT_SUBVIEW1]] as %arg1: memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>,
    // CHECK-SAME:                       [[INPUT_SUBVIEW0]] as %arg2: memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>)
    // CHECK-SAME:                outputs([[OUTPUT_BUF_SUBVIEW1]] as %arg3: memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>,
    // CHECK-SAME:                        [[OUTPUT_BUF_SUBVIEW0]] as %arg4: memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>)
    // CHECK-SAME:                -> (!VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Abs
    // CHECK-SAME:                      inputs(%arg1 as %arg5: memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>,
    // CHECK-SAME:                             %arg2 as %arg6: memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>)
    // CHECK-SAME:                      outputs(%arg3 as %arg7: memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>,
    // CHECK-SAME{LITERAL}:                     %arg4 as %arg8: memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>) strides([[5100, 1, 300, 5], [5100, 1, 300, 5]]) on tile 0
    // CHECK-SAME:                      -> (memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>, memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}(%arg5, %arg7) : memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>, memref<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = []}(%arg6, %arg8) : memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>, memref<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN>
    // CHECK:                     }
    // CHECK:               }

    // CHECK:           [[CONCAT_VIEW:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:                inputs([[CLUSTER_ABS]]#0, [[CLUSTER_ABS]]#1 :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x5x18x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x5x16x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x5x34x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:           [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs([[CONCAT_VIEW]] as %arg1: memref<1x5x34x60xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                outputs([[OUTPUT_DDR]] as %arg2: memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x5x34x60xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    // CHECK:              }
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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x18x160x288xf16, #NHWC, @CMX_NN>,
                                       %1 as %arg1: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg2: memref<1x18x160x288xf16, #NHWC, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
                    inputs(%arg0 as %arg3: memref<1x18x160x288xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x18x160x288xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x18x160x288xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x18x160x288xf16, #NHWC, @CMX_NN>, memref<1x18x1x1xf16, #NHWC, @CMX_NN>, memref<1x18x160x288xf16, #NHWC, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x18x160x288xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x18x160x288xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x18x160x288xf16>) -> memref<1x18x160x288xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x18x160x288xf16, @CMX_NN>) outputs(%arg2 : memref<1x18x160x288xf16>) -> memref<1x18x160x288xf16>
    }

    return %5: memref<1x18x160x288xf16>

    // For PRelu First Input
    // CHECK:    [[INPUT0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 80, 0] [1, 18, 80, 288]
    // CHECK:         !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.*]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 18, 80, 288]
    // CHECK:         !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For PRelu Second Input
    // CHECK:    [[INPUT1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 18, 1, 1]
    // CHECK:         !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x18x1x1xf16, {order = #NHWC, strides = [18, 1, 18, 18]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.*]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 18, 1, 1]
    // CHECK:         !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x18x1x1xf16, {order = #NHWC, strides = [18, 1, 18, 18]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:    [[PRELU_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[PRELU_OUT_TILE0:%.*]] = VPUIP.SubView [[PRELU_OUT]] [0, 0, 80, 0] [1, 18, 80, 288]
    // CHECK:    [[PRELU_OUT_TILE1:%.*]] = VPUIP.SubView [[PRELU_OUT]] [0, 0, 0, 0] [1, 18, 80, 288]
    // CHECK:    [[PRELU:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE1]] as [[ARG1:[^:]+]]: memref<1x18x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG3:[^:]+]]: memref<1x18x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[PRELU_OUT_TILE1]] as [[ARG4:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                            [[PRELU_OUT_TILE0]] as [[ARG5:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PRelu
    // CHECK:                     inputs([[ARG0]] as [[ARG6:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                            [[ARG1]] as [[ARG7:[^:]+]]: memref<1x18x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:                            [[ARG2]] as [[ARG8:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                            [[ARG3]] as [[ARG9:[^:]+]]: memref<1x18x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:                     outputs([[ARG4]] as [[ARG10:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:                             [[ARG5]] as [[ARG11:[^:]+]]: memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG6]], [[ARG7]], [[ARG10]]) : memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>, memref<1x18x1x1xf16, #NHWC, @CMX_NN>, memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG8]], [[ARG9]], [[ARG11]]) : memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>, memref<1x18x1x1xf16, #NHWC, @CMX_NN>, memref<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN>

    // CHECK:    [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[PRELU]]#0, [[PRELU]]#1
    // CHECK:                     !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     !VPUIP.DistributedBuffer<1x18x80x288xf16, {order = #NHWC, strides = [829440, 1, 5184, 18]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     outputs([[PRELU_OUT]] : !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x18x160x288xf16>
    // CHECK:    [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCATVIEW]] as [[ARG12:[^:]+]]: memref<1x18x160x288xf16, @CMX_NN>)
    // CHECK:                                                 outputs([[OUTPUT_BUF]] as [[ARG13:[^:]+]]: memref<1x18x160x288xf16>) -> memref<1x18x160x288xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x16x16x512xf16, @CMX_NN>)
                                  outputs(%1 as %arg1: memref<1x16x16x512xf16, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Floor
                    inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, @CMX_NN>)
                   outputs(%arg1 as %arg4: memref<1x16x16x512xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg3, %arg4) : memref<1x16x16x512xf16, @CMX_NN>, memref<1x16x16x512xf16, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x16x16x512xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x16x16x512xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x16x16x512xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16>
    }

    return %5 : memref<1x16x16x512xf16>

    // CHECK:    [[IN_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV0:%.*]] = VPUIP.SubView [[IN_BUF]] [0, 8, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.*]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 8, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.*]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 8, 16, 512] : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[FLOOR:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as %arg0: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, [[IN_SV0]] as %arg1: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as %arg2: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, [[OUT_SV0]] as %arg3: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                         %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Floor inputs(%arg0 as %arg4: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, %arg1 as %arg5: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:              outputs(%arg2 as %arg6: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, %arg3 as %arg7: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>) strides([[65536, 4096, 512, 1], [65536, 4096, 512, 1]]) on tile 0 -> (memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>){
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg4, %arg6) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg5, %arg7) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]]   = VPUIP.ConcatView inputs([[FLOOR]]#0, [[FLOOR]]#1 : !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%3 : !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.*]]  = memref.alloc() : memref<1x16x16x512xf16>
    // CHECK:    [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg0: memref<1x16x16x512xf16, @CMX_NN>) outputs([[OUT_DDR]] as %arg1: memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg0 : memref<1x16x16x512xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x16x16x512xf16, @CMX_NN>)
                                  outputs(%1 as %arg1: memref<1x16x16x512xf16, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Round
                    inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, @CMX_NN>)
                   outputs(%arg1 as %arg4: memref<1x16x16x512xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg3, %arg4) : memref<1x16x16x512xf16, @CMX_NN>, memref<1x16x16x512xf16, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x16x16x512xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x16x16x512xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x16x16x512xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16>
    }

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

    // CHECK:     [[CLUSTER_LOG:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN1]] as {{[^:]+}}: memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>,
    // CHECK-SAME:                 [[SUBVIEW_IN0]] as {{[^:]+}}: memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[SUBVIEW_OUT1]] as {{[^:]+}}: memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>,
    // CHECK-SAME:                  [[SUBVIEW_OUT0]] as {{[^:]+}}: memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:           %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Round
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}({{[^:]+}}, {{[^:]+}}) : memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>, memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}({{[^:]+}}, {{[^:]+}}) : memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>, memref<1x16x8x512xf16, {order = #NHCW, strides = [131072, 512, 8192, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sin inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as {{[^:]+}}: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16>) outputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[SIN:%.+]]:2  = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:                %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sin
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[SIN]]#0, [[SIN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as {{[^:]+}}: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) outputs({{[^:]+}} : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Cos inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as {{[^:]+}}: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16>) outputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[COS:%.+]]:2  = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:                %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Cos
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[COS]]#0, [[COS]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as {{[^:]+}}: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) outputs({{[^:]+}} : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Exp inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as {{[^:]+}}: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16>) outputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[EXP:%.+]]:2  = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:                %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Exp
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[EXP]]#0, [[EXP]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as {{[^:]+}}: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) outputs({{[^:]+}} : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

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

    %3 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x16x16x512xf16, @CMX_NN>)
                                  outputs(%1 as %arg1: memref<1x16x16x512xf16, @CMX_NN>)
                                      -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_LogSoftmax
                    inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, @CMX_NN>)
                   outputs(%arg1 as %arg4: memref<1x16x16x512xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [2]}(%arg3, %arg4) : memref<1x16x16x512xf16, @CMX_NN>, memref<1x16x16x512xf16, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x16x16x512xf16>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x16x16x512xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x16x16x512xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x16x512xf16>) -> memref<1x16x16x512xf16>
    }

    return %5 : memref<1x16x16x512xf16>

    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1x16x16x512xf16, @DDR>
    // CHECK:    [[COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT0]] as [[COPY_ARG0:[^:]+]]: memref<1x16x16x512xf16, @CMX_NN>) outputs([[ALLOC0]] as [[COPY_ARG1:[^:]+]]: memref<1x16x16x512xf16, @DDR>) -> memref<1x16x16x512xf16, @DDR> {
    // CHECK:                  VPUIP.Copy inputs([[COPY_ARG0]] : memref<1x16x16x512xf16, @CMX_NN>) outputs([[COPY_ARG1]] : memref<1x16x16x512xf16, @DDR>) -> memref<1x16x16x512xf16, @DDR>
    // CHECK:    }
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 8, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16, @DDR> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>
    // CHECK:    [[BUF0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT0_TILE0]] as [[COPY_ARG0:[^:]+]]: memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>) outputs([[BUF0]] as [[COPY_ARG1:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                  VPUIP.Copy inputs([[COPY_ARG0]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>) outputs([[COPY_ARG1]] : memref<1x16x8x512xf16, @CMX_NN>) -> memref<1x16x8x512xf16, @CMX_NN>
    // CHECK:    }

    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x16x512xf16, @DDR>
    // CHECK:    [[COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT0]] as [[COPY_ARG0:[^:]+]]: memref<1x16x16x512xf16, @CMX_NN>) outputs([[ALLOC1]] as [[COPY_ARG1:[^:]+]]: memref<1x16x16x512xf16, @DDR>) -> memref<1x16x16x512xf16, @DDR> {
    // CHECK:                  VPUIP.Copy inputs([[COPY_ARG0]] : memref<1x16x16x512xf16, @CMX_NN>) outputs([[COPY_ARG1]] : memref<1x16x16x512xf16, @DDR>) -> memref<1x16x16x512xf16, @DDR>
    // CHECK:    }
    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16, @DDR> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>
    // CHECK:    [[BUF1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT1_TILE1]] as [[COPY_ARG0:[^:]+]]: memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>) outputs([[BUF1]] as [[COPY_ARG1:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>)
    // CHECK-SAME:                -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                  VPUIP.Copy inputs([[COPY_ARG0]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, @DDR>) outputs([[COPY_ARG1]] : memref<1x16x8x512xf16, @CMX_NN>) -> memref<1x16x8x512xf16, @CMX_NN>
    // CHECK:    }

    // CHECK:    [[BUF9:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[BUF10:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[CLUSTER_LOG_SOFTMAX:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs([[INPUT1_COPY1]] as [[COPY_ARG0:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>, [[INPUT0_COPY0]] as [[COPY_ARG1:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>)
    // CHECK-SAME:                outputs([[BUF10]] as [[COPY_ARG2:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>, [[BUF9]] as [[COPY_ARG3:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>)
    // CHECK-SAME:                -> (!VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x8x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:         %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LogSoftmax
    // CHECK-SAME:                inputs([[COPY_ARG0]] as [[COPY_ARG4:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>, [[COPY_ARG1]] as [[COPY_ARG5:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>)
    // CHECK-SAME:                outputs([[COPY_ARG2]] as [[COPY_ARG6:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>, [[COPY_ARG3]] as [[COPY_ARG7:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>) on tile 0 -> (memref<1x16x8x512xf16, @CMX_NN>, memref<1x16x8x512xf16, @CMX_NN>){
    // CHECK:                      VPUIP.SW.Kernel.run {attrs = [2]}([[COPY_ARG4]], [[COPY_ARG6]]) : memref<1x16x8x512xf16, @CMX_NN>, memref<1x16x8x512xf16, @CMX_NN>
    // CHECK:                      VPUIP.SW.Kernel.run {attrs = [2]}([[COPY_ARG5]], [[COPY_ARG7]]) : memref<1x16x8x512xf16, @CMX_NN>, memref<1x16x8x512xf16, @CMX_NN>
    // CHECK:         }
    // CHECK:    }

    // CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<1x16x16x512xf16>
    // CHECK:    [[SUBVIEW12:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    [[COPY13:%.+]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_LOG_SOFTMAX]]#0 as [[COPY_ARG0:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>) outputs([[SUBVIEW12]] as [[COPY_ARG1:[^:]+]]: memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>)
    // CHECK-SAME:                -> memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}> {
    // CHECK:                 VPUIP.Copy inputs([[COPY_ARG0]] : memref<1x16x8x512xf16, @CMX_NN>) outputs([[COPY_ARG1]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>) -> memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    }

    // CHECK:    [[SUBVIEW14:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 8, 0] [1, 16, 8, 512] : memref<1x16x16x512xf16> to memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    [[COPY15:%.+]] = VPUIP.NCEClusterTiling inputs([[CLUSTER_LOG_SOFTMAX]]#1 as [[COPY_ARG0:[^:]+]]: memref<1x16x8x512xf16, @CMX_NN>) outputs(%14 as [[COPY_ARG1:[^:]+]]: memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>)
    // CHECK-SAME:                -> memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}> {
    // CHECK:                 VPUIP.Copy inputs([[COPY_ARG0]] : memref<1x16x8x512xf16, @CMX_NN>) outputs([[COPY_ARG1]] : memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>) -> memref<1x16x8x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}>
    // CHECK:    }
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
    %in_copy = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x4096x18x4xf16, #NHWC>) outputs(%in_alloc as %arg3: memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %inner_op = VPUIP.Copy inputs(%arg2 : memref<1x4096x18x4xf16, #NHWC>) outputs(%arg3 : memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) -> memref<1x4096x18x4xf16, #NHWC, @CMX_NN>
    }
    %out_alloc = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %sw = VPUIP.NCEClusterTiling inputs(%in_copy as %arg2: memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) outputs(%out_alloc as %arg3: memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg2 as %arg4: memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x4096x18x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg4, %arg5) : memref<1x4096x18x4xf16, #NHWC, @CMX_NN>, memref<1x4096x18x4xf16, #NHWC, @CMX_NN>
      }
    }
    %res_alloc = memref.alloc() : memref<1x4096x18x4xf16, #NHWC>
    %out_copy = VPUIP.NCEClusterTiling inputs(%sw as %arg2: memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) outputs(%res_alloc as %arg3: memref<1x4096x18x4xf16, #NHWC>) -> memref<1x4096x18x4xf16, #NHWC> {
      %inner_op = VPUIP.Copy inputs(%arg2 : memref<1x4096x18x4xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x4096x18x4xf16, #NHWC>) -> memref<1x4096x18x4xf16, #NHWC>
    }
    return %out_copy : memref<1x4096x18x4xf16, #NHWC>

    // CHECK:   [[IN_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x18x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[IN_COPY:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:   [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME:      shape = [1, 4096, 72, 1]
    // CHECK-SAME:      inputs([[IN_COPY]]
    // CHECK-SAME-LITERAL:      -> !VPUIP.DistributedBuffer<1x4096x72x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-DAG:   [[IN_SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 4096, 36, 1]
    // CHECK-DAG:   [[IN_SUBVIEW_2:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 36, 0] [1, 4096, 36, 1]
    // CHECK:   [[OUT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4096x72x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-DAG:   [[OUT_SUBVIEW_2:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 36, 0] [1, 4096, 36, 1]
    // CHECK-DAG:   [[OUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0, 0] [1, 4096, 36, 1]
    // CHECK:   [[SW_CLUSTERING:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:          [[IN_SUBVIEW_1]] as [[SW_CLUSTERING_ARG1:%.+]]
    // CHECK-SAME-LITERAL:  [[IN_SUBVIEW_2]] as [[SW_CLUSTERING_ARG2:%.+]]
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}
    // CHECK-SAME:      memref<1x4096x36x1xf16, {order = #NHWC, strides = [294912, 1, 4096, 4096]}, @CMX_NN>, memref<1x4096x36x1xf16, {order = #NHWC, strides = [294912, 1, 4096, 4096]}, @CMX_NN>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}
    // CHECK-SAME:      memref<1x4096x36x1xf16, {order = #NHWC, strides = [294912, 1, 4096, 4096]}, @CMX_NN>, memref<1x4096x36x1xf16, {order = #NHWC, strides = [294912, 1, 4096, 4096]}, @CMX_NN>
    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_CLUSTERING]]#0, [[SW_CLUSTERING]]#1
    // CHECK:   [[OUT_COPY:%.+]] = VPUIP.NCEClusterTiling
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Mish inputs(%arg1 as %arg6: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x128x64x32xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run(%arg6, %arg7) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }
    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[IN_BUF:%.+]]  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as {{[^:]+}}: memref<1x128x64x32xf16>) outputs([[IN_BUF]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16>) outputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>

    // CHECK:    [[IN_SV0:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[IN_SV1:%.+]]  = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUT_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV0:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_SV1:%.+]] = VPUIP.SubView [[OUT_BUF]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[MISH:%.+]]:2  = VPUIP.NCEClusterTiling inputs([[IN_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[IN_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[OUT_SV1]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[OUT_SV0]] as {{[^:]+}}: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK{LITERAL}:                %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Mish
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                         VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MISH]]#0, [[MISH]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as {{[^:]+}}: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUT_DDR]] as {{[^:]+}}: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs({{[^:]+}} : memref<1x128x64x32xf16, @CMX_NN>) outputs({{[^:]+}} : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>

    // CHECK:    return [[OUT_COPY]] : memref<1x128x64x32xf16>
}